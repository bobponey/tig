use crate::{context::*, error::*};
use logging_timer::time;
use std::collections::HashMap;
use tig_structs::core::*;
use tig_utils::*;

#[time]
pub(crate) async fn execute<T: Context>(
    ctx: &T,
    player: &Player,
    settings: &BenchmarkSettings,
    solutions_meta_data: &Vec<SolutionMetaData>,
    solution_data: &SolutionData,
) -> ProtocolResult<(String, Result<(), String>)> {
    verify_player_owns_benchmark(player, settings)?;
    verify_block(ctx, &settings.block_id).await?;
    verify_sufficient_lifespan(ctx, &settings.block_id).await?;
    verify_challenge(ctx, &settings.challenge_id, &settings.block_id).await?;
    verify_algorithm(ctx, &settings.algorithm_id, &settings.block_id).await?;
    verify_sufficient_solutions(ctx, &settings.block_id, solutions_meta_data).await?;
    verify_benchmark_settings_are_unique(ctx, settings).await?;
    verify_nonces_are_unique(solutions_meta_data)?;
    verify_solutions_signatures(ctx, &settings.challenge_id, solutions_meta_data).await?;
    verify_benchmark_difficulty(
        ctx,
        &settings.difficulty,
        &settings.challenge_id,
        &settings.block_id,
    )
    .await?;
    let benchmark_id = ctx
        .add_benchmark_to_mempool(
            &settings,
            &BenchmarkDetails {
                block_started: ctx
                    .read_blocks()
                    .await
                    .get(&settings.block_id)
                    .unwrap()
                    .details
                    .height,
                num_solutions: solutions_meta_data.len() as u32,
            },
            solutions_meta_data,
            solution_data,
        )
        .await;
    let mut verified = Ok(());
    if let Err(e) =
        verify_solution_is_valid(ctx, settings, solutions_meta_data, solution_data).await
    {
        ctx.add_fraud_to_mempool(&benchmark_id, &e.to_string())
            .await;
        verified = Err(e.to_string());
    }
    Ok((benchmark_id, verified))
}

#[time]
fn verify_player_owns_benchmark(
    player: &Player,
    settings: &BenchmarkSettings,
) -> ProtocolResult<()> {
    if player.id != settings.player_id {
        return Err(ProtocolError::InvalidSubmittingPlayer {
            actual_player_id: player.id.clone(),
            expected_player_id: settings.player_id.clone(),
        });
    }
    Ok(())
}

#[time]
async fn verify_block<T: Context>(ctx: &T, block_id: &String) -> ProtocolResult<()> {
    if ctx.read_blocks().await.get(block_id).is_none() {
        Err(ProtocolError::InvalidBlock {
            block_id: block_id.clone(),
        })
    } else {
        Ok(())
    }
}

#[time]
async fn verify_sufficient_lifespan<T: Context>(ctx: &T, block_id: &String) -> ProtocolResult<()> {
    let latest_block_id = ctx
        .get_block_id(BlockFilter::Latest)
        .await
        .expect("Expecting latest block to exist");
    let read_blocks = ctx.read_blocks().await;
    let latest_block = read_blocks
        .get(&latest_block_id)
        .expect("Expecting latest block to exist");
    let block = read_blocks.get(block_id).unwrap();
    let config = block.config();
    let submission_delay = latest_block.details.height - block.details.height + 1;
    if submission_delay * (config.benchmark_submissions.submission_delay_multiplier + 1)
        >= config.benchmark_submissions.lifespan_period
    {
        return Err(ProtocolError::InsufficientLifespan);
    }
    Ok(())
}

#[time]
async fn verify_challenge<T: Context>(
    ctx: &T,
    challenge_id: &String,
    block_id: &String,
) -> ProtocolResult<()> {
    if ctx.read_challenges().await.get(challenge_id).is_none() {
        return Err(ProtocolError::InvalidChallenge {
            challenge_id: challenge_id.clone(),
        });
    }
    if !ctx
        .read_blocks()
        .await
        .get(block_id)
        .unwrap()
        .data()
        .active_challenge_ids
        .contains(challenge_id)
    {
        return Err(ProtocolError::InvalidChallenge {
            challenge_id: challenge_id.clone(),
        });
    }
    Ok(())
}

#[time]
async fn verify_algorithm<T: Context>(
    ctx: &T,
    algorithm_id: &String,
    block_id: &String,
) -> ProtocolResult<()> {
    if ctx.read_algorithms().await.get(algorithm_id).is_none() {
        return Err(ProtocolError::InvalidAlgorithm {
            algorithm_id: algorithm_id.clone(),
        });
    }
    if !ctx
        .read_blocks()
        .await
        .get(block_id)
        .unwrap()
        .data()
        .active_algorithm_ids
        .contains(algorithm_id)
    {
        return Err(ProtocolError::InvalidAlgorithm {
            algorithm_id: algorithm_id.clone(),
        });
    }
    Ok(())
}

#[time]
async fn verify_sufficient_solutions<T: Context>(
    ctx: &T,
    block_id: &String,
    solutions_meta_data: &Vec<SolutionMetaData>,
) -> ProtocolResult<()> {
    let min_num_solutions = ctx
        .read_blocks()
        .await
        .get(block_id)
        .unwrap()
        .config()
        .benchmark_submissions
        .min_num_solutions as usize;
    if solutions_meta_data.len() < min_num_solutions {
        return Err(ProtocolError::InsufficientSolutions {
            num_solutions: solutions_meta_data.len(),
            min_num_solutions,
        });
    }
    Ok(())
}

#[time]
async fn verify_benchmark_settings_are_unique<T: Context>(
    ctx: &T,
    settings: &BenchmarkSettings,
) -> ProtocolResult<()> {
    if ctx
        .get_benchmark_ids(BenchmarksFilter::Settings(settings.clone()))
        .await
        .len()
        > 0
    {
        return Err(ProtocolError::DuplicateBenchmarkSettings {
            settings: settings.clone(),
        });
    }

    Ok(())
}

#[time]
fn verify_nonces_are_unique(solutions_meta_data: &Vec<SolutionMetaData>) -> ProtocolResult<()> {
    let nonces: HashMap<u32, u32> =
        solutions_meta_data
            .iter()
            .fold(HashMap::new(), |mut acc, s| {
                *acc.entry(s.nonce).or_insert(0) += 1;
                acc
            });

    if let Some((&nonce, _)) = nonces.iter().find(|(_, &count)| count > 1) {
        return Err(ProtocolError::DuplicateNonce { nonce });
    }

    Ok(())
}

#[time]
async fn verify_solutions_signatures<T: Context>(
    ctx: &T,
    challenge_id: &String,
    solutions_meta_data: &Vec<SolutionMetaData>,
) -> ProtocolResult<()> {
    let solution_signature_threshold = *ctx
        .read_challenges()
        .await
        .get(challenge_id)
        .unwrap()
        .block_data()
        .solution_signature_threshold();
    if let Some(s) = solutions_meta_data
        .iter()
        .find(|&s| s.solution_signature > solution_signature_threshold)
    {
        return Err(ProtocolError::InvalidSolutionSignature {
            nonce: s.nonce,
            solution_signature: s.solution_signature,
            threshold: solution_signature_threshold,
        });
    }

    Ok(())
}

#[time]
async fn verify_benchmark_difficulty<T: Context>(
    ctx: &T,
    difficulty: &Vec<i32>,
    challenge_id: &String,
    block_id: &String,
) -> ProtocolResult<()> {
    let difficulty_parameters = &ctx
        .read_blocks()
        .await
        .get(block_id)
        .unwrap()
        .config()
        .difficulty
        .parameters[challenge_id]
        .clone();

    if difficulty.len() != difficulty_parameters.len()
        || difficulty
            .iter()
            .zip(difficulty_parameters.iter())
            .any(|(d, p)| *d < p.min_value || *d > p.max_value)
    {
        return Err(ProtocolError::InvalidDifficulty {
            difficulty: difficulty.clone(),
            difficulty_parameters: difficulty_parameters.clone(),
        });
    }

    let read_challenges = ctx.read_challenges().await;
    let challenge_data = read_challenges.get(challenge_id).unwrap().block_data();
    let (lower_frontier, upper_frontier) = if *challenge_data.scaling_factor() > 1f64 {
        (
            challenge_data.base_frontier(),
            challenge_data.scaled_frontier(),
        )
    } else {
        (
            challenge_data.scaled_frontier(),
            challenge_data.base_frontier(),
        )
    };
    match difficulty.within(lower_frontier, upper_frontier) {
        PointCompareFrontiers::Above => {
            return Err(ProtocolError::DifficultyAboveHardestFrontier {
                difficulty: difficulty.clone(),
            });
        }
        PointCompareFrontiers::Below => {
            return Err(ProtocolError::DifficultyBelowEasiestFrontier {
                difficulty: difficulty.clone(),
            });
        }
        PointCompareFrontiers::Within => {}
    }

    Ok(())
}

#[time]
async fn verify_solution_is_valid<T: Context>(
    ctx: &T,
    settings: &BenchmarkSettings,
    solutions_meta_data: &Vec<SolutionMetaData>,
    solution_data: &SolutionData,
) -> ProtocolResult<()> {
    let solutions_map: HashMap<u32, u32> = solutions_meta_data
        .iter()
        .map(|d| (d.nonce, d.solution_signature))
        .collect();

    if let Some(&expected_signature) = solutions_map.get(&solution_data.nonce) {
        let signature = solution_data.calc_solution_signature();

        if expected_signature != signature {
            return Err(ProtocolError::InvalidSignatureFromSolutionData {
                nonce: solution_data.nonce,
                expected_signature,
                actual_signature: signature,
            });
        }
    } else {
        return Err(ProtocolError::InvalidBenchmarkNonce {
            nonce: solution_data.nonce,
        });
    }

    if ctx
        .verify_solution(settings, solution_data.nonce, &solution_data.solution)
        .await
        .unwrap_or_else(|e| panic!("verify_solution error: {:?}", e))
        .is_err()
    {
        return Err(ProtocolError::InvalidSolution {
            nonce: solution_data.nonce,
        });
    }

    Ok(())
}
