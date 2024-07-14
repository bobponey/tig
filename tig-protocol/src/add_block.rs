use crate::context::*;
use logging_timer::time;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use std::collections::{HashMap, HashSet};
use tig_structs::{config::*, core::*};
use tig_utils::*;

#[time]
pub(crate) async fn execute<T: Context>(ctx: &T) -> String {
    let block = create_block(ctx).await;
    confirm_mempool_algorithms(ctx, &block).await;
    confirm_mempool_benchmarks(ctx, &block).await;
    confirm_mempool_proofs(ctx, &block).await;
    confirm_mempool_frauds(ctx, &block).await;
    confirm_mempool_wasms(ctx, &block).await;
    update_deposits(ctx, &block).await;
    update_cutoffs(ctx, &block).await;
    update_qualifiers(ctx, &block).await;
    update_frontiers(ctx, &block).await;
    update_solution_signature_thresholds(ctx, &block).await;
    update_influence(ctx, &block).await;
    update_adoption(ctx, &block).await;
    update_innovator_rewards(ctx, &block).await;
    update_benchmarker_rewards(ctx, &block).await;
    update_merge_points(ctx, &block).await;
    update_merges(ctx, &block).await;
    block.id
}

#[time]
async fn create_block<T: Context>(ctx: &T) -> Block {
    let latest_block_id = ctx
        .get_block_id(BlockFilter::Latest)
        .await
        .expect("No latest block id");
    let latest_block = {
        let read_blocks = ctx.read_blocks().await;
        read_blocks
            .get(&latest_block_id)
            .map(|b| Block {
                id: b.id.clone(),
                config: b.config.clone(),
                details: b.details.clone(),
                data: None,
            })
            .unwrap()
    };
    let config = ctx.get_config().await;
    let height = latest_block.details.height + 1;
    let details = BlockDetails {
        prev_block_id: latest_block.id.clone(),
        height,
        round: height / config.rounds.blocks_per_round + 1,
        eth_block_num: Some(ctx.get_latest_eth_block_num().await),
    };
    let from_block_started = details
        .height
        .saturating_sub(config.benchmark_submissions.lifespan_period);
    let mut data = BlockData {
        mempool_challenge_ids: ctx
            .get_challenge_ids(ChallengesFilter::Mempool)
            .await
            .into_iter()
            .collect(),
        mempool_algorithm_ids: ctx
            .get_algorithm_ids(AlgorithmsFilter::Mempool)
            .await
            .into_iter()
            .collect(),
        mempool_benchmark_ids: ctx
            .get_benchmark_ids(BenchmarksFilter::Mempool { from_block_started })
            .await
            .into_iter()
            .collect(),
        mempool_fraud_ids: ctx
            .get_fraud_ids(FraudsFilter::Mempool { from_block_started })
            .await
            .into_iter()
            .collect(),
        mempool_proof_ids: ctx
            .get_proof_ids(ProofsFilter::Mempool { from_block_started })
            .await
            .into_iter()
            .collect(),
        mempool_wasm_ids: ctx
            .get_wasm_ids(WasmsFilter::Mempool)
            .await
            .into_iter()
            .collect(),
        active_challenge_ids: HashSet::<String>::new(),
        active_algorithm_ids: HashSet::<String>::new(),
        active_benchmark_ids: HashSet::<String>::new(),
        active_player_ids: HashSet::<String>::new(),
    };

    {
        let read_challenges = ctx.read_challenges().await;
        for challenge_id in ctx.get_challenge_ids(ChallengesFilter::Confirmed).await {
            if read_challenges[&challenge_id]
                .state()
                .round_active
                .is_some_and(|r| r <= details.round)
            {
                data.active_challenge_ids.insert(challenge_id);
            }
        }
    }
    {
        let read_wasms = ctx.read_wasms().await;
        let algorithm_ids = ctx.get_algorithm_ids(AlgorithmsFilter::Confirmed).await;
        let mut write_algorithms = ctx.write_algorithms().await;
        for algorithm_id in algorithm_ids {
            let wasm = match read_wasms.get(&algorithm_id) {
                Some(w) => w,
                None => continue,
            };
            let algorithm = write_algorithms.get_mut(&algorithm_id).unwrap();
            let state = algorithm.state.as_mut().unwrap();
            let round_pushed = state
                .round_pushed
                .unwrap_or(state.round_submitted() + config.algorithm_submissions.push_delay);
            if !state.banned && details.round >= round_pushed && wasm.details.compile_success {
                data.active_algorithm_ids.insert(algorithm.id.clone());
                if state.round_pushed.is_none() {
                    state.round_pushed = Some(round_pushed);
                }
            }
        }
    }
    {
        let read_frauds = ctx.read_frauds().await;
        let read_proofs = ctx.read_proofs().await;
        let read_benchmarks = ctx.read_benchmarks().await;
        for benchmark_id in ctx
            .get_benchmark_ids(BenchmarksFilter::Confirmed { from_block_started })
            .await
        {
            let fraud = read_frauds.get(&benchmark_id);
            let proof = read_proofs.get(&benchmark_id);
            if proof.is_none() || fraud.is_some() {
                continue;
            }
            let proof_state = proof.unwrap().state();
            let submission_delay = proof_state.submission_delay();
            let block_confirmed = proof_state.block_confirmed();
            let block_active = block_confirmed
                + submission_delay * config.benchmark_submissions.submission_delay_multiplier;
            if details.height >= block_active {
                data.active_player_ids
                    .insert(read_benchmarks[&benchmark_id].settings.player_id.clone());
                data.active_benchmark_ids.insert(benchmark_id);
            }
        }
    }

    let block_id = ctx.add_block(&details, &data, &config).await;
    Block {
        id: block_id,
        config: Some(config.clone()),
        details,
        data: Some(data),
    }
}

#[time]
async fn confirm_mempool_challenges<T: Context>(ctx: &T, block: &Block) {
    let mut write_challenges = ctx.write_challenges().await;
    for challenge_id in block.data().mempool_challenge_ids.iter() {
        let state = write_challenges
            .get_mut(challenge_id)
            .unwrap()
            .state
            .as_mut()
            .unwrap();
        state.block_confirmed = Some(block.details.height);
    }
}

#[time]
async fn confirm_mempool_algorithms<T: Context>(ctx: &T, block: &Block) {
    let mut write_algorithms = ctx.write_algorithms().await;
    for algorithm_id in block.data().mempool_algorithm_ids.iter() {
        let state = write_algorithms
            .get_mut(algorithm_id)
            .unwrap()
            .state
            .as_mut()
            .unwrap();
        state.block_confirmed = Some(block.details.height);
        state.round_submitted = Some(block.details.round);
    }
}

#[time]
async fn confirm_mempool_benchmarks<T: Context>(ctx: &T, block: &Block) {
    let config = block.config();
    let mut write_benchmarks = ctx.write_benchmarks().await;
    for benchmark_id in block.data().mempool_benchmark_ids.iter() {
        let benchmark = write_benchmarks.get_mut(benchmark_id).unwrap();

        let seed = u32_from_str(format!("{:?}|{:?}", block.id, benchmark_id).as_str());
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let solutions_meta_data = benchmark.solutions_meta_data();
        let mut indexes: Vec<usize> = (0..solutions_meta_data.len()).collect();
        indexes.shuffle(&mut rng);
        let sampled_nonces = indexes
            .into_iter()
            .take(config.benchmark_submissions.max_samples)
            .map(|i| solutions_meta_data[i].nonce)
            .collect();

        let state = benchmark.state.as_mut().unwrap();
        state.sampled_nonces = Some(sampled_nonces);
        state.block_confirmed = Some(block.details.height);
    }
}

#[time]
async fn confirm_mempool_proofs<T: Context>(ctx: &T, block: &Block) {
    let mut write_proofs = ctx.write_proofs().await;
    let read_benchmarks = ctx.read_benchmarks().await;
    for benchmark_id in block.data().mempool_proof_ids.iter() {
        let benchmark = &read_benchmarks[benchmark_id];
        let state = write_proofs
            .get_mut(benchmark_id)
            .unwrap()
            .state
            .as_mut()
            .unwrap();
        state.block_confirmed = Some(block.details.height);
        state.submission_delay = Some(block.details.height - benchmark.details.block_started);
    }
}

#[time]
async fn confirm_mempool_frauds<T: Context>(ctx: &T, block: &Block) {
    let mut write_frauds = ctx.write_frauds().await;
    for benchmark_id in block.data().mempool_fraud_ids.iter() {
        let state = write_frauds
            .get_mut(benchmark_id)
            .unwrap()
            .state
            .as_mut()
            .unwrap();
        state.block_confirmed = Some(block.details.height);
    }
}

#[time]
async fn confirm_mempool_wasms<T: Context>(ctx: &T, block: &Block) {
    let mut write_wasms = ctx.write_wasms().await;
    for algorithm_id in block.data().mempool_wasm_ids.iter() {
        let state = write_wasms
            .get_mut(algorithm_id)
            .unwrap()
            .state
            .as_mut()
            .unwrap();
        state.block_confirmed = Some(block.details.height);
    }
}

#[time]
async fn update_deposits<T: Context>(ctx: &T, block: &Block) {
    let decay = match &block
        .config()
        .optimisable_proof_of_work
        .rolling_deposit_decay
    {
        Some(decay) => PreciseNumber::from_f64(*decay),
        None => return, // Proof of deposit not implemented for these blocks
    };
    let eth_block_num = block.details.eth_block_num();
    let zero = PreciseNumber::from(0);
    let one = PreciseNumber::from(1);
    {
        let mut write_block_data = ctx.write_players_block_data().await;
        for player_id in block.data().active_player_ids.iter() {
            let rolling_deposit =
                match write_block_data[&block.details.prev_block_id].get(player_id) {
                    Some(data) => data.rolling_deposit,
                    None => None,
                }
                .unwrap_or_else(|| zero.clone());

            let block_data = write_block_data
                .get_mut(&block.id)
                .unwrap()
                .get_mut(player_id)
                .unwrap();

            let deposit = ctx
                .get_player_deposit(eth_block_num, player_id)
                .await
                .unwrap_or_else(|| zero.clone());
            block_data.rolling_deposit = Some(decay * rolling_deposit + (one - decay) * deposit);
            block_data.deposit = Some(deposit);
        }
    }
}

#[time]
async fn update_cutoffs<T: Context>(ctx: &T, block: &Block) {
    let config = block.config();
    let num_challenges = block.data().active_challenge_ids.len() as f64;

    let mut total_solutions_by_player = HashMap::<String, f64>::new();
    {
        let read_benchmarks = ctx.read_benchmarks().await;
        for benchmark_id in block.data().active_benchmark_ids.iter() {
            let benchmark = &read_benchmarks[benchmark_id];
            *total_solutions_by_player
                .entry(benchmark.settings.player_id.clone())
                .or_default() += benchmark.details.num_solutions as f64;
        }
    }

    {
        let mut write_block_data = ctx.write_players_block_data().await;
        for (player_id, total_solutions) in total_solutions_by_player.iter() {
            let block_data = write_block_data
                .get_mut(&block.id)
                .unwrap()
                .get_mut(player_id)
                .unwrap();
            block_data.cutoff = Some(
                (total_solutions / num_challenges * config.qualifiers.cutoff_multiplier).ceil()
                    as u32,
            );
        }
    }
}

#[time]
async fn update_solution_signature_thresholds<T: Context>(ctx: &T, block: &Block) {
    let config = block.config();
    let num_challenges = block.data().active_challenge_ids.len() as f64;

    let mut total_solutions_by_player_and_challenge =
        HashMap::<String, HashMap<String, u32>>::new();
    {
        let read_benchmarks = ctx.read_benchmarks().await;
        for benchmark_id in block.data().active_benchmark_ids.iter() {
            let benchmark = &read_benchmarks[benchmark_id];
            *total_solutions_by_player_and_challenge
                .entry(benchmark.settings.player_id.clone())
                .or_default()
                .entry(benchmark.settings.challenge_id.clone())
                .or_default() += benchmark.details.num_solutions;
        }
    }

    let mut solutions_rate_multiplier_by_player = HashMap::<String, HashMap<String, f64>>::new();
    for player_id in block.data().active_player_ids.iter() {
        let total_solutions_by_challenge = &total_solutions_by_player_and_challenge[player_id];
        let player_avg_solutions = (total_solutions_by_challenge
            .values()
            .fold(0, |acc, v| acc + *v) as f64
            / num_challenges)
            .ceil();
        solutions_rate_multiplier_by_player.insert(
            player_id.clone(),
            total_solutions_by_challenge
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        if *v == 0 {
                            1.0
                        } else {
                            (player_avg_solutions / (*v as f64)).min(1.0)
                        },
                    )
                })
                .collect(),
        );
    }

    let mut solutions_rate_by_challenge = HashMap::<String, f64>::new();
    {
        let read_benchmarks = ctx.read_benchmarks().await;
        for benchmark_id in block.data().mempool_proof_ids.iter() {
            let benchmark = &read_benchmarks[benchmark_id];
            let scale = match solutions_rate_multiplier_by_player.get(&benchmark.settings.player_id)
            {
                Some(fraction_qualifiers) => {
                    match fraction_qualifiers.get(&benchmark.settings.challenge_id) {
                        Some(fraction) => fraction.clone(),
                        None => 1.0,
                    }
                }
                None => 1.0,
            };
            *solutions_rate_by_challenge
                .entry(benchmark.settings.challenge_id.clone())
                .or_default() += scale * benchmark.details.num_solutions as f64;
        }
    }

    {
        let mut write_block_data = ctx.write_challenges_block_data().await;
        for challenge_id in block.data().active_challenge_ids.iter() {
            let max_threshold = u32::MAX as f64;
            let current_threshold =
                match write_block_data[&block.details.prev_block_id].get(challenge_id) {
                    Some(data) => *data.solution_signature_threshold() as f64,
                    None => max_threshold,
                };
            let current_rate = *solutions_rate_by_challenge
                .get(challenge_id)
                .unwrap_or(&0.0);

            let equilibrium_rate = config.qualifiers.total_qualifiers_threshold as f64
                / config.benchmark_submissions.lifespan_period as f64;
            let target_rate =
                config.solution_signature.equilibrium_rate_multiplier * equilibrium_rate;
            let target_threshold = if current_rate == 0.0 {
                max_threshold
            } else {
                (current_threshold * target_rate / current_rate).clamp(0.0, max_threshold)
            };

            let threshold_decay = config.solution_signature.threshold_decay.unwrap_or(0.99);
            let block_data = write_block_data
                .get_mut(&block.id)
                .unwrap()
                .get_mut(challenge_id)
                .unwrap();
            block_data.solution_signature_threshold = Some(
                (current_threshold * threshold_decay + target_threshold * (1.0 - threshold_decay))
                    .clamp(0.0, max_threshold) as u32,
            );
        }
    }
}

#[time]
async fn update_qualifiers<T: Context>(ctx: &T, block: &Block) {
    let config = block.config();
    let BlockData {
        active_benchmark_ids,
        active_algorithm_ids,
        active_challenge_ids,
        active_player_ids,
        ..
    } = block.data();

    let mut benchmark_ids_by_challenge = HashMap::<String, Vec<String>>::new();
    {
        let read_benchmarks = ctx.read_benchmarks().await;
        for benchmark_id in active_benchmark_ids.iter() {
            let benchmark = &read_benchmarks[benchmark_id];
            benchmark_ids_by_challenge
                .entry(benchmark.settings.challenge_id.clone())
                .or_default()
                .push(benchmark_id.clone());
        }
    }

    {
        let mut write_block_data = ctx.write_challenges_block_data().await;
        let block_data = write_block_data.get_mut(&block.id).unwrap();
        for challenge_id in active_challenge_ids.iter() {
            let block_data = block_data.get_mut(challenge_id).unwrap();
            block_data.num_qualifiers = Some(0);
            block_data.qualifier_difficulties = Some(HashSet::new());
        }
    }
    {
        let mut write_block_data = ctx.write_algorithms_block_data().await;
        let block_data = write_block_data.get_mut(&block.id).unwrap();
        for algorithm_id in active_algorithm_ids.iter() {
            let block_data = block_data.get_mut(algorithm_id).unwrap();
            block_data.num_qualifiers_by_player = Some(HashMap::new());
        }
    }
    let mut max_qualifiers_by_player = HashMap::<String, u32>::new();
    {
        let mut write_block_data = ctx.write_players_block_data().await;
        let block_data = write_block_data.get_mut(&block.id).unwrap();
        for player_id in active_player_ids.iter() {
            let block_data = block_data.get_mut(player_id).unwrap();
            max_qualifiers_by_player.insert(player_id.clone(), block_data.cutoff().clone());
            block_data.num_qualifiers_by_challenge = Some(HashMap::new());
        }
    }

    for challenge_id in active_challenge_ids.iter() {
        if !benchmark_ids_by_challenge.contains_key(challenge_id) {
            continue;
        }
        let mut points = HashSet::new();
        let benchmark_ids = benchmark_ids_by_challenge.get_mut(challenge_id).unwrap();
        {
            let read_benchmarks = ctx.read_benchmarks().await;
            for benchmark_id in benchmark_ids.iter() {
                let benchmark = &read_benchmarks[benchmark_id];
                points.insert(benchmark.settings.difficulty.clone());
            }
        }
        let mut frontier_indexes = HashMap::<Point, usize>::new();
        let mut frontier_index = 0;
        while !points.is_empty() {
            let frontier = points.pareto_frontier();
            points = points.difference(&frontier).cloned().collect();
            frontier.iter().for_each(|p| {
                frontier_indexes.insert(p.clone(), frontier_index);
            });
            frontier_index += 1;
        }
        let mut frontier_indexes_by_benchmark_id = HashMap::<String, usize>::new();
        {
            let read_benchmarks = ctx.read_benchmarks().await;
            for benchmark_id in benchmark_ids.iter() {
                let benchmark = &read_benchmarks[benchmark_id];
                frontier_indexes_by_benchmark_id.insert(
                    benchmark_id.clone(),
                    frontier_indexes[&benchmark.settings.difficulty],
                );
            }
        }
        benchmark_ids.sort_by(|a_id, b_id| {
            let a_index = frontier_indexes_by_benchmark_id[a_id];
            let b_index = frontier_indexes_by_benchmark_id[b_id];
            a_index.cmp(&b_index)
        });

        let mut max_qualifiers_by_player = max_qualifiers_by_player.clone();
        let mut curr_frontier_index = 0;
        for benchmark_id in benchmark_ids.iter() {
            let (settings, details) = {
                let read_benchmarks = ctx.read_benchmarks().await;
                let benchmark = &read_benchmarks[benchmark_id];
                (benchmark.settings.clone(), benchmark.details.clone())
            };

            {
                let read_block_data = ctx.read_challenges_block_data().await;
                let block_data = &read_block_data[&block.id];
                if curr_frontier_index != frontier_indexes_by_benchmark_id[benchmark_id]
                    && *block_data[challenge_id].num_qualifiers()
                        > config.qualifiers.total_qualifiers_threshold
                {
                    break;
                }
            }
            let difficulty_parameters = &config.difficulty.parameters[&settings.challenge_id];
            let min_difficulty = difficulty_parameters.min_difficulty();
            let max_difficulty = difficulty_parameters.max_difficulty();
            if (0..settings.difficulty.len()).into_iter().any(|i| {
                settings.difficulty[i] < min_difficulty[i]
                    || settings.difficulty[i] > max_difficulty[i]
            }) {
                continue;
            }
            curr_frontier_index = frontier_indexes_by_benchmark_id[benchmark_id];

            let max_qualifiers = max_qualifiers_by_player[&settings.player_id].clone();
            let num_qualifiers = details.num_solutions.min(max_qualifiers);
            max_qualifiers_by_player
                .insert(settings.player_id.clone(), max_qualifiers - num_qualifiers);

            {
                let mut write_block_data = ctx.write_players_block_data().await;
                *write_block_data
                    .get_mut(&block.id)
                    .unwrap()
                    .get_mut(&settings.player_id)
                    .unwrap()
                    .num_qualifiers_by_challenge
                    .as_mut()
                    .unwrap()
                    .entry(settings.challenge_id.clone())
                    .or_default() += num_qualifiers;
            }
            {
                let mut write_block_data = ctx.write_algorithms_block_data().await;
                *write_block_data
                    .get_mut(&block.id)
                    .unwrap()
                    .get_mut(&settings.algorithm_id)
                    .unwrap()
                    .num_qualifiers_by_player
                    .as_mut()
                    .unwrap()
                    .entry(settings.player_id.clone())
                    .or_default() += num_qualifiers;
            }
            {
                let mut write_block_data = ctx.write_challenges_block_data().await;
                let challenge_block_data = write_block_data
                    .get_mut(&block.id)
                    .unwrap()
                    .get_mut(challenge_id)
                    .unwrap();
                *challenge_block_data.num_qualifiers.as_mut().unwrap() += num_qualifiers;
                challenge_block_data
                    .qualifier_difficulties
                    .as_mut()
                    .unwrap()
                    .insert(settings.difficulty.clone());
            }
        }
    }
}

#[time]
async fn update_frontiers<T: Context>(ctx: &T, block: &Block) {
    let config = block.config();

    for challenge_id in block.data().active_challenge_ids.iter() {
        let (qualifier_difficulties, num_qualifiers) = {
            let read_block_data = ctx.read_challenges_block_data().await;
            let block_data = &read_block_data[&block.id][challenge_id];
            (
                block_data.qualifier_difficulties().clone(),
                block_data.num_qualifiers().clone(),
            )
        };

        let difficulty_parameters = &config.difficulty.parameters[challenge_id];
        let min_difficulty = difficulty_parameters.min_difficulty();
        let max_difficulty = difficulty_parameters.max_difficulty();

        let cutoff_frontier = qualifier_difficulties
            .into_iter()
            .map(|d| d.into_iter().map(|x| -x).collect()) // mirror the points so easiest difficulties are first
            .collect::<Frontier>()
            .pareto_frontier()
            .iter()
            .map(|d| d.iter().map(|x| -x).collect())
            .collect::<Frontier>() // mirror the points back;
            .extend(&min_difficulty, &max_difficulty);

        let scaling_factor =
            num_qualifiers as f64 / config.qualifiers.total_qualifiers_threshold as f64;
        let (scaling_factor, base_frontier) = match &config.difficulty.min_frontiers_gaps {
            Some(min_gaps) => {
                let min_gap = min_gaps[challenge_id];
                if scaling_factor >= 1.0 {
                    (
                        (scaling_factor / (1.0 - min_gap))
                            .min(config.difficulty.max_scaling_factor),
                        cutoff_frontier
                            .scale(&min_difficulty, &max_difficulty, 1.0 - min_gap)
                            .extend(&min_difficulty, &max_difficulty),
                    )
                } else {
                    (scaling_factor.min(1.0 - min_gap), cutoff_frontier.clone())
                }
            }
            None => (
                scaling_factor.min(config.difficulty.max_scaling_factor),
                cutoff_frontier.clone(),
            ),
        };
        let scaled_frontier = base_frontier
            .scale(&min_difficulty, &max_difficulty, scaling_factor)
            .extend(&min_difficulty, &max_difficulty);

        {
            let mut write_block_data = ctx.write_challenges_block_data().await;
            let block_data = write_block_data
                .get_mut(&block.id)
                .unwrap()
                .get_mut(challenge_id)
                .unwrap();
            block_data.cutoff_frontier = Some(cutoff_frontier);
            block_data.base_frontier = Some(base_frontier);
            block_data.scaled_frontier = Some(scaled_frontier);
            block_data.scaling_factor = Some(scaling_factor);
        }
    }
}

#[time]
async fn update_influence<T: Context>(ctx: &T, block: &Block) {
    let config = block.config();
    let BlockData {
        active_challenge_ids,
        active_player_ids,
        ..
    } = block.data();

    if active_player_ids.len() == 0 {
        return;
    }

    let mut num_qualifiers_by_challenge = HashMap::<String, u32>::new();
    {
        let read_block_data = ctx.read_challenges_block_data().await;
        let block_data = &read_block_data[&block.id];
        for challenge_id in active_challenge_ids.iter() {
            num_qualifiers_by_challenge.insert(
                challenge_id.clone(),
                *block_data[challenge_id].num_qualifiers(),
            );
        }
    }

    let mut total_deposit = PreciseNumber::from(0);
    {
        let read_block_data = ctx.read_players_block_data().await;
        for player_id in active_player_ids.iter() {
            let block_data = &read_block_data[&block.id][player_id];
            total_deposit = total_deposit + block_data.deposit().clone();
        }
    }

    let zero = PreciseNumber::from(0);
    let one = PreciseNumber::from(1);
    let imbalance_multiplier =
        PreciseNumber::from_f64(config.optimisable_proof_of_work.imbalance_multiplier);
    let num_challenges = PreciseNumber::from(active_challenge_ids.len());

    let mut weights = Vec::<PreciseNumber>::new();
    for player_id in active_player_ids.iter() {
        let (num_qualifiers_by_challenge, deposit) = {
            let read_block_data = ctx.read_players_block_data().await;
            let block_data = &read_block_data[&block.id][player_id];
            (
                block_data.num_qualifiers_by_challenge().clone(),
                block_data.deposit().clone(),
            )
        };

        let mut percent_qualifiers = Vec::<PreciseNumber>::new();
        for challenge_id in active_challenge_ids.iter() {
            let num_qualifiers = num_qualifiers_by_challenge[challenge_id];
            let num_qualifiers_by_player =
                *num_qualifiers_by_challenge.get(challenge_id).unwrap_or(&0);

            percent_qualifiers.push(if num_qualifiers_by_player == 0 {
                PreciseNumber::from(0)
            } else {
                PreciseNumber::from(num_qualifiers_by_player) / PreciseNumber::from(num_qualifiers)
            });
        }
        let OptimisableProofOfWorkConfig {
            rolling_deposit_decay,
            enable_proof_of_deposit,
            ..
        } = &config.optimisable_proof_of_work;
        if rolling_deposit_decay.is_some() && enable_proof_of_deposit.is_some_and(|x| x) {
            percent_qualifiers.push(if total_deposit == zero {
                zero.clone()
            } else {
                deposit / total_deposit
            });
        }

        let mean = percent_qualifiers.arithmetic_mean();
        let variance = percent_qualifiers.variance();
        let cv_sqr = if mean == zero {
            zero.clone()
        } else {
            variance / (mean * mean)
        };

        let imbalance = cv_sqr / (num_challenges - one);
        let imbalance_penalty =
            one - PreciseNumber::approx_inv_exp(imbalance_multiplier * imbalance);

        weights.push(mean * (one - imbalance_penalty));

        {
            let mut write_block_data = ctx.write_players_block_data().await;
            let block_data = write_block_data
                .get_mut(&block.id)
                .unwrap()
                .get_mut(player_id)
                .unwrap();
            block_data.imbalance = Some(imbalance);
            block_data.imbalance_penalty = Some(imbalance_penalty);
        }
    }

    let influences = weights.normalise();
    {
        let mut write_block_data = ctx.write_players_block_data().await;
        let block_data = write_block_data.get_mut(&block.id).unwrap();
        for (player_id, influence) in active_player_ids.iter().zip(influences) {
            let block_data = block_data.get_mut(player_id).unwrap();
            block_data.influence = Some(influence);
        }
    }
}

#[time]
async fn update_adoption<T: Context>(ctx: &T, block: &Block) {
    let BlockData {
        active_algorithm_ids,
        active_challenge_ids,
        ..
    } = block.data();

    let mut algorithm_ids_by_challenge = HashMap::<String, Vec<String>>::new();
    {
        let read_algorithms = ctx.read_algorithms().await;
        for algorithm_id in active_algorithm_ids.iter() {
            let algorithm = &read_algorithms[algorithm_id];
            algorithm_ids_by_challenge
                .entry(algorithm.details.challenge_id.clone())
                .or_default()
                .push(algorithm_id.clone());
        }
    }

    for challenge_id in active_challenge_ids.iter() {
        if !algorithm_ids_by_challenge.contains_key(challenge_id) {
            continue;
        }

        let mut weights = Vec::<PreciseNumber>::new();
        {
            let read_algorithms_block_data = ctx.read_algorithms_block_data().await;
            let read_players_block_data = ctx.read_players_block_data().await;
            for algorithm_id in algorithm_ids_by_challenge[challenge_id].iter() {
                let algorithm_block_data = &read_algorithms_block_data[&block.id][algorithm_id];
                let mut weight = PreciseNumber::from(0);
                for (player_id, &num_qualifiers) in
                    algorithm_block_data.num_qualifiers_by_player().iter()
                {
                    let num_qualifiers = PreciseNumber::from(num_qualifiers);
                    let player_block_data = &read_players_block_data[&block.id][player_id];
                    let influence = player_block_data.influence.as_ref().unwrap().clone();
                    let player_num_qualifiers = PreciseNumber::from(
                        *player_block_data
                            .num_qualifiers_by_challenge
                            .as_ref()
                            .unwrap()
                            .get(challenge_id)
                            .unwrap(),
                    );

                    weight = weight + influence * num_qualifiers / player_num_qualifiers;
                }
                weights.push(weight);
            }
        }

        let adoption = weights.normalise();
        {
            let mut write_block_data = ctx.write_algorithms_block_data().await;
            let block_data = write_block_data.get_mut(&block.id).unwrap();
            for (algorithm_id, adoption) in algorithm_ids_by_challenge[challenge_id]
                .iter()
                .zip(adoption)
            {
                let block_data = block_data.get_mut(algorithm_id).unwrap();
                block_data.adoption = Some(adoption);
            }
        }
    }
}

#[time]
async fn update_innovator_rewards<T: Context>(ctx: &T, block: &Block) {
    let config = block.config();

    let adoption_threshold =
        PreciseNumber::from_f64(config.algorithm_submissions.adoption_threshold);
    let zero = PreciseNumber::from(0);
    let mut eligible_algorithm_ids_by_challenge = HashMap::<String, Vec<String>>::new();
    {
        let read_algorithms = ctx.read_algorithms().await;
        let mut write_block_data = ctx.write_algorithms_block_data().await;
        let block_data = write_block_data.get_mut(&block.id).unwrap();
        for algorithm_id in block.data().active_algorithm_ids.iter() {
            let algorithm = &read_algorithms[algorithm_id];
            let block_data = block_data.get_mut(algorithm_id).unwrap();

            if *block_data.adoption() >= adoption_threshold
                || (algorithm.state().round_merged.is_some() && *block_data.adoption() > zero)
            {
                eligible_algorithm_ids_by_challenge
                    .entry(algorithm.details.challenge_id.clone())
                    .or_default()
                    .push(algorithm_id.clone());
            }

            block_data.reward = Some(zero.clone());
        }
    }
    if eligible_algorithm_ids_by_challenge.len() == 0 {
        return;
    }

    let reward_pool_per_challenge = PreciseNumber::from_f64(get_block_reward(block))
        * PreciseNumber::from_f64(config.rewards.distribution.optimisations)
        / PreciseNumber::from(eligible_algorithm_ids_by_challenge.len());

    let zero = PreciseNumber::from(0);
    {
        let mut write_block_data = ctx.write_algorithms_block_data().await;
        let block_data = write_block_data.get_mut(&block.id).unwrap();
        for (_, algorithm_ids) in eligible_algorithm_ids_by_challenge.iter() {
            let mut total_adoption = zero.clone();
            for algorithm_id in algorithm_ids.iter() {
                let block_data = block_data.get_mut(algorithm_id).unwrap();
                total_adoption = total_adoption + *block_data.adoption();
            }

            for algorithm_id in algorithm_ids.iter() {
                let block_data = block_data.get_mut(algorithm_id).unwrap();
                let adoption = *block_data.adoption();

                block_data.reward = Some(reward_pool_per_challenge * adoption / total_adoption);
            }
        }
    }
}

#[time]
async fn update_benchmarker_rewards<T: Context>(ctx: &T, block: &Block) {
    let config = block.config();

    let reward_pool = PreciseNumber::from_f64(get_block_reward(block))
        * PreciseNumber::from_f64(config.rewards.distribution.benchmarkers);

    {
        let mut write_block_data = ctx.write_players_block_data().await;
        let block_data = write_block_data.get_mut(&block.id).unwrap();
        for player_id in block.data().active_player_ids.iter() {
            let block_data = block_data.get_mut(player_id).unwrap();
            let influence = *block_data.influence();
            block_data.reward = Some(influence * reward_pool);
        }
    }
}

#[time]
async fn update_merge_points<T: Context>(ctx: &T, block: &Block) {
    let config = block.config();

    let adoption_threshold =
        PreciseNumber::from_f64(config.algorithm_submissions.adoption_threshold);
    {
        let read_algorithms = ctx.read_algorithms().await;
        let mut write_block_data = ctx.write_algorithms_block_data().await;
        for algorithm_id in block.data().active_algorithm_ids.iter() {
            let algorithm = &read_algorithms[algorithm_id];

            // first block of the round
            let prev_merge_points = if block.details.height % config.rounds.blocks_per_round == 0 {
                0
            } else {
                match write_block_data
                    .get(&block.details.prev_block_id)
                    .unwrap()
                    .get(algorithm_id)
                {
                    Some(block_data) => *block_data.merge_points(),
                    None => 0,
                }
            };
            let block_data = write_block_data
                .get_mut(&block.id)
                .unwrap()
                .get_mut(algorithm_id)
                .unwrap();
            block_data.merge_points = Some(
                if algorithm.state().round_merged.is_some()
                    || *block_data.adoption() < adoption_threshold
                {
                    prev_merge_points
                } else {
                    prev_merge_points + 1
                },
            );
        }
    }
}

#[time]
async fn update_merges<T: Context>(ctx: &T, block: &Block) {
    let config = block.config();

    // last block of the round
    if (block.details.height + 1) % config.rounds.blocks_per_round != 0 {
        return;
    }

    let mut merge_algorithm_ids_by_challenge = HashMap::<String, (String, u32)>::new();
    {
        let read_algorithms = ctx.read_algorithms().await;
        let read_block_data = ctx.read_algorithms_block_data().await;
        let block_data = &read_block_data[&block.id];
        for algorithm_id in block.data().active_algorithm_ids.iter() {
            let algorithm = &read_algorithms[algorithm_id];
            let challenge_id = algorithm.details.challenge_id.clone();
            let block_data = &block_data[algorithm_id];

            if algorithm.state().round_merged.is_some()
                || *block_data.merge_points() < config.algorithm_submissions.merge_points_threshold
            {
                continue;
            }
            if !merge_algorithm_ids_by_challenge.contains_key(&challenge_id)
                || merge_algorithm_ids_by_challenge[&challenge_id].1 < *block_data.merge_points()
            {
                merge_algorithm_ids_by_challenge.insert(
                    challenge_id,
                    (algorithm_id.clone(), *block_data.merge_points()),
                );
            }
        }
    }

    let round_merged = block.details.round + 1;
    {
        let mut write_algorithms = ctx.write_algorithms().await;
        for (_, (algorithm_id, _)) in merge_algorithm_ids_by_challenge.iter() {
            let state = write_algorithms
                .get_mut(algorithm_id)
                .unwrap()
                .state
                .as_mut()
                .unwrap();

            state.round_merged = Some(round_merged);
        }
    }
}

fn get_block_reward(block: &Block) -> f64 {
    let config = block.config();

    config
        .rewards
        .schedule
        .iter()
        .filter(|s| s.round_start <= block.details.round)
        .last()
        .unwrap_or_else(|| {
            panic!(
                "get_block_reward error: Expecting a reward schedule for round {}",
                block.details.round
            )
        })
        .block_reward
}
