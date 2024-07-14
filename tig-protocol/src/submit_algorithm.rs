use crate::{context::*, error::*};
use logging_timer::time;
use std::collections::HashSet;
use tig_structs::core::*;
use tig_utils::*;

#[time]
pub(crate) async fn execute<T: Context>(
    ctx: &T,
    player: &Player,
    details: &AlgorithmDetails,
    code: &String,
) -> ProtocolResult<String> {
    verify_challenge_exists(ctx, details).await?;
    verify_submission_fee(ctx, player, details).await?;
    let algorithm_id = ctx.add_algorithm_to_mempool(details, code).await;
    Ok(algorithm_id)
}

#[time]
async fn verify_challenge_exists<T: Context>(
    ctx: &T,
    details: &AlgorithmDetails,
) -> ProtocolResult<()> {
    if ctx
        .read_challenges()
        .await
        .get(&details.challenge_id)
        .is_none()
    {
        return Err(ProtocolError::InvalidChallenge {
            challenge_id: details.challenge_id.clone(),
        });
    }
    Ok(())
}

#[time]
async fn verify_submission_fee<T: Context>(
    ctx: &T,
    player: &Player,
    details: &AlgorithmDetails,
) -> ProtocolResult<()> {
    let config = {
        let latest_block_id = ctx
            .get_block_id(BlockFilter::Latest)
            .await
            .expect("No latest block id");
        let read_blocks = ctx.read_blocks().await;
        read_blocks[&latest_block_id].config().clone()
    };

    if ctx
        .get_algorithm_ids(AlgorithmsFilter::TxHash(details.tx_hash.clone()))
        .await
        .first()
        .is_some()
    {
        return Err(ProtocolError::DuplicateSubmissionFeeTx {
            tx_hash: details.tx_hash.clone(),
        });
    }
    let mut valid_senders = HashSet::<String>::new();
    valid_senders.insert(player.id.clone());
    if player.details.is_multisig {
        let multisig_owners = ctx.get_multisig_owners(&player.id).await;
        valid_senders.extend(multisig_owners.into_iter());
    }

    let transaction = ctx.get_transaction(&details.tx_hash).await;
    if !valid_senders.contains(&transaction.sender) {
        return Err(ProtocolError::InvalidSubmissionFeeSender {
            tx_hash: details.tx_hash.clone(),
            expected_sender: player.id.clone(),
            actual_sender: transaction.sender.clone(),
        });
    }
    let burn_address = config.erc20.burn_address.clone();
    if transaction.receiver != burn_address {
        return Err(ProtocolError::InvalidSubmissionFeeReceiver {
            tx_hash: details.tx_hash.clone(),
            expected_receiver: burn_address,
            actual_receiver: transaction.receiver.clone(),
        });
    }

    let expected_amount = config.algorithm_submissions.submission_fee;
    if transaction.amount != expected_amount {
        return Err(ProtocolError::InvalidSubmissionFeeAmount {
            tx_hash: details.tx_hash.clone(),
            expected_amount: jsonify(&expected_amount),
            actual_amount: jsonify(&transaction.amount),
        });
    }
    Ok(())
}
