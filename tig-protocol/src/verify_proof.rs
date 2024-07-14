use crate::{context::*, error::*};
use logging_timer::time;

#[time]
pub(crate) async fn execute<T: Context>(
    ctx: &T,
    benchmark_id: &String,
) -> ProtocolResult<Result<(), String>> {
    let mut verified = Ok(());
    if let Err(e) = verify_solutions_with_algorithm(ctx, benchmark_id).await {
        ctx.add_fraud_to_mempool(benchmark_id, &e.to_string()).await;
        verified = Err(e.to_string());
    }
    Ok(verified)
}

#[time]
async fn verify_solutions_with_algorithm<T: Context>(
    ctx: &T,
    benchmark_id: &String,
) -> ProtocolResult<()> {
    let read_benchmarks = ctx.read_benchmarks().await;
    let benchmark = &read_benchmarks[benchmark_id];
    let read_proofs = ctx.read_proofs().await;
    let proof = &read_proofs[benchmark_id];
    let settings = &benchmark.settings;
    let wasm_vm_config = ctx.read_blocks().await[&settings.block_id]
        .config
        .as_ref()
        .unwrap()
        .wasm_vm
        .clone();

    for solution_data in proof.solutions_data() {
        if let Ok(actual_solution_data) = ctx
            .compute_solution(settings, solution_data.nonce, &wasm_vm_config)
            .await
            .unwrap_or_else(|e| panic!("compute_solution error: {:?}", e))
        {
            if actual_solution_data == *solution_data {
                continue;
            }
        }

        return Err(ProtocolError::InvalidSolutionData {
            algorithm_id: settings.algorithm_id.clone(),
            nonce: solution_data.nonce,
        });
    }

    Ok(())
}
