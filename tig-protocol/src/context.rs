use std::collections::HashMap;

pub use anyhow::{Error as ContextError, Result as ContextResult};
use tig_structs::{config::*, core::*};
use tokio::sync::{RwLockReadGuard, RwLockWriteGuard};

#[derive(Debug, Clone, PartialEq)]
pub enum SubmissionType {
    Algorithm,
    Benchmark,
    Proof,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlgorithmsFilter {
    Name(String),
    TxHash(String),
    Mempool,
    Confirmed,
}
#[derive(Debug, Clone, PartialEq)]
pub enum BenchmarksFilter {
    Id(String),
    Settings(BenchmarkSettings),
    Mempool { from_block_started: u32 },
    Confirmed { from_block_started: u32 },
}
#[derive(Debug, Clone, PartialEq)]
pub enum BlockFilter {
    Height(u32),
    Latest,
    Round(u32),
}
#[derive(Debug, Clone, PartialEq)]
pub enum ChallengesFilter {
    Name(String),
    Mempool,
    Confirmed,
}
#[derive(Debug, Clone, PartialEq)]
pub enum FraudsFilter {
    Mempool { from_block_started: u32 },
    Confirmed { from_block_started: u32 },
}
#[derive(Debug, Clone, PartialEq)]
pub enum PlayersFilter {
    Name(String),
    Active { block_id: String },
}
#[derive(Debug, Clone, PartialEq)]
pub enum ProofsFilter {
    Mempool { from_block_started: u32 },
    Confirmed { from_block_started: u32 },
}
#[derive(Debug, Clone, PartialEq)]
pub enum WasmsFilter {
    Mempool,
    Confirmed,
}
#[allow(async_fn_in_trait)]
pub trait Context {
    async fn get_config(&self) -> ProtocolConfig;
    async fn get_algorithm_ids(&self, filter: AlgorithmsFilter) -> Vec<String>;
    async fn read_algorithms(&self) -> RwLockReadGuard<HashMap<String, Algorithm>>;
    async fn read_algorithms_block_data(
        &self,
    ) -> RwLockReadGuard<HashMap<String, HashMap<String, AlgorithmBlockData>>>;
    async fn write_algorithms(&self) -> RwLockWriteGuard<HashMap<String, Algorithm>>;
    async fn write_algorithms_block_data(
        &self,
    ) -> RwLockWriteGuard<HashMap<String, HashMap<String, AlgorithmBlockData>>>;
    async fn get_benchmark_ids(&self, filter: BenchmarksFilter) -> Vec<String>;
    async fn read_benchmarks(&self) -> RwLockReadGuard<HashMap<String, Benchmark>>;
    async fn write_benchmarks(&self) -> RwLockWriteGuard<HashMap<String, Benchmark>>;
    async fn get_block_id(&self, filter: BlockFilter) -> Option<String>;
    async fn read_blocks(&self) -> RwLockReadGuard<HashMap<String, Block>>;
    async fn get_challenge_ids(&self, filter: ChallengesFilter) -> Vec<String>;
    async fn read_challenges(&self) -> RwLockReadGuard<HashMap<String, Challenge>>;
    async fn read_challenges_block_data(
        &self,
    ) -> RwLockReadGuard<HashMap<String, HashMap<String, ChallengeBlockData>>>;
    async fn write_challenges(&self) -> RwLockWriteGuard<HashMap<String, Challenge>>;
    async fn write_challenges_block_data(
        &self,
    ) -> RwLockWriteGuard<HashMap<String, HashMap<String, ChallengeBlockData>>>;
    async fn get_fraud_ids(&self, filter: FraudsFilter) -> Vec<String>;
    async fn read_frauds(&self) -> RwLockReadGuard<HashMap<String, Fraud>>;
    async fn write_frauds(&self) -> RwLockWriteGuard<HashMap<String, Fraud>>;
    async fn get_player_ids(&self, filter: PlayersFilter) -> Vec<String>;
    async fn read_players(&self) -> RwLockReadGuard<HashMap<String, Player>>;
    async fn read_players_block_data(
        &self,
    ) -> RwLockReadGuard<HashMap<String, HashMap<String, PlayerBlockData>>>;
    async fn write_players(&self) -> RwLockWriteGuard<HashMap<String, Player>>;
    async fn write_players_block_data(
        &self,
    ) -> RwLockWriteGuard<HashMap<String, HashMap<String, PlayerBlockData>>>;
    async fn get_proof_ids(&self, filter: ProofsFilter) -> Vec<String>;
    async fn read_proofs(&self) -> RwLockReadGuard<HashMap<String, Proof>>;
    async fn write_proofs(&self) -> RwLockWriteGuard<HashMap<String, Proof>>;
    async fn get_wasm_ids(&self, filter: WasmsFilter) -> Vec<String>;
    async fn read_wasms(&self) -> RwLockReadGuard<HashMap<String, Wasm>>;
    async fn write_wasms(&self) -> RwLockWriteGuard<HashMap<String, Wasm>>;

    async fn verify_solution(
        &self,
        settings: &BenchmarkSettings,
        nonce: u32,
        solution: &Solution,
    ) -> ContextResult<anyhow::Result<()>>;
    async fn compute_solution(
        &self,
        settings: &BenchmarkSettings,
        nonce: u32,
        wasm_vm_config: &WasmVMConfig,
    ) -> ContextResult<anyhow::Result<SolutionData>>;
    async fn get_transaction(&self, tx_hash: &String) -> Transaction;
    async fn get_multisig_owners(&self, address: &String) -> Vec<String>;
    async fn get_latest_eth_block_num(&self) -> String;
    async fn get_player_deposit(
        &self,
        eth_block_num: &String,
        player_id: &String,
    ) -> Option<PreciseNumber>;

    // Mempool
    async fn add_block(
        &self,
        details: &BlockDetails,
        data: &BlockData,
        config: &ProtocolConfig,
    ) -> String;
    async fn add_challenge_to_mempool(&self, details: &ChallengeDetails) -> String;
    async fn add_algorithm_to_mempool(&self, details: &AlgorithmDetails, code: &String) -> String;
    async fn add_benchmark_to_mempool(
        &self,
        settings: &BenchmarkSettings,
        details: &BenchmarkDetails,
        solutions_metadata: &Vec<SolutionMetaData>,
        solution_data: &SolutionData,
    ) -> String;
    async fn add_proof_to_mempool(&self, benchmark_id: &String, solutions_data: &Vec<SolutionData>);
    async fn add_fraud_to_mempool(&self, benchmark_id: &String, allegation: &String);
    async fn add_wasm_to_mempool(
        &self,
        algorithm_id: &String,
        details: &WasmDetails,
        wasm_blob: &Option<Vec<u8>>,
    );
}
