pub mod cudarc_example;
pub mod knapsack;
pub mod satisfiability;
pub mod vector_search;
pub mod vehicle_routing;

#[cfg(feature = "gpu_cudarc")]
pub struct Kernel {
    pub src: &'static str,
    pub funcs: &'static [&'static str],
}
