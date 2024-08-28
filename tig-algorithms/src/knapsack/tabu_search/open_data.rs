/*!
Copyright 2024 Chad Blanchard

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::VecDeque;
use tig_challenges::knapsack::*;

#[derive(Clone, Debug)]
struct Individual {
    items: Vec<bool>,
    fitness: u32,
}

impl Individual {
    fn new(items: Vec<bool>, fitness: u32) -> Self {
        Self { items, fitness }
    }

    fn calculate_fitness(&mut self, values: &[u32], weights: &[u32], max_weight: u32) {
        let mut total_value = 0;
        let mut total_weight = 0;
        for i in 0..self.items.len() {
            if self.items[i] {
                total_value += values[i];
                total_weight += weights[i];
            }
        }
        if total_weight <= max_weight {
            self.fitness = total_value;
        } else {
            self.fitness = 0;
        }
    }
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let max_weight = challenge.max_weight;
    let min_value = challenge.min_value;
    let num_items = challenge.difficulty.num_items;
    let values = &challenge.values;
    let weights = &challenge.weights;

    let tabu_tenure = 10;
    let max_iterations = 100;
    let mut rng = StdRng::seed_from_u64(challenge.seeds[0] as u64);

    let initial_items: Vec<bool> = (0..num_items).map(|_| rng.gen_bool(0.5)).collect();
    let mut current_solution = Individual::new(initial_items.clone(), 0);
    current_solution.calculate_fitness(values, weights, max_weight);

    let mut best_solution = current_solution.clone();
    let mut tabu_list: VecDeque<Vec<bool>> = VecDeque::new();

    for _ in 0..max_iterations {
        let mut neighborhood = Vec::new();

        for i in 0..num_items {
            let mut neighbor_items = current_solution.items.clone();
            neighbor_items[i] = !neighbor_items[i];
            let mut neighbor = Individual::new(neighbor_items, 0);
            neighbor.calculate_fitness(values, weights, max_weight);
            if !tabu_list.contains(&neighbor.items) {
                neighborhood.push(neighbor);
            }
        }

        if let Some(best_neighbor) = neighborhood.iter().max_by_key(|ind| ind.fitness) {
            current_solution = best_neighbor.clone();
            if current_solution.fitness > best_solution.fitness {
                best_solution = current_solution.clone();
            }
        }

        tabu_list.push_back(current_solution.items.clone());
        if tabu_list.len() > tabu_tenure {
            tabu_list.pop_front();
        }
    }

    if best_solution.fitness >= min_value {
        let items = best_solution.items.iter().enumerate()
            .filter_map(|(i, &included)| if included { Some(i) } else { None })
            .collect();
        Ok(Some(Solution { items }))
    } else {
        Ok(None)
    }
}
#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = None;

    // Important! your GPU and CPU version of the algorithm should return the same result
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
