use rand::{rngs::StdRng, Rng, SeedableRng};
mod example;

pub struct Challenge {
    pub n: i32,
    pub matrix_a: Vec<f32>, // flattened N x N matrix
    pub matrix_b: Vec<f32>, // flattened N x N matrix
}

impl Challenge {
    pub fn new(n: i32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let matrix_a = (0..n * n).map(|_| rng.gen()).collect();
        let matrix_b = (0..n * n).map(|_| rng.gen()).collect();
        Self {
            n,
            matrix_a,
            matrix_b,
        }
    }
}

pub struct Solution {
    pub matrix_c: Vec<f32>, // flattened N x N matrix
}

#[cfg(feature = "gpu_cudarc")]
mod gpu_tests {
    use super::*;
    use cudarc::driver::*;
    use cudarc::nvrtc::compile_ptx;
    use std::collections::HashMap;
    use std::thread;

    #[test]
    fn test_example() {
        println!("Compiling CUDA kernel");
        let start = std::time::Instant::now();
        let ptx_and_funcs = example::KERNEL.as_ref().map(|k| {
            (
                compile_ptx(k.src).expect("Cuda Kernel failed to compile"),
                k.funcs,
            )
        });
        println!("Took {:?}ms", start.elapsed().as_millis());

        thread::scope(|s| {
            for worker in 0..4 {
                let ptx_and_funcs = ptx_and_funcs.clone();

                s.spawn(move || {
                    let dev = CudaDevice::new(0).expect("Failed to create CudaDevice");
                    let funcs: HashMap<&'static str, CudaFunction> = ptx_and_funcs
                        .map(|(ptx, funcs)| {
                            dev.load_ptx(ptx, "example", funcs).unwrap();
                            funcs
                                .iter()
                                .map(|&name| (name, dev.get_func("example", name).unwrap()))
                                .collect()
                        })
                        .unwrap_or_default();

                    for nonce in 0..10 {
                        let seed = nonce + worker * 100;
                        println!("GPU Worker {}: starting nonce {}", worker, nonce);
                        let start = std::time::Instant::now();
                        let challenge = Challenge::new(1000, seed);
                        example::cuda_solve_challenge(&challenge, &dev, funcs.clone())
                            .expect("Error solving challenge");
                        println!(
                            "GPU Worker {}: took {:?}ms",
                            worker,
                            start.elapsed().as_millis()
                        );
                    }
                });
            }
        });
    }
}

#[cfg(not(feature = "gpu_cudarc"))]
mod cpu_tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_example() {
        thread::scope(|s| {
            for worker in 0..4 {
                s.spawn(move || {
                    for nonce in 0..10 {
                        let seed = nonce + worker * 100;
                        println!("CPU Worker {}: starting nonce {}", worker, nonce);
                        let start = std::time::Instant::now();
                        let challenge = Challenge::new(1000, seed);
                        example::solve_challenge(&challenge).expect("Error solving challenge");
                        println!(
                            "CPU Worker {}: took {:?}ms",
                            worker,
                            start.elapsed().as_millis()
                        );
                    }
                });
            }
        });
    }
}
