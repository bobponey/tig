use super::{Challenge, Solution};

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut solution = Solution {
        matrix_c: vec![0.0; challenge.matrix_a.len()],
    };

    let n = challenge.n as usize;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                solution.matrix_c[i * n + j] +=
                    challenge.matrix_a[i * n + k] * challenge.matrix_b[k * n + j];
            }
        }
    }

    Ok(Some(solution))
}

#[cfg(feature = "gpu_cudarc")]
mod cudarc_stuff {
    use super::*;
    use crate::Kernel;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<Kernel> = Some(Kernel {
        src: r#"
    extern "C" __global__ void hello_world(int i) {
        printf("Hello from the cuda kernel in thread %d\n", i);
    }

    extern "C" __global__ void matmul(float* A, float* B, float* C, int N) {
        int ROW = blockIdx.y*blockDim.y+threadIdx.y;
        int COL = blockIdx.x*blockDim.x+threadIdx.x;
    
        float tmpSum = 0;
    
        if (ROW < N && COL < N) {
            // each thread computes one element of the block sub-matrix
            for (int i = 0; i < N; i++) {
                tmpSum += A[ROW * N + i] * B[i * N + COL];
            }
        }
        C[ROW * N + COL] = tmpSum;
    }
    "#,
        funcs: &["hello_world", "matmul"],
    });

    // replace body with solve_challenge(challenge) if algorithm only has a CPU implementation
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        let mut solution = Solution {
            matrix_c: vec![0.0; challenge.matrix_a.len()],
        };

        let a_dev = dev.htod_sync_copy(challenge.matrix_a.as_slice()).unwrap();
        let b_dev = dev.htod_sync_copy(challenge.matrix_b.as_slice()).unwrap();
        let mut c_dev = dev.htod_sync_copy(solution.matrix_c.as_slice()).unwrap();

        let cfg = LaunchConfig {
            block_dim: (2, 2, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            funcs
                .remove("matmul")
                .unwrap()
                .launch(cfg, (&a_dev, &b_dev, &mut c_dev, challenge.n))
        }?;

        dev.dtoh_sync_copy_into(&c_dev, &mut solution.matrix_c)?;

        Ok(Some(solution))
    }
}
#[cfg(feature = "gpu_cudarc")]
pub use cudarc_stuff::{cuda_solve_challenge, KERNEL};
