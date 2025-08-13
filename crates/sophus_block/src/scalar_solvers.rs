pub(crate) mod dense_ldlt;
pub(crate) mod dense_lu;
pub(crate) mod sparse_ldlt;
pub(crate) mod sparse_lu;
pub(crate) mod sparse_qr;
pub use dense_ldlt::*;
pub use dense_lu::*;
pub use sparse_ldlt::*;
pub use sparse_lu::*;
pub use sparse_qr::*;

use crate::IsDenseLinearSystem;

mod tests {
    use super::*;

    #[test]
    fn scalar_solver_tests() {
        use faer::sparse::Triplet;
        use log::info;
        use nalgebra::DMatrix;
        use sophus_autodiff::{
            linalg::MatF64,
            prelude::*,
        };

        use crate::{
            IsSparseSymmetricLinearSystem,
            SymmetricBlockSparseMatrixBuilder,
        };

        pub fn from_triplets_nxn(
            n: usize,
            triplets: &[Triplet<usize, usize, f64>],
        ) -> DMatrix<f64> {
            let mut mat = DMatrix::zeros(n, n);
            for t in triplets {
                assert!(
                    t.row < n && t.col < n,
                    "Triplet index ({}, {}) out of bounds for an {n}x{n} matrix",
                    t.row,
                    t.row,
                );
                mat[(t.row, t.col)] += t.val;
            }
            mat
        }

        let partitions = vec![crate::PartitionSpec {
            num_blocks: 2,
            block_dim: 3,
        }];
        let mut symmetric_matrix_builder = SymmetricBlockSparseMatrixBuilder::zero(&partitions);

        let block_a00 = MatF64::<3, 3>::from_array2([
            [2.0, 1.0, 0.0], //
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ]);
        let block_a01 = MatF64::<3, 3>::from_array2([
            [0.0, 0.0, 0.0], //
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);
        let block_a11 = MatF64::<3, 3>::from_array2([
            [1.0, 0.0, 0.0], //
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);

        symmetric_matrix_builder.add_block(&[0, 0], [0, 0], &block_a00.as_view());
        symmetric_matrix_builder.add_block(&[0, 0], [1, 0], &block_a01.as_view());
        symmetric_matrix_builder.add_block(&[0, 0], [1, 1], &block_a11.as_view());
        let mat_a = symmetric_matrix_builder.to_symmetric_dense();
        let mat_a_from_triplets = from_triplets_nxn(
            symmetric_matrix_builder.scalar_dimension(),
            &symmetric_matrix_builder.to_symmetric_scalar_triplets(),
        );
        assert_eq!(mat_a, mat_a_from_triplets);

        let b = nalgebra::DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        info!(
            "symmetric_matrix_builder. {:?}",
            symmetric_matrix_builder.to_symmetric_scalar_triplets()
        );
        let x_dense_lu = dense_lu::DenseLU {}
            .solve_dense(symmetric_matrix_builder.to_symmetric_dense(), &b)
            .unwrap();
        let x_dense_ldlt = dense_ldlt::DenseLDLt {}
            .solve_dense(symmetric_matrix_builder.to_symmetric_dense(), &b)
            .unwrap();
        let x_sparse_qr = sparse_qr::SparseQr {}
            .solve(&symmetric_matrix_builder, &b)
            .unwrap();
        let x_sparse_lu = sparse_lu::SparseLu {}
            .solve(&symmetric_matrix_builder, &b)
            .unwrap();
        let x_sparse_ldlt = sparse_ldlt::SparseLDLt {}
            .solve_dense(symmetric_matrix_builder.to_symmetric_dense(), &b)
            .unwrap();

        let x_block_sparse = symmetric_matrix_builder.builder.ldlt_solve(&b);

        info!("x_dense_lu {x_dense_lu}");
        info!("x_sparse_qr {x_sparse_qr}");
        info!("x_sparse_lu {x_sparse_lu}");
        print!("x_sparse_ldlt {x_sparse_ldlt}");
        print!("x_block_sparse {x_block_sparse}");

        approx::assert_abs_diff_eq!(mat_a.clone() * x_dense_lu, b.clone(), epsilon = 1e-6);
        approx::assert_abs_diff_eq!(mat_a.clone() * x_dense_ldlt, b.clone(), epsilon = 1e-6);
        approx::assert_abs_diff_eq!(mat_a.clone() * x_sparse_qr, b.clone(), epsilon = 1e-6);
        approx::assert_abs_diff_eq!(mat_a.clone() * x_sparse_lu, b.clone(), epsilon = 1e-6);
        approx::assert_abs_diff_eq!(mat_a.clone() * x_sparse_ldlt, b.clone(), epsilon = 1e-6);
        approx::assert_abs_diff_eq!(mat_a.clone() * x_block_sparse, b.clone(), epsilon = 1e-6);
    }

    #[test]
    fn block_ldlt_spd_single_partition_band1() {
        use nalgebra as na;

        use crate::{
            IsSparseSymmetricLinearSystem,
            PartitionSpec,
            SymmetricBlockSparseMatrixBuilder,
        };

        // ----- partitions: one family, 4 blocks of dim 2 (total n = 8)
        let parts = vec![PartitionSpec {
            num_blocks: 4,
            block_dim: 2,
        }];
        let mut sb = SymmetricBlockSparseMatrixBuilder::zero(&parts);

        // Build block-lower L (bandwidth 1 in block sense), diag = I
        let n_scalar = parts
            .iter()
            .map(|p| p.block_dim * p.num_blocks)
            .sum::<usize>();
        let mut L = na::DMatrix::<f64>::zeros(n_scalar, n_scalar);

        let nb = parts[0].num_blocks;
        let m = parts[0].block_dim;
        let off = 0; // only one partition

        // helper: scalar offset of (block idx b)
        let so = |b: usize| off + b * m;

        for i in 0..nb {
            // diagonal block = I
            L.slice_mut((so(i), so(i)), (m, m)).fill(0.0);
            for d in 0..m {
                L[(so(i) + d, so(i) + d)] = 1.0;
            }

            // immediate subdiagonal block (i, i-1)
            if i > 0 {
                let mut blk = L.slice_mut((so(i), so(i - 1)), (m, m));
                // small dense block to keep SPD well-conditioned
                for c in 0..m {
                    for r in 0..m {
                        blk[(r, c)] = 0.05 * (1.0 + (r + c) as f64);
                    }
                }
            }
        }

        let A = &L * L.transpose();

        // Fill lower-triangular blocks of A into the builder
        for br in 0..nb {
            for bc in 0..=br {
                let r0 = so(br);
                let c0 = so(bc);
                sb.add_block(&[0, 0], [br, bc], &A.slice((r0, c0), (m, m)).as_view());
            }
        }

        // Sanity: A must be SPD
        assert!(sb.to_symmetric_dense().cholesky().is_some());

        // Solve Ax=b with all solvers
        let b = na::DVector::<f64>::from_iterator(
            n_scalar,
            (0..n_scalar).map(|i| 1.0 + 0.1 * i as f64),
        );

        let x_dense = dense_lu::DenseLU {}
            .solve_dense(sb.to_symmetric_dense(), &b)
            .unwrap();
        let x_qr = sparse_qr::SparseQr {}.solve(&sb, &b).unwrap();
        let x_lu = sparse_lu::SparseLu {}.solve(&sb, &b).unwrap();
        let x_ldlt = sparse_ldlt::SparseLDLt {}
            .solve_dense(sb.to_symmetric_dense(), &b)
            .unwrap();
        let x_block = sb.builder.ldlt_solve(&b);

        let Ad = sb.to_symmetric_dense();

        approx::assert_abs_diff_eq!(Ad.clone() * x_dense, b.clone(), epsilon = 1e-8);
        approx::assert_abs_diff_eq!(Ad.clone() * x_qr, b.clone(), epsilon = 1e-8);
        approx::assert_abs_diff_eq!(Ad.clone() * x_lu, b.clone(), epsilon = 1e-8);
        approx::assert_abs_diff_eq!(Ad.clone() * x_ldlt, b.clone(), epsilon = 1e-8);
        approx::assert_abs_diff_eq!(Ad.clone() * x_block, b.clone(), epsilon = 1e-8);
    }

    #[test]
    fn block_ldlt_spd_multi_partitions_mixed_dims_full_lower() {
        use nalgebra as na;

        use crate::{
            IsSparseSymmetricLinearSystem,
            PartitionSpec,
            SymmetricBlockSparseMatrixBuilder,
        };

        // ----- partitions: [ (dim=3, blocks=2), (dim=1, blocks=3), (dim=2, blocks=1) ]
        let parts = vec![
            PartitionSpec {
                block_dim: 3,
                num_blocks: 2,
            }, // 6
            PartitionSpec {
                block_dim: 1,
                num_blocks: 3,
            }, // 3
            PartitionSpec {
                block_dim: 2,
                num_blocks: 1,
            }, // 2
        ];
        let mut sb = SymmetricBlockSparseMatrixBuilder::zero(&parts);

        // scalar offsets per partition
        let mut part_off = Vec::with_capacity(parts.len() + 1);
        let mut acc = 0usize;
        part_off.push(acc);
        for p in &parts {
            acc += p.block_dim * p.num_blocks;
            part_off.push(acc);
        }
        let n_scalar = acc;

        // global block index helpers
        let nb_per_part: Vec<usize> = parts.iter().map(|p| p.num_blocks).collect();
        let blk_off: Vec<usize> = {
            let mut o = vec![0usize];
            for p in &parts {
                o.push(o.last().unwrap() + p.num_blocks);
            }
            o
        };
        let n_blocks_total = *blk_off.last().unwrap();

        let global_blk_dim = |g: usize| {
            // which partition?
            let mut pr = 0usize;
            while pr + 1 < blk_off.len() && g >= blk_off[pr + 1] {
                pr += 1;
            }
            parts[pr].block_dim
        };
        let global_scalar_off = |g: usize| {
            let mut pr = 0usize;
            while pr + 1 < blk_off.len() && g >= blk_off[pr + 1] {
                pr += 1;
            }
            let local = g - blk_off[pr];
            part_off[pr] + local * parts[pr].block_dim
        };

        // Build block-lower L with full lower pattern, diag = I
        let mut L = na::DMatrix::<f64>::zeros(n_scalar, n_scalar);
        for i in 0..n_blocks_total {
            let mi = global_blk_dim(i);
            let i0 = global_scalar_off(i);
            // diag = I
            for d in 0..mi {
                L[(i0 + d, i0 + d)] = 1.0;
            }
            for j in 0..i {
                let mj = global_blk_dim(j);
                let j0 = global_scalar_off(j);
                // small dense block
                let mut blk = L.slice_mut((i0, j0), (mi, mj));
                for c in 0..mj {
                    for r in 0..mi {
                        blk[(r, c)] = 0.04 * ((r + 1 + c + 1) as f64); // gentle values
                    }
                }
            }
        }
        let A = &L * L.transpose();

        // Fill *lower* blocks of A into the builder (all regions/br,bc in lower)
        for rp in 0..parts.len() {
            let mr = parts[rp].block_dim;
            let nbr = parts[rp].num_blocks;
            let rbase = part_off[rp];
            for cp in 0..=rp {
                let mc = parts[cp].block_dim;
                let nbc = parts[cp].num_blocks;
                let cbase = part_off[cp];

                for br in 0..nbr {
                    let r0 = rbase + br * mr;
                    let bc_max = if rp == cp { br } else { nbc - 1 };
                    for bc in 0..=bc_max {
                        let c0 = cbase + bc * mc;
                        sb.add_block(&[rp, cp], [br, bc], &A.slice((r0, c0), (mr, mc)).as_view());
                    }
                }
            }
        }

        assert!(sb.to_symmetric_dense().cholesky().is_some());

        let b = na::DVector::<f64>::from_iterator(
            n_scalar,
            (0..n_scalar).map(|k| (k as f64).sin() + 2.0),
        );

        let x_dense = dense_lu::DenseLU {}
            .solve_dense(sb.to_symmetric_dense(), &b)
            .unwrap();
        let x_qr = sparse_qr::SparseQr {}.solve(&sb, &b).unwrap();
        let x_lu = sparse_lu::SparseLu {}.solve(&sb, &b).unwrap();
        let x_ldlt = sparse_ldlt::SparseLDLt {}
            .solve_dense(sb.to_symmetric_dense(), &b)
            .unwrap();
        let x_block = sb.builder.ldlt_solve(&b);

        let Ad = sb.to_symmetric_dense();

        approx::assert_abs_diff_eq!(Ad.clone() * x_dense, b.clone(), epsilon = 1e-8);
        approx::assert_abs_diff_eq!(Ad.clone() * x_qr, b.clone(), epsilon = 1e-8);
        approx::assert_abs_diff_eq!(Ad.clone() * x_lu, b.clone(), epsilon = 1e-8);
        approx::assert_abs_diff_eq!(Ad.clone() * x_ldlt, b.clone(), epsilon = 1e-8);
        approx::assert_abs_diff_eq!(Ad.clone() * x_block, b.clone(), epsilon = 1e-8);
    }

    #[test]
    fn block_ldlt_spd_all_scalar_blocks() {
        use nalgebra as na;

        use crate::{
            IsSparseSymmetricLinearSystem,
            PartitionSpec,
            SymmetricBlockSparseMatrixBuilder,
        };

        // ----- partitions: 1x1 blocks (scalar) — 8 blocks total
        let parts = vec![PartitionSpec {
            block_dim: 1,
            num_blocks: 8,
        }];
        let mut sb = SymmetricBlockSparseMatrixBuilder::zero(&parts);

        let n = parts[0].block_dim * parts[0].num_blocks;
        let mut L = na::DMatrix::<f64>::zeros(n, n);

        // diag = 1, subdiagonal = 0.2, next subdiag = 0.05 (bandwidth 2)
        for i in 0..n {
            L[(i, i)] = 1.0;
            if i > 0 {
                L[(i, i - 1)] = 0.2;
            }
            if i > 1 {
                L[(i, i - 2)] = 0.05;
            }
        }
        let A = &L * L.transpose();

        // fill lower
        for i in 0..parts[0].num_blocks {
            for j in 0..=i {
                sb.add_block(&[0, 0], [i, j], &A.slice((i, j), (1, 1)).as_view());
            }
        }

        assert!(sb.to_symmetric_dense().cholesky().is_some());

        let b = na::DVector::<f64>::from_row_slice(&[1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0]);

        let x_dense = dense_lu::DenseLU {}
            .solve_dense(sb.to_symmetric_dense(), &b)
            .unwrap();
        let x_qr = sparse_qr::SparseQr {}.solve(&sb, &b).unwrap();
        let x_lu = sparse_lu::SparseLu {}.solve(&sb, &b).unwrap();
        let x_ldlt = sparse_ldlt::SparseLDLt {}
            .solve_dense(sb.to_symmetric_dense(), &b)
            .unwrap();
        let x_block = sb.builder.ldlt_solve(&b);

        let Ad = sb.to_symmetric_dense();

        approx::assert_abs_diff_eq!(Ad.clone() * x_dense, b.clone(), epsilon = 1e-10);
        approx::assert_abs_diff_eq!(Ad.clone() * x_qr, b.clone(), epsilon = 1e-10);
        approx::assert_abs_diff_eq!(Ad.clone() * x_lu, b.clone(), epsilon = 1e-10);
        approx::assert_abs_diff_eq!(Ad.clone() * x_ldlt, b.clone(), epsilon = 1e-10);
        approx::assert_abs_diff_eq!(Ad.clone() * x_block, b.clone(), epsilon = 1e-10);
    }
}
