pub(crate) mod block_sparse_ldlt;
pub(crate) mod dense_ldlt;
pub(crate) mod faer_sparse_ldlt;
pub(crate) mod sparse_ldlt;

mod tests {
    use nalgebra::DVector;
    use sophus_autodiff::linalg::{
        IsMatrix,
        MatF64,
    };

    use crate::{
        DenseLdlt,
        DenseLu,
        FaerSparseLu,
        FaerSparseQr,
        IsCompressableMatrix,
        IsLinearSolver,
        IsSymmetricMatrixBuilder,
        SparseLdlt,
        block_sparse::PartitionSpec,
        dense::DenseSymmetricMatrixBuilder,
        positive_semi_definite::{
            block_sparse_ldlt::BlockSparseLdlt,
            faer_sparse_ldlt::FaerSparseLdlt,
        },
        sparse::{
            SparseSymmetricMatrixBuilder,
            faer_sparse_matrix::{
                FaerTripletsMatrix,
                FaerUpperCompressedMatrix,
                FaerUpperTripletsMatrix,
            },
        },
    };

    pub fn all_examples<Builder: IsSymmetricMatrixBuilder>() -> Vec<(Builder::Matrix, DVector<f64>)>
    {
        let mut examples = Vec::new();

        // --- Example 1 ---
        {
            let partitions = vec![PartitionSpec {
                num_blocks: 2,
                block_dim: 3,
            }];
            let mut builder = Builder::zero(&partitions);
            let block_a00 =
                MatF64::<3, 3>::from_array2([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]);
            let block_a01 =
                MatF64::<3, 3>::from_array2([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
            let block_a11 =
                MatF64::<3, 3>::from_array2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

            builder.add_lower_block(&[0, 0], [0, 0], &block_a00.as_view());
            builder.add_lower_block(&[0, 0], [1, 0], &block_a01.as_view());
            builder.add_lower_block(&[0, 0], [1, 1], &block_a11.as_view());

            let b = nalgebra::DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            examples.push((builder.build(), b));
        }

        // --- Single partition, band-1 ---
        {
            use nalgebra as na;
            let parts = vec![PartitionSpec {
                num_blocks: 4,
                block_dim: 2,
            }];
            let mut builder = Builder::zero(&parts);

            let n_scalar = parts
                .iter()
                .map(|p| p.block_dim * p.num_blocks)
                .sum::<usize>();
            let mut mat_l = na::DMatrix::<f64>::zeros(n_scalar, n_scalar);

            let nb = parts[0].num_blocks;
            let m = parts[0].block_dim;
            let so = |b: usize| b * m;

            for i in 0..nb {
                for d in 0..m {
                    mat_l[(so(i) + d, so(i) + d)] = 1.0;
                }
                if i > 0 {
                    let mut blk = mat_l.slice_mut((so(i), so(i - 1)), (m, m));
                    for c in 0..m {
                        for r in 0..m {
                            blk[(r, c)] = 0.05 * (1.0 + (r + c) as f64);
                        }
                    }
                }
            }

            let A = &mat_l * mat_l.transpose();
            for br in 0..nb {
                for bc in 0..=br {
                    builder.add_lower_block(
                        &[0, 0],
                        [br, bc],
                        &A.slice((so(br), so(bc)), (m, m)).as_view(),
                    );
                }
            }

            let b = na::DVector::<f64>::from_iterator(
                n_scalar,
                (0..n_scalar).map(|i| 1.0 + 0.1 * i as f64),
            );
            examples.push((builder.build(), b));
        }

        // --- Multi partitions, mixed dims, full lower ---
        {
            use nalgebra as na;
            let parts = vec![
                PartitionSpec {
                    block_dim: 3,
                    num_blocks: 2,
                },
                PartitionSpec {
                    block_dim: 1,
                    num_blocks: 3,
                },
                PartitionSpec {
                    block_dim: 2,
                    num_blocks: 1,
                },
            ];
            let mut builder = Builder::zero(&parts);

            let mut part_off = vec![0];
            let mut acc = 0usize;
            for p in &parts {
                acc += p.block_dim * p.num_blocks;
                part_off.push(acc);
            }
            let n_scalar = acc;

            let blk_off: Vec<usize> = {
                let mut o = vec![0];
                for p in &parts {
                    o.push(o.last().unwrap() + p.num_blocks);
                }
                o
            };
            let n_blocks_total = *blk_off.last().unwrap();

            let global_blk_dim = |g: usize| {
                let mut pr = 0;
                while pr + 1 < blk_off.len() && g >= blk_off[pr + 1] {
                    pr += 1;
                }
                parts[pr].block_dim
            };
            let global_scalar_off = |g: usize| {
                let mut pr = 0;
                while pr + 1 < blk_off.len() && g >= blk_off[pr + 1] {
                    pr += 1;
                }
                let local = g - blk_off[pr];
                part_off[pr] + local * parts[pr].block_dim
            };

            let mut L = na::DMatrix::<f64>::zeros(n_scalar, n_scalar);
            for i in 0..n_blocks_total {
                let mi = global_blk_dim(i);
                let i0 = global_scalar_off(i);
                for d in 0..mi {
                    L[(i0 + d, i0 + d)] = 1.0;
                }
                for j in 0..i {
                    let mj = global_blk_dim(j);
                    let j0 = global_scalar_off(j);
                    let mut blk = L.slice_mut((i0, j0), (mi, mj));
                    for c in 0..mj {
                        for r in 0..mi {
                            blk[(r, c)] = 0.04 * ((r + 1 + c + 1) as f64);
                        }
                    }
                }
            }
            let A = &L * L.transpose();
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
                            builder.add_lower_block(
                                &[rp, cp],
                                [br, bc],
                                &A.slice((r0, c0), (mr, mc)).as_view(),
                            );
                        }
                    }
                }
            }

            let b = na::DVector::<f64>::from_iterator(
                n_scalar,
                (0..n_scalar).map(|k| (k as f64).sin() + 2.0),
            );
            examples.push((builder.build(), b));
        }

        // --- All scalar blocks ---
        {
            use nalgebra as na;
            let parts = vec![PartitionSpec {
                block_dim: 1,
                num_blocks: 8,
            }];
            let mut builder = Builder::zero(&parts);

            let n = parts[0].block_dim * parts[0].num_blocks;
            let mut L = na::DMatrix::<f64>::zeros(n, n);
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
            for i in 0..parts[0].num_blocks {
                for j in 0..=i {
                    builder.add_lower_block(&[0, 0], [i, j], &A.slice((i, j), (1, 1)).as_view());
                }
            }

            let b =
                na::DVector::<f64>::from_row_slice(&[1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0]);
            examples.push((builder.build(), b));
        }

        examples
    }

    #[test]
    fn scalar_solver_tests() {
        use crate::BlockSparseLowerMatrixBuilder;

        let dense_ex = all_examples::<DenseSymmetricMatrixBuilder>();
        let sparse_ex = all_examples::<SparseSymmetricMatrixBuilder>();
        let block_sparse_ex = all_examples::<BlockSparseLowerMatrixBuilder>();

        for i in 0..dense_ex.len() {
            let (dense_mat_a, b) = &dense_ex[i];
            let (sparse_mat_a, _b) = &sparse_ex[i];
            let (block_sparse_mat_a, _b) = &block_sparse_ex[i];

            let faer_mat_a = FaerTripletsMatrix::from_lower(sparse_mat_a);
            let faer_upper_mat_a = FaerUpperTripletsMatrix::from_lower(sparse_mat_a);

            let x_dense_lu = DenseLu {}.solve(dense_mat_a, b).unwrap();
            let x_faer_sparse_lu = FaerSparseLu {}.solve(&faer_mat_a.compress(), b).unwrap();
            let x_faer_sparse_qr = FaerSparseQr {}.solve(&faer_mat_a.compress(), b).unwrap();

            let x_dense_ldlt = DenseLdlt {}.solve(dense_mat_a, b).unwrap();
            let x_sparse_ldlt = SparseLdlt { tol_rel: 1e-12_f64 }
                .solve(&sparse_mat_a.compress(), b)
                .unwrap();
            let x_block_sparse_ldlt = BlockSparseLdlt {}
                .solve(&block_sparse_mat_a.compress(), b)
                .unwrap();
            let x_faer_sparse_ldlt = FaerSparseLdlt::default()
                .solve(&faer_upper_mat_a.compress(), b)
                .unwrap();

            print!("dense LU: x = {x_dense_lu}");
            print!("faer sparse LU: x = {x_faer_sparse_lu}");

            print!("faer sparse QR: x = {x_faer_sparse_qr}");

            print!("dense LDLt x = {x_dense_ldlt}");
            print!("sparse LDLt x = {x_sparse_ldlt}");
            print!("block sparse LDLt x = {x_block_sparse_ldlt}");
            print!("faer sparse LDLt x = {x_faer_sparse_ldlt}");

            approx::assert_abs_diff_eq!(
                dense_mat_a.clone() * x_dense_lu,
                b.clone(),
                epsilon = 1e-6
            );
            approx::assert_abs_diff_eq!(
                dense_mat_a.clone() * x_faer_sparse_lu,
                b.clone(),
                epsilon = 1e-6
            );

            approx::assert_abs_diff_eq!(
                dense_mat_a.clone() * x_faer_sparse_qr,
                b.clone(),
                epsilon = 1e-6
            );

            approx::assert_abs_diff_eq!(
                dense_mat_a.clone() * x_dense_ldlt,
                b.clone(),
                epsilon = 1e-6
            );
            approx::assert_abs_diff_eq!(
                dense_mat_a.clone() * x_sparse_ldlt,
                b.clone(),
                epsilon = 1e-6
            );
            approx::assert_abs_diff_eq!(
                dense_mat_a.clone() * x_faer_sparse_ldlt,
                b.clone(),
                epsilon = 1e-6
            );
            approx::assert_abs_diff_eq!(
                dense_mat_a.clone() * x_block_sparse_ldlt,
                b.clone(),
                epsilon = 1e-6
            );
        }
    }
}
