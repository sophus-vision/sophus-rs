pub(crate) mod block_diag_ldlt;
pub(crate) mod block_sparse_ldlt;
pub(crate) mod dense_ldlt;
pub(crate) mod elimination_tree;
pub(crate) mod faer_sparse_ldlt;
pub(crate) mod sparse_ldlt;

use std::marker::PhantomData;

pub use block_diag_ldlt::*;
pub use block_sparse_ldlt::*;
pub use dense_ldlt::*;
pub use elimination_tree::*;
pub use faer_sparse_ldlt::*;
pub use sparse_ldlt::*;

/// Workspace for sparse LDLᵀ such as [SparseLdlt] or [BlockSparseLdlt].
pub trait IsLdltWorkspace: Sized {
    /// Decomposition error;
    type Error;

    /// Type of matrix A, we want to decompose: `A = L D Lᵀ`.
    type Matrix;

    /// Type of lower-triangular matrix `L`.
    type MatLBuilder;
    /// Type of matrix diagonal or block-diagonal `D`.
    type Diag;

    /// Type of a matrix `L[i,k]`.
    type MatrixEntry;
    /// Type of diagonal entry `d[k]`.
    type DiagnalEntry;

    /// Calculate symbolic elimination tree from matrix A.
    fn calc_etree(a_lower: &Self::Matrix) -> EliminationTree;

    /// Activate column j.
    fn activate_col(&mut self, col_j: usize);

    /// Load A(i,j) into accumulator `C(i,j)`:  `C(i,j) := A[i,j]`.
    fn load_column(&mut self, a_lower: &Self::Matrix);

    /// Apply columns in reach of column j.
    fn apply_to_col_k_in_reach(
        &mut self,
        col_k: usize,
        mat_l_builder: &Self::MatLBuilder,
        diag: &Self::Diag,
        tracer: &mut impl IsLdltTracer<Self>,
    );

    /// Append data from accumulator C to L and D.
    fn append_to_ldlt(
        &mut self,
        mat_l_builder: &mut Self::MatLBuilder,
        diag: &mut Self::Diag,
    ) -> Result<(), Self::Error>;

    /// Clear accumulator entries `C[:,j]` that were touched during column j.
    fn clear(&mut self);
}

/// Builder for L factor.
pub trait IsLMatBuilder {
    /// Matrix type to represent L.
    type Matrix;

    /// Return compressed matrix form.
    fn compress(self) -> Self::Matrix;
}

/// Indices used by LdltTracer
pub struct LdltIndices {
    /// the column of interest `j``
    pub col_j: usize,
    /// column connect to `j` through elimination tree reach.
    pub col_k: usize,
    /// row `i`
    pub row_i: usize,
}

/// Tracer - for optional debug insights.
pub trait IsLdltTracer<Workspace: IsLdltWorkspace> {
    /// Trace to show the elimination tree.
    #[inline]
    fn after_etree(&mut self, _etree: &EliminationTree) {}

    /// Trace to show the loaded column and etree reach for column `j`.
    #[inline]
    fn after_load_column_and_reach(&mut self, _j: usize, _reach: &[usize], _ws: &Workspace) {}

    /// Update on reach for column `j`.
    #[inline]
    fn after_update(
        &mut self,
        _indices: LdltIndices,
        _d: Workspace::DiagnalEntry,
        _l_ik: Workspace::MatrixEntry,
        _l_jk: Workspace::MatrixEntry,
        _c: Workspace::MatrixEntry,
    ) {
    }

    /// Show final L for column `j`.
    #[inline]
    fn after_append_and_sort(&mut self, _j: usize, _l_storage: &SparseLFactorBuilder, _d: &[f64]) {}
}

/// No-op tracer
#[derive(Debug)]
pub struct NoopLdltTracer<Workspace: IsLdltWorkspace> {
    phantom: PhantomData<Workspace>,
}
impl<Workspace: IsLdltWorkspace> Default for NoopLdltTracer<Workspace> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Workspace: IsLdltWorkspace> NoopLdltTracer<Workspace> {
    /// New no-op tracer.
    pub fn new() -> Self {
        NoopLdltTracer {
            phantom: Default::default(),
        }
    }
}
impl<Workspace: IsLdltWorkspace> IsLdltTracer<Workspace> for NoopLdltTracer<Workspace> {}

mod tests {

    #[test]
    fn scalar_solver_tests() {
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
            IsLinearSolver,
            SparseLdlt,
            matrix::{
                LowerBlockSparseMatrixBuilder,
                block_sparse::PartitionSpec,
                dense::DenseSymmetricMatrixBuilder,
                sparse::{
                    SparseSymmetricMatrixBuilder,
                    faer_sparse_matrix::{
                        FaerTripletMatrix,
                        FaerUpperTripletMatrix,
                    },
                },
            },
            positive_semidefinite::{
                block_sparse_ldlt::BlockSparseLdlt,
                faer_sparse_ldlt::FaerSparseLdlt,
            },
            prelude::*,
        };

        pub fn all_examples<Builder: IsSymmetricMatrixBuilder>()
        -> Vec<(Builder::Matrix, DVector<f64>)> {
            let mut examples = Vec::new();

            // --- first example ---
            {
                let partitions = vec![PartitionSpec {
                    block_count: 2,
                    block_dimension: 3,
                }];
                let mut builder = Builder::zero(&partitions);
                let block_a00 = MatF64::<3, 3>::from_array2([
                    [3.3, 1.0, 0.0],
                    [1.0, 3.2, 1.0],
                    [0.0, 1.0, 3.1],
                ]);
                let block_a01 = MatF64::<3, 3>::from_array2([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]);
                let block_a11 = MatF64::<3, 3>::from_array2([
                    [2.3, 0.0, 0.0],
                    [0.0, 2.2, 0.0],
                    [0.0, 0.0, 2.1],
                ]);

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
                    block_count: 4,
                    block_dimension: 2,
                }];
                let mut builder = Builder::zero(&parts);

                let n_scalar = parts
                    .iter()
                    .map(|p| p.block_dimension * p.block_count)
                    .sum::<usize>();
                let mut mat_l = na::DMatrix::<f64>::zeros(n_scalar, n_scalar);

                let nb = parts[0].block_count;
                let m = parts[0].block_dimension;
                let so = |b: usize| b * m;

                for i in 0..nb {
                    for d in 0..m {
                        mat_l[(so(i) + d, so(i) + d)] = 1.0;
                    }
                    if i > 0 {
                        let mut blk = mat_l.view_mut((so(i), so(i - 1)), (m, m));
                        for c in 0..m {
                            for r in 0..m {
                                blk[(r, c)] = 0.05 * (1.0 + (r + c) as f64);
                            }
                        }
                    }
                }

                let mat_a = &mat_l * mat_l.transpose();
                for br in 0..nb {
                    for bc in 0..=br {
                        builder.add_lower_block(
                            &[0, 0],
                            [br, bc],
                            &mat_a.view((so(br), so(bc)), (m, m)),
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
                        block_dimension: 3,
                        block_count: 2,
                    },
                    PartitionSpec {
                        block_dimension: 1,
                        block_count: 3,
                    },
                    PartitionSpec {
                        block_dimension: 2,
                        block_count: 1,
                    },
                ];
                let mut builder = Builder::zero(&parts);

                let mut part_off = vec![0];
                let mut acc = 0usize;
                for p in &parts {
                    acc += p.block_dimension * p.block_count;
                    part_off.push(acc);
                }
                let n_scalar = acc;

                let blk_off: Vec<usize> = {
                    let mut o = vec![0];
                    for p in &parts {
                        o.push(o.last().unwrap() + p.block_count);
                    }
                    o
                };
                let n_blocks_total = *blk_off.last().unwrap();

                let global_blk_dim = |g: usize| {
                    let mut pr = 0;
                    while pr + 1 < blk_off.len() && g >= blk_off[pr + 1] {
                        pr += 1;
                    }
                    parts[pr].block_dimension
                };
                let global_scalar_off = |g: usize| {
                    let mut pr = 0;
                    while pr + 1 < blk_off.len() && g >= blk_off[pr + 1] {
                        pr += 1;
                    }
                    let local = g - blk_off[pr];
                    part_off[pr] + local * parts[pr].block_dimension
                };

                let mut mat_l = na::DMatrix::<f64>::zeros(n_scalar, n_scalar);
                for i in 0..n_blocks_total {
                    let mi = global_blk_dim(i);
                    let i0 = global_scalar_off(i);
                    for d in 0..mi {
                        mat_l[(i0 + d, i0 + d)] = 1.0;
                    }
                    for j in 0..i {
                        let mj = global_blk_dim(j);
                        let j0 = global_scalar_off(j);
                        let mut blk = mat_l.view_mut((i0, j0), (mi, mj));
                        for c in 0..mj {
                            for r in 0..mi {
                                blk[(r, c)] = 0.04 * ((r + 1 + c + 1) as f64);
                            }
                        }
                    }
                }
                let mat_a = &mat_l * mat_l.transpose();
                for rp in 0..parts.len() {
                    let mr = parts[rp].block_dimension;
                    let nbr = parts[rp].block_count;
                    let rbase = part_off[rp];
                    for cp in 0..=rp {
                        let mc = parts[cp].block_dimension;
                        let nbc = parts[cp].block_count;
                        let cbase = part_off[cp];

                        for br in 0..nbr {
                            let r0 = rbase + br * mr;
                            let bc_max = if rp == cp { br } else { nbc - 1 };
                            for bc in 0..=bc_max {
                                let c0 = cbase + bc * mc;
                                builder.add_lower_block(
                                    &[rp, cp],
                                    [br, bc],
                                    &mat_a.view((r0, c0), (mr, mc)),
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
                    block_dimension: 1,
                    block_count: 8,
                }];
                let mut builder = Builder::zero(&parts);

                let n = parts[0].block_dimension * parts[0].block_count;
                let mut mat_l = na::DMatrix::<f64>::zeros(n, n);
                for i in 0..n {
                    mat_l[(i, i)] = 1.0;
                    if i > 0 {
                        mat_l[(i, i - 1)] = 0.2;
                    }
                    if i > 1 {
                        mat_l[(i, i - 2)] = 0.05;
                    }
                }
                let mat_a = &mat_l * mat_l.transpose();
                for i in 0..parts[0].block_count {
                    for j in 0..=i {
                        builder.add_lower_block(&[0, 0], [i, j], &mat_a.view((i, j), (1, 1)));
                    }
                }

                let b = na::DVector::<f64>::from_row_slice(&[
                    1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0,
                ]);
                examples.push((builder.build(), b));
            }

            examples
        }

        let dense_ex = all_examples::<DenseSymmetricMatrixBuilder>();
        let sparse_ex = all_examples::<SparseSymmetricMatrixBuilder>();
        let block_sparse_ex = all_examples::<LowerBlockSparseMatrixBuilder>();

        let parallelize = false;

        for i in 0..dense_ex.len() {
            println!("example {i}\n");
            let (dense_mat_a, b) = &dense_ex[i];
            let (sparse_mat_a, _b) = &sparse_ex[i];
            let (block_sparse_mat_a, _b) = &block_sparse_ex[i];

            let faer_mat_a = FaerTripletMatrix::from_lower(sparse_mat_a);
            let faer_upper_mat_a = FaerUpperTripletMatrix::from_lower(sparse_mat_a);

            let x_dense_lu = DenseLu {}.solve(parallelize, dense_mat_a, b).unwrap();
            let x_faer_sparse_lu = FaerSparseLu {}
                .solve(parallelize, &faer_mat_a.compress(), b)
                .unwrap();
            let x_faer_sparse_qr = FaerSparseQr {}
                .solve(parallelize, &faer_mat_a.compress(), b)
                .unwrap();

            let x_dense_ldlt = DenseLdlt::default()
                .solve(parallelize, dense_mat_a, b)
                .unwrap();
            let x_sparse_ldlt = SparseLdlt::default()
                .solve(parallelize, &sparse_mat_a.compress(), b)
                .unwrap();
            let x_block_sparse_ldlt = BlockSparseLdlt::default()
                .solve(parallelize, &block_sparse_mat_a.compress(), b)
                .unwrap();
            let x_faer_sparse_ldlt = FaerSparseLdlt::default()
                .solve(parallelize, &faer_upper_mat_a.compress(), b)
                .unwrap();

            print!("dense LU: x = {x_dense_lu}");
            print!("faer sparse LU: x = {x_faer_sparse_lu}");

            print!("faer sparse QR: x = {x_faer_sparse_qr}");

            print!("dense LDLᵀ x = {x_dense_ldlt}");
            print!("sparse LDLᵀ x = {x_sparse_ldlt}");
            print!("block-sparse LDLᵀ x = {x_block_sparse_ldlt}");
            print!("faer sparse LDLᵀ x = {x_faer_sparse_ldlt}");

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
                dense_mat_a.clone() * x_block_sparse_ldlt,
                b.clone(),
                epsilon = 1e-6
            );
            approx::assert_abs_diff_eq!(
                dense_mat_a.clone() * x_faer_sparse_ldlt,
                b.clone(),
                epsilon = 1e-6
            );
        }
    }
}
