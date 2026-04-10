/// Block-sparse backend for the min-norm LDLᵀ pseudo-inverse.
pub mod block_sparse_min_norm_ldlt;
/// Dense backend for the min-norm LDLᵀ pseudo-inverse.
pub mod dense_min_norm_ldlt;
/// Sparse backend for the min-norm LDLᵀ pseudo-inverse.
pub mod sparse_min_norm_ldlt;

use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    IsInvertible,
    kernel::{
        diag_matsolve_inplaced,
        diag_right_matsolve_inplace,
    },
    ldlt::{
        ldlt_decompose_inplace,
        ldlt_matsolve_inplace,
        ldlt_right_matsolve_inplace,
    },
    matrix::{
        BlockRange,
        PartitionBlockIndex,
    },
};

/// Min-norm `LDLᵀ` pseudo-inverse solver.
///
/// If `A` is full-rank, then `A x = b` has a unique solution `x`. Furthermore,
/// the inverse of `A` can be obtained by solving `A X = I`.
///
/// If `A` is rank-deficient, then `A x = b` has infinitely many solutions and `A` has no
/// matrix inverse. However, one can compute the Moore–Penrose pseudo-inverse `A⁺` instead.
/// It is the min-norm solution to `A X = I`, i.e.:
///
/// ```text
///   A⁺ = argmin |X|  subject to  A X A = A
/// ```
///
/// Equivalently, `A⁺` can be defined as the limit:
///
/// ```text
///   A⁺ := lim_{ε→0} (AᵀA + εI)⁻¹ Aᵀ
/// ```
///
/// ## Algorithm
///
/// Given `A = L D Lᵀ` with `LDLᵀ` factorization, let `J` be the set of indices with positive
/// pivots `d_j > 0` (i.e. the rank-`r` subspace), and let `E = L[:,J]` be the `n×r` submatrix
/// of `L` restricted to those columns.
///
/// Build the Gram matrix `G = EᵀE` (which is `r×r` symmetric positive definite) and factorize it:
/// `G = Lᴳ Dᴳ (Lᴳ)ᵀ`. Let `M = G⁻¹`.
///
/// The pseudo-inverse is then:
///
/// ```text
///   A⁺ = E · M · diag(d_J)⁻¹ · M · Eᵀ
/// ```
///
/// In the implementation, `M` is applied via right-solves of the `LDLᵀ` factorization of `G`
/// (for the full inverse) or via left-solves on column tiles of `E` (for block queries).
#[derive(Clone, Debug)]
pub struct MinNormLdlt<Backend: IsMinNormLdltBackend> {
    backend: Backend,
    gram_mat_l: DMatrix<f64>,
    gram_diag: DVector<f64>,
    mat_e: DMatrix<f64>,
}

impl<F: IsMinNormLdltBackend> MinNormLdlt<F> {
    /// Create a min-norm LDLᵀ pseudo-inverse solver from an LDLᵀ factorization.
    pub fn new(fact: F::LdltFactor) -> Self {
        let f = F::new(fact);

        let n = f.scalar_dim();
        let r = f.rank();

        // Build E = L[:,J] (n × r)
        let mut e = DMatrix::<f64>::zeros(n, r);
        for (c, &t) in f.positive_pivot_idx().iter().enumerate() {
            f.column_of_mat_e(t, e.column_mut(c).as_mut_slice());
        }
        // G = Eᵀ E (r × r), then LDLᵀ(G)
        let mut g = e.transpose() * &e;
        let mut g_diag = DVector::<f64>::zeros(r);
        ldlt_decompose_inplace(g.as_view_mut(), g_diag.as_view_mut(), f.tol_rel())
            .expect("G must be symmetric positive definite on the positive-pivot subspace");
        Self {
            backend: f,
            gram_diag: g_diag,
            gram_mat_l: g,
            mat_e: e,
        }
    }
}

impl<F: IsMinNormLdltBackend> IsInvertible for MinNormLdlt<F> {
    fn pseudo_inverse(&mut self) -> DMatrix<f64> {
        let n = self.backend.scalar_dim();
        let r = self.backend.rank();

        if n == 0 || r == 0 {
            return DMatrix::<f64>::zeros(n, n);
        }

        // Fast symmetric positive definite path if the backend can do it:
        if let Some(spd_inv) = self.backend.try_inverse() {
            return spd_inv;
        }
        // When r == n (full rank), the following path reduces to the exact inverse.

        // W = E · G⁻¹ · D_J⁻¹ · G⁻¹  (n × r)
        let mut w = self.mat_e.clone();
        ldlt_right_matsolve_inplace(
            self.gram_mat_l.as_view(),
            self.gram_diag.as_view(),
            w.as_view_mut(),
        );
        diag_right_matsolve_inplace(
            self.backend.positive_pivot_values().as_view(),
            &mut w.as_view_mut(),
        );
        ldlt_right_matsolve_inplace(
            self.gram_mat_l.as_view(),
            self.gram_diag.as_view(),
            w.as_view_mut(),
        );

        // A⁺ = W · Eᵀ  (n × n)
        let mut out = DMatrix::<f64>::zeros(n, n);
        w.mul_to(&self.mat_e.transpose(), &mut out);
        out
    }

    fn pseudo_inverse_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> DMatrix<f64> {
        let r = self.backend.rank();

        let row_range = self.backend.block_range(row_idx);
        let col_range = self.backend.block_range(col_idx);
        let mut out = DMatrix::<f64>::zeros(row_range.block_dim, col_range.block_dim);

        if r == 0 || row_range.block_dim == 0 || col_range.block_dim == 0 {
            return out;
        }

        // If the backend provides a fast symmetric positive definite inverse, slice the block out
        // of it.
        if let Some(cols) = self.backend.try_column_of_inverse(&col_range) {
            // cols is n × dj, extract the wanted row-block.
            return cols
                .view(
                    (row_range.start_idx, 0),
                    (row_range.block_dim, col_range.block_dim),
                )
                .into_owned();
        }

        // E_i = E[rows_i, :]    (di × r)
        // b   = E[rows_j, :]ᵀ  (r  × dj)
        let e_i = self
            .mat_e
            .view((row_range.start_idx, 0), (row_range.block_dim, r))
            .into_owned();
        let mut b = self
            .mat_e
            .view((col_range.start_idx, 0), (col_range.block_dim, r))
            .transpose()
            .into_owned();

        // Apply G⁻¹ D_J⁻¹ G⁻¹ to b on the left:
        // b <- G⁻¹ b
        ldlt_matsolve_inplace(
            self.gram_mat_l.as_view(),
            self.gram_diag.as_view(),
            b.as_view_mut(),
        );
        // b <- D_J⁻¹ b   (row-wise scaling by positive pivots)
        diag_matsolve_inplaced(
            self.backend.positive_pivot_values().as_view(),
            &mut b.as_view_mut(),
        );
        // b <- G⁻¹ b
        ldlt_matsolve_inplace(
            self.gram_mat_l.as_view(),
            self.gram_diag.as_view(),
            b.as_view_mut(),
        );

        // A⁺[i,j] = E_i · b
        e_i.mul_to(&b, &mut out);

        out
    }
}

/// What the min-norm algorithm needs from an LDLᵀ backend.
pub trait IsMinNormLdltBackend {
    /// The underlying LDLᵀ factorization of matrix A.
    type LdltFactor;

    /// Create a new min-norm LDLᵀ backend.
    fn new(ldlt: Self::LdltFactor) -> Self;

    /// The scalar dimension of the square matrices A = L D Lᵀ.
    fn scalar_dim(&self) -> usize;

    /// The rank of the matrix A = L D Lᵀ.
    fn rank(&self) -> usize {
        debug_assert_eq!(
            self.positive_pivot_idx().len(),
            self.positive_pivot_values().len()
        );

        self.positive_pivot_idx().len()
    }

    /// Relative tolerance, used to decompose the gram matrix G.
    fn tol_rel(&self) -> f64;

    /// Indices of positive pivots (len = rank).
    fn positive_pivot_idx(&self) -> &[usize];

    /// Vector of positive pivot values (len = rank).
    fn positive_pivot_values(&self) -> &DVector<f64>;

    /// Emit the scalar column `L[:, col_j]` into `out` (length n), with the **unit-diagonal
    /// convention**: `out[col_j]` = 1.0; entries above `col_j` are 0.0; for rows > `col_j`,
    /// the stored strict-lower values.
    fn column_of_mat_e(&self, col_j: usize, out: &mut [f64]);

    /// Tries to calculate the inverse of `A = L D Lᵀ`. If `A` is rank-deficient, then None is
    /// returned.
    fn try_inverse(&self) -> Option<DMatrix<f64>>;

    /// Tries to calculate a block-column of the inverse of `A = L D Lᵀ`. If `A` is rank-deficient,
    /// then None is returned.
    fn try_column_of_inverse(&self, col_range: &BlockRange) -> Option<nalgebra::DMatrix<f64>>;

    /// Return the block range for a given partition of A.
    fn block_range(&self, idx: PartitionBlockIndex) -> BlockRange;
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;

    use crate::{
        IsInvertible,
        IsLinearSolver,
        ldlt::{
            BlockSparseLdlt,
            DenseLdlt,
            SparseLdlt,
            min_norm_ldlt::{
                block_sparse_min_norm_ldlt::BlockSparseMinNormPsd,
                dense_min_norm_ldlt::DenseMinNormFactor,
                sparse_min_norm_ldlt::SparseMinNormPsd,
            },
        },
        matrix::{
            IsSymmetricMatrixBuilder,
            PartitionBlockIndex,
            PartitionSet,
            PartitionSpec,
            block_sparse::BlockSparseSymmetricMatrixBuilder,
            dense::DenseSymmetricMatrixBuilder,
            sparse::SparseSymmetricMatrixBuilder,
        },
    };

    /// Same SPD input to dense/sparse/block-sparse backends produces matching
    /// pseudo_inverse() and pseudo_inverse_block() results.
    #[test]
    fn cross_backend_min_norm_equivalence() {
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            eliminate_last: false,
                },
            PartitionSpec {
                block_count: 2,
                block_dim: 1,
            eliminate_last: false,
                },
        ]);
        let n = 4;

        // SPD H = A^T A + μI
        let a = DMatrix::<f64>::from_row_slice(
            3,
            n,
            &[1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 4.0, 0.5, 2.0, 0.0, 0.0, 3.0],
        );
        let mu = 0.2;
        let h_dense = a.transpose() * &a + mu * DMatrix::<f64>::identity(n, n);

        let specs = parts.specs().to_vec();
        let mut offsets = vec![0usize; specs.len()];
        let mut off = 0;
        for (i, p) in specs.iter().enumerate() {
            offsets[i] = off;
            off += p.block_count * p.block_dim;
        }

        // Helper: iterate lower-triangular blocks
        macro_rules! fill_builder {
            ($builder:expr) => {
                for (pi_idx, pi) in specs.iter().enumerate() {
                    let oi = offsets[pi_idx];
                    for bi in 0..pi.block_count {
                        let si = oi + bi * pi.block_dim;
                        let row_idx = PartitionBlockIndex {
                            partition: pi_idx,
                            block: bi,
                        };
                        $builder.add_lower_block(
                            row_idx,
                            row_idx,
                            &h_dense.view((si, si), (pi.block_dim, pi.block_dim)),
                        );
                        for bj in 0..bi {
                            let sj = oi + bj * pi.block_dim;
                            $builder.add_lower_block(
                                row_idx,
                                PartitionBlockIndex {
                                    partition: pi_idx,
                                    block: bj,
                                },
                                &h_dense.view((si, sj), (pi.block_dim, pi.block_dim)),
                            );
                        }
                        for (pj_idx, pj) in specs.iter().enumerate().take(pi_idx) {
                            let oj = offsets[pj_idx];
                            for bj in 0..pj.block_count {
                                let sj = oj + bj * pj.block_dim;
                                $builder.add_lower_block(
                                    row_idx,
                                    PartitionBlockIndex {
                                        partition: pj_idx,
                                        block: bj,
                                    },
                                    &h_dense.view((si, sj), (pi.block_dim, pj.block_dim)),
                                );
                            }
                        }
                    }
                }
            };
        }

        // Dense backend
        let mut dense_builder = DenseSymmetricMatrixBuilder::zero(parts.clone());
        fill_builder!(dense_builder);
        let dense_mat = dense_builder.build();
        let dense_ldlt = DenseLdlt::default().factorize(&dense_mat).unwrap();
        let mut dense_gs = DenseMinNormFactor::new(dense_ldlt);

        // Sparse backend
        let mut sparse_builder = SparseSymmetricMatrixBuilder::zero(parts.clone());
        fill_builder!(sparse_builder);
        let sparse_mat = sparse_builder.build();
        let sparse_ldlt = SparseLdlt::default().factorize(&sparse_mat).unwrap();
        let mut sparse_gs = SparseMinNormPsd::new(sparse_ldlt);

        // Block-sparse backend
        let mut bsparse_builder = BlockSparseSymmetricMatrixBuilder::zero(parts.clone());
        fill_builder!(bsparse_builder);
        let bsparse_mat = bsparse_builder.build();
        let bsparse_ldlt = BlockSparseLdlt::default().factorize(&bsparse_mat).unwrap();
        let mut bsparse_gs = BlockSparseMinNormPsd::new(bsparse_ldlt);

        // Compare pseudo_inverse()
        let inv_dense = dense_gs.pseudo_inverse();
        let inv_sparse = sparse_gs.pseudo_inverse();
        let inv_bsparse = bsparse_gs.pseudo_inverse();

        assert_relative_eq!(inv_dense, inv_sparse, epsilon = 1e-9, max_relative = 1e-9);
        assert_relative_eq!(inv_dense, inv_bsparse, epsilon = 1e-9, max_relative = 1e-9);

        // Compare pseudo_inverse_block() for an off-diagonal tile
        let idx00 = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };
        let idx10 = PartitionBlockIndex {
            partition: 1,
            block: 0,
        };
        let blk_dense = dense_gs.pseudo_inverse_block(idx00, idx10);
        let blk_sparse = sparse_gs.pseudo_inverse_block(idx00, idx10);
        let blk_bsparse = bsparse_gs.pseudo_inverse_block(idx00, idx10);

        assert_relative_eq!(blk_dense, blk_sparse, epsilon = 1e-9, max_relative = 1e-9);
        assert_relative_eq!(blk_dense, blk_bsparse, epsilon = 1e-9, max_relative = 1e-9);
    }
}
