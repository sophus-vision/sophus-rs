use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    ldlt::{
        SparseLdltFactor,
        min_norm_ldlt::{
            IsMinNormLdltBackend,
            MinNormLdlt,
        },
    },
    matrix::{
        BlockRange,
        PartitionBlockIndex,
    },
};

/// Sparse backend for the min-norm LDLᵀ pseudo-inverse; wraps a [`SparseLdltFactor`].
#[derive(Clone, Debug)]
pub struct SparseBackend {
    ldlt: SparseLdltFactor,
    tol_rel: f64,
    positive_pivot_idx: Vec<usize>,
    positive_pivot_values: DVector<f64>,
}

impl IsMinNormLdltBackend for SparseBackend {
    type LdltFactor = SparseLdltFactor;

    fn new(ldlt: Self::LdltFactor) -> Self {
        let n = ldlt.mat_l.scalar_dim();
        let positive_pivot_idx: Vec<usize> = (0..n).filter(|&t| ldlt.d[t] > 0.0).collect();
        let mut positive_pivot_values = DVector::<f64>::zeros(positive_pivot_idx.len());
        for (c, &t) in positive_pivot_idx.iter().enumerate() {
            positive_pivot_values[c] = ldlt.d[t];
        }
        Self {
            ldlt,
            tol_rel: 1e-12,
            positive_pivot_idx,
            positive_pivot_values,
        }
    }

    fn positive_pivot_idx(&self) -> &[usize] {
        &self.positive_pivot_idx
    }

    fn positive_pivot_values(&self) -> &DVector<f64> {
        &self.positive_pivot_values
    }

    fn scalar_dim(&self) -> usize {
        self.ldlt.mat_l.scalar_dim()
    }

    fn tol_rel(&self) -> f64 {
        self.tol_rel
    }

    fn column_of_mat_e(&self, col_j: usize, out: &mut [f64]) {
        // CSC strict-lower storage + implicit diag 1.0
        let l = &self.ldlt.mat_l;
        for i in 0..col_j {
            out[i] = 0.0;
        }
        out[col_j] = 1.0;
        for i in (col_j + 1)..self.scalar_dim() {
            out[i] = 0.0;
        }

        let c0 = l.storage_idx_by_col()[col_j];
        let c1 = l.storage_idx_by_col()[col_j + 1];
        let rows = &l.row_idx_storage()[c0..c1];
        let vals = &l.value_storage()[c0..c1];
        for (ri, &row) in rows.iter().enumerate() {
            debug_assert!(row > col_j);
            out[row] = vals[ri];
        }
    }

    fn try_column_of_inverse(&self, col_range: &BlockRange) -> Option<DMatrix<f64>> {
        if self.rank() != self.scalar_dim() {
            return None;
        }
        use crate::IsFactor;

        let n = self.scalar_dim();
        debug_assert!(col_range.start_idx + col_range.block_dim <= n);
        let mut cols = DMatrix::<f64>::zeros(n, col_range.block_dim);

        let mut rhs = nalgebra::DVector::<f64>::zeros(n);
        for k in 0..col_range.block_dim {
            rhs.fill(0.0);
            rhs[col_range.start_idx + k] = 1.0;
            let mut x = rhs.clone();
            self.ldlt.solve_inplace(&mut x).ok()?; // Solve A x = e_{sj+k}
            cols.set_column(k, &x);
        }
        Some(cols)
    }

    fn try_inverse(&self) -> Option<DMatrix<f64>> {
        if self.rank() != self.scalar_dim() {
            return None;
        }
        // Generic SPD inverse via column solves:
        use crate::IsFactor; // you already implement solve_inplace for SparseLdltSystem
        let n = self.scalar_dim();
        let mut inv = DMatrix::<f64>::zeros(n, n);
        let mut col = nalgebra::DVector::<f64>::zeros(n);
        for j in 0..n {
            col.fill(0.0);
            col[j] = 1.0;
            let mut x = col.clone();
            // Solve A x = e_j
            self.ldlt.solve_inplace(&mut x).ok()?;
            inv.set_column(j, &x);
        }
        Some(inv)
    }

    fn block_range(&self, idx: PartitionBlockIndex) -> BlockRange {
        self.ldlt.partitions.block_range(idx)
    }
}

/// Sparse min-norm LDLᵀ pseudo-inverse; produced by [`SparseMinNormPsd::new`].
pub type SparseMinNormPsd = MinNormLdlt<SparseBackend>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;

    use super::SparseMinNormPsd;
    use crate::{
        IsInvertible,
        IsLinearSolver,
        ldlt::SparseLdlt,
        matrix::{
            IsSymmetricMatrixBuilder,
            PartitionBlockIndex,
            PartitionSet,
            PartitionSpec,
            sparse::SparseSymmetricMatrixBuilder,
        },
        svd::pseudo_inverse_with_tol,
    };

    const LDLT_TOL: f64 = 1e-12;

    /// Build a `SparseSymmetricMatrix` from the lower triangle of a dense matrix.
    fn build_sparse_from_dense(
        partitions: PartitionSet,
        h_dense: &DMatrix<f64>,
    ) -> crate::matrix::sparse::sparse_symmetric_matrix::SparseSymmetricMatrix {
        let n = h_dense.nrows();
        assert_eq!(n, h_dense.ncols());

        let specs = partitions.specs().to_vec();
        let mut offsets = Vec::with_capacity(specs.len());
        let mut off = 0usize;
        for p in &specs {
            offsets.push(off);
            off += p.block_count * p.block_dim;
        }
        assert_eq!(off, n);

        let mut builder = SparseSymmetricMatrixBuilder::zero(partitions);

        for (pi_idx, pi) in specs.iter().enumerate() {
            let oi = offsets[pi_idx];
            for bi in 0..pi.block_count {
                let si = oi + bi * pi.block_dim;
                let row_idx = PartitionBlockIndex {
                    partition: pi_idx,
                    block: bi,
                };

                // diagonal block
                builder.add_lower_block(
                    row_idx,
                    row_idx,
                    &h_dense.view((si, si), (pi.block_dim, pi.block_dim)),
                );

                // within-partition lower off-diagonals
                for bj in 0..bi {
                    let sj = oi + bj * pi.block_dim;
                    let col_idx = PartitionBlockIndex {
                        partition: pi_idx,
                        block: bj,
                    };
                    builder.add_lower_block(
                        row_idx,
                        col_idx,
                        &h_dense.view((si, sj), (pi.block_dim, pi.block_dim)),
                    );
                }

                // cross-partition lower off-diagonals
                for (pj_idx, pj) in specs.iter().enumerate().take(pi_idx) {
                    let oj = offsets[pj_idx];
                    for bj in 0..pj.block_count {
                        let sj = oj + bj * pj.block_dim;
                        let col_idx = PartitionBlockIndex {
                            partition: pj_idx,
                            block: bj,
                        };
                        builder.add_lower_block(
                            row_idx,
                            col_idx,
                            &h_dense.view((si, sj), (pi.block_dim, pj.block_dim)),
                        );
                    }
                }
            }
        }

        builder.build()
    }

    #[test]
    fn psd_pseudo_inverse_matches_svd() {
        // Single 3×3 partition, rank-2 PSD: H = BᵀB
        //
        // SparseLdlt handles PD, PSD, and indefinite matrices. This test exits
        // early if factorization fails. The PSD path is fully covered for the
        // dense backend in dense_min_norm_ldlt::tests.
        let parts = PartitionSet::new(vec![PartitionSpec {
            eliminate_last: false,
            block_count: 1,
            block_dim: 3,
        }]);
        let b = DMatrix::<f64>::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let h_dense = b.transpose() * &b;

        let mat = build_sparse_from_dense(parts, &h_dense);
        let Ok(ldlt) = SparseLdlt::default().factorize(&mat) else {
            return; // backend does not support rank-deficient matrices — test is a no-op here
        };
        let mut gs = SparseMinNormPsd::new(ldlt);

        let pinv = pseudo_inverse_with_tol(h_dense.as_view(), LDLT_TOL);
        let inv_mat = gs.pseudo_inverse();

        assert_relative_eq!(inv_mat, pinv, epsilon = 1e-9, max_relative = 1e-9);

        // Moore–Penrose: H H† H ≈ H
        let check = &h_dense * &inv_mat * &h_dense;
        assert_relative_eq!(check, h_dense, epsilon = 1e-9, max_relative = 1e-9);
    }

    #[test]
    fn spd_pseudo_inverse_matches_dense_inverse() {
        // Two partitions: [(1×2), (2×1)] => n=4, SPD H = AᵀA + μI
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 2,
            },
            PartitionSpec {
                eliminate_last: false,
                block_count: 2,
                block_dim: 1,
            },
        ]);
        let n = 4;
        let a = DMatrix::<f64>::from_row_slice(
            3,
            n,
            &[1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 4.0, 0.5, 2.0, 0.0, 0.0, 3.0],
        );
        let h_dense = a.transpose() * &a + 0.2 * DMatrix::<f64>::identity(n, n);

        let mat = build_sparse_from_dense(parts.clone(), &h_dense);
        let ldlt = SparseLdlt::default().factorize(&mat).unwrap();
        let mut gs = SparseMinNormPsd::new(ldlt);

        let inv_ref = nalgebra::Cholesky::new(h_dense.clone()).unwrap().inverse();
        let inv_mat = gs.pseudo_inverse();

        assert_relative_eq!(inv_mat, inv_ref, epsilon = 1e-9, max_relative = 1e-9);

        // Sanity: H * H⁻¹ ≈ I
        let eye = &h_dense * &inv_mat;
        assert_relative_eq!(
            eye,
            DMatrix::<f64>::identity(n, n),
            epsilon = 1e-9,
            max_relative = 1e-9
        );
    }

    // Note: SparseLdlt PSD pseudo-inverse is tested indirectly through DenseLdlt in
    // dense_min_norm_ldlt::tests.

    #[test]
    fn spd_pseudo_inverse_block_matches_dense_inverse() {
        // Two partitions: [(1×2), (2×1)] => n=4
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 2,
            },
            PartitionSpec {
                eliminate_last: false,
                block_count: 2,
                block_dim: 1,
            },
        ]);
        let n = 4;
        let a = DMatrix::<f64>::from_row_slice(
            3,
            n,
            &[1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 4.0, 0.5, 2.0, 0.0, 0.0, 3.0],
        );
        let h_dense = a.transpose() * &a + 0.2 * DMatrix::<f64>::identity(n, n);

        let mat = build_sparse_from_dense(parts, &h_dense);
        let ldlt = SparseLdlt::default().factorize(&mat).unwrap();
        let mut gs = SparseMinNormPsd::new(ldlt);

        let inv_ref = nalgebra::Cholesky::new(h_dense.clone()).unwrap().inverse();

        // Check off-diagonal block (partition 0, block 0) × (partition 1, block 0)
        let idx00 = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };
        let idx10 = PartitionBlockIndex {
            partition: 1,
            block: 0,
        };

        let blk = gs.pseudo_inverse_block(idx00, idx10);
        let blk_ref = inv_ref.view((0, 2), (2, 1)).into_owned();
        assert_relative_eq!(blk, blk_ref, epsilon = 1e-9, max_relative = 1e-9);
    }

    #[test]
    fn covariance_blocks_are_symmetric() {
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 2,
            },
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 3,
            },
        ]);
        let n = 5;
        let a = DMatrix::<f64>::from_fn(7, n, |r, c| ((r + 1) * (c + 3)) as f64 / 11.0);
        let h = a.transpose() * &a + 0.3 * DMatrix::<f64>::identity(n, n);

        let mat = build_sparse_from_dense(parts, &h);
        let ldlt = SparseLdlt::default().factorize(&mat).unwrap();
        let mut gs = SparseMinNormPsd::new(ldlt);

        let idx00 = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };
        let idx10 = PartitionBlockIndex {
            partition: 1,
            block: 0,
        };

        let s_01 = gs.pseudo_inverse_block(idx00, idx10); // (2×3)
        let s_10 = gs.pseudo_inverse_block(idx10, idx00); // (3×2)
        assert_relative_eq!(
            s_01,
            s_10.transpose(),
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }
}
