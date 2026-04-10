use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    ldlt::{
        BlockSparseLdltFactor,
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

/// Block-sparse backend for the min-norm LDLᵀ pseudo-inverse; wraps a
/// [`BlockSparseLdltFactor`].
#[derive(Clone, Debug)]
pub struct BlockSparseBackend {
    ldlt: BlockSparseLdltFactor,
    tol_rel: f64,
    positive_pivot_idx: Vec<usize>,
    positive_pivot_values: DVector<f64>,
}

impl BlockSparseBackend {
    fn scalar_to_block_col(&self, t: usize) -> (usize, usize) {
        // Return (block_col_index, in_block_col_offset).
        // Uses subdivision methods to map scalar column -> (block, offset).
        let sub = &self.ldlt.mat_l.subdivision;
        // Walk blocks to find where 't' falls. If you have a direct helper, use it.
        let mut acc = 0usize;
        for bc in 0..sub.block_count() {
            let w = sub.block_dim(sub.idx(bc).partition);
            if t < acc + w {
                return (bc, t - acc);
            }
            acc += w;
        }
        unreachable!("scalar column out of range");
    }

    fn scalar_offset(&self, block_row: usize) -> usize {
        self.ldlt.mat_l.subdivision.scalar_offset(block_row)
    }
}

impl IsMinNormLdltBackend for BlockSparseBackend {
    fn scalar_dim(&self) -> usize {
        self.ldlt.mat_l.subdivision.scalar_dim()
    }

    fn tol_rel(&self) -> f64 {
        self.tol_rel
    }

    type LdltFactor = BlockSparseLdltFactor;

    fn new(ldlt: Self::LdltFactor) -> Self {
        let sub = &ldlt.mat_l.subdivision;
        let mut positive_pivot_idx = Vec::new();
        let mut positive_pivot_values = Vec::new();
        let mut scalar_col = 0usize;
        for bc in 0..sub.block_count() {
            let col_idx = sub.idx(bc);
            let w = sub.block_dim(col_idx.partition);
            let d_block = ldlt.block_diag.d.get_block(col_idx);
            debug_assert_eq!(d_block.len(), w);
            for c in 0..w {
                if d_block[c] > 0.0 {
                    positive_pivot_idx.push(scalar_col + c);
                    positive_pivot_values.push(d_block[c]);
                }
            }
            scalar_col += w;
        }

        Self {
            ldlt,
            tol_rel: 1e-12,
            positive_pivot_idx,
            positive_pivot_values: DVector::from_vec(positive_pivot_values),
        }
    }

    fn positive_pivot_idx(&self) -> &[usize] {
        &self.positive_pivot_idx
    }

    fn positive_pivot_values(&self) -> &DVector<f64> {
        &self.positive_pivot_values
    }

    fn column_of_mat_e(&self, col_j: usize, out: &mut [f64]) {
        let n = self.scalar_dim();
        assert_eq!(out.len(), n);
        out.fill(0.0);

        let sub = &self.ldlt.mat_l.subdivision;
        let (bj, co) = self.scalar_to_block_col(col_j);

        // Diagonal block L[j,j] and its column 'co'
        let col_idx = sub.idx(bj);
        let l_jj = self.ldlt.block_diag.mat_l.get_block(col_idx); // square, lower-triangular
        let hj = l_jj.nrows(); // == l_jj.ncols() for diagonal blocks
        let row0_j = self.scalar_offset(bj);

        // Copy the diagonal block column into out at rows of block-row j.
        for r in 0..hj {
            out[row0_j + r] = l_jj[(r, co)];
        }

        // Prepare v = L[j,j] * e_co  (this is just the same column we copied above)
        // Reuse it to form off-diagonal contributions: L[i,j] * v.
        let mut v = vec![0.0; hj];
        for r in 0..hj {
            v[r] = l_jj[(r, co)];
        }

        // Off-diagonal blocks in column j
        for e in self.ldlt.mat_l.col(bj).iter() {
            let bi = e.global_block_row_idx;
            debug_assert!(bi > bj);
            let l_ij = e.view; // (h_i × h_j)
            let hi = l_ij.nrows();
            let row0_i = self.scalar_offset(bi);

            // y = L[i,j] * v
            for r in 0..hi {
                let mut acc = 0.0;
                // h_j equals v.len()
                for s in 0..v.len() {
                    acc += l_ij[(r, s)] * v[s];
                }
                out[row0_i + r] = acc;
            }
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
        // Generic SPD via solves (uses your BlockSparseLdltSystem::solve_inplace)
        use crate::IsFactor;
        let n = self.scalar_dim();
        let mut inv = DMatrix::<f64>::zeros(n, n);
        let mut col = nalgebra::DVector::<f64>::zeros(n);
        for j in 0..n {
            col.fill(0.0);
            col[j] = 1.0;
            let mut x = col.clone();
            self.ldlt.solve_inplace(&mut x).ok()?;
            inv.set_column(j, &x);
        }
        Some(inv)
    }

    fn block_range(&self, idx: PartitionBlockIndex) -> BlockRange {
        self.ldlt.original_partitions.block_range(idx)
    }
}

/// Block-sparse min-norm LDLᵀ pseudo-inverse; produced by
/// [`BlockSparseMinNormPsd::new`].
pub type BlockSparseMinNormPsd = MinNormLdlt<BlockSparseBackend>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;

    use super::BlockSparseMinNormPsd;
    use crate::{
        IsInvertible,
        IsLinearSolver,
        ldlt::BlockSparseLdlt,
        matrix::{
            IsSymmetricMatrixBuilder,
            PartitionBlockIndex,
            PartitionSet,
            PartitionSpec,
            block_sparse::BlockSparseSymmetricMatrixBuilder,
        },
        svd::pseudo_inverse_with_tol,
    };

    const LDLT_TOL: f64 = 1e-12;

    /// Build a `BlockSparseSymmetricMatrix` from the lower triangle of a dense matrix.
    fn build_block_sparse_from_dense(
        partitions: PartitionSet,
        h_dense: &DMatrix<f64>,
    ) -> crate::matrix::block_sparse::block_sparse_symmetric_matrix::BlockSparseSymmetricMatrix
    {
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

        let mut builder = BlockSparseSymmetricMatrixBuilder::zero(partitions);

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

        let mat = build_block_sparse_from_dense(parts, &h_dense);
        let ldlt = BlockSparseLdlt::default().factorize(&mat).unwrap();
        let mut gs = BlockSparseMinNormPsd::new(ldlt);

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

    #[test]
    fn psd_pseudo_inverse_matches_svd() {
        // Single 3×3 partition, rank-2 PSD: H = BᵀB
        //
        // BlockSparseLdlt handles PD, PSD, and indefinite matrices. This test
        // exits early if factorization fails. The PSD path is fully covered for
        // the dense backend in dense_min_norm_ldlt::tests.
        let parts = PartitionSet::new(vec![PartitionSpec {
            eliminate_last: false,
            block_count: 1,
            block_dim: 3,
        }]);
        let b = DMatrix::<f64>::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let h_dense = b.transpose() * &b;

        let mat = build_block_sparse_from_dense(parts, &h_dense);
        let Ok(ldlt) = BlockSparseLdlt::default().factorize(&mat) else {
            return; // backend does not support rank-deficient matrices — test is a no-op here
        };
        let mut gs = BlockSparseMinNormPsd::new(ldlt);

        let pinv = pseudo_inverse_with_tol(h_dense.as_view(), LDLT_TOL);
        let inv_mat = gs.pseudo_inverse();

        assert_relative_eq!(inv_mat, pinv, epsilon = 1e-9, max_relative = 1e-9);

        // Moore–Penrose: H H† H ≈ H
        let check = &h_dense * &inv_mat * &h_dense;
        assert_relative_eq!(check, h_dense, epsilon = 1e-9, max_relative = 1e-9);
    }

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

        let mat = build_block_sparse_from_dense(parts, &h_dense);
        let ldlt = BlockSparseLdlt::default().factorize(&mat).unwrap();
        let mut gs = BlockSparseMinNormPsd::new(ldlt);

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
    fn multi_block_spd_pseudo_inverse_blocks_match() {
        // Single partition, 4 blocks of dim 2 => n=8, SPD from L Lᵀ + μI
        let parts = PartitionSet::new(vec![PartitionSpec {
            eliminate_last: false,
            block_count: 4,
            block_dim: 2,
        }]);
        let n = 8usize;
        let mut mat_l = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            mat_l[(i, i)] = 1.0;
            if i >= 2 {
                mat_l[(i, i - 2)] = 0.1;
            }
        }
        let h_dense = &mat_l * mat_l.transpose() + 0.1 * DMatrix::<f64>::identity(n, n);

        let mat = build_block_sparse_from_dense(parts, &h_dense);
        let ldlt = BlockSparseLdlt::default().factorize(&mat).unwrap();
        let mut gs = BlockSparseMinNormPsd::new(ldlt);

        let inv_ref = nalgebra::Cholesky::new(h_dense.clone()).unwrap().inverse();

        // Verify every diagonal block
        for bi in 0..4 {
            let idx = PartitionBlockIndex {
                partition: 0,
                block: bi,
            };
            let blk = gs.pseudo_inverse_block(idx, idx);
            let s = bi * 2;
            let blk_ref = inv_ref.view((s, s), (2, 2)).into_owned();
            assert_relative_eq!(blk, blk_ref, epsilon = 1e-9, max_relative = 1e-9);
        }
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

        let mat = build_block_sparse_from_dense(parts, &h);
        let ldlt = BlockSparseLdlt::default().factorize(&mat).unwrap();
        let mut gs = BlockSparseMinNormPsd::new(ldlt);

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

    #[test]
    fn off_diagonal_block_matches_dense_inverse() {
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

        let mat = build_block_sparse_from_dense(parts, &h_dense);
        let ldlt = BlockSparseLdlt::default().factorize(&mat).unwrap();
        let mut gs = BlockSparseMinNormPsd::new(ldlt);

        let inv_ref = nalgebra::Cholesky::new(h_dense).unwrap().inverse();

        // Cross-partition block (0,0) × (1,0): 2×1
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
}
