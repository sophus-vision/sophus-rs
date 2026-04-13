use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    ldlt::{
        DenseLdltFactor,
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

/// Dense backend for the min-norm LDLᵀ pseudo-inverse; wraps a [`DenseLdltFactor`].
#[derive(Clone, Debug)]
pub struct DenseBackend {
    ldlt: DenseLdltFactor,
    positive_pivot_idx: Vec<usize>,
    positive_pivot_values: DVector<f64>,
}

impl IsMinNormLdltBackend for DenseBackend {
    type LdltFactor = DenseLdltFactor;

    fn new(ldlt: Self::LdltFactor) -> Self {
        let n = ldlt.mat_l.scalar_dimension();
        let positive_pivot_idx: Vec<usize> = (0..n).filter(|&t| ldlt.diag_d[t] > 0.0).collect();
        let mut positive_pivot_values = DVector::<f64>::zeros(positive_pivot_idx.len());
        for (c, &t) in positive_pivot_idx.iter().enumerate() {
            positive_pivot_values[c] = ldlt.diag_d[t];
        }
        DenseBackend {
            ldlt,
            positive_pivot_idx,
            positive_pivot_values,
        }
    }

    fn scalar_dim(&self) -> usize {
        self.ldlt.mat_l.scalar_dimension()
    }

    fn tol_rel(&self) -> f64 {
        self.ldlt.tol_rel
    }

    fn column_of_mat_e(&self, col_j: usize, out: &mut [f64]) {
        // L is dense unit-lower with stored diagonal.
        let n = self.scalar_dim();
        let l = self.ldlt.mat_l.view();
        for i in 0..col_j {
            out[i] = 0.0;
        }
        for i in col_j..n {
            out[i] = l[(i, col_j)];
        } // includes out[t] = 1.0
    }

    fn try_column_of_inverse(&self, col_range: &BlockRange) -> Option<DMatrix<f64>> {
        if self.rank() != self.scalar_dim() {
            return None;
        }
        use crate::kernel::{
            diag_matsolve_inplaced,
            lower_matsolve_inplace,
            lower_transpose_matsolve_inplace,
        };

        let n = self.scalar_dim();
        debug_assert!(col_range.start_idx + col_range.block_dim <= n);

        // Build RHS = E_j (n × dj), i.e., selected identity columns.
        let mut w = DMatrix::<f64>::zeros(n, col_range.block_dim);
        for k in 0..col_range.block_dim {
            w[(col_range.start_idx + k, k)] = 1.0;
        }

        // H^{-1}[:, sj..sj+dj] = L^{-T} D^{-1} L^{-1} E_j
        let l = self.ldlt.mat_l.view();
        lower_matsolve_inplace(&l, &mut w.as_view_mut()); // W = L^{-1} E_j
        diag_matsolve_inplaced(self.ldlt.diag_d.as_view(), &mut w.as_view_mut()); // W = D^{-1} W
        lower_transpose_matsolve_inplace(&l, &mut w.as_view_mut()); // W = L^{-T} W
        Some(w)
    }

    fn try_inverse(&self) -> Option<DMatrix<f64>> {
        if self.rank() != self.scalar_dim() {
            return None;
        }
        // H^{-1} = L^{-T} D^{-1} L^{-1} using your kernels
        use crate::kernel::{
            diag_matsolve_inplaced,
            lower_matsolve_inplace,
            lower_transpose_matsolve_inplace,
        };
        let n = self.scalar_dim();
        let l = self.ldlt.mat_l.view();
        let mut w = DMatrix::<f64>::identity(n, n);
        lower_matsolve_inplace(&l, &mut w.as_view_mut());
        diag_matsolve_inplaced(self.ldlt.diag_d.as_view(), &mut w.as_view_mut());
        lower_transpose_matsolve_inplace(&l, &mut w.as_view_mut());
        Some(w)
    }

    fn block_range(&self, idx: PartitionBlockIndex) -> BlockRange {
        self.ldlt.mat_l.partitions().block_range(idx)
    }

    fn positive_pivot_idx(&self) -> &[usize] {
        &self.positive_pivot_idx
    }

    fn positive_pivot_values(&self) -> &DVector<f64> {
        &self.positive_pivot_values
    }
}

/// Dense min-norm LDLᵀ pseudo-inverse; produced by [`DenseMinNormFactor::new`].
pub type DenseMinNormFactor = MinNormLdlt<DenseBackend>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{
        Cholesky,
        DMatrix,
    };

    use super::DenseMinNormFactor;
    use crate::{
        IsInvertible,
        IsLinearSolver,
        ldlt::DenseLdlt,
        matrix::{
            IsSymmetricMatrixBuilder,
            PartitionBlockIndex,
            PartitionSet,
            PartitionSpec,
            dense::{
                DenseSymmetricMatrix,
                DenseSymmetricMatrixBuilder,
            },
        },
        svd::pseudo_inverse_with_tol,
    };

    const LDLT_TOL: f64 = 1e-12;

    fn build_square_from_dense(
        partitions: PartitionSet,
        h_dense: &DMatrix<f64>,
    ) -> DenseSymmetricMatrix {
        // Compute region offsets
        let mut region_offsets = Vec::with_capacity(partitions.len());
        let mut off = 0usize;
        for p in partitions.specs() {
            region_offsets.push(off);
            off += p.block_count * p.block_dim;
        }
        assert_eq!(off, h_dense.nrows());
        assert_eq!(off, h_dense.ncols());

        let mut b = DenseSymmetricMatrixBuilder::zero(partitions.clone());

        for partition_idx in 0..partitions.len() {
            let pi = &partitions.specs()[partition_idx];
            let oi = region_offsets[partition_idx];

            for block_row_idx in 0..pi.block_count {
                let di = pi.block_dim;
                let si = oi + block_row_idx * di;

                // 1) diagonal block inside region ri
                let idx = PartitionBlockIndex {
                    partition: partition_idx,
                    block: block_row_idx,
                };
                b.add_lower_block(idx, idx, &h_dense.view((si, si), (di, di)));

                // 2) within-region lower off-diagonals (same region, earlier block index)
                for block_col_idx in 0..block_row_idx {
                    let dj = pi.block_dim;
                    let sj = oi + block_col_idx * dj;
                    // lower triangle -> (bi,bj) with bj < bi
                    b.add_lower_block(
                        idx,
                        PartitionBlockIndex {
                            partition: partition_idx,
                            block: block_col_idx,
                        },
                        &h_dense.view((si, sj), (di, dj)),
                    );
                }

                // 3) cross-region lower off-diagonals (regions rj < ri)
                for rj in 0..partition_idx {
                    let pj = &partitions.specs()[rj];
                    let oj = region_offsets[rj];
                    for bj in 0..pj.block_count {
                        let dj = pj.block_dim;
                        let sj = oj + bj * dj;
                        b.add_lower_block(
                            idx,
                            PartitionBlockIndex {
                                partition: rj,
                                block: bj,
                            },
                            &h_dense.view((si, sj), (di, dj)),
                        );
                    }
                }
            }
        }

        b.build()
    }

    fn block_from_dense(
        partitions: &[PartitionSpec],
        h_dense: &DMatrix<f64>,
        ri: usize,
        bi: usize,
        rj: usize,
        bj: usize,
    ) -> DMatrix<f64> {
        let mut offs = Vec::with_capacity(partitions.len());
        let mut o = 0usize;
        for p in partitions {
            offs.push(o);
            o += p.block_count * p.block_dim;
        }

        let (di, dj) = (partitions[ri].block_dim, partitions[rj].block_dim);
        let si = offs[ri] + bi * di;
        let sj = offs[rj] + bj * dj;
        h_dense.view((si, sj), (di, dj)).into_owned()
    }

    #[test]
    fn spd_precision_and_covariance_blocks_match_dense_inverse() {
        // partitions: [(1×2), (2×1)] => n=4
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            },
            PartitionSpec {
                block_count: 2,
                block_dim: 1,
            },
        ]);
        let n = 4;

        // SPD H = A^T A + μ I
        let a = DMatrix::<f64>::from_row_slice(
            3,
            n,
            &[1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 4.0, 0.5, 2.0, 0.0, 0.0, 3.0],
        );
        let mu = 0.2;
        let h_dense = a.transpose() * &a + mu * DMatrix::<f64>::identity(n, n);

        let mat = build_square_from_dense(parts.clone(), &h_dense);
        let ldlt = DenseLdlt::default();
        let ldlt = ldlt.factorize(&mat).unwrap();
        let r = ldlt.rank();

        let mut gram_schmidt = DenseMinNormFactor::new(ldlt);
        let _ = r;

        let inv_dense = Cholesky::new(h_dense.clone()).unwrap().inverse();

        // a few tiles
        let tiles = [
            ([0usize, 0usize], [0usize, 0usize]), // (0,0) 2×2
            ([1, 1], [0, 0]),                     // (1,1) blk0×blk0 1×1
            ([1, 1], [1, 0]),                     // (1,1) blk1×blk0 1×1
            ([0, 1], [0, 1]),                     // (0,1) 2×1 off-diag
        ];

        for (region_idx, block_idx) in tiles {
            let row_idx = PartitionBlockIndex {
                partition: region_idx[0],
                block: block_idx[0],
            };
            let col_idx = PartitionBlockIndex {
                partition: region_idx[1],
                block: block_idx[1],
            };

            let sij = gram_schmidt.pseudo_inverse_block(row_idx, col_idx);

            let sref = block_from_dense(
                parts.specs(),
                &inv_dense,
                region_idx[0],
                block_idx[0],
                region_idx[1],
                block_idx[1],
            );
            assert_relative_eq!(sij, sref, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn psd_precision_and_covariance_blocks_match_pseudoinverse() {
        // single 3×3 block, rank-2 PSD: H = B^T B
        let parts = PartitionSet::new(vec![PartitionSpec {
            block_count: 1,
            block_dim: 3,
        }]);
        let n = 3;

        let b = DMatrix::<f64>::from_row_slice(2, n, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let h_dense = b.transpose() * &b;

        let mat = build_square_from_dense(parts.clone(), &h_dense);
        let ldlt = DenseLdlt::default();
        let ldlt = ldlt.factorize(&mat).unwrap();
        let mut gram_schmidt = DenseMinNormFactor::new(ldlt);

        let pinv = pseudo_inverse_with_tol(h_dense.as_view(), LDLT_TOL);

        let idx00 = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };

        let sij = gram_schmidt.pseudo_inverse_block(idx00, idx00);
        assert_relative_eq!(sij, pinv, epsilon = 1e-9, max_relative = 1e-9);
    }

    #[test]
    fn off_diagonal_covariance_tile_psd() {
        // two regions: [(1×1), (1×2)] ; n=3 ; rank-2 PSD
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 1,
            },
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            },
        ]);
        let b = DMatrix::<f64>::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let h_dense = b.transpose() * &b;

        let mat = build_square_from_dense(parts.clone(), &h_dense);
        let ldlt = DenseLdlt::default();
        let ldlt = ldlt.factorize(&mat).unwrap();
        let mut gram_schmidt = DenseMinNormFactor::new(ldlt);

        let pinv = pseudo_inverse_with_tol(h_dense.as_view(), LDLT_TOL);

        let idx00 = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };
        let idx10 = PartitionBlockIndex {
            partition: 1,
            block: 0,
        };

        // Tile (region0,blk0) × (region1,blk0)
        let sij = gram_schmidt.pseudo_inverse_block(idx00, idx10);
        let sref = block_from_dense(parts.specs(), &pinv, 0, 0, 1, 0);
        assert_relative_eq!(sij, sref, epsilon = 1e-9, max_relative = 1e-9);
    }

    #[test]
    fn covariance_blocks_are_symmetric() {
        // random SPD system, check S_ij == S_ji^T
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            },
            PartitionSpec {
                block_count: 1,
                block_dim: 3,
            },
        ]);
        let n = 5;
        let a = DMatrix::<f64>::from_fn(7, n, |r, c| ((r + 1) * (c + 3)) as f64 / 11.0);
        let h = a.transpose() * &a + 0.3 * DMatrix::<f64>::identity(n, n);

        let mat = build_square_from_dense(parts, &h);
        let ldlt = DenseLdlt::default();
        let ldlt = ldlt.factorize(&mat).unwrap();
        let mut gram_schmidt = DenseMinNormFactor::new(ldlt);

        let idx00 = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };
        let idx10 = PartitionBlockIndex {
            partition: 1,
            block: 0,
        };

        let s_01 = gram_schmidt.pseudo_inverse_block(idx00, idx10); // (2×3)
        let s_10 = gram_schmidt.pseudo_inverse_block(idx10, idx00); // (3×2)
        assert_relative_eq!(
            s_01,
            s_10.transpose(),
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn full_inverse_matches_dense_spd() {
        // partitions: [(1×2), (2×1)] => n=4
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            },
            PartitionSpec {
                block_count: 2,
                block_dim: 1,
            },
        ]);
        let n = 4;

        // SPD H = Aᵀ A + μ I
        let a = DMatrix::<f64>::from_row_slice(
            3,
            n,
            &[1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 4.0, 0.5, 2.0, 0.0, 0.0, 3.0],
        );
        let mu = 0.2;
        let h_dense = a.transpose() * &a + mu * DMatrix::<f64>::identity(n, n);

        // Factorize & build Gram–Schmidt min-norm system
        let mat = build_square_from_dense(parts, &h_dense);
        let ldlt = DenseLdlt::default().factorize(&mat).unwrap();
        let mut gs = DenseMinNormFactor::new(ldlt);

        // Our blockwise inverse
        let inv_mat = gs.pseudo_inverse();
        // Convert to a dense matrix for comparison
        let inv_dense_from_blocks = inv_mat;

        // Reference (Cholesky inverse)
        let inv_ref = Cholesky::new(h_dense.clone()).unwrap().inverse();

        assert_relative_eq!(
            inv_dense_from_blocks,
            inv_ref,
            epsilon = 1e-9,
            max_relative = 1e-9
        );
        // sanity: H * H^{-1} ≈ I
        let eye = h_dense * inv_dense_from_blocks;
        assert_relative_eq!(
            eye,
            DMatrix::<f64>::identity(n, n),
            epsilon = 1e-9,
            max_relative = 1e-9
        );
    }

    #[test]
    fn full_inverse_matches_pseudoinverse_psd() {
        // single 3×3 block, rank-2 PSD: H = Bᵀ B
        let parts = PartitionSet::new(vec![PartitionSpec {
            block_count: 1,
            block_dim: 3,
        }]);
        let n = 3;

        let b = DMatrix::<f64>::from_row_slice(2, n, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let h_dense = b.transpose() * &b;

        let mat = build_square_from_dense(parts, &h_dense);
        let ldlt = DenseLdlt::default().factorize(&mat).unwrap();
        let mut gs = DenseMinNormFactor::new(ldlt);

        let inv_mat = gs.pseudo_inverse();
        let inv_dense_from_blocks = inv_mat;

        let pinv_ref = pseudo_inverse_with_tol(h_dense.as_view(), LDLT_TOL);

        assert_relative_eq!(
            inv_dense_from_blocks,
            pinv_ref,
            epsilon = 1e-9,
            max_relative = 1e-9
        );
        // Moore–Penrose consistency: H * H^+ * H ≈ H
        let check = &h_dense * &inv_dense_from_blocks * &h_dense;
        assert_relative_eq!(check, h_dense, epsilon = 1e-9, max_relative = 1e-9);
    }

    #[test]
    fn rank_zero_returns_zero_matrix() {
        // All-zero 2x2 matrix → rank 0 → pseudo-inverse is zero
        let parts = PartitionSet::new(vec![PartitionSpec {
            block_count: 1,
            block_dim: 2,
        }]);
        let h_dense = DMatrix::<f64>::zeros(2, 2);

        let mat = build_square_from_dense(parts, &h_dense);
        let ldlt = DenseLdlt::default().factorize(&mat).unwrap();
        assert_eq!(ldlt.rank(), 0);
        let mut gs = DenseMinNormFactor::new(ldlt);

        let inv = gs.pseudo_inverse();
        assert_eq!(inv, DMatrix::<f64>::zeros(2, 2));

        let block = gs.pseudo_inverse_block(
            PartitionBlockIndex {
                partition: 0,
                block: 0,
            },
            PartitionBlockIndex {
                partition: 0,
                block: 0,
            },
        );
        assert_eq!(block, DMatrix::<f64>::zeros(2, 2));
    }

    #[test]
    fn scalar_1x1_spd() {
        let parts = PartitionSet::new(vec![PartitionSpec {
            block_count: 1,
            block_dim: 1,
        }]);
        let h_dense = DMatrix::<f64>::from_element(1, 1, 4.0);

        let mat = build_square_from_dense(parts, &h_dense);
        let ldlt = DenseLdlt::default().factorize(&mat).unwrap();
        let mut gs = DenseMinNormFactor::new(ldlt);

        let inv = gs.pseudo_inverse();
        assert_relative_eq!(
            inv,
            DMatrix::<f64>::from_element(1, 1, 0.25),
            epsilon = 1e-12
        );
    }

    #[test]
    fn scalar_1x1_psd_zero() {
        let parts = PartitionSet::new(vec![PartitionSpec {
            block_count: 1,
            block_dim: 1,
        }]);
        let h_dense = DMatrix::<f64>::zeros(1, 1);

        let mat = build_square_from_dense(parts, &h_dense);
        let ldlt = DenseLdlt::default().factorize(&mat).unwrap();
        let mut gs = DenseMinNormFactor::new(ldlt);

        let inv = gs.pseudo_inverse();
        assert_eq!(inv, DMatrix::<f64>::zeros(1, 1));
    }
}
