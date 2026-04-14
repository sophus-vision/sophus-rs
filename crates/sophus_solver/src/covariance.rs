//! Covariance and precision (information) matrix from a factored Hessian.
//!
//! - `covariance_block(i,j)` — Σ(i,j): covariance with constraint projection (if present)
//! - `precision_block(i,j)` — H(i,j): precision (information) matrix block
//!
//! The constraint projection formula is:
//!   `Σ(i,j) = H⁺(i,j) - αᵢ · M⁺ · βⱼᵀ`
//! where `M = G H⁺ Gᵀ`, `αᵢ = (H⁺Gᵀ)_i`, `βⱼ = (H⁺Gᵀ)_j`.

use nalgebra::DMatrix;

use crate::{
    InvertibleMatrix,
    IsInvertible,
    matrix::{
        PartitionBlockIndex,
        PartitionSet,
    },
};

/// Covariance computed from a factored Hessian, with optional constraint projection.
///
/// Provides block-wise access to:
/// - `covariance_block(i,j)` — Σ(i,j): covariance with constraint projection (if present)
///
/// The min-norm pseudo-inverse handles rank-deficient H (gauge freedom).
pub struct Covariance {
    invertible: InvertibleMatrix,
    partitions: PartitionSet,
    /// Cached H⁺Gᵀ (n × c). None when no constraints.
    h_inv_gt: Option<DMatrix<f64>>,
    /// Cached M⁺ = (G H⁺ Gᵀ)⁺ (c × c). None when no constraints.
    m_inv: Option<DMatrix<f64>>,
}

impl Covariance {
    /// Create a `Covariance` from a factorized matrix and optional constraint Jacobian.
    ///
    /// - `invertible`: the factorized H (from LDLᵀ, Schur, etc.)
    /// - `partitions`: block layout of the variable-only matrix
    /// - `constraint_jacobian`: optional dense G (num_constraints × num_active_scalars). When
    ///   `Some`, precomputes the rank-k correction for constrained covariance.
    pub fn new(
        mut invertible: InvertibleMatrix,
        partitions: PartitionSet,
        constraint_jacobian: Option<&DMatrix<f64>>,
    ) -> Self {
        let (h_inv_gt, m_inv) = if let Some(g) = constraint_jacobian {
            let n = partitions.scalar_dim();
            debug_assert_eq!(
                g.ncols(),
                n,
                "constraint_jacobian has {} columns but partitions have {} scalars",
                g.ncols(),
                n
            );
            let c = g.nrows();
            let num_var_partitions = partitions.len();

            // Identify constrained blocks: partition/block pairs where G has nonzero columns.
            let mut constrained_blocks: Vec<(PartitionBlockIndex, DMatrix<f64>)> = Vec::new();
            for p in 0..num_var_partitions {
                let spec = &partitions.specs()[p];
                for b in 0..spec.block_count {
                    let idx = PartitionBlockIndex {
                        partition: p,
                        block: b,
                    };
                    let range = partitions.block_range(idx);
                    let gt_block = g
                        .columns(range.start_idx, range.block_dim)
                        .transpose()
                        .into_owned();
                    if gt_block.iter().any(|&v| v.abs() > 1e-15) {
                        constrained_blocks.push((idx, gt_block));
                    }
                }
            }

            // Build H⁺Gᵀ block-row by block-row using pseudo_inverse_block.
            let mut h_inv_gt = DMatrix::zeros(n, c);
            for p in 0..num_var_partitions {
                let spec = &partitions.specs()[p];
                for b in 0..spec.block_count {
                    let row_idx = PartitionBlockIndex {
                        partition: p,
                        block: b,
                    };
                    let row_range = partitions.block_range(row_idx);
                    for (col_idx, gt_block) in &constrained_blocks {
                        let h_inv_block = invertible.pseudo_inverse_block(row_idx, *col_idx);
                        let contribution = h_inv_block * gt_block;
                        let mut target =
                            h_inv_gt.view_mut((row_range.start_idx, 0), (row_range.block_dim, c));
                        target += contribution;
                    }
                }
            }

            // M = G H⁺ Gᵀ  (c × c), then pseudo-invert.
            let m = g * &h_inv_gt;
            let m_inv = nalgebra::SVD::new(m, true, true)
                .pseudo_inverse(1e-12)
                .expect("M pseudo-inverse for constrained covariance");

            (Some(h_inv_gt), Some(m_inv))
        } else {
            (None, None)
        };

        Self {
            invertible,
            partitions,
            h_inv_gt,
            m_inv,
        }
    }

    /// Σ(i,j): covariance block with constraint projection (if present).
    ///
    /// Without constraints: returns `H⁺(i,j)` (min-norm pseudo-inverse).
    /// With constraints: returns `H⁺(i,j) - αᵢ · M⁺ · βⱼᵀ`.
    pub fn covariance_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> DMatrix<f64> {
        let h_inv_block = self.invertible.pseudo_inverse_block(row_idx, col_idx);

        if let (Some(h_inv_gt), Some(m_inv)) = (&self.h_inv_gt, &self.m_inv) {
            let row_range = self.partitions.block_range(row_idx);
            let col_range = self.partitions.block_range(col_idx);

            let alpha = h_inv_gt.rows(row_range.start_idx, row_range.block_dim);
            let beta = h_inv_gt.rows(col_range.start_idx, col_range.block_dim);

            h_inv_block - alpha * m_inv * beta.transpose()
        } else {
            h_inv_block
        }
    }

    /// The partition layout.
    pub fn partitions(&self) -> &PartitionSet {
        &self.partitions
    }

    /// Whether equality constraints are applied.
    pub fn has_constraints(&self) -> bool {
        self.m_inv.is_some()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{
        DMatrix,
        Matrix3,
    };

    use super::*;
    use crate::{
        IsLinearSolver,
        ldlt::{
            DenseLdlt,
            min_norm_ldlt::dense_min_norm_ldlt::DenseMinNormFactor,
        },
        matrix::{
            IsSymmetricMatrixBuilder,
            PartitionSpec,
            dense::DenseSymmetricMatrixBuilder,
        },
    };

    /// Build an InvertibleMatrix from a dense symmetric matrix and partition specs.
    fn invertible_from_dense(h: &DMatrix<f64>, specs: Vec<PartitionSpec>) -> InvertibleMatrix {
        let partitions = PartitionSet::new(specs);
        let n = partitions.scalar_dim();
        assert_eq!(h.nrows(), n);

        let mut builder = DenseSymmetricMatrixBuilder::zero(partitions.clone());
        for p in 0..partitions.len() {
            let bcount_p = partitions.specs()[p].block_count;
            for b in 0..bcount_p {
                let row_idx = PartitionBlockIndex {
                    partition: p,
                    block: b,
                };
                let row_range = partitions.block_range(row_idx);
                for q in 0..partitions.len() {
                    let bcount_q = partitions.specs()[q].block_count;
                    for c in 0..bcount_q {
                        let col_idx = PartitionBlockIndex {
                            partition: q,
                            block: c,
                        };
                        let col_range = partitions.block_range(col_idx);
                        let block = h
                            .view(
                                (row_range.start_idx, col_range.start_idx),
                                (row_range.block_dim, col_range.block_dim),
                            )
                            .into_owned();
                        builder.add_lower_block(row_idx, col_idx, &block.as_view());
                    }
                }
            }
        }
        let mat = builder.build();
        let factor = DenseLdlt::default().factorize(&mat).unwrap();
        InvertibleMatrix::Dense(DenseMinNormFactor::new(factor))
    }

    fn idx(partition: usize, block: usize) -> PartitionBlockIndex {
        PartitionBlockIndex { partition, block }
    }

    // ── Unconstrained, full rank ────────────────────────────────────────

    /// Unconstrained covariance of a full-rank SPD matrix matches dense inverse.
    #[test]
    fn unconstrained_full_rank_matches_inverse() {
        // 4×4 SPD: two partitions of dim 2.
        let h = DMatrix::from_row_slice(
            4,
            4,
            &[
                4.0, 1.0, 0.5, 0.0, 1.0, 3.0, 0.0, 0.5, 0.5, 0.0, 5.0, 1.0, 0.0, 0.5, 1.0, 4.0,
            ],
        );
        let h_inv_expected = h.clone().try_inverse().unwrap();
        let specs = vec![
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 2,
            },
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 2,
            },
        ];
        let partitions = PartitionSet::new(specs.clone());
        let h_inv = invertible_from_dense(&h, specs);
        let mut cov = Covariance::new(h_inv, partitions, None);

        assert!(!cov.has_constraints());

        // Check all blocks match dense inverse.
        for (ri, ci) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let block = cov.covariance_block(idx(ri, 0), idx(ci, 0));
            let expected = h_inv_expected.view((ri * 2, ci * 2), (2, 2)).into_owned();
            assert_abs_diff_eq!(block, expected, epsilon = 1e-10);
        }
    }

    // ── Unconstrained, rank-deficient (gauge freedom) ───────────────────

    /// Rank-deficient H: pseudo-inverse block is symmetric and H H⁺ H = H.
    #[test]
    fn unconstrained_rank_deficient_pseudo_inverse() {
        // Rank-2 matrix in 3D (gauge: null space along [1,1,1]).
        let v1 = DMatrix::from_row_slice(3, 1, &[1.0, -1.0, 0.0]);
        let v2 = DMatrix::from_row_slice(3, 1, &[0.0, 1.0, -1.0]);
        let h = &v1 * v1.transpose() + &v2 * v2.transpose();
        let specs = vec![PartitionSpec {
            eliminate_last: false,
            block_count: 1,
            block_dim: 3,
        }];
        let partitions = PartitionSet::new(specs.clone());
        let h_inv = invertible_from_dense(&h, specs);
        let mut cov = Covariance::new(h_inv, partitions, None);

        let block = cov.covariance_block(idx(0, 0), idx(0, 0));

        // H H⁺ H = H  (Moore-Penrose condition)
        let product = &h * &block * &h;
        assert_abs_diff_eq!(product, h, epsilon = 1e-10);

        // Symmetric
        assert_abs_diff_eq!(&block, &block.transpose(), epsilon = 1e-12);
    }

    // ── Constrained, single constraint ──────────────────────────────────

    /// Constraint along one direction zeros out that eigenvalue.
    #[test]
    fn constrained_single_constraint_zeros_direction() {
        // H = diag(2, 3) — full rank, two partitions of dim 1.
        let h = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![2.0, 3.0]));
        let specs = vec![
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 1,
            },
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 1,
            },
        ];
        let partitions = PartitionSet::new(specs.clone());
        let h_inv = invertible_from_dense(&h, specs);

        // Constraint: G = [1, 0] — constrains the first variable.
        let g = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let mut cov = Covariance::new(h_inv, partitions, Some(&g));

        assert!(cov.has_constraints());

        // First variable: constrained → variance should be ~0.
        let block_00 = cov.covariance_block(idx(0, 0), idx(0, 0));
        assert!(
            block_00[(0, 0)].abs() < 1e-10,
            "constrained var should have ~0 variance"
        );

        // Second variable: unconstrained → variance = 1/3.
        let block_11 = cov.covariance_block(idx(1, 0), idx(1, 0));
        assert_abs_diff_eq!(block_11[(0, 0)], 1.0 / 3.0, epsilon = 1e-10);

        // Cross-block should be zero (independent variables, one constrained).
        let block_01 = cov.covariance_block(idx(0, 0), idx(1, 0));
        assert!(block_01[(0, 0)].abs() < 1e-10);
    }

    // ── Constrained covariance is symmetric ─────────────────────────────

    /// block(i,j) == block(j,i)ᵀ for constrained covariance.
    #[test]
    fn constrained_blocks_are_symmetric() {
        // 4×4 SPD with off-diagonal coupling.
        let h = DMatrix::from_row_slice(
            4,
            4,
            &[
                4.0, 1.0, 0.5, 0.0, 1.0, 3.0, 0.0, 0.5, 0.5, 0.0, 5.0, 1.0, 0.0, 0.5, 1.0, 4.0,
            ],
        );
        let specs = vec![
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 2,
            },
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 2,
            },
        ];
        let partitions = PartitionSet::new(specs.clone());
        let h_inv = invertible_from_dense(&h, specs);

        // Constraint on first partition: G = [1 0 | 0 0].
        let g = DMatrix::from_row_slice(1, 4, &[1.0, 0.0, 0.0, 0.0]);
        let mut cov = Covariance::new(h_inv, partitions, Some(&g));

        // Diagonal blocks should be symmetric.
        let b00 = cov.covariance_block(idx(0, 0), idx(0, 0));
        assert_abs_diff_eq!(&b00, &b00.transpose(), epsilon = 1e-12);
        let b11 = cov.covariance_block(idx(1, 0), idx(1, 0));
        assert_abs_diff_eq!(&b11, &b11.transpose(), epsilon = 1e-12);

        // Cross-blocks: block(0,1) == block(1,0)ᵀ.
        let b01 = cov.covariance_block(idx(0, 0), idx(1, 0));
        let b10 = cov.covariance_block(idx(1, 0), idx(0, 0));
        assert_abs_diff_eq!(b01, b10.transpose(), epsilon = 1e-12);
    }

    // ── Constrained matches dense formula ───────────────────────────────

    /// Constrained covariance blocks match the dense formula Σ = H⁻¹ - H⁻¹Gᵀ(GH⁻¹Gᵀ)⁻¹GH⁻¹.
    #[test]
    fn constrained_matches_dense_formula() {
        let h = DMatrix::from_row_slice(3, 3, &[5.0, 1.0, 0.5, 1.0, 4.0, 1.0, 0.5, 1.0, 6.0]);
        let h_inv = h.clone().try_inverse().unwrap();
        let g = DMatrix::from_row_slice(1, 3, &[0.0, 1.0, 0.0]); // constrain var 1

        // Dense formula
        let m = &g * &h_inv * g.transpose();
        let m_inv = m.try_inverse().unwrap();
        let correction = &h_inv * g.transpose() * &m_inv * &g * &h_inv;
        let sigma_expected = &h_inv - &correction;

        // Block-wise via Hessian
        let specs = vec![
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 1,
            },
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 1,
            },
            PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 1,
            },
        ];
        let partitions = PartitionSet::new(specs.clone());
        let h_invertible = invertible_from_dense(&h, specs);
        let mut cov = Covariance::new(h_invertible, partitions, Some(&g));

        for i in 0..3 {
            for j in 0..3 {
                let block = cov.covariance_block(idx(i, 0), idx(j, 0));
                let expected = sigma_expected[(i, j)];
                assert_abs_diff_eq!(block[(0, 0)], expected, epsilon = 1e-10);
            }
        }
    }

    // ── Gauge + constraint ──────────────────────────────────────────────

    /// Rank-deficient H with constraint that partially resolves gauge.
    #[test]
    fn gauge_with_constraint_partial_resolution() {
        // Rank-2 in 3D: null space along [1,1,1].
        let v1 = DMatrix::from_row_slice(3, 1, &[1.0, -1.0, 0.0]);
        let v2 = DMatrix::from_row_slice(3, 1, &[0.0, 1.0, -1.0]);
        let h = 2.0 * &v1 * v1.transpose() + 3.0 * &v2 * v2.transpose();

        let specs = vec![PartitionSpec {
            eliminate_last: false,
            block_count: 1,
            block_dim: 3,
        }];
        let partitions = PartitionSet::new(specs.clone());
        let h_invertible = invertible_from_dense(&h, specs);

        // Constraint: fix the sum x0 + x1 + x2 = 0 (constrains the gauge direction).
        let g = DMatrix::from_row_slice(1, 3, &[1.0, 1.0, 1.0]);
        let mut cov = Covariance::new(h_invertible, partitions, Some(&g));

        let sigma = cov.covariance_block(idx(0, 0), idx(0, 0));

        // After constraining the gauge, all 3 eigenvalues of the constrained
        // covariance should reflect: 2 from the observable space (now well-defined)
        // and 1 near-zero from the constraint.
        let eig = nalgebra::SymmetricEigen::new(Matrix3::from_iterator(sigma.iter().copied()));
        let mut evs: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        evs.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Two positive eigenvalues from the observable space.
        assert!(
            evs[0] > 1e-6,
            "largest eigenvalue should be positive: {}",
            evs[0]
        );
        assert!(
            evs[1] > 1e-6,
            "second eigenvalue should be positive: {}",
            evs[1]
        );
        // Third near-zero from the constraint resolving the gauge.
        assert!(
            evs[2].abs() < 1e-6,
            "constrained direction should be ~0: {}",
            evs[2]
        );

        // Symmetric.
        assert_abs_diff_eq!(&sigma, &sigma.transpose(), epsilon = 1e-12);
    }

    // ── Multi-block with constraint ─────────────────────────────────────

    /// BA-like structure: 2 pose blocks + 3 point blocks, constraint on pose 1.
    #[test]
    fn multi_block_constrained() {
        // Simple 5×5 SPD: poses (2 × dim 1) + points (3 × dim 1).
        let a = DMatrix::from_row_slice(
            3,
            5,
            &[
                1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            ],
        );
        let h = a.transpose() * &a + 0.1 * DMatrix::identity(5, 5);

        let specs = vec![
            PartitionSpec {
                eliminate_last: false,
                block_count: 2,
                block_dim: 1,
            }, // "points" (2 vars)
            PartitionSpec {
                eliminate_last: false,
                block_count: 3,
                block_dim: 1,
            }, // "poses" (3 vars)
        ];
        let partitions = PartitionSet::new(specs.clone());
        let h_invertible = invertible_from_dense(&h, specs);

        // Constraint on pose 0 (partition 1, block 0): G = [0 0 | 1 0 0].
        let g = DMatrix::from_row_slice(1, 5, &[0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut cov = Covariance::new(h_invertible, partitions, Some(&g));

        // Pose 0 (partition 1, block 0) should have ~0 variance.
        let pose0_cov = cov.covariance_block(idx(1, 0), idx(1, 0));
        assert!(
            pose0_cov[(0, 0)].abs() < 1e-6,
            "constrained pose should have ~0 variance: {}",
            pose0_cov[(0, 0)]
        );

        // Pose 1 (partition 1, block 1) should have positive variance.
        let pose1_cov = cov.covariance_block(idx(1, 1), idx(1, 1));
        assert!(
            pose1_cov[(0, 0)] > 1e-6,
            "free pose should have positive variance"
        );

        // Point × pose cross-block should exist and be finite.
        let cross = cov.covariance_block(idx(0, 0), idx(1, 1));
        assert!(cross[(0, 0)].is_finite());
    }

    /// Two simultaneous equality constraints: constrained variables have ~0 variance,
    /// unconstrained have positive variance, and the result matches the dense formula.
    #[test]
    fn multi_constraint_covariance() {
        // 4×4 SPD matrix with 4 partitions of dim 1.
        let h = DMatrix::from_row_slice(
            4,
            4,
            &[
                5.0, 1.0, 0.5, 0.2, 1.0, 4.0, 0.3, 0.1, 0.5, 0.3, 6.0, 0.4, 0.2, 0.1, 0.4, 3.0,
            ],
        );
        let h_inv = h.clone().try_inverse().unwrap();

        let specs = vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 1,
                eliminate_last: false,
            },
            PartitionSpec {
                block_count: 1,
                block_dim: 1,
                eliminate_last: false,
            },
            PartitionSpec {
                block_count: 1,
                block_dim: 1,
                eliminate_last: false,
            },
            PartitionSpec {
                block_count: 1,
                block_dim: 1,
                eliminate_last: false,
            },
        ];
        let partitions = PartitionSet::new(specs.clone());
        let h_inv_mat = invertible_from_dense(&h, specs);

        // Two constraints: pin variables 0 and 1.
        // G = [[1 0 0 0]; [0 1 0 0]]
        let g = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut cov = Covariance::new(h_inv_mat, partitions, Some(&g));

        // Variables 0, 1: constrained → ~0 variance.
        let s00 = cov.covariance_block(idx(0, 0), idx(0, 0));
        let s11 = cov.covariance_block(idx(1, 0), idx(1, 0));
        assert!(
            s00[(0, 0)].abs() < 1e-10,
            "constrained var 0 should have ~0 variance: {}",
            s00[(0, 0)]
        );
        assert!(
            s11[(0, 0)].abs() < 1e-10,
            "constrained var 1 should have ~0 variance: {}",
            s11[(0, 0)]
        );

        // Variables 2, 3: unconstrained → positive variance.
        let s22 = cov.covariance_block(idx(2, 0), idx(2, 0));
        let s33 = cov.covariance_block(idx(3, 0), idx(3, 0));
        assert!(
            s22[(0, 0)] > 1e-6,
            "free var 2 should have positive variance"
        );
        assert!(
            s33[(0, 0)] > 1e-6,
            "free var 3 should have positive variance"
        );

        // G * Σ * G^T ≈ 0 (constraint satisfaction in covariance).
        // Assemble full Σ from blocks.
        let mut sigma = DMatrix::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                let block = cov.covariance_block(idx(i, 0), idx(j, 0));
                sigma[(i, j)] = block[(0, 0)];
            }
        }
        let g_sigma_gt = &g * &sigma * g.transpose();
        assert_abs_diff_eq!(g_sigma_gt, DMatrix::zeros(2, 2), epsilon = 1e-10);

        // Matches dense formula: Σ = H⁻¹ - H⁻¹ Gᵀ (G H⁻¹ Gᵀ)⁻¹ G H⁻¹
        let m = &g * &h_inv * g.transpose();
        let m_inv = m.try_inverse().unwrap();
        let correction = &h_inv * g.transpose() * &m_inv * &g * &h_inv;
        let sigma_expected = &h_inv - &correction;

        assert_abs_diff_eq!(sigma, sigma_expected, epsilon = 1e-10);
    }
}
