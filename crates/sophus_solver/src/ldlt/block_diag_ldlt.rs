use nalgebra::{
    DMatrixView,
    DMatrixViewMut,
    DVectorViewMut,
};
use snafu::ResultExt;

use crate::{
    BlockDiagLdltSnafu,
    BlockSparseLdltError,
    Definiteness,
    kernel::{
        diag_solve_inplaced,
        lower_solve_inplace,
        lower_transpose_solve_inplace,
    },
    ldlt::{
        BK_FALLBACK_THRESHOLD,
        dense_bunch_kaufman,
        ldlt_decompose_inplace,
        ldlt_right_matsolve_inplace,
    },
    matrix::{
        PartitionBlockIndex,
        PartitionSpec,
        block::{
            BlockDiag,
            BlockVector,
        },
    },
};

/// Block-diagonal LDLᵀ factors with optional BK fallback per block.
#[derive(Clone, Debug)]
pub struct BlockDiagonalLdltSystem {
    /// Unit lower-triangular matrices `L[j]`.
    pub mat_l: BlockDiag,
    /// Diagonal vectors `d[j]`.
    pub d: BlockVector,
    tol_rel: f64,
    /// Cumulative block offsets per partition (for flat indexing).
    partition_block_offsets: Vec<usize>,
    /// Optional BK factors for blocks that needed fallback.
    bk_factors: Vec<Option<dense_bunch_kaufman::BunchKaufmanFactor>>,
    /// Whether any block used BK fallback.
    pub(crate) used_bk_fallback: bool,
}

impl BlockDiagonalLdltSystem {
    /// Create zero block-diagonal.
    #[inline]
    pub fn zero(partition_specs: &[PartitionSpec], tol_rel: f64) -> Self {
        let mut offsets = Vec::with_capacity(partition_specs.len() + 1);
        offsets.push(0);
        for spec in partition_specs {
            offsets.push(offsets.last().unwrap() + spec.block_count);
        }
        let total_blocks = *offsets.last().unwrap();
        Self {
            mat_l: BlockDiag::zero(partition_specs),
            d: BlockVector::zero(partition_specs),
            tol_rel,
            partition_block_offsets: offsets,
            bk_factors: vec![None; total_blocks],
            used_bk_fallback: false,
        }
    }

    /// Flat index for a partition block.
    #[inline]
    fn flat_idx(&self, idx: PartitionBlockIndex) -> usize {
        self.partition_block_offsets[idx.partition] + idx.block
    }

    /// Factor block `A[j,j]` into unit lower-triangular `L[j]` and diagonal `d[j]`.
    ///
    /// Falls back to Bunch-Kaufman if the standard LDLᵀ produces a poor pivot
    /// condition on an indefinite block.
    #[inline]
    pub fn decompose(
        &mut self,
        idx: PartitionBlockIndex,
        mat_a: DMatrixView<'_, f64>,
    ) -> Result<(), BlockSparseLdltError> {
        // Populate L[j,j] from A[j,j].
        let mut mat_l_block = self.mat_l.get_block_mut(idx);
        mat_l_block.copy_from(&mat_a);

        // Decompose L[j,j] in-place.
        let result = ldlt_decompose_inplace(
            mat_l_block.as_view_mut(),
            self.d.get_block_mut(idx),
            self.tol_rel,
        )
        .context(BlockDiagLdltSnafu { idx })?;

        // Check if BK fallback is needed: indefinite with poor pivot condition.
        if result.definiteness == Definiteness::Indefinite
            && result.pivot_condition < BK_FALLBACK_THRESHOLD
        {
            let a_owned = mat_a.clone_owned();
            let bk_factor = dense_bunch_kaufman::factorize(&a_owned).map_err(|_| {
                BlockSparseLdltError::BlockDiagLdltError {
                    source: crate::error::LdltDecompositionError::NonFinitePivot {
                        j: 0,
                        d_jj: result.pivot_condition,
                    },
                    idx,
                }
            })?;
            let flat = self.flat_idx(idx);
            self.bk_factors[flat] = Some(bk_factor);
            self.used_bk_fallback = true;
        }

        Ok(())
    }

    /// Vector solve in-place: `y ← A[j,j]⁻¹ y`.
    #[inline]
    pub fn solve_inplace_vec(&self, mut y: DVectorViewMut<'_, f64>, idx: PartitionBlockIndex) {
        let flat = self.flat_idx(idx);
        if let Some(ref bk) = self.bk_factors[flat] {
            // BK fallback path.
            bk.solve_slice_inplace(y.as_mut_slice());
        } else {
            // Standard LDLᵀ path.
            let mat_l_jj = self.mat_l.get_block(idx);
            let d_j = self.d.get_block(idx);
            lower_solve_inplace(&mat_l_jj, &mut y);
            diag_solve_inplaced(d_j.as_view(), &mut y);
            lower_transpose_solve_inplace(&mat_l_jj, &mut y);
        }
    }

    /// Right-solve in-place: `X ← X * A[j,j]⁻¹`.
    #[inline]
    pub fn right_solve_inplace(&self, x: DMatrixViewMut<'_, f64>, idx: PartitionBlockIndex) {
        let flat = self.flat_idx(idx);
        if let Some(ref bk) = self.bk_factors[flat] {
            // BK fallback path: right-solve row by row.
            let nrows = x.nrows();
            let ncols = x.ncols();
            let mut x_owned = x.clone_owned();
            let mut buf = vec![0.0_f64; ncols];
            for r in 0..nrows {
                for c in 0..ncols {
                    buf[c] = x_owned[(r, c)];
                }
                bk.solve_slice_inplace(&mut buf[..ncols]);
                for c in 0..ncols {
                    x_owned[(r, c)] = buf[c];
                }
            }
            // Copy back to the mutable view.
            let mut x = x;
            x.copy_from(&x_owned);
        } else {
            // Standard LDLᵀ path.
            let mat_l_jj = self.mat_l.get_block(idx);
            let d_j = self.d.get_block(idx);
            ldlt_right_matsolve_inplace(mat_l_jj, d_j.as_view(), x);
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{
        DMatrix,
        DVector,
    };

    use super::*;

    #[test]
    fn bk_fallback_on_ill_conditioned_indefinite_block() {
        // Create a 2×2 indefinite block with near-zero pivot to trigger BK fallback.
        // A = [[ε, 1], [1, -ε]] where ε is very small.
        // Standard LDLᵀ: d[0] = ε (tiny), L[1,0] = 1/ε (huge), d[1] = -ε - 1/ε (huge negative).
        // This has terrible pivot condition ≈ ε² → BK fallback should activate.
        let eps = 1e-12;
        let mat_a = DMatrix::from_row_slice(2, 2, &[eps, 1.0, 1.0, -eps]);

        let specs = vec![PartitionSpec {
            eliminate_last: false,
            block_count: 1,
            block_dim: 2,
        }];
        let mut system = BlockDiagonalLdltSystem::zero(&specs, 1e-15);

        let idx = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };
        system.decompose(idx, mat_a.as_view()).unwrap();

        // Should have triggered BK fallback.
        assert!(
            system.used_bk_fallback,
            "expected BK fallback for ill-conditioned indefinite block"
        );

        // Solve should still give correct answer.
        let b = DVector::from_row_slice(&[1.0, 2.0]);
        let mut x = b.clone();
        let n = x.len();
        system.solve_inplace_vec(x.rows_mut(0, n), idx);

        let x_ref = mat_a.clone().lu().solve(&b).unwrap();
        approx::assert_abs_diff_eq!(x, x_ref, epsilon = 1e-6);
    }

    #[test]
    fn no_bk_fallback_on_well_conditioned_indefinite_block() {
        // Well-conditioned indefinite: [[1, 2], [2, 1]] — eigs: 3, -1.
        // Pivot condition ≈ 0.33 — no BK needed.
        let mat_a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);

        let specs = vec![PartitionSpec {
            eliminate_last: false,
            block_count: 1,
            block_dim: 2,
        }];
        let mut system = BlockDiagonalLdltSystem::zero(&specs, 1e-15);

        let idx = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };
        system.decompose(idx, mat_a.as_view()).unwrap();

        assert!(
            !system.used_bk_fallback,
            "should NOT trigger BK fallback for well-conditioned indefinite block"
        );

        // Solve should be correct without BK.
        let b = DVector::from_row_slice(&[1.0, 2.0]);
        let mut x = b.clone();
        let n = x.len();
        system.solve_inplace_vec(x.rows_mut(0, n), idx);

        let x_ref = mat_a.clone().lu().solve(&b).unwrap();
        approx::assert_abs_diff_eq!(x, x_ref, epsilon = 1e-10);
    }
}
