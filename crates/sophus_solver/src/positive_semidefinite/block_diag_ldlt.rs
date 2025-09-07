use nalgebra::{
    DMatrixView,
    DMatrixViewMut,
    DVectorViewMut,
};
use snafu::ResultExt;

use crate::{
    BlockDiagLdltSnafu,
    BlockSparseLdltError,
    kernel::{
        diag_solve_inplaced,
        lower_solve_inplace,
        lower_transpose_solve_inplace,
    },
    matrix::{
        BlockDiag,
        BlockVector,
        PartitionSpec,
    },
    positive_semidefinite::{
        ldlt_decompose_inplace,
        ldlt_right_solve_inplace,
    },
};

/// Block-diagonal LDLᵀ factors.
#[derive(Debug)]
pub struct BlockDiagonalLdltFactors {
    /// Unit lower-triangular matrices `L[j]`.
    pub mat_l: BlockDiag,
    /// Diagonal vectors `d[j]`.
    pub d: BlockVector,
    tol_rel: f64,
}

impl BlockDiagonalLdltFactors {
    /// Create zero block-diagonal.
    #[inline]
    pub fn zero(partition_specs: &[PartitionSpec], tol_rel: f64) -> Self {
        Self {
            mat_l: BlockDiag::zero(partition_specs),
            d: BlockVector::zero(partition_specs),
            tol_rel,
        }
    }

    /// Factor block `A[j,j]` into unit lower-triangular `L[j]` and diagonal `d[j]`,
    ///
    /// with j = (partition_idx, local_block_idx).
    #[inline]
    pub fn decompose(
        &mut self,
        partition_idx: usize,
        local_block_idx: usize,
        mat_a: DMatrixView<'_, f64>,
    ) -> Result<(), BlockSparseLdltError> {
        // Populate L[j,j] from A[j,j].
        let mut mat_l_block = self.mat_l.get_block_mut(partition_idx, local_block_idx);
        mat_l_block.copy_from(&mat_a);

        // Decompose L[j,j] in-place.
        ldlt_decompose_inplace(
            mat_l_block.as_view_mut(),
            self.d.get_block_mut(partition_idx, local_block_idx),
            self.tol_rel,
        )
        .context(BlockDiagLdltSnafu {
            partition_idx,
            local_block_idx,
        })?;

        // Zero the strict upper part of L[j, j].
        for r in 0..mat_l_block.nrows() {
            for c in (r + 1)..mat_l_block.ncols() {
                mat_l_block[(r, c)] = 0.0;
            }
        }
        Ok(())
    }

    /// Vector solve in-place: `y ← L[j,j]⁻ᵀ diag(d[j])⁻¹ L[j,j]⁻¹ y`.
    #[inline]
    pub fn solve_inplace_vec(
        &self,
        mut y: DVectorViewMut<'_, f64>,
        partition_idx: usize,
        local_block_idx: usize,
    ) {
        let mat_l_jj = self.mat_l.get_block(partition_idx, local_block_idx);
        let d_j = self.d.get_block(partition_idx, local_block_idx);

        lower_solve_inplace(&mat_l_jj, &mut y);
        diag_solve_inplaced(d_j.as_view(), &mut y, self.tol_rel);
        lower_transpose_solve_inplace(&mat_l_jj, &mut y);
    }

    /// Right-solve in-place: `X ← X * L[j,j]⁻ᵀ diag(d[j])⁻¹ L[j,j]⁻¹`.
    #[inline]
    pub fn right_solve_inplace(
        &self,
        x: DMatrixViewMut<'_, f64>,
        part_idx: usize,
        local_block_idx: usize,
    ) {
        let mat_l_jj = self.mat_l.get_block(part_idx, local_block_idx);
        let d_j = self.d.get_block(part_idx, local_block_idx);
        ldlt_right_solve_inplace(mat_l_jj, d_j.as_view(), x, self.tol_rel);
    }
}
