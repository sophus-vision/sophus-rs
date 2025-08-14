use sophus_autodiff::linalg::MatF64;

use crate::block::BlockRange;

/// Jacobian matrix, split into several blocks
///
/// ```ascii
/// -------------------------------
/// |         |         |         |
/// |   J_0   |  . . .  | J_{N-1} |
/// |         |         |         |
/// -------------------------------
/// ```
///
/// The `(RESIDUAL_DIM  x  INPUT_DIM)` matrix is partitioned into `N` blocks horizontally.
/// The shape of each of are specified by the `ranges` array: The Jacobian sub-block `J_i` is a
/// `(RESIDUAL_DIM  x  ranges(i).dim)` matrix.
#[derive(Clone, Debug)]
pub struct BlockJacobian<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const N: usize> {
    /// matrix storage
    pub mat: nalgebra::SMatrix<f64, RESIDUAL_DIM, INPUT_DIM>,
    /// ranges, one for each block
    pub ranges: [BlockRange; N],
}

impl<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const N: usize>
    BlockJacobian<RESIDUAL_DIM, INPUT_DIM, N>
{
    /// create a new Jacobian matrix with N blocks with the given dimensions
    pub fn new(dims: &[usize]) -> Self {
        debug_assert!(!dims.is_empty());
        debug_assert_eq!(dims.iter().sum::<usize>(), INPUT_DIM);

        let num_blocks = dims.len();
        let mut col_ranges = [BlockRange::default(); N];
        let mut num_cols: usize = 0;

        for i in 0..num_blocks {
            let dim = dims[i];
            col_ranges[i] = BlockRange {
                index: num_cols as i64,
                dim,
            };
            num_cols += dim;
        }
        Self {
            mat: nalgebra::SMatrix::zeros(),
            ranges: col_ranges,
        }
    }

    /// Number of blocks (along the column/input direction).
    pub fn num_blocks(&self) -> usize {
        self.ranges.len()
    }

    /// set block given by `col_block_idx` and static matrix `submat`
    pub fn set_block<const C: usize>(
        &mut self,
        col_block_idx: usize,
        submat: MatF64<RESIDUAL_DIM, C>,
    ) {
        debug_assert!(col_block_idx < self.num_blocks());
        debug_assert_eq!(C, self.ranges[col_block_idx].dim);

        let col_offset = self.ranges[col_block_idx].index as usize;
        let mut block_view = self.mat.fixed_view_mut::<RESIDUAL_DIM, C>(0, col_offset);
        block_view.copy_from(&submat);
    }

    /// Get a (dynamic) view of the `block_idx`-th column block as a matrix slice (read-only).
    pub fn block(
        &self,
        col_block_idx: usize,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Dyn,
        nalgebra::Dyn,
        nalgebra::ViewStorage<
            '_,
            f64,
            nalgebra::Dyn,
            nalgebra::Dyn,
            nalgebra::Const<1>,
            nalgebra::Const<RESIDUAL_DIM>,
        >,
    > {
        debug_assert!(col_block_idx < self.num_blocks());
        let col_offset = self.ranges[col_block_idx].index as usize;
        let cdim = self.ranges[col_block_idx].dim;
        self.mat.view((0, col_offset), (RESIDUAL_DIM, cdim))
    }

    /// Get mutable reference to the block (of static size), if you want direct in-place
    /// modifications.
    pub fn mut_block<const C: usize>(
        &mut self,
        col_block_idx: usize,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Const<RESIDUAL_DIM>,
        nalgebra::Const<C>,
        nalgebra::ViewStorageMut<
            '_,
            f64,
            nalgebra::Const<RESIDUAL_DIM>,
            nalgebra::Const<C>,
            nalgebra::Const<1>,
            nalgebra::Const<RESIDUAL_DIM>,
        >,
    > {
        debug_assert!(col_block_idx < self.num_blocks());
        debug_assert_eq!(C, self.ranges[col_block_idx].dim);
        let col_offset = self.ranges[col_block_idx].index as usize;
        self.mat.fixed_view_mut::<RESIDUAL_DIM, C>(0, col_offset)
    }
}
