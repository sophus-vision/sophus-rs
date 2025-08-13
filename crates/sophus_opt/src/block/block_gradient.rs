use nalgebra::Const;
use sophus_autodiff::linalg::VecF64;

use super::BlockRange;

/// Block gradient vector
///
/// ```ascii
/// -----------
/// |         |
/// |   g_0   |
/// |         |
/// |---------|
/// |    .    |
/// |    .    |
/// |    .    |
/// |---------|
/// |         |
/// | g_{N-1} |
/// |         |
/// -----------
/// ```
///
/// The `INPUT_DIM`-dimensional vector is partitioned into `N` blocks, each of
/// which has an offset and dimension specified by the `ranges` array.
///
/// Hence, the gradient sub-block `g_i` has a dimensionality of `ranges(i).dim`.
#[derive(Debug, Clone)]
pub struct BlockGradient<const INPUT_DIM: usize, const N: usize> {
    /// vector storage
    pub vec: nalgebra::SVector<f64, INPUT_DIM>,
    /// ranges, one for each block
    pub ranges: [BlockRange; N],
}

impl<const INPUT_DIM: usize, const N: usize> BlockGradient<INPUT_DIM, N> {
    /// create a new block vector
    pub fn new(dims: &[usize; N]) -> Self {
        debug_assert!(!dims.is_empty());

        let num_blocks = dims.len();

        let mut ranges = [BlockRange::default(); N];

        let mut num_rows: usize = 0;

        for i in 0..num_blocks {
            let dim = dims[i];
            ranges[i] = BlockRange {
                index: num_rows as i64,
                dim,
            };
            num_rows += dim;
        }
        Self {
            vec: nalgebra::SVector::zeros(),
            ranges,
        }
    }

    /// Number of blocks
    pub fn num_blocks(&self) -> usize {
        self.ranges.len()
    }

    /// set the ith block
    pub fn set_block<const R: usize>(&mut self, ith: usize, v: VecF64<R>) {
        debug_assert!(ith < self.num_blocks());
        debug_assert_eq!(R, self.ranges[ith].dim);
        self.mut_block::<R>(ith).copy_from(&v);
    }

    /// get the ith block
    pub fn block(
        &self,
        ith: usize,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Dyn,
        nalgebra::Const<1>,
        nalgebra::ViewStorage<
            '_,
            f64,
            nalgebra::Dyn,
            nalgebra::Const<1>,
            nalgebra::Const<1>,
            Const<INPUT_DIM>,
        >,
    > {
        let idx = self.ranges[ith].index as usize;
        self.vec.rows(idx, self.ranges[ith].dim)
    }

    /// mutable reference to the ith block
    pub fn mut_block<const ROWS: usize>(
        &mut self,
        ith: usize,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Const<ROWS>,
        nalgebra::Const<1>,
        nalgebra::ViewStorageMut<
            '_,
            f64,
            nalgebra::Const<ROWS>,
            nalgebra::Const<1>,
            nalgebra::Const<1>,
            Const<INPUT_DIM>,
        >,
    > {
        let idx = self.ranges[ith].index as usize;
        self.vec.fixed_rows_mut::<ROWS>(idx)
    }
}
