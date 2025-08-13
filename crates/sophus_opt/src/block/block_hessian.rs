use sophus_autodiff::linalg::MatF64;

use super::BlockRange;

/// Hessian matrix, partitioned into several blocks
///
/// ```ascii
/// -------------------------------------------------
/// |               |               |               |
/// |     H_0,0     |   .   .   .   |   H_0,{N-1}   |
/// |               |               |               |
/// |-------------------------------|---------------|
/// |       .       |   .           |       .       |
/// |       .       |       .       |       .       |
/// |       .       |           .   |       .       |
/// |---------------|-------------------------------|
/// |               |               |               |
/// |  H_{N-1},{0}  |   .   .   .   | H_{N-1},{N-1} |
/// |               |               |               |
/// -------------------------------------------------
/// ```
///
/// The `(INPUT_DIM  x  INPUT_DIM)` symmetric matrix is partitioned into `N` blocks horizontally and
/// vertically. The shape of each of are specified by the `ranges` array: The Hessian sub-block
/// `H_i,j` is a `(ranges(i).dim  x  ranges(j).dim)` matrix.
#[derive(Clone, Debug)]
pub struct BlockHessian<const INPUT_DIM: usize, const N: usize> {
    /// matrix storage
    pub mat: nalgebra::SMatrix<f64, INPUT_DIM, INPUT_DIM>,
    /// ranges, one for each block
    pub ranges: [BlockRange; N],
}

impl<const INPUT_DIM: usize, const N: usize> BlockHessian<INPUT_DIM, N> {
    /// create a new block matrix
    pub fn new(dims: &[usize]) -> Self {
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
            mat: nalgebra::SMatrix::zeros(),
            ranges,
        }
    }

    /// Number of blocks
    pub fn num_blocks(&self) -> usize {
        self.ranges.len()
    }

    /// set block (i, j)
    pub fn set_block<const R: usize, const C: usize>(
        &mut self,
        ith: usize,
        jth: usize,
        m: MatF64<R, C>,
    ) {
        debug_assert!(ith < self.num_blocks());
        debug_assert!(jth < self.num_blocks());
        debug_assert_eq!(R, self.ranges[ith].dim);
        debug_assert_eq!(C, self.ranges[jth].dim);

        if ith == jth {
            debug_assert_eq!(R, C);

            let mut mut_block = self.mut_block::<R, C>(ith, jth);
            mut_block.copy_from(&m);
        } else {
            debug_assert!(ith < jth);
            {
                self.mut_block::<R, C>(ith, jth).copy_from(&m);
            }
            {
                self.mut_block::<C, R>(jth, ith).copy_from(&m.transpose());
            }
        }
    }

    /// get block (i, j)
    pub fn block(
        &self,
        ith: usize,
        jth: usize,
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
            nalgebra::Const<INPUT_DIM>,
        >,
    > {
        let idx_i = self.ranges[ith].index as usize;
        let idx_j = self.ranges[jth].index as usize;
        self.mat
            .view((idx_i, idx_j), (self.ranges[ith].dim, self.ranges[jth].dim))
    }

    /// mutable reference to block (i, j)
    pub fn mut_block<const R: usize, const C: usize>(
        &mut self,
        ith: usize,
        jth: usize,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Const<R>,
        nalgebra::Const<C>,
        nalgebra::ViewStorageMut<
            '_,
            f64,
            nalgebra::Const<R>,
            nalgebra::Const<C>,
            nalgebra::Const<1>,
            nalgebra::Const<INPUT_DIM>,
        >,
    > {
        let idx_i = self.ranges[ith].index as usize;
        let idx_j = self.ranges[jth].index as usize;
        self.mat.fixed_view_mut::<R, C>(idx_i, idx_j)
    }
}
