use nalgebra::Const;
use sophus_core::linalg::MatF64;
use sophus_core::linalg::VecF64;

/// Range of a block
#[derive(Clone, Debug, Copy, Default)]
pub struct BlockRange {
    /// Index of the first element of the block
    pub index: i64,
    /// Dimension of the block
    pub dim: usize,
}

/// Block vector
#[derive(Debug, Clone)]
pub struct BlockVector<const NUM: usize, const NUM_ARGS: usize> {
    /// vector storage
    pub vec: nalgebra::SVector<f64, NUM>,
    /// ranges, one for each block
    pub ranges: [BlockRange; NUM_ARGS],
}

impl<const NUM: usize, const NUM_ARGS: usize> BlockVector<NUM, NUM_ARGS> {
    /// create a new block vector
    pub fn new(dims: &[usize; NUM_ARGS]) -> Self {
        debug_assert!(!dims.is_empty());

        let num_blocks = dims.len();

        let mut ranges = [BlockRange::default(); NUM_ARGS];

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
            Const<NUM>,
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
            Const<NUM>,
        >,
    > {
        let idx = self.ranges[ith].index as usize;
        self.vec.fixed_rows_mut::<ROWS>(idx)
    }
}

/// Block matrix
#[derive(Clone, Debug)]
pub struct BlockMatrix<const NUM: usize, const NUM_ARGS: usize> {
    /// matrix storage
    pub mat: nalgebra::SMatrix<f64, NUM, NUM>,
    /// ranges, one for each block
    pub ranges: [BlockRange; NUM_ARGS],
}

impl<const NUM: usize, const NUM_ARGS: usize> BlockMatrix<NUM, NUM_ARGS> {
    /// create a new block matrix
    pub fn new(dims: &[usize]) -> Self {
        debug_assert!(!dims.is_empty());

        let num_blocks = dims.len();

        let mut ranges = [BlockRange::default(); NUM_ARGS];
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
            nalgebra::Const<NUM>,
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
            nalgebra::Const<NUM>,
        >,
    > {
        let idx_i = self.ranges[ith].index as usize;
        let idx_j = self.ranges[jth].index as usize;
        self.mat.fixed_view_mut::<R, C>(idx_i, idx_j)
    }
}
