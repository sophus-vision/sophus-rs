use nalgebra::Const;

use crate::calculus::types::M;
use crate::calculus::types::V;

pub trait SophusAddAssign<Rhs = Self> {
    // Required method
    fn add_assign(&mut self, rhs: Rhs);
}

impl SophusAddAssign for faer_core::Mat<f64> {
    fn add_assign(&mut self, other: Self) {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                self.write(i, j, self.read(i, j) + other.read(i, j));
            }
        }
    }
}
#[derive(Clone, Debug, Copy)]
pub struct BlockRange {
    pub index: i64,
    pub dim: usize,
}

#[derive(Debug, Clone)]
pub struct BlockVector<const NUM: usize> {
    pub vec: nalgebra::SVector<f64, NUM>,
    pub ranges: Vec<BlockRange>,
}

impl<const NUM: usize> BlockVector<NUM> {
    pub fn num_blocks(&self) -> usize {
        self.ranges.len()
    }

    pub fn set_block<const R: usize>(&mut self, ith: usize, v: V<R>) {
        assert!(ith < self.num_blocks());
        assert_eq!(R, self.ranges[ith].dim);
        self.mut_block::<R>(ith).copy_from(&v);
    }

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

    pub fn new(dims: &Vec<usize>) -> Self {
        assert!(!dims.is_empty());

        let num_blocks = dims.len();

        let mut ranges = Vec::with_capacity(num_blocks);

        let mut num_rows: usize = 0;

        for i in 0..num_blocks {
            let dim = dims[i];
            ranges.push(BlockRange {
                index: num_rows as i64,
                dim,
            });
            num_rows += dim;
        }
        Self {
            vec: nalgebra::SVector::zeros(),
            ranges,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NewBlockMatrix<const NUM: usize> {
    pub mat: nalgebra::SMatrix<f64, NUM, NUM>,
    pub ranges: Vec<BlockRange>,
}

impl<const NUM: usize> NewBlockMatrix<NUM> {
    pub fn num_blocks(&self) -> usize {
        self.ranges.len()
    }

    pub fn set_block<const R: usize, const C: usize>(
        &mut self,
        ith: usize,
        jth: usize,
        m: M<R, C>,
    ) {
        assert!(ith < self.num_blocks());
        assert!(jth < self.num_blocks());
        assert_eq!(R, self.ranges[ith].dim);
        assert_eq!(C, self.ranges[jth].dim);

        if ith == jth {
            assert_eq!(R, C);

            let mut mut_block = self.mut_block::<R, C>(ith, jth);
            mut_block.copy_from(&m);
        } else {
            assert!(ith < jth);
            {
                self.mut_block::<R, C>(ith, jth).copy_from(&m);
            }
            {
                self.mut_block::<C, R>(jth, ith).copy_from(&m.transpose());
            }
        }
    }

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

    pub fn new(dims: &Vec<usize>) -> Self {
        assert!(!dims.is_empty());

        let num_blocks = dims.len();

        let mut ranges = Vec::with_capacity(num_blocks);

        let mut num_rows: usize = 0;

        for i in 0..num_blocks {
            let dim = dims[i];
            ranges.push(BlockRange {
                index: num_rows as i64,
                dim,
            });
            num_rows += dim;
        }
        Self {
            mat: nalgebra::SMatrix::zeros(),
            ranges,
        }
    }
}
