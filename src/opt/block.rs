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
pub struct NewBlockVector {
    pub vec: faer_core::Mat<f64>,
    pub ranges: Vec<BlockRange>,
}

impl NewBlockVector {
    pub fn num_blocks(&self) -> usize {
        self.ranges.len()
    }

    pub fn set_block<const R: usize>(&mut self, ith: usize, v: V<R>) {
        assert!(ith < self.num_blocks());
        assert_eq!(R, self.ranges[ith].dim);
        let mut mut_block = self.mut_block(ith);

        for i in 0..R {
            mut_block.write(i, 0, v[i]);
        }
    }

    pub fn block(&self, ith: usize) -> faer_core::MatRef<f64> {
        let idx = self.ranges[ith].index as usize;
        self.vec.as_ref().submatrix(idx, 0, self.ranges[ith].dim, 1)
    }

    pub fn mut_block(&mut self, ith: usize) -> faer_core::MatMut<f64> {
        let idx = self.ranges[ith].index as usize;
        self.vec
            .as_mut()
            .submatrix_mut(idx, 0, self.ranges[ith].dim, 1)
    }
}

impl NewBlockVector {
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
            vec: faer_core::Mat::<f64>::zeros(num_rows, 1),
            ranges,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NewBlockMatrix {
    pub mat: faer_core::Mat<f64>,
    pub ranges: Vec<BlockRange>,
}

impl NewBlockMatrix {
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

        let mut mut_block = self.mut_block(ith, jth);

        for i in 0..R {
            for j in 0..C {
                mut_block.write(i, j, m[(i, j)]);
            }
        }
    }

    pub fn block(&self, ith: usize, jth: usize) -> faer_core::MatRef<f64> {
        let idx_i = self.ranges[ith].index as usize;
        let idx_j = self.ranges[jth].index as usize;
        self.mat
            .as_ref()
            .submatrix(idx_i, idx_j, self.ranges[ith].dim, self.ranges[jth].dim)
    }

    pub fn mut_block(&mut self, ith: usize, jth: usize) -> faer_core::MatMut<f64> {
        let idx_i = self.ranges[ith].index as usize;
        let idx_j = self.ranges[jth].index as usize;
        self.mat
            .as_mut()
            .submatrix_mut(idx_i, idx_j, self.ranges[ith].dim, self.ranges[jth].dim)
    }
}

impl NewBlockMatrix {
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
            mat: faer_core::Mat::<f64>::zeros(num_rows, num_rows),
            ranges,
        }
    }
}
