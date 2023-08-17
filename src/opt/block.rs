use dfdx::prelude::*;

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

#[derive(Debug)]
pub struct BlockVector<const NUM_BLOCKS: usize> {
    pub vec: faer_core::Mat<f64>,
    pub indices: [i64; NUM_BLOCKS],
    pub dims: [usize; NUM_BLOCKS],
}

impl<const NUM_BLOCKS: usize> BlockVector<NUM_BLOCKS> {
    pub fn set_block(&mut self, ith: usize, v: Tensor<(usize,), f64, Cpu>) {
        assert!(ith<NUM_BLOCKS);
        assert_eq!(v.shape().0, self.dims[ith]);
        let mut mut_block = self.mut_block(ith);

        for i in 0..v.shape().0 {
            mut_block.write(i, 0, v[[i]]);
        }
    }

    pub fn block(&self, ith: usize) -> faer_core::MatRef<f64> {
        let idx = self.indices[ith] as usize;
        self.vec.as_ref().submatrix(idx, 0, self.dims[ith], 1)
    }

    pub fn mut_block(&mut self, ith: usize) -> faer_core::MatMut<f64> {
        let idx = self.indices[ith] as usize;
        self.vec.as_mut().submatrix(idx, 0, self.dims[ith], 1)
    }
}

impl<const NUM_BLOCKS: usize> BlockVector<NUM_BLOCKS> {
    pub fn new(dims: &[usize; NUM_BLOCKS]) -> Self {
        let mut idx = [0; NUM_BLOCKS];

        let mut num_rows = 0;
        for i in 1..NUM_BLOCKS {
            idx[i] = idx[i - 1] + dims[i - 1] as i64;
            num_rows += dims[i - 1];
        }
        num_rows += dims[NUM_BLOCKS - 1];
        Self {
            vec: faer_core::Mat::<f64>::zeros(num_rows, 1),
            indices: idx,
            dims: *dims,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BlockMatrix<const NUM_BLOCKS: usize> {
    pub mat: faer_core::Mat<f64>,
    pub indices: [i64; NUM_BLOCKS],
    pub dims: [usize; NUM_BLOCKS],
}

impl<const NUM_BLOCKS: usize> BlockMatrix<NUM_BLOCKS> {
    pub fn set_block(&mut self, ith: usize, jth: usize, v: Tensor<(usize, usize), f64, Cpu>) {
        assert!(ith<NUM_BLOCKS);
        assert!(jth<NUM_BLOCKS);
        assert_eq!(v.shape().0, self.dims[ith]);
        assert_eq!(v.shape().1, self.dims[jth]);

        let mut mut_block = self.mut_block(ith, jth);

        for i in 0..v.shape().0 {
            for j in 0..v.shape().0 {
                mut_block.write(i, j, v[[i, j]]);
            }
        }
    }

    pub fn block(&self, ith: usize, jth: usize) -> faer_core::MatRef<f64> {
        let idx_i = self.indices[ith] as usize;
        let idx_j = self.indices[jth] as usize;
        self.mat
            .as_ref()
            .submatrix(idx_i, idx_j, self.dims[ith], self.dims[jth])
    }

    pub fn mut_block(&mut self, ith: usize, jth: usize) -> faer_core::MatMut<f64> {
        let idx_i = self.indices[ith] as usize;
        let idx_j = self.indices[jth] as usize;
        self.mat
            .as_mut()
            .submatrix(idx_i, idx_j, self.dims[ith], self.dims[jth])
    }
}

impl<const NUM_BLOCKS: usize> BlockMatrix<NUM_BLOCKS> {
    pub fn new(dims: &[usize; NUM_BLOCKS]) -> Self {
        let mut idx = [0; NUM_BLOCKS];

        let mut num_rows = 0;
        for i in 1..NUM_BLOCKS {
            idx[i] = idx[i - 1] + dims[i - 1] as i64;
            num_rows += dims[i - 1];
        }
        num_rows += dims[NUM_BLOCKS - 1];
        Self {
            mat: faer_core::Mat::<f64>::zeros(num_rows, num_rows),
            indices: idx,
            dims: *dims,
        }
    }
}
