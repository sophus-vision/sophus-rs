#[derive(Copy, Clone, Debug)]
pub struct BlockVector<const MAX_DIM: usize, const NUM_BLOCKS: usize> {
    pub vec: nalgebra::SVector<f64, MAX_DIM>,
    pub indices: [i64; NUM_BLOCKS],
    pub dims: [usize; NUM_BLOCKS],
}

impl<const MAX_DIM: usize, const NUM_BLOCKS: usize> BlockVector<MAX_DIM, NUM_BLOCKS> {
    pub fn block(
        &self,
        i: usize,
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
            nalgebra::Const<MAX_DIM>,
        >,
    > {
        let idx = self.indices[i] as usize;
        self.vec.index((idx..idx + self.dims[i], ..))
    }

    pub fn mut_block(
        &mut self,
        i: usize,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Dyn,
        nalgebra::Const<1>,
        nalgebra::ViewStorageMut<
            '_,
            f64,
            nalgebra::Dyn,
            nalgebra::Const<1>,
            nalgebra::Const<1>,
            nalgebra::Const<MAX_DIM>,
        >,
    > {
        let idx = self.indices[i] as usize;
        self.vec.index_mut((idx..idx + self.dims[i], ..))
    }
}

impl<const MAX_DIM: usize, const NUM_BLOCKS: usize> BlockVector<MAX_DIM, NUM_BLOCKS> {
    pub fn new(dims: &[usize; NUM_BLOCKS]) -> Self {
        let mut idx = [0; NUM_BLOCKS];
        for i in 1..NUM_BLOCKS {
            idx[i] = idx[i - 1] + dims[i - 1] as i64;
        }
        Self {
            vec: nalgebra::SMatrix::repeat(0.0),
            indices: idx,
            dims: *dims,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BlockMatrix<const MAX_DIM: usize, const NUM_BLOCKS: usize> {
    pub mat: nalgebra::SMatrix<f64, MAX_DIM, MAX_DIM>,
    pub indices: [i64; NUM_BLOCKS],
    pub dims: [usize; NUM_BLOCKS],
}

impl<const MAX_DIM: usize, const NUM_BLOCKS: usize> BlockMatrix<MAX_DIM, NUM_BLOCKS> {
    pub fn block(
        &self,
        i: usize,
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
            nalgebra::Const<MAX_DIM>,
        >,
    > {
        let idx = self.indices[i] as usize;
        self.mat
            .index((idx..idx + self.dims[i], idx..idx + self.dims[i]))
    }

    pub fn mut_block(
        &mut self,
        i: usize,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Dyn,
        nalgebra::Dyn,
        nalgebra::ViewStorageMut<
            '_,
            f64,
            nalgebra::Dyn,
            nalgebra::Dyn,
            nalgebra::Const<1>,
            nalgebra::Const<MAX_DIM>,
        >,
    > {
        let idx = self.indices[i] as usize;
        self.mat
            .index_mut((idx..idx + self.dims[i], idx..idx + self.dims[i]))
    }
}

impl<const MAX_DIM: usize, const NUM_BLOCKS: usize> BlockMatrix<MAX_DIM, NUM_BLOCKS> {
    pub fn new(dims: &[usize; NUM_BLOCKS]) -> Self {
        let mut idx = [0; NUM_BLOCKS];
        for i in 1..NUM_BLOCKS {
            idx[i] = idx[i - 1] + dims[i - 1] as i64;
        }
        Self {
            mat: nalgebra::SMatrix::repeat(0.0),
            indices: idx,
            dims: *dims,
        }
    }
}
