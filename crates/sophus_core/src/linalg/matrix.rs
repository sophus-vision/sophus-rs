use crate::calculus::dual::DualBatchMatrix;
use crate::calculus::dual::DualMatrix;
use crate::linalg::BatchMatF64;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::linalg::MatF64;
use crate::linalg::VecF64;
use crate::prelude::*;
use approx::AbsDiffEq;
use approx::RelativeEq;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;
use std::simd::LaneCount;
use std::simd::Mask;
use std::simd::SupportedLaneCount;

/// Matrix trait
///  - either a real (f64) or a dual number matrix
///  - either a single matrix or a batch matrix
pub trait IsMatrix<
    S: IsScalar<BATCH_SIZE>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH_SIZE: usize,
>:
    Debug
    + Clone
    + Sized
    + Mul<S::Vector<COLS>, Output = S::Vector<ROWS>>
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg
    + AbsDiffEq<Epsilon = f64>
    + RelativeEq<Epsilon = f64>
{
    /// creates matrix from a left and right block columns
    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: S::Matrix<ROWS, C0>,
        righ_col: S::Matrix<ROWS, C1>,
    ) -> Self;

    /// creates matrix from a top and bottom block rows
    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: S::Matrix<R0, COLS>,
        bot_row: S::Matrix<R1, COLS>,
    ) -> Self;

    /// creates matrix from a 2x2 block of matrices
    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (S::Matrix<R0, C0>, S::Matrix<R0, C1>),
        bot_row: (S::Matrix<R1, C0>, S::Matrix<R1, C1>),
    ) -> Self;

    /// creates matrix from a 2d array of scalars
    fn from_array2(vals: [[S; COLS]; ROWS]) -> Self;

    /// creates matrix with all real elements (and lanes) set to the given value
    ///
    /// (for dual numbers, the infinitesimal part is set to zero)
    fn from_f64(val: f64) -> Self;

    /// creates matrix from a 2d array of real values
    ///
    ///  - all lanes are set to the same value
    ///  - for dual numbers, the infinitesimal part is set to zero
    fn from_f64_array2(vals: [[f64; COLS]; ROWS]) -> Self;

    /// creates matrix from a 2d array of real scalars
    ///
    /// (for dual numbers, the infinitesimal part is set to zero)
    fn from_real_scalar_array2(vals: [[S::RealScalar; COLS]; ROWS]) -> Self;

    /// create a constant matrix
    fn from_real_matrix(val: S::RealMatrix<ROWS, COLS>) -> Self;

    /// create a constant scalar
    fn from_scalar(val: S) -> Self;

    /// extract column vector
    fn get_col_vec(&self, c: usize) -> S::Vector<ROWS>;

    /// extract row vector
    fn get_row_vec(&self, r: usize) -> S::Vector<COLS>;

    /// get element
    fn get_elem(&self, idx: [usize; 2]) -> S;

    /// get fixed submatrix
    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> S::Matrix<R, C>;

    /// create an identity matrix
    fn identity() -> Self;

    /// matrix multiplication
    fn mat_mul<const C2: usize>(&self, other: S::Matrix<COLS, C2>) -> S::Matrix<ROWS, C2>;

    /// ones
    fn ones() -> Self {
        Self::from_f64(1.0)
    }

    /// Return a real matrix
    fn real_matrix(&self) -> &S::RealMatrix<ROWS, COLS>;

    /// return scaled matrix
    fn scaled(&self, v: S) -> Self;

    /// Returns self if mask is true, otherwise returns other
    ///
    /// For batch matrices, this is a lane-wise operation
    fn select(self, mask: &S::Mask, other: Self) -> Self;

    /// set i-th element
    fn set_elem(&mut self, idx: [usize; 2], val: S);

    /// set column vectors
    fn set_col_vec(&mut self, c: usize, v: S::Vector<ROWS>);

    /// Return dual matrix
    ///
    /// If self is a real matrix, this will return a dual matrix with the infinitesimal part set to
    /// zero: (self, 0ϵ)
    fn to_dual(self) -> S::DualMatrix<ROWS, COLS>;

    /// zeros
    fn zeros() -> Self {
        Self::from_f64(0.0)
    }
}

/// Is real matrix?
pub trait IsRealMatrix<
    S: IsRealScalar<BATCH_SIZE> + IsScalar<BATCH_SIZE>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH_SIZE: usize,
>:
    IsMatrix<S, ROWS, COLS, BATCH_SIZE>
    + Index<(usize, usize), Output = S>
    + IndexMut<(usize, usize), Output = S>
    + Copy
{
}

/// Is single matrix? (not batch)
pub trait IsSingleMatrix<S: IsSingleScalar, const ROWS: usize, const COLS: usize>:
    IsMatrix<S, ROWS, COLS, 1> + Mul<S::SingleVector<COLS>, Output = S::SingleVector<ROWS>>
{
}

impl<const ROWS: usize, const COLS: usize> IsRealMatrix<f64, ROWS, COLS, 1> for MatF64<ROWS, COLS> {}

impl<const ROWS: usize, const COLS: usize> IsSingleMatrix<f64, ROWS, COLS> for MatF64<ROWS, COLS> {}

impl<const ROWS: usize, const COLS: usize> IsMatrix<f64, ROWS, COLS, 1> for MatF64<ROWS, COLS> {
    fn from_real_matrix(val: MatF64<ROWS, COLS>) -> Self {
        val
    }

    fn from_scalar(val: f64) -> Self {
        Self::from_f64(val)
    }

    fn from_array2(vals: [[f64; COLS]; ROWS]) -> MatF64<ROWS, COLS> {
        let mut m = MatF64::<ROWS, COLS>::zeros();

        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c];
            }
        }
        m
    }

    fn from_real_scalar_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        let mut m = MatF64::<ROWS, COLS>::zeros();
        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c];
            }
        }
        m
    }

    fn get_elem(&self, idx: [usize; 2]) -> f64 {
        self[(idx[0], idx[1])]
    }

    fn identity() -> Self {
        Self::identity()
    }

    fn real_matrix(&self) -> &Self {
        self
    }

    fn mat_mul<const C2: usize>(&self, other: MatF64<COLS, C2>) -> MatF64<ROWS, C2> {
        self * other
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: MatF64<R0, COLS>,
        bot_row: MatF64<R1, COLS>,
    ) -> Self {
        assert_eq!(ROWS, R0 + R1);
        let mut m = Self::zeros();

        m.fixed_view_mut::<R0, COLS>(0, 0).copy_from(&top_row);
        m.fixed_view_mut::<R1, COLS>(R0, 0).copy_from(&bot_row);
        m
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (MatF64<R0, C0>, MatF64<R0, C1>),
        bot_row: (MatF64<R1, C0>, MatF64<R1, C1>),
    ) -> Self {
        assert_eq!(ROWS, R0 + R1);
        assert_eq!(COLS, C0 + C1);
        let mut m = Self::zeros();

        m.fixed_view_mut::<R0, C0>(0, 0).copy_from(&top_row.0);
        m.fixed_view_mut::<R0, C1>(0, C0).copy_from(&top_row.1);

        m.fixed_view_mut::<R1, C0>(R0, 0).copy_from(&bot_row.0);
        m.fixed_view_mut::<R1, C1>(R0, C0).copy_from(&bot_row.1);
        m
    }

    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: MatF64<ROWS, C0>,
        righ_col: MatF64<ROWS, C1>,
    ) -> Self {
        assert_eq!(COLS, C0 + C1);
        let mut m = Self::zeros();

        m.fixed_view_mut::<ROWS, C0>(0, 0).copy_from(&left_col);
        m.fixed_view_mut::<ROWS, C1>(0, C0).copy_from(&righ_col);

        m
    }

    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> MatF64<R, C> {
        self.fixed_view::<R, C>(start_r, start_c).into()
    }

    fn get_col_vec(&self, c: usize) -> VecF64<ROWS> {
        self.fixed_view::<ROWS, 1>(0, c).into()
    }

    fn get_row_vec(&self, r: usize) -> VecF64<COLS> {
        self.fixed_view::<1, COLS>(r, 0).transpose()
    }

    fn scaled(&self, v: f64) -> Self {
        self * v
    }

    fn from_f64_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        let mut m = MatF64::<ROWS, COLS>::zeros();
        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c];
            }
        }
        m
    }

    fn from_f64(val: f64) -> Self {
        MatF64::<ROWS, COLS>::from_element(val)
    }

    fn set_col_vec(&mut self, c: usize, v: VecF64<ROWS>) {
        self.fixed_columns_mut::<1>(c).copy_from(&v);
    }

    fn to_dual(self) -> <f64 as IsScalar<1>>::DualMatrix<ROWS, COLS> {
        DualMatrix::from_real_matrix(self)
    }

    fn select(self, mask: &bool, other: Self) -> Self {
        if *mask {
            self
        } else {
            other
        }
    }

    fn set_elem(&mut self, idx: [usize; 2], val: f64) {
        self[(idx[0], idx[1])] = val;
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize>
    IsMatrix<BatchScalarF64<BATCH>, ROWS, COLS, BATCH> for BatchMatF64<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn from_scalar(val: BatchScalarF64<BATCH>) -> Self {
        Self::from_element(val)
    }

    fn from_real_matrix(val: Self) -> Self {
        val
    }

    fn real_matrix(&self) -> &Self {
        self
    }

    fn scaled(&self, v: BatchScalarF64<BATCH>) -> Self {
        self * v
    }

    fn identity() -> Self {
        nalgebra::SMatrix::<BatchScalarF64<BATCH>, ROWS, COLS>::identity()
    }

    fn from_array2(vals: [[BatchScalarF64<BATCH>; COLS]; ROWS]) -> Self {
        Self::from_fn(|r, c| vals[r][c])
    }

    fn from_real_scalar_array2(vals: [[BatchScalarF64<BATCH>; COLS]; ROWS]) -> Self {
        Self::from_fn(|r, c| vals[r][c])
    }

    fn from_f64_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        Self::from_fn(|r, c| BatchScalarF64::<BATCH>::from_f64(vals[r][c]))
    }

    fn get_elem(&self, idx: [usize; 2]) -> BatchScalarF64<BATCH> {
        self[(idx[0], idx[1])]
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: BatchMatF64<R0, COLS, BATCH>,
        bot_row: BatchMatF64<R1, COLS, BATCH>,
    ) -> Self {
        Self::from_fn(|r, c| {
            if r < R0 {
                top_row[(r, c)]
            } else {
                bot_row[(r - R0, c)]
            }
        })
    }

    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: BatchMatF64<ROWS, C0, BATCH>,
        righ_col: BatchMatF64<ROWS, C1, BATCH>,
    ) -> Self {
        Self::from_fn(|r, c| {
            if c < C0 {
                left_col[(r, c)]
            } else {
                righ_col[(r, c - C0)]
            }
        })
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (BatchMatF64<R0, C0, BATCH>, BatchMatF64<R0, C1, BATCH>),
        bot_row: (BatchMatF64<R1, C0, BATCH>, BatchMatF64<R1, C1, BATCH>),
    ) -> Self {
        Self::from_fn(|r, c| {
            if r < R0 {
                if c < C0 {
                    top_row.0[(r, c)]
                } else {
                    top_row.1[(r, c - C0)]
                }
            } else if c < C0 {
                bot_row.0[(r - R0, c)]
            } else {
                bot_row.1[(r - R0, c - C0)]
            }
        })
    }

    fn mat_mul<const C2: usize>(
        &self,
        other: BatchMatF64<COLS, C2, BATCH>,
    ) -> BatchMatF64<ROWS, C2, BATCH> {
        self * other
    }

    fn set_col_vec(&mut self, c: usize, v: BatchVecF64<ROWS, BATCH>) {
        self.fixed_columns_mut::<1>(c).copy_from(&v);
    }

    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> BatchMatF64<R, C, BATCH> {
        self.fixed_view::<R, C>(start_r, start_c).into()
    }

    fn get_col_vec(&self, c: usize) -> BatchVecF64<ROWS, BATCH> {
        self.fixed_view::<ROWS, 1>(0, c).into()
    }

    fn get_row_vec(&self, r: usize) -> BatchVecF64<COLS, BATCH> {
        self.fixed_view::<1, COLS>(r, 0).transpose()
    }

    fn from_f64(val: f64) -> Self {
        Self::from_element(BatchScalarF64::<BATCH>::from_f64(val))
    }

    fn to_dual(self) -> <BatchScalarF64<BATCH> as IsScalar<BATCH>>::DualMatrix<ROWS, COLS> {
        DualBatchMatrix::from_real_matrix(self)
    }

    fn select(self, mask: &Mask<i64, BATCH>, other: Self) -> Self {
        self.zip_map(&other, |a, b| a.select(mask, b))
    }

    fn set_elem(&mut self, idx: [usize; 2], val: BatchScalarF64<BATCH>) {
        self[(idx[0], idx[1])] = val;
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize>
    IsRealMatrix<BatchScalarF64<BATCH>, ROWS, COLS, BATCH> for BatchMatF64<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
}
