use crate::types::scalar::IsScalar;
use crate::types::vector::IsVectorLike;
use crate::types::MatF64;
use crate::types::VecF64;

use std::fmt::Debug;
use std::ops::Mul;

/// Matrix - either a real (f64) or a dual number matrix
pub trait IsMatrix<
    S: IsScalar<BATCH_SIZE>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH_SIZE: usize,
>: Debug + Clone + Sized + Mul<S::Vector<COLS>, Output = S::Vector<ROWS>> + IsVectorLike
{
    /// create a constant matrix
    fn c(val: MatF64<ROWS, COLS>) -> Self;

    /// return the real part
    fn real(&self) -> &MatF64<ROWS, COLS>;

    /// return scaled matrix
    fn scaled(&self, v: S) -> Self;

    /// create an identity matrix
    fn identity() -> Self;

    /// create from 2d array
    fn from_array2(vals: [[S; COLS]; ROWS]) -> Self;

    /// create from constant 2d array
    fn from_c_array2(vals: [[f64; COLS]; ROWS]) -> Self;

    /// get element
    fn get(&self, idx: (usize, usize)) -> S;

    /// create 2x1 block matrix
    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: S::Matrix<R0, COLS>,
        bot_row: S::Matrix<R1, COLS>,
    ) -> Self;

    /// create 1x2 block matrix
    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: S::Matrix<ROWS, C0>,
        righ_col: S::Matrix<ROWS, C1>,
    ) -> Self;

    /// create 2x2 block matrix
    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (S::Matrix<R0, C0>, S::Matrix<R0, C1>),
        bot_row: (S::Matrix<R1, C0>, S::Matrix<R1, C1>),
    ) -> Self;

    /// matrix multiplication
    fn mat_mul<const C2: usize>(&self, other: S::Matrix<COLS, C2>) -> S::Matrix<ROWS, C2>;

    /// get fixed submatrix
    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> S::Matrix<R, C>;

    /// extract column vector
    fn get_col_vec(&self, r: usize) -> S::Vector<ROWS>;

    /// extract row vector
    fn get_row_vec(&self, r: usize) -> S::Vector<ROWS>;
}

impl<const ROWS: usize, const COLS: usize> IsVectorLike for MatF64<ROWS, COLS> {
    fn zero() -> Self {
        MatF64::zeros()
    }
}

impl<const ROWS: usize, const COLS: usize> IsMatrix<f64, ROWS, COLS, 1> for MatF64<ROWS, COLS> {
    fn c(val: MatF64<ROWS, COLS>) -> Self {
        val
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

    fn from_c_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        let mut m = MatF64::<ROWS, COLS>::zeros();
        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c];
            }
        }
        m
    }

    fn get(&self, idx: (usize, usize)) -> f64 {
        self[idx]
    }

    fn identity() -> Self {
        Self::identity()
    }

    fn real(&self) -> &Self {
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
        let mut m = Self::zero();

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
        let mut m = Self::zero();

        m.fixed_view_mut::<R0, C0>(0, 0).copy_from(&top_row.0);
        m.fixed_view_mut::<R0, C1>(0, C0).copy_from(&top_row.1);

        m.fixed_view_mut::<R1, C0>(R0, 0).copy_from(&bot_row.0);
        m.fixed_view_mut::<R1, C1>(R0, C0).copy_from(&bot_row.1);
        m
    }

    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: <f64 as IsScalar<1>>::Matrix<ROWS, C0>,
        righ_col: <f64 as IsScalar<1>>::Matrix<ROWS, C1>,
    ) -> Self {
        assert_eq!(COLS, C0 + C1);
        let mut m = Self::zero();

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

    fn get_row_vec(&self, r: usize) -> VecF64<ROWS> {
        self.fixed_view::<1, ROWS>(0, r).transpose()
    }

    fn scaled(&self, v: f64) -> Self {
        self * v
    }
}
