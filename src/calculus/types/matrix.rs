use super::scalar::IsScalar;
use super::vector::IsVectorLike;
use super::M;
use super::V;
use std::fmt::Debug;
use std::ops::Mul;

pub trait IsMatrix<S: IsScalar, const ROWS: usize, const COLS: usize>:
    Debug + Clone + Sized + Mul<S::Vector<COLS>, Output = S::Vector<ROWS>> + IsVectorLike
{
    fn c(val: M<ROWS, COLS>) -> Self;
    fn real(&self) -> &M<ROWS, COLS>;

    fn scaled(&self, v: S) -> Self;

    fn identity() -> Self;
    fn from_array2(vals: [[S; COLS]; ROWS]) -> Self;
    fn from_c_array2(vals: [[f64; COLS]; ROWS]) -> Self;

    fn get(&self, idx: (usize, usize)) -> S;

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: S::Matrix<R0, COLS>,
        bot_row: S::Matrix<R1, COLS>,
    ) -> Self;

    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: S::Matrix<ROWS, C0>,
        righ_col: S::Matrix<ROWS, C1>,
    ) -> Self;

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (S::Matrix<R0, C0>, S::Matrix<R0, C1>),
        bot_row: (S::Matrix<R1, C0>, S::Matrix<R1, C1>),
    ) -> Self;

    fn mat_mul<const C2: usize>(&self, other: S::Matrix<COLS, C2>) -> S::Matrix<ROWS, C2>;

    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> S::Matrix<R, C>;

    fn get_col_vec(&self, r: usize) -> S::Vector<ROWS>;

    fn get_row_vec(&self, r: usize) -> S::Vector<ROWS>;
}

impl<const ROWS: usize, const COLS: usize> IsVectorLike for M<ROWS, COLS> {
    fn zero() -> Self {
        M::zeros()
    }
}

impl<const ROWS: usize, const COLS: usize> IsMatrix<f64, ROWS, COLS> for M<ROWS, COLS> {
    fn c(val: M<ROWS, COLS>) -> Self {
        val
    }

    fn from_array2(vals: [[f64; COLS]; ROWS]) -> M<ROWS, COLS> {
        let mut m = M::<ROWS, COLS>::zeros();

        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c];
            }
        }
        m
    }

    fn from_c_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        let mut m = M::<ROWS, COLS>::zeros();
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

    fn mat_mul<const C2: usize>(&self, other: M<COLS, C2>) -> M<ROWS, C2> {
        self * other
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: M<R0, COLS>,
        bot_row: M<R1, COLS>,
    ) -> Self {
        assert_eq!(ROWS, R0 + R1);
        let mut m = Self::zero();

        m.fixed_view_mut::<R0, COLS>(0, 0).copy_from(&top_row);
        m.fixed_view_mut::<R1, COLS>(R0, 0).copy_from(&bot_row);
        m
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (M<R0, C0>, M<R0, C1>),
        bot_row: (M<R1, C0>, M<R1, C1>),
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
        left_col: <f64 as IsScalar>::Matrix<ROWS, C0>,
        righ_col: <f64 as IsScalar>::Matrix<ROWS, C1>,
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
    ) -> M<R, C> {
        self.fixed_view::<R, C>(start_r, start_c).into()
    }

    fn get_col_vec(&self, c: usize) -> V<ROWS> {
        self.fixed_view::<ROWS, 1>(0, c).into()
    }

    fn get_row_vec(&self, r: usize) -> V<ROWS> {
        self.fixed_view::<1, ROWS>(0, r).transpose()
    }

    fn scaled(&self, v: f64) -> Self {
        self * v
    }
}
