use crate::dual::dual_matrix::DualMatrix;
use crate::linalg::MatF64;
use crate::linalg::VecF64;
use crate::prelude::*;
use approx::AbsDiffEq;
use approx::RelativeEq;
use core::borrow::Borrow;
use core::fmt::Debug;
use core::ops::Add;
use core::ops::Index;
use core::ops::IndexMut;
use core::ops::Mul;
use core::ops::Neg;
use core::ops::Sub;

/// Matrix trait
///  - either a real (f64) or a dual number matrix
///  - either a single matrix or a batch matrix
pub trait IsMatrix<
    S: IsScalar<BATCH, DM, DN>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>:
    Debug
    + Clone
    + Copy
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
    fn from_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[S; COLS]; ROWS]>;

    /// creates matrix with all real elements (and lanes) set to the given value
    ///
    /// (for dual numbers, the infinitesimal part is set to zero)
    fn from_f64(val: f64) -> Self;

    /// creates matrix from a 2d array of real values
    ///
    ///  - all lanes are set to the same value
    ///  - for dual numbers, the infinitesimal part is set to zero
    fn from_f64_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[f64; COLS]; ROWS]>;

    /// creates matrix from a 2d array of real scalars
    ///
    /// (for dual numbers, the infinitesimal part is set to zero)
    fn from_real_scalar_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[S::RealScalar; COLS]; ROWS]>;

    /// create a constant matrix
    fn from_real_matrix<A>(val: A) -> Self
    where
        A: Borrow<S::RealMatrix<ROWS, COLS>>;

    /// create a constant scalar
    fn from_scalar<Q>(val: Q) -> Self
    where
        Q: Borrow<S>;

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
    fn mat_mul<const C2: usize, M>(&self, other: M) -> S::Matrix<ROWS, C2>
    where
        M: Borrow<S::Matrix<COLS, C2>>;

    /// ones
    fn ones() -> Self {
        Self::from_f64(1.0)
    }

    /// Return a real matrix
    fn real_matrix(&self) -> S::RealMatrix<ROWS, COLS>;

    /// return scaled matrix
    fn scaled<Q>(&self, v: Q) -> Self
    where
        Q: Borrow<S>;

    /// Returns self if mask is true, otherwise returns other
    ///
    /// For batch matrices, this is a lane-wise operation
    fn select<Q>(&self, mask: &S::Mask, other: Q) -> Self
    where
        Q: Borrow<Self>;

    /// set i-th element
    fn set_elem(&mut self, idx: [usize; 2], val: S);

    /// set column vectors
    fn set_col_vec(&mut self, c: usize, v: S::Vector<ROWS>);

    /// transpose
    fn transposed(&self) -> S::Matrix<COLS, ROWS>;

    /// Return dual matrix
    ///
    /// If self is a real matrix, this will return a dual matrix with the infinitesimal part set to
    /// zero: (self, 0ϵ)
    fn to_dual_const<const M: usize, const N: usize>(&self) -> S::DualMatrix<ROWS, COLS, M, N>;

    /// zeros
    fn zeros() -> Self {
        Self::from_f64(0.0)
    }
}

/// Is real matrix?
pub trait IsRealMatrix<
    S: IsRealScalar<BATCH> + IsScalar<BATCH, 0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH: usize,
>:
    IsMatrix<S, ROWS, COLS, BATCH, 0, 0>
    + Index<(usize, usize), Output = S>
    + IndexMut<(usize, usize), Output = S>
    + Copy
{
}

/// Is single matrix? (not batch)
pub trait IsSingleMatrix<
    S: IsSingleScalar<DM, DN>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
>:
    IsMatrix<S, ROWS, COLS, 1, DM, DN> + Mul<S::SingleVector<COLS>, Output = S::SingleVector<ROWS>>
{
    /// returns single real matrix
    fn single_real_matrix(&self) -> MatF64<ROWS, COLS>;
}

impl<const ROWS: usize, const COLS: usize> IsRealMatrix<f64, ROWS, COLS, 1> for MatF64<ROWS, COLS> {}

impl<const ROWS: usize, const COLS: usize> IsSingleMatrix<f64, ROWS, COLS, 0, 0>
    for MatF64<ROWS, COLS>
{
    fn single_real_matrix(&self) -> MatF64<ROWS, COLS> {
        *self
    }
}

impl<const ROWS: usize, const COLS: usize> IsMatrix<f64, ROWS, COLS, 1, 0, 0>
    for MatF64<ROWS, COLS>
{
    fn from_real_matrix<A>(val: A) -> Self
    where
        A: Borrow<MatF64<ROWS, COLS>>,
    {
        *val.borrow()
    }

    fn from_scalar<S>(val: S) -> Self
    where
        S: Borrow<f64>,
    {
        Self::from_f64(*val.borrow())
    }

    fn from_array2<A>(vals: A) -> MatF64<ROWS, COLS>
    where
        A: Borrow<[[f64; COLS]; ROWS]>,
    {
        let vals = vals.borrow();
        let mut m = MatF64::<ROWS, COLS>::zeros();

        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c];
            }
        }
        m
    }

    fn from_real_scalar_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[f64; COLS]; ROWS]>,
    {
        let vals = vals.borrow();
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

    fn real_matrix(&self) -> Self {
        *self
    }

    fn mat_mul<const C2: usize, M>(&self, other: M) -> MatF64<ROWS, C2>
    where
        M: Borrow<MatF64<COLS, C2>>,
    {
        self * other.borrow()
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

    fn scaled<Q>(&self, v: Q) -> Self
    where
        Q: Borrow<f64>,
    {
        *self * *v.borrow()
    }

    fn from_f64_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[f64; COLS]; ROWS]>,
    {
        let vals = vals.borrow();
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

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <f64 as IsScalar<1, 0, 0>>::DualMatrix<ROWS, COLS, M, N> {
        DualMatrix::from_real_matrix(self)
    }

    fn select<Q>(&self, mask: &bool, other: Q) -> Self
    where
        Q: Borrow<Self>,
    {
        if *mask {
            *self
        } else {
            *other.borrow()
        }
    }

    fn set_elem(&mut self, idx: [usize; 2], val: f64) {
        self[(idx[0], idx[1])] = val;
    }

    fn transposed(&self) -> MatF64<COLS, ROWS> {
        self.transpose()
    }
}
