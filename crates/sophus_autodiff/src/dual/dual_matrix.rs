use core::{
    borrow::Borrow,
    fmt::Debug,
    ops::{
        Add,
        Mul,
        Neg,
        Sub,
    },
};

use approx::{
    AbsDiffEq,
    RelativeEq,
};
use num_traits::Zero;

use super::matrix::MatrixValuedDerivative;
use crate::{
    dual::{
        DualScalar,
        DualVector,
    },
    linalg::{
        MatF64,
        SMat,
    },
    prelude::*,
};

/// DualScalarLike matrix
#[derive(Clone, Debug, Copy)]
pub struct DualMatrix<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize> {
    pub(crate) inner: SMat<DualScalar<DM, DN>, ROWS, COLS>,
}

impl<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize>
    IsSingleMatrix<DualScalar<DM, DN>, ROWS, COLS, DM, DN> for DualMatrix<ROWS, COLS, DM, DN>
{
    fn single_real_matrix(&self) -> MatF64<ROWS, COLS> {
        let mut m: SMat<f64, ROWS, COLS> = SMat::zeros();

        for i in 0..ROWS {
            for j in 0..COLS {
                m[(i, j)] = self.inner[(i, j)].real_part;
            }
        }
        m
    }
}

impl<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize>
    IsDualMatrix<DualScalar<DM, DN>, ROWS, COLS, 1, DM, DN> for DualMatrix<ROWS, COLS, DM, DN>
{
    /// Create a new dual number
    fn var(val: MatF64<ROWS, COLS>) -> Self {
        let mut dij_val: SMat<DualScalar<DM, DN>, ROWS, COLS> = SMat::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                dij_val[(i, j)].real_part = val[(i, j)];
                let mut v = SMat::<f64, DM, DN>::zeros();
                v[(i, j)] = 1.0;
                dij_val[(i, j)].infinitesimal_part = Some(v);
            }
        }

        Self { inner: dij_val }
    }

    fn derivative(self) -> MatrixValuedDerivative<f64, ROWS, COLS, 1, DM, DN> {
        let mut v = SMat::<SMat<f64, DM, DN>, ROWS, COLS>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                v[(i, j)] = self.inner[(i, j)].derivative();
            }
        }
        MatrixValuedDerivative { out_mat: v }
    }
}

impl<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize> PartialEq
    for DualMatrix<ROWS, COLS, DM, DN>
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize> AbsDiffEq
    for DualMatrix<ROWS, COLS, DM, DN>
{
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.inner.abs_diff_eq(&other.inner, epsilon)
    }
}

impl<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize> RelativeEq
    for DualMatrix<ROWS, COLS, DM, DN>
{
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.inner.relative_eq(&other.inner, epsilon, max_relative)
    }
}

impl<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize>
    IsMatrix<DualScalar<DM, DN>, ROWS, COLS, 1, DM, DN> for DualMatrix<ROWS, COLS, DM, DN>
{
    fn mat_mul<const C2: usize, M>(&self, other: M) -> DualMatrix<ROWS, C2, DM, DN>
    where
        M: Borrow<DualMatrix<COLS, C2, DM, DN>>,
    {
        let rhs = other.borrow();

        DualMatrix {
            inner: self.inner * rhs.inner,
        }
    }

    fn from_scalar<S>(val: S) -> Self
    where
        S: Borrow<DualScalar<DM, DN>>,
    {
        let mut out = Self {
            inner: SMat::zeros(),
        };
        let val = val.borrow();
        for i in 0..ROWS {
            for j in 0..COLS {
                out.inner[(i, j)] = *val;
            }
        }
        out
    }

    fn from_real_matrix<A>(val: A) -> Self
    where
        A: Borrow<MatF64<ROWS, COLS>>,
    {
        let vals = val.borrow();
        let mut m = SMat::<DualScalar<DM, DN>, ROWS, COLS>::zeros();

        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = DualScalar::from_real_scalar(vals[(r, c)]);
            }
        }

        Self { inner: m }
    }

    fn scaled<Q>(&self, s: Q) -> Self
    where
        Q: Borrow<DualScalar<DM, DN>>,
    {
        let s = s.borrow();
        Self {
            inner: self.inner * *s,
        }
    }

    fn identity() -> Self {
        DualMatrix::from_real_matrix(MatF64::<ROWS, COLS>::identity())
    }

    fn get_elem(&self, idx: [usize; 2]) -> DualScalar<DM, DN> {
        self.inner[(idx[0], idx[1])]
    }

    fn from_array2<A>(duals: A) -> Self
    where
        A: Borrow<[[DualScalar<DM, DN>; COLS]; ROWS]>,
    {
        let vals = duals.borrow();
        let mut m = SMat::<DualScalar<DM, DN>, ROWS, COLS>::zeros();

        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c]
            }
        }

        Self { inner: m }
    }

    fn real_matrix(&self) -> MatF64<ROWS, COLS> {
        let mut m = MatF64::<ROWS, COLS>::zeros();
        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = self.inner[(r, c)].real_part;
            }
        }
        m
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (DualMatrix<R0, C0, DM, DN>, DualMatrix<R0, C1, DM, DN>),
        bot_row: (DualMatrix<R1, C0, DM, DN>, DualMatrix<R1, C1, DM, DN>),
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
        assert_eq!(C0 + C1, COLS);

        Self::block_mat2x1(
            DualMatrix::<R0, COLS, DM, DN>::block_mat1x2(top_row.0, top_row.1),
            DualMatrix::<R1, COLS, DM, DN>::block_mat1x2(bot_row.0, bot_row.1),
        )
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: DualMatrix<R0, COLS, DM, DN>,
        bot_row: DualMatrix<R1, COLS, DM, DN>,
    ) -> Self {
        assert_eq!(ROWS, R0 + R1);
        let mut m = Self::zeros();

        m.inner
            .fixed_view_mut::<R0, COLS>(0, 0)
            .copy_from(&top_row.inner);
        m.inner
            .fixed_view_mut::<R1, COLS>(R0, 0)
            .copy_from(&bot_row.inner);
        m
    }

    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: DualMatrix<ROWS, C0, DM, DN>,
        righ_col: DualMatrix<ROWS, C1, DM, DN>,
    ) -> Self {
        assert_eq!(COLS, C0 + C1);
        let mut m = Self::zeros();

        m.inner
            .fixed_view_mut::<ROWS, C0>(0, 0)
            .copy_from(&left_col.inner);
        m.inner
            .fixed_view_mut::<ROWS, C1>(0, C0)
            .copy_from(&righ_col.inner);

        m
    }

    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> DualMatrix<R, C, DM, DN> {
        DualMatrix {
            inner: self.inner.fixed_view::<R, C>(start_r, start_c).into(),
        }
    }

    fn get_col_vec(&self, c: usize) -> DualVector<ROWS, DM, DN> {
        DualVector {
            inner: self.inner.fixed_view::<ROWS, 1>(0, c).into(),
        }
    }

    fn get_row_vec(&self, r: usize) -> DualVector<COLS, DM, DN> {
        DualVector {
            inner: self.inner.fixed_view::<1, COLS>(r, 0).transpose(),
        }
    }

    fn from_real_scalar_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[f64; COLS]; ROWS]>,
    {
        let vals = vals.borrow();

        let mut out = Self::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                out.inner[(i, j)].real_part = vals[i][j];
            }
        }
        out
    }

    fn from_f64_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[f64; COLS]; ROWS]>,
    {
        let vals = vals.borrow();

        let mut out = Self::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                out.inner[(i, j)].real_part = vals[i][j];
            }
        }
        out
    }

    fn from_f64(val: f64) -> Self {
        let mut out = Self {
            inner: SMat::zeros(),
        };
        for i in 0..ROWS {
            for j in 0..COLS {
                out.inner[(i, j)].real_part = val;
            }
        }
        out
    }

    fn set_col_vec(&mut self, c: usize, v: DualVector<ROWS, DM, DN>) {
        self.inner.fixed_columns_mut::<1>(c).copy_from(&v.inner);
    }

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <DualScalar<DM, DN> as IsScalar<1, DM, DN>>::DualMatrix<ROWS, COLS, M, N> {
        DualMatrix::from_real_matrix(self.real_matrix())
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

    fn set_elem(&mut self, idx: [usize; 2], val: DualScalar<DM, DN>) {
        self.inner[(idx[0], idx[1])] = val;
    }

    fn transposed(&self) -> <DualScalar<DM, DN> as IsScalar<1, DM, DN>>::Matrix<COLS, ROWS> {
        DualMatrix {
            inner: self.inner.transpose(),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize> Add
    for DualMatrix<ROWS, COLS, DM, DN>
{
    type Output = DualMatrix<ROWS, COLS, DM, DN>;

    fn add(self, rhs: Self) -> Self::Output {
        DualMatrix {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize> Sub
    for DualMatrix<ROWS, COLS, DM, DN>
{
    type Output = DualMatrix<ROWS, COLS, DM, DN>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualMatrix {
            inner: self.inner - rhs.inner,
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize> Neg
    for DualMatrix<ROWS, COLS, DM, DN>
{
    type Output = DualMatrix<ROWS, COLS, DM, DN>;

    fn neg(self) -> Self::Output {
        DualMatrix { inner: -self.inner }
    }
}

impl<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize> Zero
    for DualMatrix<ROWS, COLS, DM, DN>
{
    fn zero() -> Self {
        Self::from_real_matrix(MatF64::zeros())
    }

    fn is_zero(&self) -> bool {
        self.real_matrix().is_zero()
    }
}

impl<const ROWS: usize, const COLS: usize, const DM: usize, const DN: usize>
    Mul<DualVector<COLS, DM, DN>> for DualMatrix<ROWS, COLS, DM, DN>
{
    type Output = DualVector<ROWS, DM, DN>;

    fn mul(self, rhs: DualVector<COLS, DM, DN>) -> Self::Output {
        Self::Output {
            inner: self.inner * rhs.inner,
        }
    }
}
