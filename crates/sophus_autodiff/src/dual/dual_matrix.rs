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
use nalgebra::SMatrix;
use num_traits::Zero;

use super::matrix::MatrixValuedDerivative;
use crate::{
    dual::{
        DualScalar,
        DualVector,
        matrix::IsDualMatrixFromCurve,
    },
    linalg::SMat,
    prelude::*,
};

/// A dual matrix, storing a set of dual scalars (with partial derivatives) for each element.
///
/// Conceptually, this is the forward-mode AD version of a ROWSxCOLS matrix, where each element is
/// a [`DualScalar<R, DM, DN>`], i.e., each element carries its own infinitesimal part.
///
/// # Private fields
/// - `inner`: A `ROWS x COLS` [`SMat`] of [`DualScalar<R, DM, DN>`]s.
///
/// # Generic Parameters
/// - `R`: The underlying real scalar type (e.g. `f64` or `f32`).
/// - `ROWS`, `COLS`: Matrix dimensions.
/// - `DM`, `DN`: Dimensions for each component's derivative (infinitesimal) matrix. For example,
///   `DM=3, DN=1` might store partial derivatives w.r.t. a 3D input for each element.
///
/// See [crate::dual::IsDualMatrix] for more details.
#[derive(Clone, Debug, Copy)]
pub struct DualMatrix<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> {
    pub(crate) inner: SMat<DualScalar<R, DM, DN>, ROWS, COLS>,
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> IsSingleMatrix<DualScalar<R, DM, DN>, ROWS, COLS, DM, DN> for DualMatrix<R, ROWS, COLS, DM, DN>
{
    fn single_real_matrix(&self) -> R::RealMatrix<ROWS, COLS> {
        let mut m = R::RealMatrix::<ROWS, COLS>::zeros();

        for i in 0..ROWS {
            for j in 0..COLS {
                m[(i, j)] = self.inner[(i, j)].real_part;
            }
        }
        m
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> IsDualMatrix<DualScalar<R, DM, DN>, ROWS, COLS, 1, DM, DN> for DualMatrix<R, ROWS, COLS, DM, DN>
{
    /// Create a new dual number
    fn var(val: R::RealMatrix<ROWS, COLS>) -> Self {
        let mut dij_val: SMat<DualScalar<R, DM, DN>, ROWS, COLS> = SMat::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                dij_val[(i, j)].real_part = val[(i, j)];
                let mut v = SMatrix::<R, DM, DN>::zeros();
                v[(i, j)] = <R as IsScalar<1, 0, 0>>::from_f64(1.0);
                dij_val[(i, j)].infinitesimal_part = Some(v);
            }
        }

        Self { inner: dij_val }
    }

    fn derivative(self) -> MatrixValuedDerivative<R, ROWS, COLS, 1, DM, DN> {
        let mut v = SMat::<SMat<R, DM, DN>, ROWS, COLS>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                v[(i, j)] = self.inner[(i, j)].derivative();
            }
        }
        MatrixValuedDerivative { out_mat: v }
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
> IsDualMatrixFromCurve<DualScalar<R, 1, 1>, ROWS, COLS, 1> for DualMatrix<R, ROWS, COLS, 1, 1>
{
    fn curve_derivative(&self) -> nalgebra::SMatrix<R, ROWS, COLS> {
        let mut out = SMat::<R, ROWS, COLS>::zeros();

        for i in 0..ROWS {
            for j in 0..COLS {
                out[(i, j)] = self.inner[(i, j)].derivative()[(0, 0)];
            }
        }
        out
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> PartialEq for DualMatrix<R, ROWS, COLS, DM, DN>
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> AbsDiffEq for DualMatrix<R, ROWS, COLS, DM, DN>
{
    type Epsilon = R;

    fn default_epsilon() -> Self::Epsilon {
        R::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.inner.abs_diff_eq(&other.inner, epsilon)
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> RelativeEq for DualMatrix<R, ROWS, COLS, DM, DN>
{
    fn default_max_relative() -> Self::Epsilon {
        R::default_max_relative()
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

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> IsMatrix<DualScalar<R, DM, DN>, ROWS, COLS, 1, DM, DN> for DualMatrix<R, ROWS, COLS, DM, DN>
{
    fn mat_mul<const C2: usize, M>(&self, other: M) -> DualMatrix<R, ROWS, C2, DM, DN>
    where
        M: Borrow<DualMatrix<R, COLS, C2, DM, DN>>,
    {
        let rhs = other.borrow();

        DualMatrix {
            inner: self.inner * rhs.inner,
        }
    }

    fn from_scalar<S>(val: S) -> Self
    where
        S: Borrow<DualScalar<R, DM, DN>>,
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
        A: Borrow<R::RealMatrix<ROWS, COLS>>,
    {
        let vals = val.borrow();
        let mut m = SMat::<DualScalar<R, DM, DN>, ROWS, COLS>::zeros();

        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = DualScalar::from_real_scalar(vals[(r, c)]);
            }
        }

        Self { inner: m }
    }

    fn scaled<Q>(&self, s: Q) -> Self
    where
        Q: Borrow<DualScalar<R, DM, DN>>,
    {
        let s = s.borrow();
        Self {
            inner: self.inner * *s,
        }
    }

    fn identity() -> Self {
        DualMatrix::from_real_matrix(R::RealMatrix::<ROWS, COLS>::identity())
    }

    fn elem(&self, idx: [usize; 2]) -> DualScalar<R, DM, DN> {
        self.inner[(idx[0], idx[1])]
    }

    fn elem_mut(&mut self, idx: [usize; 2]) -> &mut DualScalar<R, DM, DN> {
        &mut self.inner[(idx[0], idx[1])]
    }

    fn from_array2<A>(duals: A) -> Self
    where
        A: Borrow<[[DualScalar<R, DM, DN>; COLS]; ROWS]>,
    {
        let vals = duals.borrow();
        let mut m = SMat::<DualScalar<R, DM, DN>, ROWS, COLS>::zeros();

        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c]
            }
        }

        Self { inner: m }
    }

    fn real_matrix(&self) -> R::RealMatrix<ROWS, COLS> {
        let mut m = R::RealMatrix::<ROWS, COLS>::zeros();
        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = self.inner[(r, c)].real_part;
            }
        }
        m
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (DualMatrix<R, R0, C0, DM, DN>, DualMatrix<R, R0, C1, DM, DN>),
        bot_row: (DualMatrix<R, R1, C0, DM, DN>, DualMatrix<R, R1, C1, DM, DN>),
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
        assert_eq!(C0 + C1, COLS);

        Self::block_mat2x1(
            DualMatrix::<R, R0, COLS, DM, DN>::block_mat1x2(top_row.0, top_row.1),
            DualMatrix::<R, R1, COLS, DM, DN>::block_mat1x2(bot_row.0, bot_row.1),
        )
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: DualMatrix<R, R0, COLS, DM, DN>,
        bot_row: DualMatrix<R, R1, COLS, DM, DN>,
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
        left_col: DualMatrix<R, ROWS, C0, DM, DN>,
        righ_col: DualMatrix<R, ROWS, C1, DM, DN>,
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

    fn get_fixed_submat<const RR: usize, const CC: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> DualMatrix<R, RR, CC, DM, DN> {
        DualMatrix {
            inner: self.inner.fixed_view::<RR, CC>(start_r, start_c).into(),
        }
    }

    fn get_col_vec(&self, c: usize) -> DualVector<R, ROWS, DM, DN> {
        DualVector {
            inner: self.inner.fixed_view::<ROWS, 1>(0, c).into(),
        }
    }

    fn get_row_vec(&self, r: usize) -> DualVector<R, COLS, DM, DN> {
        DualVector {
            inner: self.inner.fixed_view::<1, COLS>(r, 0).transpose(),
        }
    }

    fn from_real_scalar_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[R; COLS]; ROWS]>,
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
                out.inner[(i, j)].real_part = <R as IsScalar<1, 0, 0>>::from_f64(vals[i][j]);
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
                out.inner[(i, j)].real_part = <R as IsScalar<1, 0, 0>>::from_f64(val);
            }
        }
        out
    }

    fn set_col_vec(&mut self, c: usize, v: DualVector<R, ROWS, DM, DN>) {
        self.inner.fixed_columns_mut::<1>(c).copy_from(&v.inner);
    }

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <DualScalar<R, DM, DN> as IsScalar<1, DM, DN>>::DualMatrix<ROWS, COLS, M, N> {
        DualMatrix::from_real_matrix(self.real_matrix())
    }

    fn select<Q>(&self, mask: &bool, other: Q) -> Self
    where
        Q: Borrow<Self>,
    {
        if *mask { *self } else { *other.borrow() }
    }

    fn transposed(&self) -> <DualScalar<R, DM, DN> as IsScalar<1, DM, DN>>::Matrix<COLS, ROWS> {
        DualMatrix {
            inner: self.inner.transpose(),
        }
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> Add for DualMatrix<R, ROWS, COLS, DM, DN>
{
    type Output = DualMatrix<R, ROWS, COLS, DM, DN>;

    fn add(self, rhs: Self) -> Self::Output {
        DualMatrix {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> Sub for DualMatrix<R, ROWS, COLS, DM, DN>
{
    type Output = DualMatrix<R, ROWS, COLS, DM, DN>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualMatrix {
            inner: self.inner - rhs.inner,
        }
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> Neg for DualMatrix<R, ROWS, COLS, DM, DN>
{
    type Output = DualMatrix<R, ROWS, COLS, DM, DN>;

    fn neg(self) -> Self::Output {
        DualMatrix { inner: -self.inner }
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> Zero for DualMatrix<R, ROWS, COLS, DM, DN>
{
    fn zero() -> Self {
        Self::from_real_matrix(R::RealMatrix::<ROWS, COLS>::zeros())
    }

    fn is_zero(&self) -> bool {
        let m = self.real_matrix();
        let zero = <R as IsScalar<1, 0, 0>>::from_f64(0.0);
        for i in 0..ROWS {
            for j in 0..COLS {
                if m[(i, j)] != zero {
                    return false;
                }
            }
        }
        true
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
> Mul<DualVector<R, COLS, DM, DN>> for DualMatrix<R, ROWS, COLS, DM, DN>
{
    type Output = DualVector<R, ROWS, DM, DN>;

    fn mul(self, rhs: DualVector<R, COLS, DM, DN>) -> Self::Output {
        Self::Output {
            inner: self.inner * rhs.inner,
        }
    }
}
