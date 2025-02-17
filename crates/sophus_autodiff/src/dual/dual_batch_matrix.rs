use core::{
    borrow::Borrow,
    fmt::Debug,
    ops::{
        Add,
        Mul,
        Neg,
        Sub,
    },
    simd::{
        LaneCount,
        SupportedLaneCount,
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
        DualBatchScalar,
        DualBatchVector,
    },
    linalg::{
        BatchMask,
        BatchMatF64,
        BatchScalarF64,
        SMat,
    },
    prelude::*,
};

/// A batch dual matrix, whose elements are [DualBatchScalar] (forward-mode AD) across multiple
/// lanes.
///
/// This implements vector functionality for `ℝʳ` *with* partial derivatives,
/// in parallel lanes (SIMD). Each element is a `DualBatchScalar<BATCH, DM, DN>` storing:
///
/// - `BATCH`: The number of SIMD lanes.
/// - `DM`, `DN`: The shape of each element’s derivative.
///
/// # Private fields
/// - `inner`: A `ROWS x COLS` [SMat], each item a [`DualBatchScalar<DM, DN>`].
///
/// # Example
/// For `ROWS=3, COLS=2, BATCH=4, DM=3, DN=1`, you have 3x2 elements, each storing 4-lane real parts
/// plus a 3×1 derivative for each lane, i.e. 4-lane forward-mode AD on a 3x2 matrix input.
///
/// See [crate::dual::IsDualMatrix] for more details.
#[cfg(feature = "simd")]
#[derive(Clone, Debug, Copy)]
pub struct DualBatchMatrix<
    const ROWS: usize,
    const COLS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    pub(crate) inner: SMat<DualBatchScalar<BATCH, DM, DN>, ROWS, COLS>,
}

impl<
        const ROWS: usize,
        const COLS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > IsDualMatrix<DualBatchScalar<BATCH, DM, DN>, ROWS, COLS, BATCH, DM, DN>
    for DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    /// Create a new dual number
    fn var(val: BatchMatF64<ROWS, COLS, BATCH>) -> Self {
        let mut dij_val: SMat<DualBatchScalar<BATCH, DM, DN>, ROWS, COLS> = SMat::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                dij_val[(i, j)].real_part = val[(i, j)];
                let mut v = SMat::<BatchScalarF64<BATCH>, DM, DN>::zeros();
                v[(i, j)] = BatchScalarF64::<BATCH>::from_f64(1.0);
                dij_val[(i, j)].infinitesimal_part = Some(v);
            }
        }

        Self { inner: dij_val }
    }

    fn derivative(
        self,
    ) -> MatrixValuedDerivative<BatchScalarF64<BATCH>, ROWS, COLS, BATCH, DM, DN> {
        let mut v = SMat::<SMat<BatchScalarF64<BATCH>, DM, DN>, ROWS, COLS>::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                v[(i, j)] = self.inner[(i, j)].derivative();
            }
        }
        MatrixValuedDerivative { out_mat: v }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize>
    IsDualMatrixFromCurve<DualBatchScalar<BATCH, 1, 1>, ROWS, COLS, BATCH>
    for DualBatchMatrix<ROWS, COLS, BATCH, 1, 1>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn curve_derivative(&self) -> BatchMatF64<ROWS, COLS, BATCH> {
        let mut out = BatchMatF64::<ROWS, COLS, BATCH>::zeros();

        for i in 0..ROWS {
            for j in 0..COLS {
                *out.elem_mut([i, j]) = self.inner[(i, j)].derivative()[(0, 0)];
            }
        }
        out
    }
}

impl<
        const ROWS: usize,
        const COLS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > PartialEq for DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<
        const ROWS: usize,
        const COLS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > AbsDiffEq for DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.inner.abs_diff_eq(&other.inner, epsilon)
    }
}

impl<
        const ROWS: usize,
        const COLS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > RelativeEq for DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
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

impl<
        const ROWS: usize,
        const COLS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > IsMatrix<DualBatchScalar<BATCH, DM, DN>, ROWS, COLS, BATCH, DM, DN>
    for DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn mat_mul<const C2: usize, M>(&self, other: M) -> DualBatchMatrix<ROWS, C2, BATCH, DM, DN>
    where
        M: Borrow<DualBatchMatrix<COLS, C2, BATCH, DM, DN>>,
    {
        let rhs = other.borrow();

        DualBatchMatrix {
            inner: self.inner * rhs.inner,
        }
    }

    fn from_scalar<S>(val: S) -> Self
    where
        S: Borrow<DualBatchScalar<BATCH, DM, DN>>,
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
        A: Borrow<BatchMatF64<ROWS, COLS, BATCH>>,
    {
        let vals = val.borrow();
        let mut m = SMat::<DualBatchScalar<BATCH, DM, DN>, ROWS, COLS>::zeros();

        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = DualBatchScalar::from_real_scalar(vals[(r, c)]);
            }
        }

        Self { inner: m }
    }

    fn scaled<Q>(&self, s: Q) -> Self
    where
        Q: Borrow<DualBatchScalar<BATCH, DM, DN>>,
    {
        let s = s.borrow();
        Self {
            inner: self.inner * *s,
        }
    }

    fn identity() -> Self {
        DualBatchMatrix::from_real_matrix(BatchMatF64::<ROWS, COLS, BATCH>::identity())
    }

    fn elem(&self, idx: [usize; 2]) -> DualBatchScalar<BATCH, DM, DN> {
        self.inner[(idx[0], idx[1])]
    }

    fn elem_mut(&mut self, idx: [usize; 2]) -> &mut DualBatchScalar<BATCH, DM, DN> {
        &mut self.inner[(idx[0], idx[1])]
    }

    fn from_array2<A>(duals: A) -> Self
    where
        A: Borrow<[[DualBatchScalar<BATCH, DM, DN>; COLS]; ROWS]>,
    {
        let vals = duals.borrow();
        let mut m = SMat::<DualBatchScalar<BATCH, DM, DN>, ROWS, COLS>::zeros();

        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c]
            }
        }

        Self { inner: m }
    }

    fn real_matrix(&self) -> BatchMatF64<ROWS, COLS, BATCH> {
        let mut m = BatchMatF64::<ROWS, COLS, BATCH>::zeros();
        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = self.inner[(r, c)].real_part;
            }
        }
        m
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (
            DualBatchMatrix<R0, C0, BATCH, DM, DN>,
            DualBatchMatrix<R0, C1, BATCH, DM, DN>,
        ),
        bot_row: (
            DualBatchMatrix<R1, C0, BATCH, DM, DN>,
            DualBatchMatrix<R1, C1, BATCH, DM, DN>,
        ),
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
        assert_eq!(C0 + C1, COLS);

        Self::block_mat2x1(
            DualBatchMatrix::<R0, COLS, BATCH, DM, DN>::block_mat1x2(top_row.0, top_row.1),
            DualBatchMatrix::<R1, COLS, BATCH, DM, DN>::block_mat1x2(bot_row.0, bot_row.1),
        )
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: DualBatchMatrix<R0, COLS, BATCH, DM, DN>,
        bot_row: DualBatchMatrix<R1, COLS, BATCH, DM, DN>,
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
        left_col: DualBatchMatrix<ROWS, C0, BATCH, DM, DN>,
        righ_col: DualBatchMatrix<ROWS, C1, BATCH, DM, DN>,
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
    ) -> DualBatchMatrix<R, C, BATCH, DM, DN> {
        DualBatchMatrix {
            inner: self.inner.fixed_view::<R, C>(start_r, start_c).into(),
        }
    }

    fn get_col_vec(&self, c: usize) -> DualBatchVector<ROWS, BATCH, DM, DN> {
        DualBatchVector {
            inner: self.inner.fixed_view::<ROWS, 1>(0, c).into(),
        }
    }

    fn get_row_vec(&self, r: usize) -> DualBatchVector<COLS, BATCH, DM, DN> {
        DualBatchVector {
            inner: self.inner.fixed_view::<1, COLS>(r, 0).transpose(),
        }
    }

    fn from_real_scalar_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[BatchScalarF64<BATCH>; COLS]; ROWS]>,
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
                out.inner[(i, j)].real_part = BatchScalarF64::from_f64(vals[i][j]);
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
                out.inner[(i, j)].real_part = BatchScalarF64::from_f64(val);
            }
        }
        out
    }

    fn set_col_vec(&mut self, c: usize, v: DualBatchVector<ROWS, BATCH, DM, DN>) {
        self.inner.fixed_columns_mut::<1>(c).copy_from(&v.inner);
    }

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <DualBatchScalar<BATCH, DM, DN> as IsScalar<BATCH, DM, DN>>::DualMatrix<ROWS, COLS, M, N>
    {
        DualBatchMatrix::from_real_matrix(self.real_matrix())
    }

    fn select<Q>(&self, mask: &BatchMask<BATCH>, other: Q) -> Self
    where
        Q: Borrow<Self>,
    {
        let mut v = SMat::<DualBatchScalar<BATCH, DM, DN>, ROWS, COLS>::zeros();
        let other = other.borrow();
        for i in 0..ROWS {
            for j in 0..COLS {
                v[(i, j)] = self.elem([i, j]).select(mask, other.elem([i, j]));
            }
        }

        Self { inner: v }
    }

    fn transposed(
        &self,
    ) -> <DualBatchScalar<BATCH, DM, DN> as IsScalar<BATCH, DM, DN>>::Matrix<COLS, ROWS> {
        DualBatchMatrix {
            inner: self.inner.transpose(),
        }
    }
}

impl<
        const ROWS: usize,
        const COLS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > Add for DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>;

    fn add(self, rhs: Self) -> Self::Output {
        DualBatchMatrix {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<
        const ROWS: usize,
        const COLS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > Sub for DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualBatchMatrix {
            inner: self.inner - rhs.inner,
        }
    }
}

impl<
        const ROWS: usize,
        const COLS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > Neg for DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>;

    fn neg(self) -> Self::Output {
        DualBatchMatrix { inner: -self.inner }
    }
}

impl<
        const ROWS: usize,
        const COLS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > Zero for DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn zero() -> Self {
        Self::from_real_matrix(BatchMatF64::zeros())
    }

    fn is_zero(&self) -> bool {
        self.real_matrix().is_zero()
    }
}

impl<
        const ROWS: usize,
        const COLS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > Mul<DualBatchVector<COLS, BATCH, DM, DN>> for DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchVector<ROWS, BATCH, DM, DN>;

    fn mul(self, rhs: DualBatchVector<COLS, BATCH, DM, DN>) -> Self::Output {
        Self::Output {
            inner: self.inner * rhs.inner,
        }
    }
}
