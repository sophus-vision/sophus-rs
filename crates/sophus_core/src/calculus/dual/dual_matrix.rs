use super::dual_scalar::IsDual;
use super::dual_scalar::IsDualScalar;
use super::dual_vector::DualVector;
use crate::calculus::dual::dual_scalar::DualBatchScalar;
use crate::calculus::dual::dual_scalar::DualScalar;
use crate::calculus::dual::dual_vector::DualBatchVector;
use crate::linalg::matrix::IsMatrix;
use crate::linalg::matrix::IsSingleMatrix;
use crate::linalg::scalar::IsCoreScalar;
use crate::linalg::scalar::IsScalar;
use crate::linalg::BatchMatF64;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::linalg::MatF64;
use crate::linalg::VecF64;
use crate::tensor::mut_tensor::MutTensorDD;
use crate::tensor::mut_tensor::MutTensorDDR;
use crate::tensor::mut_tensor::MutTensorDDRC;
use crate::tensor::mut_tensor_view::IsMutTensorLike;
use crate::tensor::tensor_view::IsTensorLike;
use approx::AbsDiffEq;
use approx::RelativeEq;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;
use std::simd::LaneCount;
use std::simd::Mask;
use std::simd::SupportedLaneCount;

/// DualScalarLike matrix
#[derive(Clone)]
pub struct DualMatrix<const ROWS: usize, const COLS: usize> {
    /// value - real matrix
    pub val: MatF64<ROWS, COLS>,
    /// derivative - infinitesimal matrix
    pub dij_val: Option<MutTensorDDRC<f64, ROWS, COLS>>,
}

/// DualScalarLike matrix
#[derive(Clone)]
pub struct DualBatchMatrix<const ROWS: usize, const COLS: usize, const BATCH: usize>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    /// value - real matrix
    pub val: BatchMatF64<ROWS, COLS, BATCH>,
    /// derivative - infinitesimal matrix
    pub dij_val: Option<MutTensorDDRC<BatchScalarF64<BATCH>, ROWS, COLS>>,
}

impl<const ROWS: usize, const COLS: usize> IsSingleMatrix<DualScalar, ROWS, COLS>
    for DualMatrix<ROWS, COLS>
{
}

/// Trait for scalar dual numbers
pub trait IsDualMatrix<
    S: IsDualScalar<BATCH>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH: usize,
>: IsMatrix<S, ROWS, COLS, BATCH> + IsDual
{
    /// Create a new dual number
    fn new(val: S::RealMatrix<ROWS, COLS>) -> Self;

    /// Get the derivative
    fn dij_val(self) -> Option<MutTensorDDRC<S::RealScalar, ROWS, COLS>>;
}

impl<const ROWS: usize, const COLS: usize> IsDual for DualMatrix<ROWS, COLS> {}

impl<const ROWS: usize, const COLS: usize> IsDualMatrix<DualScalar, ROWS, COLS, 1>
    for DualMatrix<ROWS, COLS>
{
    /// Create a new dual number
    fn new(val: MatF64<ROWS, COLS>) -> Self {
        DualMatrix { val, dij_val: None }
    }

    /// Get the derivative
    fn dij_val(self) -> Option<MutTensorDDRC<f64, ROWS, COLS>> {
        self.dij_val
    }
}

/// Pair of dual matrices
pub struct DijPairM<
    S: IsCoreScalar,
    const ROWS: usize,
    const COLS: usize,
    const ROWS2: usize,
    const COLS2: usize,
> {
    lhs: MutTensorDDRC<S, ROWS, COLS>,
    rhs: MutTensorDDRC<S, ROWS2, COLS2>,
}

/// Pair of dual matrices
pub struct DijPairMV<S: IsCoreScalar, const ROWS: usize, const COLS: usize> {
    /// left hand side
    pub lhs: MutTensorDDRC<S, ROWS, COLS>,
    /// right hand side
    pub rhs: MutTensorDDR<S, COLS>,
}

impl<
        S: IsCoreScalar,
        const ROWS: usize,
        const COLS: usize,
        const ROWS2: usize,
        const COLS2: usize,
    > DijPairM<S, ROWS, COLS, ROWS2, COLS2>
{
    fn shape(&self) -> [usize; 2] {
        self.lhs.dims()
    }
}

impl<const ROWS: usize, const COLS: usize> DualMatrix<ROWS, COLS> {
    pub(crate) fn binary_mm_dij<
        const R0: usize,
        const R1: usize,
        const C0: usize,
        const C1: usize,
        F: FnMut(&MatF64<R0, C0>) -> MatF64<ROWS, COLS>,
        G: FnMut(&MatF64<R1, C1>) -> MatF64<ROWS, COLS>,
    >(
        lhs_dx: &Option<MutTensorDDRC<f64, R0, C0>>,
        rhs_dx: &Option<MutTensorDDRC<f64, R1, C1>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDRC<f64, ROWS, COLS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDRC::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDRC::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDRC::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    fn binary_mv_dij<
        const R0: usize,
        const R1: usize,
        const C0: usize,
        F: FnMut(&MatF64<R0, C0>) -> VecF64<ROWS>,
        G: FnMut(&VecF64<R1>) -> VecF64<ROWS>,
    >(
        lhs_dx: &Option<MutTensorDDRC<f64, R0, C0>>,
        rhs_dx: &Option<MutTensorDDR<f64, R1>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDR<f64, ROWS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDR::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDR::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDR::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    fn binary_ms_dij<
        const R0: usize,
        const C0: usize,
        F: FnMut(&MatF64<R0, C0>) -> MatF64<ROWS, COLS>,
        G: FnMut(&f64) -> MatF64<ROWS, COLS>,
    >(
        lhs_dx: &Option<MutTensorDDRC<f64, R0, C0>>,
        rhs_dx: &Option<MutTensorDD<f64>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDRC<f64, ROWS, COLS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDRC::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDRC::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDRC::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    /// derivatives
    pub fn two_dx<const R1: usize, const C1: usize, const R2: usize, const C2: usize>(
        mut lhs_dx: Option<MutTensorDDRC<f64, R1, C1>>,
        mut rhs_dx: Option<MutTensorDDRC<f64, R2, C2>>,
    ) -> Option<DijPairM<f64, R1, C1, R2, C2>> {
        if lhs_dx.is_none() && rhs_dx.is_none() {
            return None;
        }

        if lhs_dx.is_some() && rhs_dx.is_some() {
            assert_eq!(
                lhs_dx.clone().unwrap().dims(),
                rhs_dx.clone().unwrap().dims()
            );
        }

        if lhs_dx.is_none() {
            lhs_dx = Some(MutTensorDDRC::<f64, R1, C1>::from_shape(
                rhs_dx.clone().unwrap().dims(),
            ))
        } else if rhs_dx.is_none() {
            rhs_dx = Some(MutTensorDDRC::<f64, R2, C2>::from_shape(
                lhs_dx.clone().unwrap().dims(),
            ))
        }

        Some(DijPairM {
            lhs: lhs_dx.unwrap(),
            rhs: rhs_dx.unwrap(),
        })
    }

    /// derivatives
    pub fn two_dx_from_vec(
        mut lhs_dx: Option<MutTensorDDRC<f64, ROWS, COLS>>,
        mut rhs_dx: Option<MutTensorDDR<f64, COLS>>,
    ) -> Option<DijPairMV<f64, ROWS, COLS>> {
        if lhs_dx.is_none() && rhs_dx.is_none() {
            return None;
        }

        if lhs_dx.is_some() && rhs_dx.is_some() {
            assert_eq!(
                lhs_dx.clone().unwrap().dims(),
                rhs_dx.clone().unwrap().dims()
            );
        }

        if lhs_dx.is_none() {
            lhs_dx = Some(MutTensorDDRC::<f64, ROWS, COLS>::from_shape(
                rhs_dx.clone().unwrap().dims(),
            ))
        } else if rhs_dx.is_none() {
            rhs_dx = Some(MutTensorDDR::<f64, COLS>::from_shape(
                lhs_dx.clone().unwrap().dims(),
            ))
        }

        Some(DijPairMV::<f64, ROWS, COLS> {
            lhs: lhs_dx.unwrap(),
            rhs: rhs_dx.unwrap(),
        })
    }

    /// Create a dual matrix
    pub fn v(val: MatF64<ROWS, COLS>) -> Self {
        let mut dij_val = MutTensorDDRC::<f64, ROWS, COLS>::from_shape([ROWS, COLS]);
        for i in 0..ROWS {
            for j in 0..COLS {
                dij_val.mut_view().get_mut([i, j])[(i, j)] = 1.0;
            }
        }

        Self {
            val,
            dij_val: Some(dij_val),
        }
    }
}

impl<const ROWS: usize, const COLS: usize> PartialEq for DualMatrix<ROWS, COLS> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val && self.dij_val == other.dij_val
    }
}

impl<const ROWS: usize, const COLS: usize> AbsDiffEq for DualMatrix<ROWS, COLS> {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.val.abs_diff_eq(&other.val, epsilon)
    }
}

impl<const ROWS: usize, const COLS: usize> RelativeEq for DualMatrix<ROWS, COLS> {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.val.relative_eq(&other.val, epsilon, max_relative)
    }
}

impl<const ROWS: usize, const COLS: usize> IsMatrix<DualScalar, ROWS, COLS, 1>
    for DualMatrix<ROWS, COLS>
{
    fn mat_mul<const COLS2: usize>(&self, rhs: DualMatrix<COLS, COLS2>) -> DualMatrix<ROWS, COLS2> {
        DualMatrix {
            val: self.val * rhs.val,
            dij_val: DualMatrix::binary_mm_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| l_dij * rhs.val,
                |r_dij| self.val * r_dij,
            ),
        }
    }

    fn from_scalar(val: DualScalar) -> Self {
        DualMatrix {
            val: MatF64::<ROWS, COLS>::from_scalar(val.val),
            dij_val: val.dij_val.map(|dij_val| {
                MutTensorDDRC::from_map(&dij_val.view(), |v| MatF64::<ROWS, COLS>::from_scalar(*v))
            }),
        }
    }

    fn from_real_matrix(val: MatF64<ROWS, COLS>) -> Self {
        Self { val, dij_val: None }
    }

    fn scaled(&self, s: DualScalar) -> Self {
        DualMatrix {
            val: self.val * s.val,
            dij_val: DualMatrix::binary_ms_dij(
                &self.dij_val,
                &s.dij_val,
                |l_dij| l_dij * s.val,
                |r_dij| self.val * *r_dij,
            ),
        }
    }

    fn identity() -> Self {
        DualMatrix::from_real_matrix(MatF64::<ROWS, COLS>::identity())
    }

    fn get_elem(&self, idx: [usize; 2]) -> DualScalar {
        DualScalar {
            val: self.val.get_elem(idx),
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDD::from_map(&dij_val.view(), |v| v[(idx[0], idx[1])])),
        }
    }

    fn from_array2(duals: [[DualScalar; COLS]; ROWS]) -> Self {
        let mut shape = None;
        let mut val_mat = MatF64::<ROWS, COLS>::zeros();
        for i in 0..duals.len() {
            let d_rows = duals[i].clone();
            for j in 0..d_rows.len() {
                let d = d_rows.clone()[j].clone();

                val_mat[(i, j)] = d.val;
                if d.dij_val.is_some() {
                    shape = Some(d.dij_val.clone().unwrap().dims());
                }
            }
        }

        if shape.is_none() {
            return DualMatrix {
                val: val_mat,
                dij_val: None,
            };
        }
        let shape = shape.unwrap();

        let mut r = MutTensorDDRC::<f64, ROWS, COLS>::from_shape(shape);

        for i in 0..duals.len() {
            let d_rows = duals[i].clone();
            for j in 0..d_rows.len() {
                let d = d_rows.clone()[j].clone();
                if d.dij_val.is_some() {
                    for d0 in 0..shape[0] {
                        for d1 in 0..shape[1] {
                            r.mut_view().get_mut([d0, d1])[(i, j)] =
                                d.dij_val.clone().unwrap().get([d0, d1]);
                        }
                    }
                }
            }
        }
        DualMatrix {
            val: val_mat,
            dij_val: Some(r),
        }
    }

    fn real_matrix(&self) -> &MatF64<ROWS, COLS> {
        &self.val
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (DualMatrix<R0, C0>, DualMatrix<R0, C1>),
        bot_row: (DualMatrix<R1, C0>, DualMatrix<R1, C1>),
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
        assert_eq!(C0 + C1, COLS);

        Self::block_mat2x1(
            DualMatrix::<R0, COLS>::block_mat1x2(top_row.0, top_row.1),
            DualMatrix::<R1, COLS>::block_mat1x2(bot_row.0, bot_row.1),
        )
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: DualMatrix<R0, COLS>,
        bot_row: DualMatrix<R1, COLS>,
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
        let maybe_dij = Self::two_dx(top_row.dij_val, bot_row.dij_val);

        Self {
            val: MatF64::<ROWS, COLS>::block_mat2x1(top_row.val, bot_row.val),
            dij_val: match maybe_dij {
                Some(dij_val) => {
                    let mut r = MutTensorDDRC::<f64, ROWS, COLS>::from_shape(dij_val.shape());
                    for d0 in 0..dij_val.shape()[0] {
                        for d1 in 0..dij_val.shape()[1] {
                            *r.mut_view().get_mut([d0, d1]) = MatF64::<ROWS, COLS>::block_mat2x1(
                                dij_val.lhs.get([d0, d1]),
                                dij_val.rhs.get([d0, d1]),
                            );
                        }
                    }
                    Some(r)
                }
                None => None,
            },
        }
    }

    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: DualMatrix<ROWS, C0>,
        righ_col: DualMatrix<ROWS, C1>,
    ) -> Self {
        assert_eq!(C0 + C1, COLS);
        let maybe_dij = Self::two_dx(left_col.dij_val, righ_col.dij_val);

        Self {
            val: MatF64::<ROWS, COLS>::block_mat1x2(left_col.val, righ_col.val),
            dij_val: match maybe_dij {
                Some(dij_val) => {
                    let mut r = MutTensorDDRC::<f64, ROWS, COLS>::from_shape(dij_val.shape());
                    for d0 in 0..dij_val.shape()[0] {
                        for d1 in 0..dij_val.shape()[1] {
                            *r.mut_view().get_mut([d0, d1]) = MatF64::<ROWS, COLS>::block_mat1x2(
                                dij_val.lhs.get([d0, d1]),
                                dij_val.rhs.get([d0, d1]),
                            );
                        }
                    }
                    Some(r)
                }
                None => None,
            },
        }
    }

    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> DualMatrix<R, C> {
        DualMatrix {
            val: self.val.get_fixed_submat(start_r, start_c),
            dij_val: self.dij_val.clone().map(|dij_val| {
                MutTensorDDRC::from_map(&dij_val.view(), |v| v.get_fixed_submat(start_r, start_c))
            }),
        }
    }

    fn get_col_vec(&self, start_r: usize) -> DualVector<ROWS> {
        DualVector {
            val: self.val.get_col_vec(start_r),
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| v.get_col_vec(start_r))),
        }
    }

    fn get_row_vec(&self, c: usize) -> DualVector<COLS> {
        DualVector {
            val: self.val.get_row_vec(c),
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| v.get_row_vec(c))),
        }
    }

    fn from_real_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        DualMatrix {
            val: MatF64::from_real_array2(vals),
            dij_val: None,
        }
    }

    fn from_f64_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        DualMatrix {
            val: MatF64::from_real_array2(vals),
            dij_val: None,
        }
    }

    fn from_f64(val: f64) -> Self {
        DualMatrix {
            val: MatF64::<ROWS, COLS>::from_f64(val),
            dij_val: None,
        }
    }

    fn set_col_vec(&mut self, c: usize, v: DualVector<ROWS>) {
        self.val.set_col_vec(c, v.val);
        todo!();
    }

    fn to_dual(self) -> <DualScalar as IsScalar<1>>::DualMatrix<ROWS, COLS> {
        self
    }

    fn select(self, mask: &bool, other: Self) -> Self {
        if *mask {
            self
        } else {
            other
        }
    }

    fn set_elem(&mut self, idx: [usize; 2], val: DualScalar) {
        self.val.set_elem(idx, val.val);
        if self.dij_val.is_some() {
            let dij = &mut self.dij_val.as_mut().unwrap();
            for i in 0..dij.dims()[0] {
                for j in 0..dij.dims()[1] {
                    dij.mut_view().get_mut([i, j])[(idx[0], idx[1])] =
                        val.dij_val.clone().unwrap().get([i, j]);
                }
            }
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Add for DualMatrix<ROWS, COLS> {
    type Output = DualMatrix<ROWS, COLS>;

    fn add(self, rhs: Self) -> Self::Output {
        DualMatrix {
            val: self.val + rhs.val,
            dij_val: Self::binary_mm_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| *l_dij,
                |r_dij| *r_dij,
            ),
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Sub for DualMatrix<ROWS, COLS> {
    type Output = DualMatrix<ROWS, COLS>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualMatrix {
            val: self.val - rhs.val,
            dij_val: Self::binary_mm_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| *l_dij,
                |r_dij| -r_dij,
            ),
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Neg for DualMatrix<ROWS, COLS> {
    type Output = DualMatrix<ROWS, COLS>;

    fn neg(self) -> Self::Output {
        DualMatrix {
            val: -self.val,
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDDRC::from_map(&dij_val.view(), |v| -v)),
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Zero for DualMatrix<ROWS, COLS> {
    fn zero() -> Self {
        Self::from_real_matrix(MatF64::zeros())
    }

    fn is_zero(&self) -> bool {
        self.val.is_zero()
    }
}

impl<const ROWS: usize, const COLS: usize> Mul<DualVector<COLS>> for DualMatrix<ROWS, COLS> {
    type Output = DualVector<ROWS>;

    fn mul(self, rhs: DualVector<COLS>) -> Self::Output {
        Self::Output {
            val: self.val * rhs.val,
            dij_val: Self::binary_mv_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| l_dij * rhs.val,
                |r_dij| self.val * r_dij,
            ),
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Debug for DualMatrix<ROWS, COLS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dij_val.is_some() {
            f.debug_struct("DualScalarLike")
                .field("val", &self.val)
                .field("dij_val", &self.dij_val.as_ref().unwrap().elem_view())
                .finish()
        } else {
            f.debug_struct("DualScalarLike")
                .field("val", &self.val)
                .finish()
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> IsDual
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize>
    IsDualMatrix<DualBatchScalar<BATCH>, ROWS, COLS, BATCH> for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    /// Create a new dual number
    fn new(val: BatchMatF64<ROWS, COLS, BATCH>) -> Self {
        DualBatchMatrix { val, dij_val: None }
    }

    /// Get the derivative
    fn dij_val(self) -> Option<MutTensorDDRC<BatchScalarF64<BATCH>, ROWS, COLS>> {
        self.dij_val
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn binary_mm_dij<
        const R0: usize,
        const R1: usize,
        const C0: usize,
        const C1: usize,
        F: FnMut(&BatchMatF64<R0, C0, BATCH>) -> BatchMatF64<ROWS, COLS, BATCH>,
        G: FnMut(&BatchMatF64<R1, C1, BATCH>) -> BatchMatF64<ROWS, COLS, BATCH>,
    >(
        lhs_dx: &Option<MutTensorDDRC<BatchScalarF64<BATCH>, R0, C0>>,
        rhs_dx: &Option<MutTensorDDRC<BatchScalarF64<BATCH>, R1, C1>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDRC<BatchScalarF64<BATCH>, ROWS, COLS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDRC::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDRC::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDRC::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    fn binary_mv_dij<
        const R0: usize,
        const R1: usize,
        const C0: usize,
        F: FnMut(&BatchMatF64<R0, C0, BATCH>) -> BatchVecF64<ROWS, BATCH>,
        G: FnMut(&BatchVecF64<R1, BATCH>) -> BatchVecF64<ROWS, BATCH>,
    >(
        lhs_dx: &Option<MutTensorDDRC<BatchScalarF64<BATCH>, R0, C0>>,
        rhs_dx: &Option<MutTensorDDR<BatchScalarF64<BATCH>, R1>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDR<BatchScalarF64<BATCH>, ROWS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDR::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDR::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDR::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    fn binary_ms_dij<
        const R0: usize,
        const C0: usize,
        F: FnMut(&BatchMatF64<R0, C0, BATCH>) -> BatchMatF64<ROWS, COLS, BATCH>,
        G: FnMut(&BatchScalarF64<BATCH>) -> BatchMatF64<ROWS, COLS, BATCH>,
    >(
        lhs_dx: &Option<MutTensorDDRC<BatchScalarF64<BATCH>, R0, C0>>,
        rhs_dx: &Option<MutTensorDD<BatchScalarF64<BATCH>>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDRC<BatchScalarF64<BATCH>, ROWS, COLS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDRC::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDRC::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDRC::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    /// derivatives
    pub fn two_dx<const R1: usize, const C1: usize, const R2: usize, const C2: usize>(
        mut lhs_dx: Option<MutTensorDDRC<BatchScalarF64<BATCH>, R1, C1>>,
        mut rhs_dx: Option<MutTensorDDRC<BatchScalarF64<BATCH>, R2, C2>>,
    ) -> Option<DijPairM<BatchScalarF64<BATCH>, R1, C1, R2, C2>> {
        if lhs_dx.is_none() && rhs_dx.is_none() {
            return None;
        }

        if lhs_dx.is_some() && rhs_dx.is_some() {
            assert_eq!(
                lhs_dx.clone().unwrap().dims(),
                rhs_dx.clone().unwrap().dims()
            );
        }

        if lhs_dx.is_none() {
            lhs_dx = Some(MutTensorDDRC::<BatchScalarF64<BATCH>, R1, C1>::from_shape(
                rhs_dx.clone().unwrap().dims(),
            ))
        } else if rhs_dx.is_none() {
            rhs_dx = Some(MutTensorDDRC::<BatchScalarF64<BATCH>, R2, C2>::from_shape(
                lhs_dx.clone().unwrap().dims(),
            ))
        }

        Some(DijPairM {
            lhs: lhs_dx.unwrap(),
            rhs: rhs_dx.unwrap(),
        })
    }

    /// derivatives
    pub fn two_dx_from_vec(
        mut lhs_dx: Option<MutTensorDDRC<BatchScalarF64<BATCH>, ROWS, COLS>>,
        mut rhs_dx: Option<MutTensorDDR<BatchScalarF64<BATCH>, COLS>>,
    ) -> Option<DijPairMV<BatchScalarF64<BATCH>, ROWS, COLS>> {
        if lhs_dx.is_none() && rhs_dx.is_none() {
            return None;
        }

        if lhs_dx.is_some() && rhs_dx.is_some() {
            assert_eq!(
                lhs_dx.clone().unwrap().dims(),
                rhs_dx.clone().unwrap().dims()
            );
        }

        if lhs_dx.is_none() {
            lhs_dx = Some(
                MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape(
                    rhs_dx.clone().unwrap().dims(),
                ),
            )
        } else if rhs_dx.is_none() {
            rhs_dx = Some(MutTensorDDR::<BatchScalarF64<BATCH>, COLS>::from_shape(
                lhs_dx.clone().unwrap().dims(),
            ))
        }

        Some(DijPairMV::<BatchScalarF64<BATCH>, ROWS, COLS> {
            lhs: lhs_dx.unwrap(),
            rhs: rhs_dx.unwrap(),
        })
    }

    /// Create a dual matrix
    pub fn v(val: BatchMatF64<ROWS, COLS, BATCH>) -> Self {
        let mut dij_val =
            MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape([ROWS, COLS]);
        for i in 0..ROWS {
            for j in 0..COLS {
                dij_val.mut_view().get_mut([i, j])[(i, j)] = BatchScalarF64::<BATCH>::from_f64(1.0);
            }
        }

        Self {
            val,
            dij_val: Some(dij_val),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> PartialEq
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val && self.dij_val == other.dij_val
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> AbsDiffEq
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.val.abs_diff_eq(&other.val, epsilon)
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> RelativeEq
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
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
        self.val.relative_eq(&other.val, epsilon, max_relative)
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize>
    IsMatrix<DualBatchScalar<BATCH>, ROWS, COLS, BATCH> for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn from_f64(val: f64) -> Self {
        DualBatchMatrix {
            val: BatchMatF64::<ROWS, COLS, BATCH>::from_f64(val),
            dij_val: None,
        }
    }

    fn set_elem(&mut self, idx: [usize; 2], val: DualBatchScalar<BATCH>) {
        self.val.set_elem(idx, val.val);
        if self.dij_val.is_some() {
            let dij = &mut self.dij_val.as_mut().unwrap();
            for i in 0..dij.dims()[0] {
                for j in 0..dij.dims()[1] {
                    dij.mut_view().get_mut([i, j])[(idx[0], idx[1])] =
                        val.dij_val.clone().unwrap().get([i, j]);
                }
            }
        }
    }

    fn from_scalar(val: DualBatchScalar<BATCH>) -> Self {
        DualBatchMatrix {
            val: BatchMatF64::<ROWS, COLS, BATCH>::from_scalar(val.val),
            dij_val: val.dij_val.map(|dij_val| {
                MutTensorDDRC::from_map(&dij_val.view(), |v| {
                    BatchMatF64::<ROWS, COLS, BATCH>::from_scalar(*v)
                })
            }),
        }
    }

    fn mat_mul<const COLS2: usize>(
        &self,
        rhs: DualBatchMatrix<COLS, COLS2, BATCH>,
    ) -> DualBatchMatrix<ROWS, COLS2, BATCH> {
        DualBatchMatrix {
            val: self.val * rhs.val,
            dij_val: DualBatchMatrix::<ROWS, COLS2, BATCH>::binary_mm_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| l_dij * rhs.val,
                |r_dij| self.val * r_dij,
            ),
        }
    }

    fn from_real_matrix(val: BatchMatF64<ROWS, COLS, BATCH>) -> Self {
        Self { val, dij_val: None }
    }

    fn scaled(&self, s: DualBatchScalar<BATCH>) -> Self {
        DualBatchMatrix {
            val: self.val * s.val,
            dij_val: DualBatchMatrix::<ROWS, COLS, BATCH>::binary_ms_dij(
                &self.dij_val,
                &s.dij_val,
                |l_dij| l_dij * s.val,
                |r_dij| self.val * *r_dij,
            ),
        }
    }

    fn identity() -> Self {
        DualBatchMatrix::from_real_matrix(BatchMatF64::<ROWS, COLS, BATCH>::identity())
    }

    fn get_elem(&self, idx: [usize; 2]) -> DualBatchScalar<BATCH> {
        DualBatchScalar::<BATCH> {
            val: self.val.get_elem(idx),
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDD::from_map(&dij_val.view(), |v| v[(idx[0], idx[1])])),
        }
    }

    fn from_array2(duals: [[DualBatchScalar<BATCH>; COLS]; ROWS]) -> Self {
        let mut shape = None;
        let mut val_mat = BatchMatF64::<ROWS, COLS, BATCH>::zeros();
        for i in 0..duals.len() {
            let d_rows = duals[i].clone();
            for j in 0..d_rows.len() {
                let d = d_rows.clone()[j].clone();

                val_mat[(i, j)] = d.val;
                if d.dij_val.is_some() {
                    shape = Some(d.dij_val.clone().unwrap().dims());
                }
            }
        }

        if shape.is_none() {
            return DualBatchMatrix {
                val: val_mat,
                dij_val: None,
            };
        }
        let shape = shape.unwrap();

        let mut r = MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape(shape);

        for i in 0..duals.len() {
            let d_rows = duals[i].clone();
            for j in 0..d_rows.len() {
                let d = d_rows.clone()[j].clone();
                if d.dij_val.is_some() {
                    for d0 in 0..shape[0] {
                        for d1 in 0..shape[1] {
                            r.mut_view().get_mut([d0, d1])[(i, j)] =
                                d.dij_val.clone().unwrap().get([d0, d1]);
                        }
                    }
                }
            }
        }
        DualBatchMatrix {
            val: val_mat,
            dij_val: Some(r),
        }
    }

    fn real_matrix(&self) -> &BatchMatF64<ROWS, COLS, BATCH> {
        &self.val
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (
            DualBatchMatrix<R0, C0, BATCH>,
            DualBatchMatrix<R0, C1, BATCH>,
        ),
        bot_row: (
            DualBatchMatrix<R1, C0, BATCH>,
            DualBatchMatrix<R1, C1, BATCH>,
        ),
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
        assert_eq!(C0 + C1, COLS);

        Self::block_mat2x1(
            DualBatchMatrix::<R0, COLS, BATCH>::block_mat1x2(top_row.0, top_row.1),
            DualBatchMatrix::<R1, COLS, BATCH>::block_mat1x2(bot_row.0, bot_row.1),
        )
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: DualBatchMatrix<R0, COLS, BATCH>,
        bot_row: DualBatchMatrix<R1, COLS, BATCH>,
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
        let maybe_dij = Self::two_dx(top_row.dij_val, bot_row.dij_val);

        Self {
            val: BatchMatF64::<ROWS, COLS, BATCH>::block_mat2x1(top_row.val, bot_row.val),
            dij_val: match maybe_dij {
                Some(dij_val) => {
                    let mut r = MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape(
                        dij_val.shape(),
                    );
                    for d0 in 0..dij_val.shape()[0] {
                        for d1 in 0..dij_val.shape()[1] {
                            *r.mut_view().get_mut([d0, d1]) =
                                BatchMatF64::<ROWS, COLS, BATCH>::block_mat2x1(
                                    dij_val.lhs.get([d0, d1]),
                                    dij_val.rhs.get([d0, d1]),
                                );
                        }
                    }
                    Some(r)
                }
                None => None,
            },
        }
    }

    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: DualBatchMatrix<ROWS, C0, BATCH>,
        righ_col: DualBatchMatrix<ROWS, C1, BATCH>,
    ) -> Self {
        assert_eq!(C0 + C1, COLS);
        let maybe_dij = Self::two_dx(left_col.dij_val, righ_col.dij_val);

        Self {
            val: BatchMatF64::<ROWS, COLS, BATCH>::block_mat1x2(left_col.val, righ_col.val),
            dij_val: match maybe_dij {
                Some(dij_val) => {
                    let mut r = MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape(
                        dij_val.shape(),
                    );
                    for d0 in 0..dij_val.shape()[0] {
                        for d1 in 0..dij_val.shape()[1] {
                            *r.mut_view().get_mut([d0, d1]) =
                                BatchMatF64::<ROWS, COLS, BATCH>::block_mat1x2(
                                    dij_val.lhs.get([d0, d1]),
                                    dij_val.rhs.get([d0, d1]),
                                );
                        }
                    }
                    Some(r)
                }
                None => None,
            },
        }
    }

    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> DualBatchMatrix<R, C, BATCH> {
        DualBatchMatrix {
            val: self.val.get_fixed_submat(start_r, start_c),
            dij_val: self.dij_val.clone().map(|dij_val| {
                MutTensorDDRC::from_map(&dij_val.view(), |v| v.get_fixed_submat(start_r, start_c))
            }),
        }
    }

    fn get_col_vec(&self, start_r: usize) -> DualBatchVector<ROWS, BATCH> {
        DualBatchVector {
            val: self.val.get_col_vec(start_r),
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| v.get_col_vec(start_r))),
        }
    }

    fn get_row_vec(&self, c: usize) -> DualBatchVector<COLS, BATCH> {
        DualBatchVector {
            val: self.val.get_row_vec(c),
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| v.get_row_vec(c))),
        }
    }

    fn from_real_array2(vals: [[BatchScalarF64<BATCH>; COLS]; ROWS]) -> Self {
        DualBatchMatrix {
            val: BatchMatF64::from_real_array2(vals),
            dij_val: None,
        }
    }

    fn from_f64_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        DualBatchMatrix {
            val: BatchMatF64::from_f64_array2(vals),
            dij_val: None,
        }
    }

    fn set_col_vec(
        &mut self,
        c: usize,
        v: <DualBatchScalar<BATCH> as IsScalar<BATCH>>::Vector<ROWS>,
    ) {
        self.val.set_col_vec(c, v.val);
        todo!();
    }

    fn to_dual(self) -> <DualBatchScalar<BATCH> as IsScalar<BATCH>>::DualMatrix<ROWS, COLS> {
        self
    }

    fn select(self, mask: &Mask<i64, BATCH>, other: Self) -> Self {
        let maybe_dij = Self::two_dx(self.dij_val, other.dij_val);

        DualBatchMatrix {
            val: self.val.select(mask, other.val),
            dij_val: match maybe_dij {
                Some(dij) => {
                    let mut r =
                        MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape(dij.shape());
                    for i in 0..dij.shape()[0] {
                        for j in 0..dij.shape()[1] {
                            *r.get_mut([i, j]) =
                                dij.lhs.get([i, j]).select(mask, dij.rhs.get([i, j]));
                        }
                    }
                    Some(r)
                }
                _ => None,
            },
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Add
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchMatrix<ROWS, COLS, BATCH>;

    fn add(self, rhs: Self) -> Self::Output {
        DualBatchMatrix {
            val: self.val + rhs.val,
            dij_val: Self::binary_mm_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| *l_dij,
                |r_dij| *r_dij,
            ),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Sub
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchMatrix<ROWS, COLS, BATCH>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualBatchMatrix {
            val: self.val - rhs.val,
            dij_val: Self::binary_mm_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| *l_dij,
                |r_dij| -r_dij,
            ),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Neg
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchMatrix<ROWS, COLS, BATCH>;

    fn neg(self) -> Self::Output {
        DualBatchMatrix {
            val: -self.val,
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDDRC::from_map(&dij_val.view(), |v| -v)),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Zero
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn zero() -> Self {
        Self::from_real_matrix(BatchMatF64::zeros())
    }

    fn is_zero(&self) -> bool {
        self.val.is_zero()
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Mul<DualBatchVector<COLS, BATCH>>
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchVector<ROWS, BATCH>;

    fn mul(self, rhs: DualBatchVector<COLS, BATCH>) -> Self::Output {
        Self::Output {
            val: self.val * rhs.val,
            dij_val: Self::binary_mv_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| l_dij * rhs.val,
                |r_dij| self.val * r_dij,
            ),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Debug
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dij_val.is_some() {
            f.debug_struct("DualScalarLike")
                .field("val", &self.val)
                .field("dij_val", &self.dij_val.as_ref().unwrap().elem_view())
                .finish()
        } else {
            f.debug_struct("DualScalarLike")
                .field("val", &self.val)
                .finish()
        }
    }
}

#[test]
fn dual_matrix_tests() {
    use crate::calculus::dual::dual_scalar::DualBatchScalar;
    use crate::calculus::dual::dual_scalar::DualScalar;
    use crate::calculus::maps::matrix_valued_maps::MatrixValuedMapFromMatrix;
    use crate::linalg::BatchScalarF64;

    #[cfg(test)]
    trait Test {
        fn run();
    }

    macro_rules! def_test_template {
        ( $scalar:ty, $dual_scalar: ty, $batch:literal
    ) => {
            #[cfg(test)]
            impl Test for $scalar {
                fn run() {
                    let m_2x4 = <$scalar as IsScalar<$batch>>::Matrix::<2, 4>::from_f64_array2([
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                    ]);
                    let m_4x1 = <$scalar as IsScalar<$batch>>::Matrix::<4, 1>::from_f64_array2([
                        [1.0],
                        [2.0],
                        [3.0],
                        [4.0],
                    ]);

                    fn mat_mul_fn<S: IsScalar<BATCH>, const BATCH: usize>(
                        x: S::Matrix<2, 4>,
                        y: S::Matrix<4, 1>,
                    ) -> S::Matrix<2, 1> {
                        x.mat_mul(y)
                    }
                    let finite_diff =
                        MatrixValuedMapFromMatrix::<$scalar, $batch>::sym_diff_quotient(
                            |x| mat_mul_fn::<$scalar, $batch>(x, m_4x1),
                            m_2x4,
                            1e-6,
                        );
                    let auto_grad = MatrixValuedMapFromMatrix::<$dual_scalar, $batch>::fw_autodiff(
                        |x| {
                            mat_mul_fn::<$dual_scalar, $batch>(
                                x,
                                <$dual_scalar as IsScalar<$batch>>::Matrix::from_real_matrix(m_4x1),
                            )
                        },
                        m_2x4,
                    );

                    for i in 0..2 {
                        for j in 0..1 {
                            approx::assert_abs_diff_eq!(
                                finite_diff.get([i, j]),
                                auto_grad.get([i, j]),
                                epsilon = 0.0001
                            );
                        }
                    }

                    let finite_diff = MatrixValuedMapFromMatrix::sym_diff_quotient(
                        |x| mat_mul_fn::<$scalar, $batch>(m_2x4, x),
                        m_4x1,
                        1e-6,
                    );
                    let auto_grad = MatrixValuedMapFromMatrix::<$dual_scalar, $batch>::fw_autodiff(
                        |x| {
                            mat_mul_fn::<$dual_scalar, $batch>(
                                <$dual_scalar as IsScalar<$batch>>::Matrix::from_real_matrix(m_2x4),
                                x,
                            )
                        },
                        m_4x1,
                    );

                    for i in 0..2 {
                        for j in 0..1 {
                            approx::assert_abs_diff_eq!(
                                finite_diff.get([i, j]),
                                auto_grad.get([i, j]),
                                epsilon = 0.0001
                            );
                        }
                    }

                    fn mat_mul2_fn<S: IsScalar<BATCH>, const BATCH: usize>(
                        x: S::Matrix<4, 4>,
                    ) -> S::Matrix<4, 4> {
                        x.mat_mul(x.clone())
                    }

                    let m_4x4 = <$scalar as IsScalar<$batch>>::Matrix::<4, 4>::from_f64_array2([
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                    ]);

                    let finite_diff =
                        MatrixValuedMapFromMatrix::<$scalar, $batch>::sym_diff_quotient(
                            mat_mul2_fn::<$scalar, $batch>,
                            m_4x4,
                            1e-6,
                        );
                    let auto_grad = MatrixValuedMapFromMatrix::<$dual_scalar, $batch>::fw_autodiff(
                        mat_mul2_fn::<$dual_scalar, $batch>,
                        m_4x4,
                    );

                    for i in 0..2 {
                        for j in 0..1 {
                            approx::assert_abs_diff_eq!(
                                finite_diff.get([i, j]),
                                auto_grad.get([i, j]),
                                epsilon = 0.0001
                            );
                        }
                    }
                }
            }
        };
    }

    def_test_template!(f64, DualScalar, 1);
    def_test_template!(BatchScalarF64<2>, DualBatchScalar<2>, 2);
    def_test_template!(BatchScalarF64<4>, DualBatchScalar<4>, 4);

    f64::run();
    BatchScalarF64::<2>::run();
    BatchScalarF64::<4>::run();
}
