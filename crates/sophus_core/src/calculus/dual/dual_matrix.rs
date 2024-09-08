use crate::calculus::dual::DualScalar;
use crate::calculus::dual::DualVector;
use crate::linalg::MatF64;
use crate::linalg::VecF64;
use crate::prelude::*;
use crate::tensor::mut_tensor::MutTensorDD;
use crate::tensor::mut_tensor::MutTensorDDR;
use crate::tensor::mut_tensor::MutTensorDDRC;
use approx::AbsDiffEq;
use approx::RelativeEq;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

/// DualScalarLike matrix
#[derive(Clone)]
pub struct DualMatrix<const ROWS: usize, const COLS: usize> {
    /// real part
    pub real_part: MatF64<ROWS, COLS>,
    /// infinitesimal part - represents derivative
    pub dij_part: Option<MutTensorDDRC<f64, ROWS, COLS>>,
}

impl<const ROWS: usize, const COLS: usize> Debug for DualMatrix<ROWS, COLS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dij_part.is_some() {
            f.debug_struct("DualScalarLike")
                .field("val", &self.real_part)
                .field("dij_val", &self.dij_part.as_ref().unwrap().elem_view())
                .finish()
        } else {
            f.debug_struct("DualScalarLike")
                .field("val", &self.real_part)
                .finish()
        }
    }
}

impl<const ROWS: usize, const COLS: usize> IsSingleMatrix<DualScalar, ROWS, COLS>
    for DualMatrix<ROWS, COLS>
{
    fn single_real_matrix(&self) -> MatF64<ROWS, COLS> {
        self.real_part
    }
}

/// Trait for scalar dual numbers
pub trait IsDualMatrix<
    S: IsDualScalar<BATCH>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH: usize,
>: IsMatrix<S, ROWS, COLS, BATCH> + IsDual
{
    /// Create a new dual matrix from a real matrix for auto-differentiation with respect to self
    ///
    /// Typically this is not called directly, but through using a map auto-differentiation call:
    ///
    ///  - ScalarValuedMapFromMatrix::fw_autodiff(...);
    ///  - VectorValuedMapFromMatrix::fw_autodiff(...);
    ///  - MatrixValuedMapFromMatrix::fw_autodiff(...);
    fn new_with_dij(val: S::RealMatrix<ROWS, COLS>) -> Self;

    /// Get the derivative
    fn dij_val(self) -> Option<MutTensorDDRC<S::RealScalar, ROWS, COLS>>;
}

impl<const ROWS: usize, const COLS: usize> IsDual for DualMatrix<ROWS, COLS> {}

impl<const ROWS: usize, const COLS: usize> IsDualMatrix<DualScalar, ROWS, COLS, 1>
    for DualMatrix<ROWS, COLS>
{
    /// Create a new dual number
    fn new_with_dij(val: MatF64<ROWS, COLS>) -> Self {
        let mut dij_val = MutTensorDDRC::<f64, ROWS, COLS>::from_shape([ROWS, COLS]);
        for i in 0..ROWS {
            for j in 0..COLS {
                dij_val.mut_view().get_mut([i, j])[(i, j)] = 1.0;
            }
        }

        Self {
            real_part: val,
            dij_part: Some(dij_val),
        }
    }

    /// Get the derivative
    fn dij_val(self) -> Option<MutTensorDDRC<f64, ROWS, COLS>> {
        self.dij_part
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
    pub(crate) lhs: MutTensorDDRC<S, ROWS, COLS>,
    pub(crate) rhs: MutTensorDDRC<S, ROWS2, COLS2>,
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
    pub(crate) fn shape(&self) -> [usize; 2] {
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
}

impl<const ROWS: usize, const COLS: usize> PartialEq for DualMatrix<ROWS, COLS> {
    fn eq(&self, other: &Self) -> bool {
        self.real_part == other.real_part && self.dij_part == other.dij_part
    }
}

impl<const ROWS: usize, const COLS: usize> AbsDiffEq for DualMatrix<ROWS, COLS> {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.real_part.abs_diff_eq(&other.real_part, epsilon)
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
        self.real_part
            .relative_eq(&other.real_part, epsilon, max_relative)
    }
}

impl<const ROWS: usize, const COLS: usize> IsMatrix<DualScalar, ROWS, COLS, 1>
    for DualMatrix<ROWS, COLS>
{
    fn mat_mul<const COLS2: usize>(&self, rhs: DualMatrix<COLS, COLS2>) -> DualMatrix<ROWS, COLS2> {
        DualMatrix {
            real_part: self.real_part * rhs.real_part,
            dij_part: DualMatrix::binary_mm_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| l_dij * rhs.real_part,
                |r_dij| self.real_part * r_dij,
            ),
        }
    }

    fn from_scalar(val: DualScalar) -> Self {
        DualMatrix {
            real_part: MatF64::<ROWS, COLS>::from_scalar(val.real_part),
            dij_part: val.dij_part.map(|dij_val| {
                MutTensorDDRC::from_map(&dij_val.view(), |v| MatF64::<ROWS, COLS>::from_scalar(*v))
            }),
        }
    }

    fn from_real_matrix(val: MatF64<ROWS, COLS>) -> Self {
        Self {
            real_part: val,
            dij_part: None,
        }
    }

    fn scaled(&self, s: DualScalar) -> Self {
        DualMatrix {
            real_part: self.real_part * s.real_part,
            dij_part: DualMatrix::binary_ms_dij(
                &self.dij_part,
                &s.dij_part,
                |l_dij| l_dij * s.real_part,
                |r_dij| self.real_part * *r_dij,
            ),
        }
    }

    fn identity() -> Self {
        DualMatrix::from_real_matrix(MatF64::<ROWS, COLS>::identity())
    }

    fn get_elem(&self, idx: [usize; 2]) -> DualScalar {
        DualScalar {
            real_part: self.real_part.get_elem(idx),
            dij_part: self
                .dij_part
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

                val_mat[(i, j)] = d.real_part;
                if d.dij_part.is_some() {
                    shape = Some(d.dij_part.clone().unwrap().dims());
                }
            }
        }

        if shape.is_none() {
            return DualMatrix {
                real_part: val_mat,
                dij_part: None,
            };
        }
        let shape = shape.unwrap();

        let mut r = MutTensorDDRC::<f64, ROWS, COLS>::from_shape(shape);

        for i in 0..duals.len() {
            let d_rows = duals[i].clone();
            for j in 0..d_rows.len() {
                let d = d_rows.clone()[j].clone();
                if d.dij_part.is_some() {
                    for d0 in 0..shape[0] {
                        for d1 in 0..shape[1] {
                            r.mut_view().get_mut([d0, d1])[(i, j)] =
                                d.dij_part.clone().unwrap().get([d0, d1]);
                        }
                    }
                }
            }
        }
        DualMatrix {
            real_part: val_mat,
            dij_part: Some(r),
        }
    }

    fn real_matrix(&self) -> &MatF64<ROWS, COLS> {
        &self.real_part
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
        let maybe_dij = Self::two_dx(top_row.dij_part, bot_row.dij_part);

        Self {
            real_part: MatF64::<ROWS, COLS>::block_mat2x1(top_row.real_part, bot_row.real_part),
            dij_part: match maybe_dij {
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
        let maybe_dij = Self::two_dx(left_col.dij_part, righ_col.dij_part);

        Self {
            real_part: MatF64::<ROWS, COLS>::block_mat1x2(left_col.real_part, righ_col.real_part),
            dij_part: match maybe_dij {
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
            real_part: self.real_part.get_fixed_submat(start_r, start_c),
            dij_part: self.dij_part.clone().map(|dij_val| {
                MutTensorDDRC::from_map(&dij_val.view(), |v| v.get_fixed_submat(start_r, start_c))
            }),
        }
    }

    fn get_col_vec(&self, start_r: usize) -> DualVector<ROWS> {
        DualVector {
            real_part: self.real_part.get_col_vec(start_r),
            dij_part: self
                .dij_part
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| v.get_col_vec(start_r))),
        }
    }

    fn get_row_vec(&self, c: usize) -> DualVector<COLS> {
        DualVector {
            real_part: self.real_part.get_row_vec(c),
            dij_part: self
                .dij_part
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| v.get_row_vec(c))),
        }
    }

    fn from_real_scalar_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        DualMatrix {
            real_part: MatF64::from_real_scalar_array2(vals),
            dij_part: None,
        }
    }

    fn from_f64_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        DualMatrix {
            real_part: MatF64::from_real_scalar_array2(vals),
            dij_part: None,
        }
    }

    fn from_f64(val: f64) -> Self {
        DualMatrix {
            real_part: MatF64::<ROWS, COLS>::from_f64(val),
            dij_part: None,
        }
    }

    fn set_col_vec(&mut self, c: usize, v: DualVector<ROWS>) {
        self.real_part.set_col_vec(c, v.real_part);
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
        self.real_part.set_elem(idx, val.real_part);
        if self.dij_part.is_some() {
            let dij = &mut self.dij_part.as_mut().unwrap();
            for i in 0..dij.dims()[0] {
                for j in 0..dij.dims()[1] {
                    dij.mut_view().get_mut([i, j])[(idx[0], idx[1])] =
                        val.dij_part.clone().unwrap().get([i, j]);
                }
            }
        }
    }

    fn transposed(self) -> <DualScalar as IsScalar<1>>::Matrix<COLS, ROWS> {
        todo!();
    }
}

impl<const ROWS: usize, const COLS: usize> Add for DualMatrix<ROWS, COLS> {
    type Output = DualMatrix<ROWS, COLS>;

    fn add(self, rhs: Self) -> Self::Output {
        DualMatrix {
            real_part: self.real_part + rhs.real_part,
            dij_part: Self::binary_mm_dij(
                &self.dij_part,
                &rhs.dij_part,
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
            real_part: self.real_part - rhs.real_part,
            dij_part: Self::binary_mm_dij(
                &self.dij_part,
                &rhs.dij_part,
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
            real_part: -self.real_part,
            dij_part: self
                .dij_part
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
        self.real_part.is_zero()
    }
}

impl<const ROWS: usize, const COLS: usize> Mul<DualVector<COLS>> for DualMatrix<ROWS, COLS> {
    type Output = DualVector<ROWS>;

    fn mul(self, rhs: DualVector<COLS>) -> Self::Output {
        Self::Output {
            real_part: self.real_part * rhs.real_part,
            dij_part: Self::binary_mv_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| l_dij * rhs.real_part,
                |r_dij| self.real_part * r_dij,
            ),
        }
    }
}

#[test]
fn dual_matrix_tests() {
    use crate::calculus::dual::DualScalar;
    use crate::calculus::maps::matrix_valued_maps::MatrixValuedMapFromMatrix;
    use crate::linalg::EPS_F64;

    #[cfg(feature = "simd")]
    use crate::calculus::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
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
                            EPS_F64,
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
                        EPS_F64,
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
                            EPS_F64,
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
    #[cfg(feature = "simd")]
    def_test_template!(BatchScalarF64<2>, DualBatchScalar<2>, 2);
    #[cfg(feature = "simd")]
    def_test_template!(BatchScalarF64<4>, DualBatchScalar<4>, 4);

    f64::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<2>::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<4>::run();
}
