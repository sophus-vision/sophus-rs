use crate::dual::dual_scalar::Dual;
use crate::dual::dual_vector::DualV;
use crate::types::matrix::IsMatrix;
use crate::types::vector::IsVectorLike;
use crate::types::MatF64;
use crate::types::VecF64;

use sophus_tensor::mut_tensor::MutTensorDD;
use sophus_tensor::mut_tensor::MutTensorDDR;
use sophus_tensor::mut_tensor::MutTensorDDRC;
use sophus_tensor::mut_view::IsMutTensorLike;
use sophus_tensor::view::IsTensorLike;

use std::fmt::Debug;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

/// Dual matrix
#[derive(Clone)]
pub struct DualM<const ROWS: usize, const COLS: usize> {
    /// value - real matrix
    pub val: MatF64<ROWS, COLS>,
    /// derivative - infinitesimal matrix
    pub dij_val: Option<MutTensorDDRC<f64, ROWS, COLS>>,
}

impl<const ROWS: usize, const COLS: usize> DualM<ROWS, COLS> {
    fn binary_mm_dij<
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
    ) -> Option<DijPairM<R1, C1, R2, C2>> {
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
    ) -> Option<DijPairMV<ROWS, COLS>> {
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

        Some(DijPairMV::<ROWS, COLS> {
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

impl<const ROWS: usize, const COLS: usize> IsMatrix<Dual, ROWS, COLS, 1> for DualM<ROWS, COLS> {
    fn mat_mul<const COLS2: usize>(&self, rhs: DualM<COLS, COLS2>) -> DualM<ROWS, COLS2> {
        DualM {
            val: self.val * rhs.val,
            dij_val: DualM::binary_mm_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| l_dij * rhs.val,
                |r_dij| self.val * r_dij,
            ),
        }
    }

    fn c(val: MatF64<ROWS, COLS>) -> Self {
        Self { val, dij_val: None }
    }

    fn scaled(&self, s: Dual) -> Self {
        DualM {
            val: self.val * s.val,
            dij_val: DualM::binary_ms_dij(
                &self.dij_val,
                &s.dij_val,
                |l_dij| l_dij * s.val,
                |r_dij| self.val * *r_dij,
            ),
        }
    }

    fn identity() -> Self {
        DualM::c(MatF64::<ROWS, COLS>::identity())
    }

    fn get(&self, idx: (usize, usize)) -> Dual {
        Dual {
            val: self.val[idx],
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDD::from_map(&dij_val.view(), |v| v[idx])),
        }
    }

    fn from_array2(duals: [[Dual; COLS]; ROWS]) -> Self {
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
            return DualM {
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
        DualM {
            val: val_mat,
            dij_val: Some(r),
        }
    }

    fn real(&self) -> &MatF64<ROWS, COLS> {
        &self.val
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (
            <Dual as crate::types::scalar::IsScalar<1>>::Matrix<R0, C0>,
            <Dual as crate::types::scalar::IsScalar<1>>::Matrix<R0, C1>,
        ),
        bot_row: (
            <Dual as crate::types::scalar::IsScalar<1>>::Matrix<R1, C0>,
            <Dual as crate::types::scalar::IsScalar<1>>::Matrix<R1, C1>,
        ),
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
        assert_eq!(C0 + C1, COLS);

        Self::block_mat2x1(
            DualM::<R0, COLS>::block_mat1x2(top_row.0, top_row.1),
            DualM::<R1, COLS>::block_mat1x2(bot_row.0, bot_row.1),
        )
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: DualM<R0, COLS>,
        bot_row: DualM<R1, COLS>,
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
        left_col: <Dual as crate::types::scalar::IsScalar<1>>::Matrix<ROWS, C0>,
        righ_col: <Dual as crate::types::scalar::IsScalar<1>>::Matrix<ROWS, C1>,
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
    ) -> DualM<R, C> {
        DualM {
            val: self.val.get_fixed_submat(start_r, start_c),
            dij_val: self.dij_val.clone().map(|dij_val| {
                MutTensorDDRC::from_map(&dij_val.view(), |v| v.get_fixed_submat(start_r, start_c))
            }),
        }
    }

    fn get_col_vec(&self, start_r: usize) -> DualV<ROWS> {
        DualV {
            val: self.val.get_col_vec(start_r),
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| v.get_col_vec(start_r))),
        }
    }

    fn get_row_vec(&self, c: usize) -> DualV<ROWS> {
        DualV {
            val: self.val.get_row_vec(c),
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| v.get_row_vec(c))),
        }
    }

    fn from_c_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        DualM {
            val: MatF64::from_c_array2(vals),
            dij_val: None,
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Add for DualM<ROWS, COLS> {
    type Output = DualM<ROWS, COLS>;

    fn add(self, rhs: Self) -> Self::Output {
        DualM {
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

impl<const ROWS: usize, const COLS: usize> Sub for DualM<ROWS, COLS> {
    type Output = DualM<ROWS, COLS>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualM {
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

impl<const ROWS: usize, const COLS: usize> Neg for DualM<ROWS, COLS> {
    type Output = DualM<ROWS, COLS>;

    fn neg(self) -> Self::Output {
        DualM {
            val: -self.val,
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDDRC::from_map(&dij_val.view(), |v| -v)),
        }
    }
}

impl<const ROWS: usize, const COLS: usize> IsVectorLike for DualM<ROWS, COLS> {
    fn zero() -> Self {
        Self::c(MatF64::zeros())
    }
}

impl<const ROWS: usize, const COLS: usize> Mul<DualV<COLS>> for DualM<ROWS, COLS> {
    type Output = DualV<ROWS>;

    fn mul(self, rhs: DualV<COLS>) -> Self::Output {
        DualV {
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

impl<const ROWS: usize, const COLS: usize> Debug for DualM<ROWS, COLS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dij_val.is_some() {
            f.debug_struct("Dual")
                .field("val", &self.val)
                .field("dij_val", &self.dij_val.as_ref().unwrap().elem_view())
                .finish()
        } else {
            f.debug_struct("Dual").field("val", &self.val).finish()
        }
    }
}

/// Pair of dual matrices
pub struct DijPairM<const ROWS: usize, const COLS: usize, const ROWS2: usize, const COLS2: usize> {
    lhs: MutTensorDDRC<f64, ROWS, COLS>,
    rhs: MutTensorDDRC<f64, ROWS2, COLS2>,
}

impl<const ROWS: usize, const COLS: usize, const ROWS2: usize, const COLS2: usize>
    DijPairM<ROWS, COLS, ROWS2, COLS2>
{
    fn shape(&self) -> [usize; 2] {
        self.lhs.dims()
    }
}

/// Pair of dual matrices
pub struct DijPairMV<const ROWS: usize, const COLS: usize> {
    /// left hand side
    pub lhs: MutTensorDDRC<f64, ROWS, COLS>,
    /// right hand side
    pub rhs: MutTensorDDR<f64, COLS>,
}

mod test {

    #[test]
    fn matrix_dual() {
        use crate::dual::dual_matrix::DualM;
        use crate::dual::dual_scalar::Dual;
        use crate::maps::matrix_valued_maps::MatrixValuedMapFromMatrix;
        use crate::types::matrix::IsMatrix;
        use crate::types::scalar::IsScalar;
        use crate::types::MatF64;
        use sophus_tensor::view::IsTensorLike;

        let m_2x4 = MatF64::<2, 4>::new_random();
        let m_4x1 = MatF64::<4, 1>::new_random();

        fn mat_mul_fn<S: IsScalar<1>>(x: S::Matrix<2, 4>, y: S::Matrix<4, 1>) -> S::Matrix<2, 1> {
            x.mat_mul(y)
        }
        let finite_diff = MatrixValuedMapFromMatrix::sym_diff_quotient(
            |x| mat_mul_fn::<f64>(x, m_4x1),
            m_2x4,
            1e-6,
        );
        let auto_grad = MatrixValuedMapFromMatrix::fw_autodiff(
            |x| mat_mul_fn::<Dual>(x, DualM::c(m_4x1)),
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
            |x| mat_mul_fn::<f64>(m_2x4, x),
            m_4x1,
            1e-6,
        );
        let auto_grad = MatrixValuedMapFromMatrix::fw_autodiff(
            |x| mat_mul_fn::<Dual>(DualM::c(m_2x4), x),
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

        fn mat_mul2_fn<S: IsScalar<1>>(x: S::Matrix<4, 4>) -> S::Matrix<4, 4> {
            x.mat_mul(x.clone())
        }

        let m_4x4 = MatF64::<4, 4>::new_random();

        let finite_diff =
            MatrixValuedMapFromMatrix::sym_diff_quotient(mat_mul2_fn::<f64>, m_4x4, 1e-6);
        let auto_grad = MatrixValuedMapFromMatrix::fw_autodiff(mat_mul2_fn::<Dual>, m_4x4);

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
