use crate::dual::dual_matrix::DualM;
use crate::dual::dual_scalar::Dual;
use crate::types::scalar::IsScalar;
use crate::types::vector::IsVector;
use crate::types::vector::IsVectorLike;
use crate::types::VecF64;

use sophus_tensor::mut_tensor::InnerVecToMat;
use sophus_tensor::mut_tensor::MutTensorDD;
use sophus_tensor::mut_tensor::MutTensorDDR;
use sophus_tensor::mut_view::IsMutTensorLike;
use sophus_tensor::view::IsTensorLike;

use std::fmt::Debug;
use std::ops::Add;
use std::ops::Neg;
use std::ops::Sub;

/// Dual vector
#[derive(Clone)]
pub struct DualV<const ROWS: usize> {
    /// value - real vector
    pub val: VecF64<ROWS>,
    /// derivative - infinitesimal vector
    pub dij_val: Option<MutTensorDDR<f64, ROWS>>,
}

impl<const ROWS: usize> DualV<ROWS> {
    /// create a dual vector
    pub fn v(val: VecF64<ROWS>) -> Self {
        let mut dij_val = MutTensorDDR::<f64, ROWS>::from_shape([ROWS, 1]);
        for i in 0..ROWS {
            dij_val.mut_view().get_mut([i, 0])[(i, 0)] = 1.0;
        }

        Self {
            val,
            dij_val: Some(dij_val),
        }
    }

    fn binary_dij<
        const R0: usize,
        const R1: usize,
        F: FnMut(&VecF64<R0>) -> VecF64<ROWS>,
        G: FnMut(&VecF64<R1>) -> VecF64<ROWS>,
    >(
        lhs_dx: &Option<MutTensorDDR<f64, R0>>,
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

    fn binary_vs_dij<
        const R0: usize,
        F: FnMut(&VecF64<R0>) -> VecF64<ROWS>,
        G: FnMut(&f64) -> VecF64<ROWS>,
    >(
        lhs_dx: &Option<MutTensorDDR<f64, R0>>,
        rhs_dx: &Option<MutTensorDD<f64>>,
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

    fn two_dx<const R0: usize, const R1: usize>(
        mut lhs_dx: Option<MutTensorDDR<f64, R0>>,
        mut rhs_dx: Option<MutTensorDDR<f64, R1>>,
    ) -> Option<DijPair<R0, R1>> {
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
            lhs_dx = Some(MutTensorDDR::from_shape(rhs_dx.clone().unwrap().dims()))
        } else if rhs_dx.is_none() {
            rhs_dx = Some(MutTensorDDR::from_shape(lhs_dx.clone().unwrap().dims()))
        }

        Some(DijPair {
            lhs: lhs_dx.unwrap(),
            rhs: rhs_dx.unwrap(),
        })
    }
}

impl<const ROWS: usize> Neg for DualV<ROWS> {
    type Output = DualV<ROWS>;

    fn neg(self) -> Self::Output {
        DualV {
            val: -self.val,
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| -v)),
        }
    }
}

impl<const ROWS: usize> Sub for DualV<ROWS> {
    type Output = DualV<ROWS>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualV {
            val: self.val - rhs.val,
            dij_val: Self::binary_dij(&self.dij_val, &rhs.dij_val, |l_dij| *l_dij, |r_dij| -r_dij),
        }
    }
}

impl<const ROWS: usize> Add for DualV<ROWS> {
    type Output = DualV<ROWS>;

    fn add(self, rhs: Self) -> Self::Output {
        DualV {
            val: self.val + rhs.val,
            dij_val: Self::binary_dij(&self.dij_val, &rhs.dij_val, |l_dij| *l_dij, |r_dij| *r_dij),
        }
    }
}

struct DijPair<const R0: usize, const R1: usize> {
    lhs: MutTensorDDR<f64, R0>,
    rhs: MutTensorDDR<f64, R1>,
}

impl<const R0: usize, const R1: usize> DijPair<R0, R1> {
    fn shape(&self) -> [usize; 2] {
        self.lhs.dims()
    }
}

impl<const ROWS: usize> Debug for DualV<ROWS> {
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

impl<const ROWS: usize> IsVectorLike for DualV<ROWS> {
    fn zero() -> Self {
        Self::c(VecF64::zeros())
    }
}

impl<const ROWS: usize> IsVector<Dual, ROWS, 1> for DualV<ROWS> {
    fn set_c(&mut self, idx: usize, v: f64) {
        self.val[idx] = v;
        if self.dij_val.is_some() {
            let dij = &mut self.dij_val.as_mut().unwrap();
            for i in 0..dij.dims()[0] {
                for j in 0..dij.dims()[1] {
                    dij.mut_view().get_mut([i, j])[idx] = 0.0;
                }
            }
        }
    }

    fn norm(&self) -> Dual {
        self.clone().dot(self.clone()).sqrt()
    }

    fn squared_norm(&self) -> Dual {
        self.clone().dot(self.clone())
    }

    fn get(&self, idx: usize) -> Dual {
        Dual {
            val: self.val[idx],
            dij_val: self
                .dij_val
                .clone()
                .map(|dij_val| MutTensorDD::from_map(&dij_val.view(), |v| v[idx])),
        }
    }

    fn from_array(duals: [Dual; ROWS]) -> Self {
        let mut shape = None;
        let mut val_v = VecF64::<ROWS>::zeros();
        for i in 0..duals.len() {
            let d = duals.clone()[i].clone();

            val_v[i] = d.val;
            if d.dij_val.is_some() {
                shape = Some(d.dij_val.clone().unwrap().dims());
            }
        }

        if shape.is_none() {
            return DualV {
                val: val_v,
                dij_val: None,
            };
        }
        let shape = shape.unwrap();

        let mut r = MutTensorDDR::<f64, ROWS>::from_shape(shape);

        for i in 0..duals.len() {
            let d = duals.clone()[i].clone();
            if d.dij_val.is_some() {
                for d0 in 0..shape[0] {
                    for d1 in 0..shape[1] {
                        r.mut_view().get_mut([d0, d1])[(i, 0)] =
                            d.dij_val.clone().unwrap().get([d0, d1]);
                    }
                }
            }
        }
        DualV {
            val: val_v,
            dij_val: Some(r),
        }
    }

    fn from_c_array(vals: [f64; ROWS]) -> Self {
        DualV {
            val: VecF64::from_c_array(vals),
            dij_val: None,
        }
    }

    fn c(val: VecF64<ROWS>) -> Self {
        Self { val, dij_val: None }
    }

    fn real(&self) -> &VecF64<ROWS> {
        &self.val
    }

    fn get_fixed_rows<const R: usize>(&self, start: usize) -> DualV<R> {
        DualV {
            val: self.val.fixed_rows::<R>(start).into(),
            dij_val: self.dij_val.clone().map(|dij_val| {
                MutTensorDDR::from_map(&dij_val.view(), |v| v.fixed_rows::<R>(start).into())
            }),
        }
    }

    fn to_mat(self) -> DualM<ROWS, 1> {
        DualM::<ROWS, 1> {
            val: self.val,
            dij_val: self.dij_val.map(|dij| dij.inner_vec_to_mat()),
        }
    }

    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: DualV<R0>,
        bot_row: DualV<R1>,
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);

        let maybe_dij = Self::two_dx(top_row.dij_val, bot_row.dij_val);
        Self {
            val: VecF64::<ROWS>::block_vec2(top_row.val, bot_row.val),
            dij_val: match maybe_dij {
                Some(dij_val) => {
                    let mut r = MutTensorDDR::<f64, ROWS>::from_shape(dij_val.shape());
                    for d0 in 0..dij_val.shape()[0] {
                        for d1 in 0..dij_val.shape()[1] {
                            *r.mut_view().get_mut([d0, d1]) = VecF64::<ROWS>::block_vec2(
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

    fn scaled(&self, s: Dual) -> Self {
        DualV {
            val: self.val * s.val,
            dij_val: Self::binary_vs_dij(
                &self.dij_val,
                &s.dij_val,
                |l_dij| l_dij * s.val,
                |r_dij| self.val * *r_dij,
            ),
        }
    }

    fn dot(self, rhs: Self) -> Dual {
        let mut sum = Dual::c(0.0);

        for i in 0..ROWS {
            sum = sum + self.get(i) * rhs.get(i);
        }

        sum
    }

    fn normalized(&self) -> Self {
        self.clone().scaled(Dual::c(1.0) / self.norm())
    }
}

mod test {

    #[test]
    fn scalar_valued() {
        use crate::dual::dual_scalar::Dual;
        use crate::dual::dual_vector::DualV;
        use crate::maps::scalar_valued_maps::ScalarValuedMapFromVector;
        use crate::maps::vector_valued_maps::VectorValuedMapFromVector;
        use crate::points::example_points;
        use crate::types::scalar::IsScalar;
        use crate::types::vector::IsVector;
        use crate::types::VecF64;
        use sophus_tensor::view::IsTensorLike;

        let points: Vec<VecF64<4>> = example_points::<f64, 4>();

        for p in points.clone() {
            for p1 in points.clone() {
                {
                    fn dot_fn<S: IsScalar<1>>(x: S::Vector<4>, y: S::Vector<4>) -> S {
                        x.dot(y)
                    }
                    let finite_diff = ScalarValuedMapFromVector::sym_diff_quotient(
                        |x| dot_fn::<f64>(x, p1),
                        p,
                        1e-6,
                    );
                    let auto_grad = ScalarValuedMapFromVector::fw_autodiff(
                        |x| dot_fn::<Dual>(x, DualV::<4>::c(p1)),
                        p,
                    );
                    approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                }

                fn dot_fn<S: IsScalar<1>>(x: S::Vector<4>, s: S) -> S::Vector<4> {
                    x.scaled(s)
                }
                let finite_diff = VectorValuedMapFromVector::sym_diff_quotient(
                    |x| dot_fn::<f64>(x, 0.99),
                    p,
                    1e-6,
                );
                let auto_grad =
                    VectorValuedMapFromVector::fw_autodiff(|x| dot_fn::<Dual>(x, Dual::c(0.99)), p);
                for i in 0..finite_diff.dims()[0] {
                    approx::assert_abs_diff_eq!(
                        finite_diff.get([i]),
                        auto_grad.get([i]),
                        epsilon = 0.0001
                    );
                }

                let finite_diff = VectorValuedMapFromVector::sym_diff_quotient(
                    |x| dot_fn::<f64>(p1, x[0]),
                    p,
                    1e-6,
                );
                let auto_grad = VectorValuedMapFromVector::fw_autodiff(
                    |x| dot_fn::<Dual>(DualV::c(p1), x.get(0)),
                    p,
                );
                for i in 0..finite_diff.dims()[0] {
                    approx::assert_abs_diff_eq!(
                        finite_diff.get([i]),
                        auto_grad.get([i]),
                        epsilon = 0.0001
                    );
                }
            }
        }
    }
}
