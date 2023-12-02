use std::fmt::Debug;
use std::ops::{Add, Div, Index, Mul, Neg, Sub};
use std::process::Output;

use crate::calculus::types::matrix::IsMatrix;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::{IsVector, IsVectorLike};
use crate::calculus::types::{M, V};
use crate::tensor::layout::{HasShape, TensorShape};
use crate::tensor::mut_tensor::{
    InnerVecToMat, IsMutTensor, MutTensorDD, MutTensorDDRC, MutTensorDDR,
};
use crate::tensor::mut_view::IsMutTensorView;
use crate::tensor::view::IsTensorView;
use nalgebra::{ArrayStorage, Const, RawStorage};
use num_traits::{One, Zero};

use super::dual_matrix::DualM;
use super::dual_scalar::Dual;

#[derive(Clone, Debug)]
pub struct DualV<const ROWS: usize> {
    pub val: V<ROWS>,
    pub dij_val: Option<MutTensorDDR<f64, ROWS>>,
}

impl<const ROWS: usize> Neg for DualV<ROWS> {
    type Output = DualV<ROWS>;

    fn neg(self) -> Self::Output {
        let n_val = -self.val;
        if self.dij_val.is_none() {
            return Self {
                val: n_val,
                dij_val: None,
            };
        }

        todo!()
    }
}

impl<const ROWS: usize> Sub for DualV<ROWS> {
    type Output = DualV<ROWS>;

    fn sub(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl<const ROWS: usize> Add for DualV<ROWS> {
    type Output = DualV<ROWS>;

    fn add(self, rhs: Self) -> Self::Output {
        let dij_pair = Self::two_dx(self.dij_val.clone(), rhs.dij_val);
        if dij_pair.is_none() {
            return DualV {
                val: self.val + rhs.val,
                dij_val: None,
            };
        }

        let dij = dij_pair.unwrap();

        let mut dij_val = MutTensorDDR::<f64, ROWS>::from_shape(dij.lhs.dims());
        for i in 0..dij_val.dims()[0] {
            for j in 0..dij_val.dims()[1] {
                *dij_val.get_mut([i, j]) = dij.lhs.get([i, j]) + dij.rhs.get([i, j]);
            }
        }

        DualV {
            val: self.val + rhs.val,
            dij_val: Some(dij_val),
        }
    }
}

pub struct DijPair<const R0: usize, const R1: usize> {
    lhs: MutTensorDDR<f64, R0>,
    rhs: MutTensorDDR<f64, R1>,
}

impl<const R0: usize, const R1: usize> DijPair<R0, R1> {
    fn shape(&self) -> TensorShape<2> {
        self.lhs.dims()
    }
}

impl<const ROWS: usize> DualV<ROWS> {
    pub fn v(val: V<ROWS>) -> Self {
        let mut dij_val = MutTensorDDR::<f64, ROWS>::from_shape([ROWS, 1]);
        for i in 0..ROWS {
            dij_val.get_mut([i, 0])[(i, 0)] = 1.0;
        }

        Self {
            val,
            dij_val: Some(dij_val),
        }
    }

    pub fn two_dx<const R0: usize, const R1: usize>(
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

impl<const ROWS: usize> IsVectorLike for DualV<ROWS> {
    fn zero() -> Self {
        Self::c(V::zeros())
    }
}

impl<const ROWS: usize> IsVector<Dual, ROWS> for DualV<ROWS> {
    fn norm(&self) -> Dual {
        let nrm = self.val.norm();

        if self.dij_val.is_none() {
            return Dual {
                val: nrm,
                dij_val: None,
            };
        }

        let dij_val = self.dij_val.clone().unwrap();
        let mut dyn_mat = MutTensorDD::<f64>::from_shape(dij_val.dims());

        for i in 0..dij_val.dims()[0] {
            for j in 0..dij_val.dims()[1] {
                *dyn_mat.get_mut([i, j]) = (1.0 / nrm) * dij_val.get([i, j])[(i, j)] * self.val[i];
            }
        }

        Dual {
            val: nrm,
            dij_val: Some(dyn_mat),
        }
    }

    fn squared_norm(&self) -> Dual {
        let sq_nrm = self.val.squared_norm();

        if self.dij_val.is_none() {
            return Dual {
                val: sq_nrm,
                dij_val: None,
            };
        }

        let dij_val = self.dij_val.clone().unwrap();
        let mut dyn_mat = MutTensorDD::<f64>::from_shape(dij_val.dims());

        for i in 0..dij_val.dims()[0] {
            for j in 0..dij_val.dims()[1] {
                *dyn_mat.get_mut([i, j]) = dij_val.get([i, j])[(i, j)] * 2.0 * self.val[i];
            }
        }

        Dual {
            val: sq_nrm,
            dij_val: Some(dyn_mat),
        }
    }

    fn get(&self, idx: usize) -> Dual {
        if self.dij_val.is_none() {
            return Dual {
                val: self.val[idx],
                dij_val: None,
            };
        }
        let dij_val = self.dij_val.clone().unwrap();
        let mut r = MutTensorDD::<f64>::from_shape(dij_val.dims());

        for i in 0..dij_val.dims()[0] {
            for j in 0..dij_val.dims()[1] {
                *r.get_mut([i, j]) = dij_val.get([i, j])[idx];
            }
        }

        Dual {
            val: self.val[idx],
            dij_val: Some(r),
        }
    }

    fn from_array(duals: [Dual; ROWS]) -> Self {
        let mut shape = None;
        let mut val_v = V::<ROWS>::zeros();
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
                        r.get_mut([d0, d1])[(i, 0)] = d.dij_val.clone().unwrap().get([d0, d1]);
                    }
                }
            }
        }
        return DualV {
            val: val_v,
            dij_val: Some(r),
        };
    }

    fn from_c_array(vals: [f64; ROWS]) -> Self {
        return DualV {
            val: V::from_c_array(vals),
            dij_val: None,
        };
    }

    fn c(val: V<ROWS>) -> Self {
        Self { val, dij_val: None }
    }

    fn real(&self) -> &V<ROWS> {
        &self.val
    }

    fn get_fixed_rows<const R: usize>(&self, start: usize) -> DualV<R> {
        DualV {
            val: self.val.fixed_rows::<R>(start).into(),
            dij_val: match self.dij_val.as_ref() {
                Some(dij_val) => {
                    let mut r = MutTensorDDR::<f64, R>::from_shape(dij_val.dims());
                    for d0 in 0..dij_val.dims()[0] {
                        for d1 in 0..dij_val.dims()[1] {
                            *r.get_mut([d0, d1]) =
                                dij_val.get([d0, d1]).fixed_rows::<R>(start).into();
                        }
                    }
                    Some(r)
                }
                None => None,
            },
        }
    }

    fn to_mat(self) -> DualM<ROWS, 1> {
        DualM::<ROWS, 1> {
            val: self.val,
            dij_val: match self.dij_val {
                Some(dij) => Some(dij.inner_to_mat()),
                None => None,
            },
        }
    }

    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: DualV<R0>,
        bot_row: DualV<R1>,
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
        let maybe_dij = Self::two_dx(top_row.dij_val, bot_row.dij_val);

        Self {
            val: V::<ROWS>::block_vec2(top_row.val, bot_row.val),
            dij_val: match maybe_dij {
                Some(dij_val) => {
                    let mut r = MutTensorDDR::<f64, ROWS>::from_shape(dij_val.shape());
                    for d0 in 0..dij_val.shape()[0] {
                        for d1 in 0..dij_val.shape()[1] {
                            *r.get_mut([d0, d1]) = V::<ROWS>::block_vec2(
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

    fn set_c(&mut self, idx: usize, v: f64) {
        self.val[idx] = v;
        if self.dij_val.is_some() {
            let dij = &mut self.dij_val.as_mut().unwrap();
            for i in 0..dij.dims()[0] {
                for j in 0..dij.dims()[1] {
                    dij.get_mut([i, j])[idx] = 0.0;
                }
            }
        }
    }

    fn scaled(&self, v: Dual) -> Self {
        let s = self.val * v.val;
        Self {
            val: s,
            dij_val: match self.dij_val.as_ref() {
                Some(dij) => {
                    let mut dyn_mat = MutTensorDDR::from_shape(dij.dims());
                    dyn_mat.map(&self.dij_val.clone().unwrap(), |dij| dij * v.val);
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn dot(self, rhs: Self) -> Dual {
        let mut sum = Dual::c(0.0);

        for i in 0..ROWS {
            sum = sum + self.get(i) * rhs.get(i);
        }

        sum
    }
}
