use crate::dual::dual_matrix::DualM;
use crate::dual::dual_vector::DualV;
use crate::types::scalar::IsScalar;

use sophus_tensor::mut_tensor::InnerScalarToVec;
use sophus_tensor::mut_tensor::MutTensorDD;
use sophus_tensor::view::IsTensorLike;

use num_traits::One;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

/// Dual number
#[derive(Clone)]
pub struct Dual {
    /// value - real number
    pub val: f64,

    /// derivative - infinitesimal number
    pub dij_val: Option<MutTensorDD<f64>>,
}

impl AsRef<Dual> for Dual {
    fn as_ref(&self) -> &Dual {
        self
    }
}

impl One for Dual {
    fn one() -> Self {
        Dual::c(1.0)
    }
}

impl Zero for Dual {
    fn zero() -> Self {
        Dual::c(0.0)
    }

    fn is_zero(&self) -> bool {
        self.val == 0.0
    }
}

impl Dual {
    /// create a dual number
    pub fn v(val: f64) -> Self {
        let dij_val = MutTensorDD::<f64>::from_shape_and_val([1, 1], 1.0);
        Self {
            val,
            dij_val: Some(dij_val),
        }
    }

    fn binary_dij<F: FnMut(&f64) -> f64, G: FnMut(&f64) -> f64>(
        lhs_dx: &Option<MutTensorDD<f64>>,
        rhs_dx: &Option<MutTensorDD<f64>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDD<f64>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDD::from_map(&rhs_dij.view(), |r_dij: &f64| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDD::from_map(&lhs_dij.view(), |l_dij: &f64| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat = MutTensorDD::from_map2(
                    &lhs_dij.view(),
                    &rhs_dij.view(),
                    |l_dij: &f64, r_dij: &f64| left_op(l_dij) + right_op(r_dij),
                );
                Some(dyn_mat)
            }
        }
    }
}

impl Neg for Dual {
    type Output = Dual;

    fn neg(self) -> Self {
        Dual {
            val: -self.val,
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat = MutTensorDD::from_map(&dij_val.view(), |v: &f64| -v);

                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }
}

impl PartialEq for Dual {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl PartialOrd for Dual {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl From<f64> for Dual {
    fn from(value: f64) -> Self {
        Dual::c(value)
    }
}

impl Debug for Dual {
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

impl IsScalar<1> for Dual {
    type Vector<const ROWS: usize> = DualV<ROWS>;
    type Matrix<const ROWS: usize, const COLS: usize> = DualM<ROWS, COLS>;

    fn c(val: f64) -> Self {
        Self { val, dij_val: None }
    }

    fn cos(self) -> Dual {
        Dual {
            val: self.val.cos(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &f64| -dij * self.val.sin());
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn sin(self) -> Dual {
        Dual {
            val: self.val.sin(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &f64| dij * self.val.cos());
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn value(self) -> f64 {
        self.val
    }

    fn abs(self) -> Self {
        Dual {
            val: self.val.abs(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat = MutTensorDD::from_map(&dij_val.view(), |dij: &f64| {
                        *dij * self.val.signum()
                    });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn atan2(self, rhs: Self) -> Self {
        let inv_sq_nrm: f64 = 1.0 / (self.val * self.val + rhs.val * rhs.val);
        Dual {
            val: self.val.atan2(rhs.val),
            dij_val: Self::binary_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| inv_sq_nrm * (l_dij * rhs.val),
                |r_dij| -inv_sq_nrm * (self.val * r_dij),
            ),
        }
    }

    fn real(&self) -> f64 {
        self.val
    }

    fn sqrt(self) -> Self {
        let sqrt = self.val.sqrt();
        Dual {
            val: sqrt,
            dij_val: match self.dij_val {
                Some(dij) => {
                    let out_dij =
                        MutTensorDD::from_map(&dij.view(), |dij: &f64| dij * 1.0 / (2.0 * sqrt));
                    Some(out_dij)
                }
                None => None,
            },
        }
    }

    fn to_vec(self) -> DualV<1> {
        DualV::<1> {
            val: self.val.to_vec(),
            dij_val: match self.dij_val {
                Some(dij) => {
                    let tmp = dij.inner_scalar_to_vec();
                    Some(tmp)
                }
                None => None,
            },
        }
    }

    fn tan(self) -> Self {
        Dual {
            val: self.val.tan(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let c = self.val.cos();
                    let sec_squared = 1.0 / (c * c);
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &f64| *dij * sec_squared);
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn acos(self) -> Self {
        Dual {
            val: self.val.acos(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dval = -1.0 / (1.0 - self.val * self.val).sqrt();
                    let dyn_mat = MutTensorDD::from_map(&dij_val.view(), |dij: &f64| *dij * dval);
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn asin(self) -> Self {
        Dual {
            val: self.val.asin(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dval = 1.0 / (1.0 - self.val * self.val).sqrt();
                    let dyn_mat = MutTensorDD::from_map(&dij_val.view(), |dij: &f64| *dij * dval);
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn atan(self) -> Self {
        Dual {
            val: self.val.atan(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dval = 1.0 / (1.0 + self.val * self.val);
                    let dyn_mat = MutTensorDD::from_map(&dij_val.view(), |dij: &f64| *dij * dval);
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn fract(self) -> Self {
        Dual {
            val: self.val.fract(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat = MutTensorDD::from_map(&dij_val.view(), |dij: &f64| *dij);
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn floor(&self) -> i64 {
        self.val.floor() as i64
    }
}

impl Add<Dual> for Dual {
    type Output = Dual;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl Add<&Dual> for Dual {
    type Output = Dual;
    fn add(self, rhs: &Self) -> Self::Output {
        let r = self.val + rhs.val;

        Dual {
            val: r,
            dij_val: Self::binary_dij(&self.dij_val, &rhs.dij_val, |l_dij| *l_dij, |r_dij| *r_dij),
        }
    }
}

impl Mul<Dual> for Dual {
    type Output = Dual;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl Mul<&Dual> for Dual {
    type Output = Dual;
    fn mul(self, rhs: &Self) -> Self::Output {
        let r = self.val * rhs.val;

        Dual {
            val: r,
            dij_val: Self::binary_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| l_dij * rhs.val,
                |r_dij| r_dij * self.val,
            ),
        }
    }
}

impl Div<Dual> for Dual {
    type Output = Dual;
    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl Div<&Dual> for Dual {
    type Output = Dual;
    fn div(self, rhs: &Self) -> Self::Output {
        let rhs_inv = 1.0 / rhs.val;
        Dual {
            val: self.val * rhs_inv,
            dij_val: Self::binary_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| l_dij * rhs_inv,
                |r_dij| -self.val * r_dij * rhs_inv * rhs_inv,
            ),
        }
    }
}

impl Sub<Dual> for Dual {
    type Output = Dual;
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl Sub<&Dual> for Dual {
    type Output = Dual;
    fn sub(self, rhs: &Self) -> Self::Output {
        Dual {
            val: self.val - rhs.val,
            dij_val: Self::binary_dij(&self.dij_val, &rhs.dij_val, |l_dij| *l_dij, |r_dij| -r_dij),
        }
    }
}

mod test {

    #[test]
    fn scalar_valued() {
        use crate::dual::dual_scalar::Dual;
        use crate::maps::curves::ScalarValuedCurve;
        use crate::types::scalar::IsScalar;

        for i in 1..10 {
            let a = 0.1 * (i as f64);

            // f(x) = x^2
            fn square_fn<S: IsScalar<1>>(x: S) -> S {
                x.clone() * x
            }
            let finite_diff = ScalarValuedCurve::sym_diff_quotient(square_fn, a, 1e-6);
            let auto_grad = ScalarValuedCurve::fw_autodiff(square_fn, a);
            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

            {
                fn add_fn<S: IsScalar<1>>(x: S, y: S) -> S {
                    x + y
                }
                let b = 12.0;
                let finite_diff =
                    ScalarValuedCurve::sym_diff_quotient(|x| add_fn::<f64>(x, b), a, 1e-6);
                let auto_grad =
                    ScalarValuedCurve::fw_autodiff(|x| add_fn::<Dual>(x, Dual::c(b)), a);
                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                let b = 12.0;
                let finite_diff =
                    ScalarValuedCurve::sym_diff_quotient(|x| add_fn::<f64>(b, x), a, 1e-6);
                let auto_grad =
                    ScalarValuedCurve::fw_autodiff(|x| add_fn::<Dual>(Dual::c(b), x), a);
                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
            }

            {
                fn sub_fn<S: IsScalar<1>>(x: S, y: S) -> S {
                    x - y
                }
                let b = 12.0;
                let finite_diff =
                    ScalarValuedCurve::sym_diff_quotient(|x| sub_fn::<f64>(x, b), a, 1e-6);
                let auto_grad =
                    ScalarValuedCurve::fw_autodiff(|x| sub_fn::<Dual>(x, Dual::c(b)), a);
                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                let b = 12.0;
                let finite_diff =
                    ScalarValuedCurve::sym_diff_quotient(|x| sub_fn::<f64>(b, x), a, 1e-6);
                let auto_grad =
                    ScalarValuedCurve::fw_autodiff(|x| sub_fn::<Dual>(Dual::c(b), x), a);
                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
            }

            {
                fn mul_fn<S: IsScalar<1>>(x: S, y: S) -> S {
                    x * y
                }
                let b = 12.0;
                let finite_diff =
                    ScalarValuedCurve::sym_diff_quotient(|x| mul_fn::<f64>(x, b), a, 1e-6);
                let auto_grad =
                    ScalarValuedCurve::fw_autodiff(|x| mul_fn::<Dual>(x, Dual::c(b)), a);
                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                let b = 12.0;
                let finite_diff =
                    ScalarValuedCurve::sym_diff_quotient(|x| mul_fn::<f64>(x, b), a, 1e-6);
                let auto_grad =
                    ScalarValuedCurve::fw_autodiff(|x| mul_fn::<Dual>(x, Dual::c(b)), a);
                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
            }

            {
                fn div_fn<S: IsScalar<1>>(x: S, y: S) -> S {
                    x / y
                }
                let b = 12.0;
                let finite_diff =
                    ScalarValuedCurve::sym_diff_quotient(|x| div_fn::<f64>(x, b), a, 1e-6);
                let auto_grad =
                    ScalarValuedCurve::fw_autodiff(|x| div_fn::<Dual>(x, Dual::c(b)), a);
                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                let b = 12.0;
                let finite_diff =
                    ScalarValuedCurve::sym_diff_quotient(|x| div_fn::<f64>(x, b), a, 1e-6);
                let auto_grad =
                    ScalarValuedCurve::fw_autodiff(|x| div_fn::<Dual>(x, Dual::c(b)), a);
                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                let finite_diff =
                    ScalarValuedCurve::sym_diff_quotient(|x| div_fn::<f64>(b, x), a, 1e-6);
                let auto_grad =
                    ScalarValuedCurve::fw_autodiff(|x| div_fn::<Dual>(Dual::c(b), x), a);
                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                let b = 12.0;
                let finite_diff =
                    ScalarValuedCurve::sym_diff_quotient(|x| div_fn::<f64>(x, b), a, 1e-6);
                let auto_grad =
                    ScalarValuedCurve::fw_autodiff(|x| div_fn::<Dual>(x, Dual::c(b)), a);
                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
            }
        }
    }
}
