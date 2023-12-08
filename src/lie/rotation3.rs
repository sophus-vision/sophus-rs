use std::marker::PhantomData;

use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::types::matrix::IsMatrix;
use crate::calculus::types::params::HasParams;
use crate::calculus::types::params::ParamsImpl;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::cross;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::M;
use crate::lie;
use crate::manifold::{self};

use super::lie_group::LieGroup;
use super::traits::IsLieGroupImpl;
use super::traits::IsTranslationProductGroup;

#[derive(Debug, Copy, Clone)]
pub struct Rotation3Impl<S: IsScalar> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar> ParamsImpl<S, 4> for Rotation3Impl<S> {
    fn params_examples() -> Vec<S::Vector<4>> {
        let mut params = vec![];

        params.push(
            Rotation3::<S>::exp(&S::Vector::<3>::from_c_array([0.0, 0.0, 0.0]))
                .params()
                .clone(),
        );
        params.push(
            Rotation3::<S>::exp(&S::Vector::<3>::from_c_array([0.1, 0.5, -0.1]))
                .params()
                .clone(),
        );
        params.push(
            Rotation3::<S>::exp(&S::Vector::<3>::from_c_array([0.0, 0.2, 1.0]))
                .params()
                .clone(),
        );
        params.push(
            Rotation3::<S>::exp(&S::Vector::<3>::from_c_array([-0.2, 0.0, 0.8]))
                .params()
                .clone(),
        );
        params
    }

    fn invalid_params_examples() -> Vec<S::Vector<4>> {
        vec![
            S::Vector::<4>::from_array([0.0.into(), 0.0.into(), 0.0.into(), 0.0.into()]),
            S::Vector::<4>::from_array([0.5.into(), 0.5.into(), 0.5.into(), 0.0.into()]),
            S::Vector::<4>::from_array([0.5.into(), (-0.5).into(), 0.5.into(), 1.0.into()]),
        ]
    }

    fn are_params_valid(params: &S::Vector<4>) -> bool {
        let norm = params.norm().real();
        (norm - 1.0).abs() < 1e-6
    }
}

impl<S: IsScalar> manifold::traits::TangentImpl<S, 3> for Rotation3Impl<S> {
    fn tangent_examples() -> Vec<S::Vector<3>> {
        vec![
            S::Vector::<3>::from_c_array([0.0, 0.0, 0.0]),
            S::Vector::<3>::from_c_array([1.0, 0.0, 0.0]),
            S::Vector::<3>::from_c_array([0.0, 1.0, 0.0]),
            S::Vector::<3>::from_c_array([0.0, 0.0, 1.0]),
            S::Vector::<3>::from_c_array([0.5, 0.5, 0.5]),
            S::Vector::<3>::from_c_array([-0.5, -0.5, -0.5]),
        ]
    }
}

impl<S: IsScalar> IsLieGroupImpl<S, 3, 4, 3, 3> for Rotation3Impl<S> {
    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = true;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<4> {
        S::Vector::<4>::from_c_array([1.0, 0.0, 0.0, 0.0])
    }

    fn adj(params: &S::Vector<4>) -> S::Matrix<3, 3> {
        Self::matrix(params)
    }

    fn exp(omega: &S::Vector<3>) -> S::Vector<4> {
        const EPS: f64 = 1e-8;
        let theta_sq = omega.squared_norm();

        let (imag_factor, real_factor) = if theta_sq.real() < EPS * EPS {
            let theta_po4 = theta_sq.clone() * theta_sq.clone();
            (
                S::c(0.5) - S::c(1.0 / 48.0) * theta_sq.clone()
                    + S::c(1.0 / 3840.0) * theta_po4.clone(),
                S::c(1.0) - S::c(1.0 / 8.0) * theta_sq + S::c(1.0 / 384.0) * theta_po4,
            )
        } else {
            let theta = theta_sq.sqrt();
            let half_theta: S = S::c(0.5) * theta.clone();
            (half_theta.clone().sin() / theta, half_theta.cos())
        };
        S::Vector::<4>::from_array([
            real_factor,
            imag_factor.clone() * omega.get(0),
            imag_factor.clone() * omega.get(1),
            imag_factor * omega.get(2),
        ])
    }

    fn log(params: &S::Vector<4>) -> S::Vector<3> {
        const EPS: f64 = 1e-8;
        let ivec: S::Vector<3> = params.get_fixed_rows::<3>(1);

        let squared_n = ivec.squared_norm();
        let w = params.get(0);
        let w_real = w.real();

        let two_atan_nbyd_by_n: S = if squared_n.real() < EPS * EPS {
            assert!(
                w_real.abs() > EPS,
                "|params| should be close to 1. (w = {})",
                w_real
            );
            let w_sq = w.clone() * w.clone();
            S::c(2.0) / w.clone() - S::c(2.0 / 3.0) * squared_n / (w_sq * w)
        } else {
            let n = squared_n.sqrt();
            let atan_nbyw = if w_real < 0.0 {
                -n.clone().atan2(-w)
            } else {
                n.clone().atan2(w)
            };
            S::c(2.0) * atan_nbyw / n
        };
        ivec.scaled(two_atan_nbyd_by_n)
    }

    fn hat(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        let o0 = omega.get(0);
        let o1 = omega.get(1);
        let o2 = omega.get(2);

        S::Matrix::from_array2([
            [S::zero(), -o2.clone(), o1.clone()],
            [o2, S::zero(), -o0.clone()],
            [-o1, o0, S::zero()],
        ])
    }

    fn vee(omega_hat: &S::Matrix<3, 3>) -> S::Vector<3> {
        S::Vector::<3>::from_array([
            omega_hat.get((2, 1)),
            omega_hat.get((0, 2)),
            omega_hat.get((1, 0)),
        ])
    }

    fn inverse(params: &S::Vector<4>) -> S::Vector<4> {
        S::Vector::from_array([
            params.get(0),
            -params.get(1),
            -params.get(2),
            -params.get(3),
        ])
    }

    fn transform(params: &S::Vector<4>, point: &S::Vector<3>) -> S::Vector<3> {
        Self::matrix(params) * point.clone()
    }

    fn to_ambient(point: &S::Vector<3>) -> S::Vector<3> {
        point.clone()
    }

    fn compact(params: &S::Vector<4>) -> S::Matrix<3, 3> {
        Self::matrix(params)
    }

    fn matrix(params: &S::Vector<4>) -> S::Matrix<3, 3> {
        let ivec = params.get_fixed_rows::<3>(1);
        let re = params.get(0);

        let unit_x = S::Vector::from_c_array([1.0, 0.0, 0.0]);
        let unit_y = S::Vector::from_c_array([0.0, 1.0, 0.0]);
        let unit_z = S::Vector::from_c_array([0.0, 0.0, 1.0]);

        let two = S::c(2.0);

        let uv_x: S::Vector<3> = cross::<S>(ivec.clone(), unit_x.clone()).scaled(two.clone());
        let uv_y: S::Vector<3> = cross::<S>(ivec.clone(), unit_y.clone()).scaled(two.clone());
        let uv_z: S::Vector<3> = cross::<S>(ivec.clone(), unit_z.clone()).scaled(two);

        let col_x = unit_x + cross::<S>(ivec.clone(), uv_x.clone()) + uv_x.scaled(re.clone());
        let col_y = unit_y + cross::<S>(ivec.clone(), uv_y.clone()) + uv_y.scaled(re.clone());
        let col_z = unit_z + cross::<S>(ivec.clone(), uv_z.clone()) + uv_z.scaled(re.clone());

        S::Matrix::block_mat1x2::<1, 2>(
            col_x.to_mat(),
            S::Matrix::block_mat1x2(col_y.to_mat(), col_z.to_mat()),
        )
    }

    fn ad(tangent: &S::Vector<3>) -> S::Matrix<3, 3> {
        Self::hat(tangent)
    }

    type GenG<S2: IsScalar> = Rotation3Impl<S2>;
    type RealG = Rotation3Impl<f64>;
    type DualG = Rotation3Impl<Dual>;

    fn group_mul(lhs_params: &S::Vector<4>, rhs_params: &S::Vector<4>) -> S::Vector<4> {
        let lhs_re = lhs_params.get(0);
        let rhs_re = rhs_params.get(0);

        let lhs_ivec = lhs_params.get_fixed_rows::<3>(1);
        let rhs_ivec = rhs_params.get_fixed_rows::<3>(1);

        let re = lhs_re.clone() * rhs_re.clone() - lhs_ivec.clone().dot(rhs_ivec.clone());
        let ivec =
            rhs_ivec.scaled(lhs_re) + lhs_ivec.scaled(rhs_re) + cross::<S>(lhs_ivec, rhs_ivec);

        S::Vector::block_vec2(re.to_vec(), ivec)
    }

    fn has_shortest_path_ambiguity(params: &<S as IsScalar>::Vector<4>) -> bool {
        let theta = Self::log(params).real().norm();
        (theta - std::f64::consts::PI).abs() < 1e-5
    }
}

impl lie::traits::IsF64LieGroupImpl<3, 4, 3, 3> for Rotation3Impl<f64> {
    fn dx_exp_x_at_0() -> M<4, 3> {
        M::from_c_array2([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ])
    }

    fn dx_exp_x_times_point_at_0(point: crate::calculus::types::V<3>) -> M<3, 3> {
        Self::hat(&-point)
    }
}

impl<S: IsScalar> lie::traits::IsLieFactorGroupImpl<S, 3, 4, 3, 3> for Rotation3Impl<S> {
    type GenFactorG<S2: IsScalar> = Rotation3Impl<S2>;
    type RealFactorG = Rotation3Impl<f64>;
    type DualFactorG = Rotation3Impl<Dual>;

    fn mat_v(_params: &S::Vector<4>, omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        let theta_sq = omega.squared_norm();
        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S>::hat(omega);
        let mat_omega_sq = mat_omega.clone().mat_mul(mat_omega.clone());
        if theta_sq.real() < 1e-6 {
            S::Matrix::<3, 3>::identity() + mat_omega.scaled(S::c(0.5))
        } else {
            let theta = theta_sq.clone().sqrt();
            S::Matrix::<3, 3>::identity()
                + mat_omega.scaled((S::c(1.0) - theta.clone().cos()) / theta_sq.clone())
                + mat_omega_sq.scaled((theta.clone() - theta.clone().sin()) / (theta_sq * theta))
        }
    }

    fn mat_v_inverse(_params: &S::Vector<4>, tangent: &S::Vector<3>) -> S::Matrix<3, 3> {
        let theta_sq = tangent.clone().dot(tangent.clone());
        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S>::hat(tangent);
        let mat_omega_sq = mat_omega.clone().mat_mul(mat_omega.clone());

        if theta_sq.real() < 1e-6 {
            S::Matrix::<3, 3>::identity() - mat_omega.scaled(S::c(0.5))
                + mat_omega_sq.scaled(S::c(1. / 12.))
        } else {
            let theta = theta_sq.clone().sqrt();
            let half_theta = S::c(0.5) * theta.clone();

            S::Matrix::<3, 3>::identity() - mat_omega.scaled(S::c(0.5))
                + mat_omega_sq.scaled(
                    (S::c(1.0)
                        - (S::c(0.5) * theta.clone() * half_theta.clone().cos())
                            / half_theta.sin())
                        / (theta.clone() * theta),
                )
        }
    }

    fn adj_of_translation(params: &S::Vector<4>, point: &S::Vector<3>) -> S::Matrix<3, 3> {
        Rotation3Impl::<S>::hat(point).mat_mul(Rotation3Impl::<S>::matrix(params))
    }

    fn ad_of_translation(point: &S::Vector<3>) -> S::Matrix<3, 3> {
        Rotation3Impl::<S>::hat(point)
    }
}

pub type Isometry3Impl<S> =
    lie::semi_direct_product::TranslationProductGroupImpl<S, 6, 7, 3, 4, 3, 4, Rotation3Impl<S>>;
pub type Rotation3<S> = LieGroup<S, 3, 4, 3, 3, Rotation3Impl<S>>;
pub type Isometry3<S> = lie::lie_group::LieGroup<S, 6, 7, 3, 4, Isometry3Impl<S>>;

impl<S: IsScalar> Isometry3<S> {
    pub fn from_translation_and_rotation(
        translation: &<S as IsScalar>::Vector<3>,
        rotation: &Rotation3<S>,
    ) -> Self {
        Self::from_translation_and_factor(translation, rotation)
    }

    pub fn set_rotation(&mut self, rotation: &Rotation3<S>) {
        self.set_factor(rotation)
    }

    pub fn rotation(&self) -> Rotation3<S> {
        self.factor()
    }
}

mod tests {

    #[test]
    fn rotation3_prop_tests() {
        use super::Rotation3;
        use crate::calculus::dual::dual_scalar::Dual;

        Rotation3::<f64>::test_suite();
        Rotation3::<Dual>::test_suite();
        Rotation3::<f64>::real_test_suite();
    }

    #[test]
    fn isometry3_prop_tests() {
        use super::Isometry3;
        use crate::calculus::dual::dual_scalar::Dual;
        use crate::lie::traits::IsTranslationProductGroup;

        Isometry3::<f64>::test_suite();
        Isometry3::<Dual>::test_suite();
        Isometry3::<f64>::real_test_suite();

        for g in Isometry3::<f64>::element_examples() {
            let translation = g.translation();
            let rotation = g.rotation();

            let g2 = Isometry3::from_translation_and_rotation(&translation, &rotation);
            assert_eq!(g2.translation(), translation);
            assert_eq!(g2.rotation().matrix(), rotation.matrix());
        }
    }
}
