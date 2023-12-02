use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::types::matrix::IsMatrix;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::vector::IsVectorLike;
use crate::lie;
use crate::manifold::traits::ParamsImpl;
use crate::manifold::{self};
use std::marker::PhantomData;

#[derive(Debug, Copy, Clone)]
pub struct Rotation2Impl<S: IsScalar> {
    phanton: PhantomData<S>,
}

impl<S: IsScalar> Rotation2Impl<S> {}

impl<S: IsScalar> ParamsImpl<S, 2> for Rotation2Impl<S> {
    fn params_examples() -> Vec<S::Vector<2>> {
        let mut params = vec![];
        for i in 0..10 {
            let angle = i as f64 * std::f64::consts::PI / 5.0;
            params.push(
                Rotation2::<S>::exp(&S::Vector::<1>::from_array([angle.into()]))
                    .params()
                    .clone(),
            );
        }
        params
    }

    fn invalid_params_examples() -> Vec<S::Vector<2>> {
        vec![
            S::Vector::<2>::from_array([S::c(0.0), S::c(0.0)]),
            S::Vector::<2>::from_array([S::c(0.5), S::c(0.5)]),
            S::Vector::<2>::from_array([S::c(0.5), S::c(-0.5)]),
        ]
    }

    fn are_params_valid(params: &S::Vector<2>) -> bool {
        let norm = params.norm();
        (norm - S::c(1.0)).abs() < S::c(1e-6)
    }
}

impl<S: IsScalar> manifold::traits::TangentImpl<S, 1> for Rotation2Impl<S> {
    fn tangent_examples() -> Vec<S::Vector<1>> {
        vec![
            S::Vector::<1>::from_array([0.0.into()]),
            S::Vector::<1>::from_array([1.0.into()]),
            S::Vector::<1>::from_array([(-1.0).into()]),
            S::Vector::<1>::from_array([0.5.into()]),
            S::Vector::<1>::from_array([(-0.5).into()]),
        ]
    }
}

impl<S: IsScalar> lie::traits::LieGroupImpl<S, 1, 2, 2, 2> for Rotation2Impl<S> {
    type GenG<S2: IsScalar> = Rotation2Impl<S2>;
    type RealG = Rotation2Impl<f64>;
    type DualG = Rotation2Impl<Dual>;

    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = true;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<2> {
        S::Vector::<2>::from_array([1.0.into(), 0.0.into()])
    }

    fn adj(_params: &S::Vector<2>) -> S::Matrix<1, 1> {
        S::Matrix::<1, 1>::identity()
    }

    fn exp(omega: &S::Vector<1>) -> S::Vector<2> {
        // angle to complex number
        let angle = omega.get(0);
        let cos = angle.clone().cos();
        let sin = angle.sin();
        S::Vector::<2>::from_array([cos, sin])
    }

    fn log(params: &S::Vector<2>) -> S::Vector<1> {
        // complex number to angle
        let angle = params.get(1).atan2(params.get(0));
        S::Vector::<1>::from_array([angle])
    }

    fn hat(omega: &S::Vector<1>) -> S::Matrix<2, 2> {
        let angle = omega.clone().get(0);
        S::Matrix::<2, 2>::from_array2([[0.0.into(), -angle.clone()], [angle, 0.0.into()]])
    }

    fn vee(hat: &S::Matrix<2, 2>) -> S::Vector<1> {
        let angle = hat.get((1, 0));
        S::Vector::<1>::from_array([angle])
    }

    fn group_mul(params1: &S::Vector<2>, params2: &S::Vector<2>) -> S::Vector<2> {
        let a = params1.get(0);
        let b = params1.get(1);
        let c = params2.get(0);
        let d = params2.get(1);

        S::Vector::<2>::from_array([a.clone() * c.clone() - d.clone() * b.clone(), a * d + b * c])
    }

    fn inverse(params: &S::Vector<2>) -> S::Vector<2> {
        S::Vector::<2>::from_array([params.get(0), -params.get(1)])
    }

    fn transform(params: &S::Vector<2>, point: &S::Vector<2>) -> S::Vector<2> {
        Self::matrix(params) * point.clone()
    }

    fn to_ambient(params: &S::Vector<2>) -> S::Vector<2> {
        // homogeneous coordinates
        params.clone()
    }

    fn compact(params: &S::Vector<2>) -> S::Matrix<2, 2> {
        Self::matrix(params)
    }

    fn matrix(params: &S::Vector<2>) -> S::Matrix<2, 2> {
        // rotation matrix
        let cos = params.get(0);
        let sin = params.get(1);
        S::Matrix::<2, 2>::from_array2([[cos.clone(), -sin.clone()], [sin, cos]])
    }

    fn ad(_tangent: &S::Vector<1>) -> S::Matrix<1, 1> {
        S::Matrix::<1, 1>::zero()
    }

    fn has_shortest_path_ambiguity(params: &<S as IsScalar>::Vector<2>) -> bool {
        (Self::log(params).real()[0].abs() - std::f64::consts::PI).abs() < 1e-5
    }
}

pub type Rotation2<S> = lie::lie_group::LieGroup<S, 1, 2, 2, 2, Rotation2Impl<S>>;

impl<S: IsScalar> lie::traits::LieFactorGroupImplTrait<S, 1, 2, 2, 2> for Rotation2Impl<S> {
    type GenFactorG<S2: IsScalar> = Rotation2Impl<S2>;
    type RealFactorG = Rotation2Impl<f64>;
    type DualFactorG = Rotation2Impl<Dual>;

    fn mat_v(params: &S::Vector<2>, v: &S::Vector<1>) -> S::Matrix<2, 2> {
        let sin_theta_by_theta;
        let one_minus_cos_theta_by_theta: S;
        let theta = v.get(0);
        let abs_theta = theta.clone().abs();
        if abs_theta.real() < 1e-6 {
            let theta_sq = theta.clone() * theta.clone();
            sin_theta_by_theta = S::c(1.0) - S::c(1.0 / 6.0) * theta_sq.clone();
            one_minus_cos_theta_by_theta =
                S::c(0.5) * theta.clone() - S::c(1.0 / 24.0) * theta * theta_sq;
        } else {
            sin_theta_by_theta = params.get(1) / theta.clone();
            one_minus_cos_theta_by_theta = (S::c(1.0) - params.get(0)) / theta;
        }
        S::Matrix::<2, 2>::from_array2([
            [
                sin_theta_by_theta.clone(),
                -one_minus_cos_theta_by_theta.clone(),
            ],
            [one_minus_cos_theta_by_theta, sin_theta_by_theta],
        ])
    }

    fn mat_v_inverse(params: &S::Vector<2>, tangent: &S::Vector<1>) -> S::Matrix<2, 2> {
        let halftheta = S::c(0.5) * tangent.get(0);
        let halftheta_by_tan_of_halftheta: S;

        let real_minus_one = params.get(0) - S::c(1.0);
        let abs_real_minus_one = real_minus_one.clone().abs();
        if abs_real_minus_one.real() < 1e-6 {
            halftheta_by_tan_of_halftheta =
                S::c(1.0) - S::c(1.0 / 12.0) * tangent.get(0) * tangent.get(0);
        } else {
            halftheta_by_tan_of_halftheta = -(halftheta.clone() * params.get(1)) / real_minus_one;
        }
        S::Matrix::<2, 2>::from_array2([
            [halftheta_by_tan_of_halftheta.clone(), halftheta.clone()],
            [-halftheta, halftheta_by_tan_of_halftheta],
        ])
    }

    fn adj_of_translation(_params: &S::Vector<2>, point: &S::Vector<2>) -> S::Matrix<2, 1> {
        S::Matrix::<2, 1>::from_array2([[point.get(1)], [-point.get(0)]])
    }

    fn ad_of_translation(point: &S::Vector<2>) -> S::Matrix<2, 1> {
        S::Matrix::<2, 1>::from_array2([[point.get(1)], [-point.get(0)]])
    }
}

pub type Isometry2Impl<S> =
    lie::semi_direct_product::SemiDirectProductImpl<S, 3, 4, 2, 3, 1, 2, Rotation2Impl<S>>;

pub type Isometry2<S> = lie::lie_group::LieGroup<S, 3, 4, 2, 3, Isometry2Impl<S>>;

mod tests {
    #[test]
    fn rotation2_prop_tests() {
        use super::Rotation2;
        use crate::calculus::dual::dual_scalar::Dual;
        Rotation2::<f64>::test_suite();
        Rotation2::<Dual>::test_suite();

        Rotation2::<f64>::real_test_suite();
    }

    #[test]
    fn isometry2_prop_tests() {
        use super::Isometry2;
        use crate::calculus::dual::dual_scalar::Dual;
        Isometry2::<f64>::test_suite();
        Isometry2::<Dual>::test_suite();
        Isometry2::<f64>::real_test_suite();
    }
}
