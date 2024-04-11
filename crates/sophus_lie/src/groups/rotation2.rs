use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::traits::IsLieFactorGroupImpl;
use crate::traits::IsLieGroupImpl;
use crate::traits::IsRealLieFactorGroupImpl;
use crate::traits::IsRealLieGroupImpl;
use sophus_core::manifold::traits::TangentImpl;
use sophus_core::params::ParamsImpl;
use std::marker::PhantomData;

/// 2D rotation group implementation struct - SO(2)
#[derive(Debug, Copy, Clone, Default)]
pub struct Rotation2Impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize> {
    phanton: PhantomData<S>,
}

impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize> Rotation2Impl<S, BATCH_SIZE> {}

impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize> ParamsImpl<S, 2, BATCH_SIZE>
    for Rotation2Impl<S, BATCH_SIZE>
{
    fn params_examples() -> Vec<S::Vector<2>> {
        let mut params = vec![];
        for i in 0..10 {
            let angle = S::from_f64(i as f64 * std::f64::consts::PI / 5.0);
            params.push(
                Rotation2::<S, BATCH_SIZE>::exp(&S::Vector::<1>::from_array([angle]))
                    .params()
                    .clone(),
            );
        }
        params
    }

    fn invalid_params_examples() -> Vec<S::Vector<2>> {
        vec![
            S::Vector::<2>::from_array([S::from_f64(0.0), S::from_f64(0.0)]),
            S::Vector::<2>::from_array([S::from_f64(0.5), S::from_f64(0.5)]),
            S::Vector::<2>::from_array([S::from_f64(0.5), S::from_f64(-0.5)]),
        ]
    }

    fn are_params_valid(params: &S::Vector<2>) -> S::Mask {
        let norm = params.norm();
        (norm - S::from_f64(1.0))
            .abs()
            .less_equal(&S::from_f64(1e-6))
    }
}

impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize> TangentImpl<S, 1, BATCH_SIZE>
    for Rotation2Impl<S, BATCH_SIZE>
{
    fn tangent_examples() -> Vec<S::Vector<1>> {
        vec![
            S::Vector::<1>::from_array([S::from_f64(0.0)]),
            S::Vector::<1>::from_array([S::from_f64(1.0)]),
            S::Vector::<1>::from_array([S::from_f64(-1.0)]),
            S::Vector::<1>::from_array([S::from_f64(0.5)]),
            S::Vector::<1>::from_array([S::from_f64(-0.4)]),
        ]
    }
}

impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize>
    crate::traits::IsLieGroupImpl<S, 1, 2, 2, 2, BATCH_SIZE> for Rotation2Impl<S, BATCH_SIZE>
{
    type GenG<S2: IsScalar<BATCH_SIZE>> = Rotation2Impl<S2, BATCH_SIZE>;
    type RealG = Rotation2Impl<S::RealScalar, BATCH_SIZE>;
    type DualG = Rotation2Impl<S::DualScalar, BATCH_SIZE>;

    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = true;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<2> {
        S::Vector::<2>::from_array([S::ones(), S::zeros()])
    }

    fn adj(_params: &S::Vector<2>) -> S::Matrix<1, 1> {
        S::Matrix::<1, 1>::identity()
    }

    fn exp(omega: &S::Vector<1>) -> S::Vector<2> {
        // angle to complex number
        let angle = omega.get_elem(0);
        let cos = angle.clone().cos();
        let sin = angle.sin();
        S::Vector::<2>::from_array([cos, sin])
    }

    fn log(params: &S::Vector<2>) -> S::Vector<1> {
        // complex number to angle
        let angle = params.get_elem(1).atan2(params.get_elem(0));
        S::Vector::<1>::from_array([angle])
    }

    fn hat(omega: &S::Vector<1>) -> S::Matrix<2, 2> {
        let angle = omega.clone().get_elem(0);
        S::Matrix::<2, 2>::from_array2([[S::zeros(), -angle.clone()], [angle, S::zeros()]])
    }

    fn vee(hat: &S::Matrix<2, 2>) -> S::Vector<1> {
        let angle = hat.get_elem([1, 0]);
        S::Vector::<1>::from_array([angle])
    }

    fn group_mul(params1: &S::Vector<2>, params2: &S::Vector<2>) -> S::Vector<2> {
        let a = params1.get_elem(0);
        let b = params1.get_elem(1);
        let c = params2.get_elem(0);
        let d = params2.get_elem(1);

        S::Vector::<2>::from_array([a.clone() * c.clone() - d.clone() * b.clone(), a * d + b * c])
    }

    fn inverse(params: &S::Vector<2>) -> S::Vector<2> {
        S::Vector::<2>::from_array([params.get_elem(0), -params.get_elem(1)])
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
        let cos = params.get_elem(0);
        let sin = params.get_elem(1);
        S::Matrix::<2, 2>::from_array2([[cos.clone(), -sin.clone()], [sin, cos]])
    }

    fn ad(_tangent: &S::Vector<1>) -> S::Matrix<1, 1> {
        S::Matrix::zeros()
    }
}

impl<S: IsRealScalar<BATCH_SIZE>, const BATCH_SIZE: usize>
    IsRealLieGroupImpl<S, 1, 2, 2, 2, BATCH_SIZE> for Rotation2Impl<S, BATCH_SIZE>
{
    fn dx_exp_x_at_0() -> S::Matrix<2, 1> {
        S::Matrix::from_real_scalar_array2([[S::RealScalar::zeros()], [S::RealScalar::ones()]])
    }

    fn dx_exp_x_times_point_at_0(point: S::Vector<2>) -> S::Matrix<2, 1> {
        S::Matrix::from_array2([[-point.get_elem(1)], [point.get_elem(0)]])
    }

    fn dx_exp(tangent: &S::Vector<1>) -> S::Matrix<2, 1> {
        let theta = tangent.get_elem(0);
        S::Matrix::<2, 1>::from_array2([[-theta.sin()], [theta.cos()]])
    }

    fn dx_log_x(params: &S::Vector<2>) -> S::Matrix<1, 2> {
        let x_0 = params.get_elem(0);
        let x_1 = params.get_elem(1);
        let x_sq = x_0 * x_0 + x_1 * x_1;
        S::Matrix::from_array2([[-x_1 / x_sq, x_0 / x_sq]])
    }

    fn da_a_mul_b(_a: &S::Vector<2>, b: &S::Vector<2>) -> S::Matrix<2, 2> {
        Self::matrix(b)
    }

    fn db_a_mul_b(a: &S::Vector<2>, _b: &S::Vector<2>) -> S::Matrix<2, 2> {
        Self::matrix(a)
    }

    fn has_shortest_path_ambiguity(params: &S::Vector<2>) -> S::Mask {
        (Self::log(params).get_elem(0).abs() - S::from_f64(std::f64::consts::PI))
            .abs()
            .less_equal(&S::from_f64(1e-5))
    }
}

/// 2d rotation group - SO(2)
pub type Rotation2<S, const B: usize> = LieGroup<S, 1, 2, 2, 2, B, Rotation2Impl<S, B>>;

impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize> IsLieFactorGroupImpl<S, 1, 2, 2, BATCH_SIZE>
    for Rotation2Impl<S, BATCH_SIZE>
{
    type GenFactorG<S2: IsScalar<BATCH_SIZE>> = Rotation2Impl<S2, BATCH_SIZE>;
    type RealFactorG = Rotation2Impl<S::RealScalar, BATCH_SIZE>;
    type DualFactorG = Rotation2Impl<S::DualScalar, BATCH_SIZE>;

    fn mat_v(v: &S::Vector<1>) -> S::Matrix<2, 2> {
        let one_minus_cos_theta_by_theta: S;
        let theta = v.get_elem(0);
        let abs_theta = theta.clone().abs();

        let near_zero = abs_theta.less_equal(&S::from_f64(1e-6));

        let theta_sq = theta.clone() * theta.clone();
        let sin_theta_by_theta = (S::from_f64(1.0) - S::from_f64(1.0 / 6.0) * theta_sq.clone())
            .select(&near_zero, theta.clone().sin() / theta.clone());
        one_minus_cos_theta_by_theta = (S::from_f64(0.5) * theta.clone()
            - S::from_f64(1.0 / 24.0) * theta.clone() * theta_sq)
            .select(&near_zero, (S::from_f64(1.0) - theta.clone().cos()) / theta);

        S::Matrix::<2, 2>::from_array2([
            [
                sin_theta_by_theta.clone(),
                -one_minus_cos_theta_by_theta.clone(),
            ],
            [one_minus_cos_theta_by_theta, sin_theta_by_theta],
        ])
    }

    fn mat_v_inverse(tangent: &S::Vector<1>) -> S::Matrix<2, 2> {
        let theta = tangent.get_elem(0);
        let halftheta = S::from_f64(0.5) * theta.clone();

        let real_minus_one = theta.clone().cos() - S::from_f64(1.0);
        let abs_real_minus_one = real_minus_one.clone().abs();

        let near_zero = abs_real_minus_one.less_equal(&S::from_f64(1e-6));

        let halftheta_by_tan_of_halftheta = (S::from_f64(1.0)
            - S::from_f64(1.0 / 12.0) * tangent.get_elem(0) * tangent.get_elem(0))
        .select(
            &near_zero,
            -(halftheta.clone() * theta.sin()) / real_minus_one,
        );

        S::Matrix::<2, 2>::from_array2([
            [halftheta_by_tan_of_halftheta.clone(), halftheta.clone()],
            [-halftheta, halftheta_by_tan_of_halftheta],
        ])
    }

    fn adj_of_translation(_params: &S::Vector<2>, point: &S::Vector<2>) -> S::Matrix<2, 1> {
        S::Matrix::<2, 1>::from_array2([[point.get_elem(1)], [-point.get_elem(0)]])
    }

    fn ad_of_translation(point: &S::Vector<2>) -> S::Matrix<2, 1> {
        S::Matrix::<2, 1>::from_array2([[point.get_elem(1)], [-point.get_elem(0)]])
    }
}

impl<S: IsRealScalar<BATCH_SIZE>, const BATCH_SIZE: usize>
    IsRealLieFactorGroupImpl<S, 1, 2, 2, BATCH_SIZE> for Rotation2Impl<S, BATCH_SIZE>
{
    fn dx_mat_v(tangent: &S::Vector<1>) -> [S::Matrix<2, 2>; 1] {
        let theta = tangent.get_elem(0);
        let theta_sq = theta * theta;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        let near_zero = theta_sq.abs().less_equal(&S::from_f64(1e-6));

        let m00 = (S::from_f64(-1.0 / 3.0) * theta + S::from_f64(1.0 / 30.0) * theta * theta_sq)
            .select(&near_zero, (theta * cos_theta - sin_theta) / theta_sq);
        let m01 = (-S::from_f64(0.5) + S::from_f64(0.125) * theta_sq).select(
            &near_zero,
            (-theta * sin_theta - cos_theta + S::from_f64(1.0)) / theta_sq,
        );

        [S::Matrix::<2, 2>::from_array2([[m00, m01], [-m01, m00]])]
    }

    fn dparams_matrix_times_point(_params: &S::Vector<2>, point: &S::Vector<2>) -> S::Matrix<2, 2> {
        let px = point.get_elem(0);
        let py = point.get_elem(1);
        S::Matrix::from_array2([[px, -py], [py, px]])
    }

    fn dx_mat_v_inverse(tangent: &S::Vector<1>) -> [S::Matrix<2, 2>; 1] {
        let theta = tangent.get_elem(0);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        let near_zero = theta.abs().less_equal(&S::from_f64(1e-6));

        let c = (S::from_f64(-1.0 / 6.0) * theta).select(
            &near_zero,
            (theta - sin_theta) / (S::from_f64(2.0) * (cos_theta - S::from_f64(1.0))),
        );

        [S::Matrix::<2, 2>::from_array2([
            [c, S::from_f64(0.5)],
            [-S::from_f64(0.5), c],
        ])]
    }
}

#[test]
fn rotation2_prop_tests() {
    use crate::factor_lie_group::RealFactorLieGroupTest;
    use crate::real_lie_group::RealLieGroupTest;
    use sophus_core::calculus::dual::dual_scalar::DualBatchScalar;
    use sophus_core::calculus::dual::dual_scalar::DualScalar;
    use sophus_core::linalg::BatchScalarF64;

    Rotation2::<f64, 1>::test_suite();
    Rotation2::<BatchScalarF64<8>, 8>::test_suite();
    Rotation2::<DualScalar, 1>::test_suite();
    Rotation2::<DualBatchScalar<8>, 8>::test_suite();

    Rotation2::<f64, 1>::run_real_tests();
    Rotation2::<BatchScalarF64<8>, 8>::run_real_tests();

    Rotation2::<f64, 1>::run_real_factor_tests();
    Rotation2::<BatchScalarF64<8>, 8>::run_real_factor_tests();
}
