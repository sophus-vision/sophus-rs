use core::{
    borrow::Borrow,
    marker::PhantomData,
};

use sophus_autodiff::{
    linalg::EPS_F64,
    manifold::IsTangent,
    params::{
        HasParams,
        IsParamsImpl,
    },
};

use crate::{
    lie_group::LieGroup,
    prelude::*,
    EmptySliceError,
    HasAverage,
    HasDisambiguate,
    IsLieFactorGroupImpl,
    IsLieGroupImpl,
    IsRealLieFactorGroupImpl,
    IsRealLieGroupImpl,
};

extern crate alloc;

/// 2D rotation group implementation struct - SO(2)
#[derive(Debug, Copy, Clone, Default)]
pub struct Rotation2Impl<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phanton: PhantomData<S>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Rotation2Impl<S, BATCH, DM, DN>
{
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    HasDisambiguate<S, 2, BATCH, DM, DN> for Rotation2Impl<S, BATCH, DM, DN>
{
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsParamsImpl<S, 2, BATCH, DM, DN> for Rotation2Impl<S, BATCH, DM, DN>
{
    fn params_examples() -> alloc::vec::Vec<S::Vector<2>> {
        let mut params = alloc::vec![];
        for i in 0..10 {
            let angle = S::from_f64(i as f64 * core::f64::consts::PI / 5.0);
            params.push(
                *Rotation2::<S, BATCH, DM, DN>::exp(S::Vector::<1>::from_array([angle])).params(),
            );
        }
        params
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<2>> {
        alloc::vec![
            S::Vector::<2>::from_array([S::from_f64(0.0), S::from_f64(0.0)]),
            S::Vector::<2>::from_array([S::from_f64(0.5), S::from_f64(0.5)]),
            S::Vector::<2>::from_array([S::from_f64(0.5), S::from_f64(-0.5)]),
        ]
    }

    fn are_params_valid<P>(params: P) -> S::Mask
    where
        P: Borrow<S::Vector<2>>,
    {
        let norm = params.borrow().norm();
        (norm - S::from_f64(1.0))
            .abs()
            .less_equal(&S::from_f64(EPS_F64))
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsTangent<S, 1, BATCH, DM, DN> for Rotation2Impl<S, BATCH, DM, DN>
{
    fn tangent_examples() -> alloc::vec::Vec<S::Vector<1>> {
        alloc::vec![
            S::Vector::<1>::from_array([S::from_f64(0.0)]),
            S::Vector::<1>::from_array([S::from_f64(1.0)]),
            S::Vector::<1>::from_array([S::from_f64(-1.0)]),
            S::Vector::<1>::from_array([S::from_f64(0.5)]),
            S::Vector::<1>::from_array([S::from_f64(-0.4)]),
        ]
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    crate::IsLieGroupImpl<S, 1, 2, 2, 2, BATCH, DM, DN> for Rotation2Impl<S, BATCH, DM, DN>
{
    type GenG<S2: IsScalar<BATCH, DM, DN>> = Rotation2Impl<S2, BATCH, DM, DN>;
    type RealG = Rotation2Impl<S::RealScalar, BATCH, 0, 0>;
    type DualG<const M: usize, const N: usize> = Rotation2Impl<S::DualScalar<M, N>, BATCH, M, N>;

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
        let angle = omega.elem(0);
        let cos = angle.cos();
        let sin = angle.sin();
        S::Vector::<2>::from_array([cos, sin])
    }

    fn log(params: &S::Vector<2>) -> S::Vector<1> {
        // complex number to angle
        let angle = params.elem(1).atan2(params.elem(0));
        S::Vector::<1>::from_array([angle])
    }

    fn hat(omega: &S::Vector<1>) -> S::Matrix<2, 2> {
        let angle = omega.elem(0);
        S::Matrix::<2, 2>::from_array2([[S::zeros(), -angle], [angle, S::zeros()]])
    }

    fn vee(hat: &S::Matrix<2, 2>) -> S::Vector<1> {
        let angle = hat.elem([1, 0]);
        S::Vector::<1>::from_array([angle])
    }

    fn group_mul(params1: &S::Vector<2>, params2: &S::Vector<2>) -> S::Vector<2> {
        let a = params1.elem(0);
        let b = params1.elem(1);
        let c = params2.elem(0);
        let d = params2.elem(1);

        S::Vector::<2>::from_array([a * c - d * b, a * d + b * c])
    }

    fn inverse(params: &S::Vector<2>) -> S::Vector<2> {
        S::Vector::<2>::from_array([params.elem(0), -params.elem(1)])
    }

    fn transform(params: &S::Vector<2>, point: &S::Vector<2>) -> S::Vector<2> {
        Self::matrix(params) * *point
    }

    fn to_ambient(params: &S::Vector<2>) -> S::Vector<2> {
        // homogeneous coordinates
        *params
    }

    fn compact(params: &S::Vector<2>) -> S::Matrix<2, 2> {
        Self::matrix(params)
    }

    fn matrix(params: &S::Vector<2>) -> S::Matrix<2, 2> {
        // rotation matrix
        let cos = params.elem(0);
        let sin = params.elem(1);
        S::Matrix::<2, 2>::from_array2([[cos, -sin], [sin, cos]])
    }

    fn ad(_tangent: &S::Vector<1>) -> S::Matrix<1, 1> {
        S::Matrix::zeros()
    }
}

impl<S: IsRealScalar<BATCH>, const BATCH: usize> IsRealLieGroupImpl<S, 1, 2, 2, 2, BATCH>
    for Rotation2Impl<S, BATCH, 0, 0>
{
    fn dx_exp_x_at_0() -> S::Matrix<2, 1> {
        S::Matrix::from_real_scalar_array2([[S::RealScalar::zeros()], [S::RealScalar::ones()]])
    }

    fn dx_exp_x_times_point_at_0(point: &S::Vector<2>) -> S::Matrix<2, 1> {
        S::Matrix::from_array2([[-point.elem(1)], [point.elem(0)]])
    }

    fn dx_exp(tangent: &S::Vector<1>) -> S::Matrix<2, 1> {
        let theta = tangent.elem(0);
        S::Matrix::<2, 1>::from_array2([[-theta.sin()], [theta.cos()]])
    }

    fn dx_log_x(params: &S::Vector<2>) -> S::Matrix<1, 2> {
        let x_0 = params.elem(0);
        let x_1 = params.elem(1);
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
        (Self::log(params).elem(0).abs() - S::from_f64(core::f64::consts::PI))
            .abs()
            .less_equal(&S::from_f64(EPS_F64))
    }

    fn dparams_matrix(_params: &<S>::Vector<2>, col_idx: usize) -> <S>::Matrix<2, 2> {
        match col_idx {
            0 => S::Matrix::identity(),
            1 => S::Matrix::from_f64_array2([[0.0, -1.0], [1.0, 0.0]]),
            _ => panic!("Invalid column index: {}", col_idx),
        }
    }
}

/// 2d rotation group - SO(2)
pub type Rotation2<S, const B: usize, const DM: usize, const DN: usize> =
    LieGroup<S, 1, 2, 2, 2, B, DM, DN, Rotation2Impl<S, B, DM, DN>>;

/// 2d rotation group with f64 scalar type
pub type Rotation2F64 = Rotation2<f64, 1, 0, 0>;

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Rotation2<S, BATCH, DM, DN>
{
    /// Rotate by angle
    pub fn rot<U>(theta: U) -> Self
    where
        U: Borrow<S>,
    {
        let theta: &S = theta.borrow();
        Rotation2::exp(S::Vector::<1>::from_array([*theta]))
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsLieFactorGroupImpl<S, 1, 2, 2, BATCH, DM, DN> for Rotation2Impl<S, BATCH, DM, DN>
{
    type GenFactorG<S2: IsScalar<BATCH, M, N>, const M: usize, const N: usize> =
        Rotation2Impl<S2, BATCH, M, N>;
    type RealFactorG = Rotation2Impl<S::RealScalar, BATCH, 0, 0>;
    type DualFactorG<const M: usize, const N: usize> =
        Rotation2Impl<S::DualScalar<M, N>, BATCH, M, N>;

    fn mat_v(v: &S::Vector<1>) -> S::Matrix<2, 2> {
        let theta = v.elem(0);
        let abs_theta = theta.abs();

        let near_zero = abs_theta.less_equal(&S::from_f64(EPS_F64));

        let theta_sq = theta * theta;
        let sin_theta_by_theta = (S::from_f64(1.0) - S::from_f64(1.0 / 6.0) * theta_sq)
            .select(&near_zero, theta.sin() / theta);
        let one_minus_cos_theta_by_theta: S = (S::from_f64(0.5) * theta
            - S::from_f64(1.0 / 24.0) * theta * theta_sq)
            .select(&near_zero, (S::from_f64(1.0) - theta.cos()) / theta);

        S::Matrix::<2, 2>::from_array2([
            [sin_theta_by_theta, -one_minus_cos_theta_by_theta],
            [one_minus_cos_theta_by_theta, sin_theta_by_theta],
        ])
    }

    fn mat_v_inverse(tangent: &S::Vector<1>) -> S::Matrix<2, 2> {
        let theta = tangent.elem(0);
        let halftheta = S::from_f64(0.5) * theta;

        let real_minus_one = theta.cos() - S::from_f64(1.0);
        let abs_real_minus_one = real_minus_one.abs();

        let near_zero = abs_real_minus_one.less_equal(&S::from_f64(EPS_F64));

        let halftheta_by_tan_of_halftheta = (S::from_f64(1.0)
            - S::from_f64(1.0 / 12.0) * tangent.elem(0) * tangent.elem(0))
        .select(&near_zero, -(halftheta * theta.sin()) / real_minus_one);

        S::Matrix::<2, 2>::from_array2([
            [halftheta_by_tan_of_halftheta, halftheta],
            [-halftheta, halftheta_by_tan_of_halftheta],
        ])
    }

    fn adj_of_translation(_params: &S::Vector<2>, point: &S::Vector<2>) -> S::Matrix<2, 1> {
        S::Matrix::<2, 1>::from_array2([[point.elem(1)], [-point.elem(0)]])
    }

    fn ad_of_translation(point: &S::Vector<2>) -> S::Matrix<2, 1> {
        S::Matrix::<2, 1>::from_array2([[point.elem(1)], [-point.elem(0)]])
    }
}

impl<S: IsRealScalar<BATCH>, const BATCH: usize> IsRealLieFactorGroupImpl<S, 1, 2, 2, BATCH>
    for Rotation2Impl<S, BATCH, 0, 0>
{
    fn dx_mat_v(tangent: &S::Vector<1>) -> [S::Matrix<2, 2>; 1] {
        let theta = tangent.elem(0);
        let theta_sq = theta * theta;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        let near_zero = theta_sq.abs().less_equal(&S::from_f64(EPS_F64));

        let m00 = (S::from_f64(-1.0 / 3.0) * theta + S::from_f64(1.0 / 30.0) * theta * theta_sq)
            .select(&near_zero, (theta * cos_theta - sin_theta) / theta_sq);
        let m01 = (-S::from_f64(0.5) + S::from_f64(0.125) * theta_sq).select(
            &near_zero,
            (-theta * sin_theta - cos_theta + S::from_f64(1.0)) / theta_sq,
        );

        [S::Matrix::<2, 2>::from_array2([[m00, m01], [-m01, m00]])]
    }

    fn dparams_matrix_times_point(_params: &S::Vector<2>, point: &S::Vector<2>) -> S::Matrix<2, 2> {
        let px = point.elem(0);
        let py = point.elem(1);
        S::Matrix::from_array2([[px, -py], [py, px]])
    }

    fn dx_mat_v_inverse(tangent: &S::Vector<1>) -> [S::Matrix<2, 2>; 1] {
        let theta = tangent.elem(0);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        let near_zero = theta.abs().less_equal(&S::from_f64(EPS_F64));

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

impl<S: IsSingleScalar<DM, DN> + PartialOrd, const DM: usize, const DN: usize>
    HasAverage<S, 1, 2, 2, 2, DM, DN> for Rotation2<S, 1, DM, DN>
{
    /// Closed form average for the Rotation2 group.
    fn average(
        parent_from_body_transforms: &[Rotation2<S, 1, DM, DN>],
    ) -> Result<Self, EmptySliceError> {
        if parent_from_body_transforms.is_empty() {
            return Err(EmptySliceError);
        }
        let parent_from_body0 = parent_from_body_transforms[0];
        let w = S::from_f64(1.0 / parent_from_body_transforms.len() as f64);

        let mut average_tangent = S::Vector::zeros();

        for parent_from_body in parent_from_body_transforms {
            average_tangent = average_tangent
                + (parent_from_body0.inverse() * parent_from_body)
                    .log()
                    .scaled(w);
        }

        Ok(parent_from_body0 * LieGroup::exp(average_tangent))
    }
}

#[test]
fn rotation2_prop_tests() {
    use sophus_autodiff::dual::dual_scalar::DualScalar;
    #[cfg(feature = "simd")]
    use sophus_autodiff::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
    use sophus_autodiff::linalg::BatchScalarF64;

    use crate::lie_group::{
        factor_lie_group::RealFactorLieGroupTest,
        real_lie_group::RealLieGroupTest,
    };

    Rotation2F64::test_suite();
    #[cfg(feature = "simd")]
    Rotation2::<BatchScalarF64<8>, 8, 0, 0>::test_suite();

    Rotation2::<DualScalar<0, 0>, 1, 0, 0>::test_suite();
    #[cfg(feature = "simd")]
    Rotation2::<DualBatchScalar<8, 1, 1>, 8, 1, 1>::test_suite();

    Rotation2F64::run_real_tests();
    #[cfg(feature = "simd")]
    Rotation2::<BatchScalarF64<8>, 8, 0, 0>::run_real_tests();

    Rotation2F64::run_real_factor_tests();
    #[cfg(feature = "simd")]
    Rotation2::<BatchScalarF64<8>, 8, 0, 0>::run_real_factor_tests();
}
