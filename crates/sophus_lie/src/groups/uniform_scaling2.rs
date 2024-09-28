use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::traits::HasDisambiguate;
use crate::traits::IsLieFactorGroupImpl;
use crate::traits::IsLieGroupImpl;
use crate::traits::IsRealLieFactorGroupImpl;
use crate::traits::IsRealLieGroupImpl;
use crate::Rotation2;
use sophus_core::manifold::traits::TangentImpl;
use sophus_core::params::ParamsImpl;
use std::marker::PhantomData;

/// 2D rotation group implementation struct - SO(2)
#[derive(Debug, Copy, Clone, Default)]
pub struct UniformScaling2Impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize> {
    phanton: PhantomData<S>,
}

impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize> UniformScaling2Impl<S, BATCH_SIZE> {}

impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize> HasDisambiguate<S, 2, BATCH_SIZE>
    for UniformScaling2Impl<S, BATCH_SIZE>
{
}

impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize> ParamsImpl<S, 2, BATCH_SIZE>
    for UniformScaling2Impl<S, BATCH_SIZE>
{
    fn params_examples() -> Vec<S::Vector<2>> {
        let mut params = vec![];
        for i in 0..10 {
            let angle = S::from_f64(i as f64 * std::f64::consts::PI / 5.0);
            for j in 1..10 {
                let scale = S::from_f64(j as f64 * 0.23);
                params.push(S::Vector::<2>::from_array([angle.clone(), scale]));
            }
        }
        params
    }

    fn invalid_params_examples() -> Vec<S::Vector<2>> {
        vec![S::Vector::<2>::from_array([
            S::from_f64(0.0),
            S::from_f64(0.0),
        ])]
    }

    fn are_params_valid(params: &S::Vector<2>) -> S::Mask {
        // let squared_norm = params.squared_norm();
        // let eps = 1e-5;

        // squared_norm
        //     .less_equal(&S::from_f64(1.0 / eps))
        //     .and(&squared_norm.greater_equal(&S::from_f64(eps)))
        todo!()
    }
}

impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize> TangentImpl<S, 2, BATCH_SIZE>
    for UniformScaling2Impl<S, BATCH_SIZE>
{
    fn tangent_examples() -> Vec<S::Vector<2>> {
        vec![
            S::Vector::<2>::from_array([S::from_f64(0.0), S::from_f64(0.0)]),
            S::Vector::<2>::from_array([S::from_f64(0.0), S::from_f64(3.0)]),
            S::Vector::<2>::from_array([S::from_f64(1.2), S::from_f64(0.0)]),
            S::Vector::<2>::from_array([S::from_f64(1.0), S::from_f64(1.0)]),
        ]
    }
}

impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize>
    crate::traits::IsLieGroupImpl<S, 2, 2, 2, 2, BATCH_SIZE>
    for UniformScaling2Impl<S, BATCH_SIZE>
{
    type GenG<S2: IsScalar<BATCH_SIZE>> = UniformScaling2Impl<S2, BATCH_SIZE>;
    type RealG = UniformScaling2Impl<S::RealScalar, BATCH_SIZE>;
    type DualG = UniformScaling2Impl<S::DualScalar, BATCH_SIZE>;

    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = true;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<2> {
        S::Vector::<2>::from_array([S::ones(), S::zeros()])
    }

    fn adj(_params: &S::Vector<2>) -> S::Matrix<2, 2> {
        S::Matrix::<2, 2>::identity()
    }

    fn exp(angle_logscale: &S::Vector<2>) -> S::Vector<2> {
        let angle: S = angle_logscale.get_elem(0);
        let sigma: S = angle_logscale.get_elem(1);
        let s: S = sigma.clone().exp();

        // let exp_plus = S::from_f64(2e-5);
        // Ensuring proper scale
        // s = s.clamp(exp_plus, S::from_f64(1.0) / exp_plus);
        let rot = Rotation2::<S, BATCH_SIZE>::exp(&S::Vector::<1>::from_array([angle]));
        let z: &S::Vector<2> = rot.params();
        z.scaled(s)
    }

    fn log(params: &S::Vector<2>) -> S::Vector<2> {
        // complex number to angle
        let angle = params.get_elem(1).atan2(params.get_elem(0));
        S::Vector::<2>::from_array([angle, params.norm().log()])
    }

    fn hat(omega: &S::Vector<2>) -> S::Matrix<2, 2> {
        let angle = omega.clone().get_elem(0);
        let log_scale = omega.clone().get_elem(1);
        S::Matrix::<2, 2>::from_array2([[log_scale.clone(), -angle.clone()], [angle, log_scale]])
    }

    fn vee(hat: &S::Matrix<2, 2>) -> S::Vector<2> {
        let angle = hat.get_elem([1, 0]);
        let log_scale = hat.get_elem([0, 0]);
        S::Vector::<2>::from_array([angle, log_scale])
    }

    fn group_mul(params1: &S::Vector<2>, params2: &S::Vector<2>) -> S::Vector<2> {
        let a = params1.get_elem(0);
        let b = params1.get_elem(1);
        let c = params2.get_elem(0);
        let d = params2.get_elem(1);

        let z = S::Vector::<2>::from_array([
            a.clone() * c.clone() - d.clone() * b.clone(),
            a * d + b * c,
        ]);

        let squared_scale = z.squared_norm();

        //let eps_plus = S::from_f64(2e-5);
        let eps = S::from_f64(1e-5);
        let eps_sq = eps.clone() * eps.clone();

        if squared_scale.less_equal(&eps_sq).any() {
            // Saturation to ensure class invariant.
            unimplemented!();
        } else if squared_scale
            .greater_equal(&(S::from_f64(1.0) / eps_sq))
            .any()
        {
            // Saturation to ensure class invariant.
            unimplemented!();
        }
        z
    }

    fn inverse(params: &S::Vector<2>) -> S::Vector<2> {
        let squared_scale = params.squared_norm();
        S::Vector::<2>::from_array([
            params.get_elem(0) / squared_scale.clone(),
            -params.get_elem(1) / squared_scale,
        ])
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

    fn ad(_tangent: &S::Vector<2>) -> S::Matrix<2, 2> {
        S::Matrix::zeros()
    }
}

// TODO: Complete the implementation of IsRealLieGroupImpl trait for UniformScaling2Impl
impl<S: IsRealScalar<BATCH_SIZE>, const BATCH_SIZE: usize>
    IsRealLieGroupImpl<S, 2, 2, 2, 2, BATCH_SIZE> for UniformScaling2Impl<S, BATCH_SIZE>
{
    fn dx_exp_x_at_0() -> S::Matrix<2, 2> {
        S::Matrix::from_real_scalar_array2([
            [S::RealScalar::zeros(), S::RealScalar::from_f64(1.0)],
            [S::RealScalar::from_f64(1.0), S::RealScalar::ones()],
        ])
    }

    fn dx_exp_x_times_point_at_0(point: S::Vector<2>) -> S::Matrix<2, 2> {
        unimplemented!()
    }

    fn dx_exp(tangent: &S::Vector<2>) -> S::Matrix<2, 2> {
        unimplemented!();
    }

    fn dx_log_x(params: &S::Vector<2>) -> S::Matrix<2, 2> {
        unimplemented!()
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
pub type SpiralSimilarity2<S, const B: usize> =
    LieGroup<S, 2, 2, 2, 2, B, UniformScaling2Impl<S, B>>;

impl<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize> IsLieFactorGroupImpl<S, 2, 2, 2, BATCH_SIZE>
    for UniformScaling2Impl<S, BATCH_SIZE>
{
    type GenFactorG<S2: IsScalar<BATCH_SIZE>> = UniformScaling2Impl<S2, BATCH_SIZE>;
    type RealFactorG = UniformScaling2Impl<S::RealScalar, BATCH_SIZE>;
    type DualFactorG = UniformScaling2Impl<S::DualScalar, BATCH_SIZE>;

    fn mat_v(v: &S::Vector<2>) -> S::Matrix<2, 2> {
        // return details::calcW<Scalar, 2>(
        //     Rotation2Impl<Scalar>::hat(angle_logscale.template head<1>()),
        //     angle_logscale[0],
        //     angle_logscale[1]);
        use crate::groups::utils::calc_mat_v;

        let angle = v.get_elem(0);
        let log_scale = v.get_elem(1);
        let omega = Rotation2::<S, BATCH_SIZE>::hat(&S::Vector::<1>::from_array([angle.clone()]));
        calc_mat_v(&omega, &angle, &log_scale)

        // unimplemented!()
    }

    fn mat_v_inverse(tangent: &S::Vector<2>) -> S::Matrix<2, 2> {
        use crate::groups::utils::calc_mat_w_inv;

        // return details::calcWInv<Scalar, 2>(
        //     Rotation2Impl<Scalar>::hat(angle_logscale.template head<1>()),
        //     angle_logscale[0],
        //     angle_logscale[1],
        //     non_zero_complex.norm());

        let angle = tangent.get_elem(0);
        let log_scale = tangent.get_elem(1);
        let nrm = tangent.norm();
        let omega = Rotation2::<S, BATCH_SIZE>::hat(&S::Vector::<1>::from_array([angle.clone()]));
        calc_mat_w_inv(&omega, &angle, &log_scale, &nrm)
    }

    fn adj_of_translation(_params: &S::Vector<2>, point: &S::Vector<2>) -> S::Matrix<2, 2> {
        //S::Matrix::<2, 1>::from_array2([[point.get_elem(1)], [-point.get_elem(0)]])

        unimplemented!()
    }

    fn ad_of_translation(point: &S::Vector<2>) -> S::Matrix<2, 2> {
        //S::Matrix::<2, 1>::from_array2([[point.get_elem(1)], [-point.get_elem(0)]])

        unimplemented!()
    }
}

// TODO: Complete the implementation of IsRealLieFactorGroupImpl trait for UniformScaling2Impl
impl<S: IsRealScalar<BATCH_SIZE>, const BATCH_SIZE: usize>
    IsRealLieFactorGroupImpl<S, 2, 2, 2, BATCH_SIZE> for UniformScaling2Impl<S, BATCH_SIZE>
{
    fn dx_mat_v(tangent: &S::Vector<2>) -> [S::Matrix<2, 2>; 2] {
        unimplemented!()
    }

    fn dparams_matrix_times_point(_params: &S::Vector<2>, point: &S::Vector<2>) -> S::Matrix<2, 2> {
        let px = point.get_elem(0);
        let py = point.get_elem(1);
        S::Matrix::from_array2([[px, -py], [py, px]])
    }

    fn dx_mat_v_inverse(tangent: &S::Vector<2>) -> [S::Matrix<2, 2>; 2] {
        unimplemented!()
    }
}

#[test]
fn rotation2_prop_tests() {
    use crate::factor_lie_group::RealFactorLieGroupTest;
    use crate::real_lie_group::RealLieGroupTest;


    // SpiralSimilarity2::<f64, 1>::test_suite();
    // SpiralSimilarity2::<BatchScalarF64<8>, 8>::test_suite();
    // SpiralSimilarity2::<DualScalar, 1>::test_suite();
    // SpiralSimilarity2::<DualBatchScalar<8>, 8>::test_suite();

    // TODO: Complete the implementation of IsRealLieGroupImpl trait for UniformScaling2Impl
    // SpiralSimilarity2::<f64, 1>::run_real_tests();
    // SpiralSimilarity2::<BatchScalarF64<8>, 8>::run_real_tests();

    //SpiralSimilarity2::<f64, 1>::run_factor_tests();
    // SpiralSimilarity2::<BatchScalarF64<8>, 8>::run_real_factor_tests();
}
