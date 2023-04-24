use nalgebra::{SMatrix, SVector};
type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

use crate::{
    calculus, lie,
    manifold::{self},
};

#[derive(Debug, Copy, Clone)]
pub struct Rotation2Impl;

impl Rotation2Impl {}

impl calculus::traits::ParamsImpl<2> for Rotation2Impl {
    fn params_examples() -> Vec<V<2>> {
        let mut params = vec![];
        for i in 0..10 {
            let angle = i as f64 * std::f64::consts::PI / 5.0;
            params.push(*Rotation2::exp(&V::<1>::new(angle)).params());
        }
        params
    }

    fn invalid_params_examples() -> Vec<V<2>> {
        vec![
            V::<2>::new(0.0, 0.0),
            V::<2>::new(0.5, 0.5),
            V::<2>::new(0.5, -0.5),
        ]
    }

    fn are_params_valid(params: &V<2>) -> bool {
        let norm = params.norm();
        (norm - 1.0).abs() < 1e-6
    }
}

impl manifold::traits::TangentImpl<1> for Rotation2Impl {
    fn tangent_examples() -> Vec<V<1>> {
        vec![
            V::<1>::new(0.0),
            V::<1>::new(1.0),
            V::<1>::new(-1.0),
            V::<1>::new(0.5),
            V::<1>::new(-0.5),
        ]
    }
}

impl lie::traits::LieGroupImpl<1, 2, 2, 2> for Rotation2Impl {
    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = true;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> V<2> {
        V::<2>::new(1.0, 0.0)
    }

    fn adj(_params: &V<2>) -> M<1, 1> {
        M::<1, 1>::identity()
    }

    fn exp(omega: &V<1>) -> V<2> {
        // angle to complex number
        let angle = omega[0];
        let cos = angle.cos();
        let sin = angle.sin();
        V::<2>::new(cos, sin)
    }

    fn log(params: &V<2>) -> V<1> {
        // complex number to angle
        let angle = params[1].atan2(params[0]);
        V::<1>::new(angle)
    }

    fn hat(omega: &V<1>) -> M<2, 2> {
        let angle = omega[0];
        M::<2, 2>::new(0.0, -angle, angle, 0.0)
    }

    fn vee(hat: &M<2, 2>) -> V<1> {
        let angle = hat[(1, 0)];
        V::<1>::new(angle)
    }

    fn multiply(params1: &V<2>, params2: &V<2>) -> V<2> {
        let z = nalgebra::geometry::UnitComplex::from_cos_sin_unchecked(params1[0], params1[1])
            * nalgebra::geometry::UnitComplex::from_cos_sin_unchecked(params2[0], params2[1]);
        V::<2>::new(z.re, z.im)
    }

    fn inverse(params: &V<2>) -> V<2> {
        V::<2>::new(params[0], -params[1])
    }

    fn transform(params: &V<2>, point: &V<2>) -> V<2> {
        Self::matrix(params) * point
    }

    fn to_ambient(params: &V<2>) -> V<2> {
        // homogeneous coordinates
        V::<2>::new(params[0], params[1])
    }

    fn compact(params: &V<2>) -> M<2, 2> {
        Self::matrix(params)
    }

    fn matrix(params: &V<2>) -> M<2, 2> {
        // rotation matrix
        let cos = params[0];
        let sin = params[1];
        M::<2, 2>::new(cos, -sin, sin, cos)
    }

    fn ad(_tangent: &V<1>) -> M<1, 1> {
        M::<1, 1>::zeros()
    }
}

pub type Rotation2 = lie::lie_group::LieGroup<1, 2, 2, 2, Rotation2Impl>;

impl lie::traits::LieSubgroupImplTrait<1, 2, 2, 2> for Rotation2Impl {
    fn mat_v(params: &V<2>, v: &V<1>) -> M<2, 2> {
        let sin_theta_by_theta;
        let one_minus_cos_theta_by_theta;
        let theta = v[0];
        let abs_theta = theta.abs();
        if abs_theta < 1e-6 {
            let theta_sq = theta * theta;
            sin_theta_by_theta = 1.0 - 1.0 / 6.0 * theta_sq;
            one_minus_cos_theta_by_theta = 0.5 * theta - 1.0 / 24.0 * theta * theta_sq;
        } else {
            sin_theta_by_theta = params[1] / theta;
            one_minus_cos_theta_by_theta = (1.0 - params[0]) / theta;
        }
        M::<2, 2>::new(
            sin_theta_by_theta,
            -one_minus_cos_theta_by_theta,
            one_minus_cos_theta_by_theta,
            sin_theta_by_theta,
        )
    }

    fn mat_v_inverse(params: &V<2>, tangent: &V<1>) -> M<2, 2> {
        let halftheta = 0.5 * tangent[0];
        let halftheta_by_tan_of_halftheta;

        let real_minus_one = params[0] - 1.0;
        let abs_real_minus_one = real_minus_one.abs();
        if abs_real_minus_one < 1e-6 {
            halftheta_by_tan_of_halftheta = 1.0 - 1.0 / 12.0 * tangent[0] * tangent[0];
        } else {
            halftheta_by_tan_of_halftheta = -(halftheta * params[1]) / real_minus_one;
        }
        M::<2, 2>::new(
            halftheta_by_tan_of_halftheta,
            halftheta,
            -halftheta,
            halftheta_by_tan_of_halftheta,
        )
    }

    fn adj_of_translation(_params: &V<2>, point: &V<2>) -> SMatrix<f64, 2, 1> {
        V::<2>::new(point[1], -point[0])
    }

    fn ad_of_translation(point: &V<2>) -> SMatrix<f64, 2, 1> {
        V::<2>::new(point[1], -point[0])
    }
}

pub type Isometry2Impl =
    lie::semi_direct_product::SemiDirectProductImpl<3, 4, 2, 3, 1, 2, Rotation2Impl>;

pub type Isometry2 = lie::lie_group::LieGroup<3, 4, 2, 3, Isometry2Impl>;

mod tests {

    use super::*;

    #[test]
    fn rotation2_prop_tests() {
        Rotation2::test_suite();
    }

    #[test]
    fn rotation2_unit_tests() {
        let angle_30_deg = V::<1>::new(30.0_f64.to_radians());
        let rot_30deg = Rotation2::exp(&angle_30_deg);
        //print!("rot_30deg: {}", rot_30deg.compact());

        let hat_30deg = Rotation2::hat(&angle_30_deg);
        //print!("hat_30deg: {}", hat_30deg);

        let vee_30deg = Rotation2::vee(&hat_30deg);
        //print!("vee_30deg: {}", vee_30deg);

        approx::assert_relative_eq!(angle_30_deg, vee_30deg, epsilon = 1e-6);
    }

    #[test]
    fn isometry2_prop_tests() {
        Isometry2::test_suite();
    }

    #[test]
    fn isometry2_unit_tests() {
        let angle_30_deg = V::<1>::new(30.0_f64.to_radians());
        let rot_30deg = Rotation2::exp(&angle_30_deg);
        let translation = V::<2>::new(1.0, 2.0);
        let isometry = Isometry2::from_t_and_subgroup(&translation, &rot_30deg);

        // print!("isometry: {}", isometry.compact());

        // let hat_30deg = Isometry2::hat(&angle_30_deg);
        // print!("hat_30deg: {}", hat_30deg);

        // let vee_30deg = Isometry2::vee(&hat_30deg);
        // print!("vee_30deg: {}", vee_30deg);

        // approx::assert_relative_eq!(angle_30_deg, vee_30deg, epsilon = 1e-6);
    }
}
