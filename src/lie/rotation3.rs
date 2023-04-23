use std::f32::consts::E;

use crate::lie::traits::LieGroupImpl;
use nalgebra::{SMatrix, SVector};

use crate::{
    calculus, lie,
    manifold::{self},
};

type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

#[derive(Debug, Copy, Clone)]
pub struct Rotation3Impl;

impl Rotation3Impl {}

impl calculus::traits::ParamsImpl<4> for Rotation3Impl {
    fn params_examples() -> Vec<V<4>> {
        let mut params = vec![];

        params.push(*Rotation3::exp(&V::<3>::new(0.0, 0.0, 0.0)).params());
        params.push(*Rotation3::exp(&V::<3>::new(1.0, 0.0, 0.0)).params());
        params.push(*Rotation3::exp(&V::<3>::new(0.0, 1.0, 0.0)).params());
        params.push(*Rotation3::exp(&V::<3>::new(0.0, 0.2, 0.2)).params());

        params
    }

    fn invalid_params_examples() -> Vec<V<4>> {
        vec![
            V::<4>::new(0.0, 0.0, 0.0, 0.0),
            V::<4>::new(0.5, 0.5, 0.5, 0.0),
            V::<4>::new(0.5, -0.5, 0.5, 1.0),
        ]
    }

    fn are_params_valid(params: &V<4>) -> bool {
        let norm = params.norm();
        (norm - 1.0).abs() < 1e-6
    }
}

impl manifold::traits::TangentImpl<3> for Rotation3Impl {
    fn tangent_examples() -> Vec<V<3>> {
        vec![
            V::<3>::new(0.0, 0.0, 0.0),
            V::<3>::new(1.0, 0.0, 0.0),
            V::<3>::new(0.0, 1.0, 0.0),
            V::<3>::new(0.0, 0.0, 1.0),
            V::<3>::new(0.5, 0.5, 0.5),
            V::<3>::new(-0.5, -0.5, -0.5),
        ]
    }
}

impl lie::traits::LieGroupImpl<3, 4, 3, 3> for Rotation3Impl {
    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = true;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> V<4> {
        V::<4>::new(1.0, 0.0, 0.0, 0.0)
    }

    fn adj(params: &V<4>) -> M<3, 3> {
        Self::matrix(params)
    }

    fn exp(omega: &V<3>) -> V<4> {
        const EPS: f64 = 1e-8;
        let theta_sq = omega.norm_squared();

        let (imag_factor, real_factor) = if theta_sq < EPS * EPS {
            let theta_po4 = theta_sq * theta_sq;
            (
                0.5 - 1.0 / 48.0 * theta_sq + 1.0 / 3840.0 * theta_po4,
                1.0 - 1.0 / 8.0 * theta_sq + 1.0 / 384.0 * theta_po4,
            )
        } else {
            let theta = theta_sq.sqrt();
            let half_theta = 0.5 * theta;
            (half_theta.sin() / theta, half_theta.cos())
        };
        V::<4>::new(
            imag_factor * omega[0],
            imag_factor * omega[1],
            imag_factor * omega[2],
            real_factor,
        )
    }

    fn log(params: &V<4>) -> V<3> {
        println!("params: {:?}", params);
        const EPS: f64 = 1e-8;
        let ivec = V::<3>::new(params[0], params[1], params[2]);

        let squared_n = ivec.norm_squared();
        let w = params[3];

        let two_atan_nbyd_by_n = if squared_n < EPS * EPS {
            assert!(w.abs() > EPS, "|params| should be close to 1. (w = {})", w);
            let w_sq = w * w;
            2.0 / w - 2.0 / 3.0 * squared_n / (w_sq * w)
        } else {
            let n = squared_n.sqrt();
            let atan_nbyw = if w < 0.0 { -n.atan2(-w) } else { n.atan2(w) };
            2.0 * atan_nbyw / n
        };
        two_atan_nbyd_by_n * ivec
    }

    fn hat(omega: &V<3>) -> M<3, 3> {
        let mut omega_hat = M::<3, 3>::zeros();
        omega_hat[(0, 1)] = -omega[2];
        omega_hat[(0, 2)] = omega[1];
        omega_hat[(1, 0)] = omega[2];
        omega_hat[(1, 2)] = -omega[0];
        omega_hat[(2, 0)] = -omega[1];
        omega_hat[(2, 1)] = omega[0];
        omega_hat
    }

    fn vee(omega_hat: &M<3, 3>) -> V<3> {
        V::<3>::new(omega_hat[(2, 1)], omega_hat[(0, 2)], omega_hat[(1, 0)])
    }

    fn multiply(params1: &V<4>, params2: &V<4>) -> V<4> {
        *(nalgebra::geometry::Quaternion::from_vector(*params1)
            * nalgebra::geometry::Quaternion::from_vector(*params2))
        .as_vector()
    }

    fn inverse(params: &V<4>) -> V<4> {
        let mut result = V::<4>::zeros();
        result[0] = -params[0];
        result[1] = -params[1];
        result[2] = -params[2];
        result[3] = params[3];

        result
    }

    fn transform(params: &V<4>, point: &V<3>) -> V<3> {
        Self::matrix(params) * point
    }

    fn to_ambient(point: &V<3>) -> V<3> {
        *point
    }

    fn compact(params: &V<4>) -> M<3, 3> {
        Self::matrix(params)
    }

    fn matrix(params: &V<4>) -> M<3, 3> {
        *nalgebra::UnitQuaternion::from_quaternion(nalgebra::geometry::Quaternion::from_vector(
            *params,
        ))
        .to_rotation_matrix()
        .matrix()
    }

    fn ad(tangent: &V<3>) -> M<3, 3> {
        Self::hat(tangent)
    }
}

pub type Rotation3 = lie::lie_group::LieGroup<3, 4, 3, 3, Rotation3Impl>;

impl lie::traits::LieSubgroupImplTrait<3, 4, 3, 3> for Rotation3Impl {
    fn mat_v(_params: &V<4>, omega: &V<3>) -> M<3, 3> {
        let theta_sq = omega.norm_squared();
        let mat_omega = Rotation3Impl::hat(omega);
        let mat_omega_sq = mat_omega * mat_omega;
        if theta_sq < 1e-6 {
            M::<3, 3>::identity() + 0.5 * mat_omega
        } else {
            let theta = theta_sq.sqrt();
            M::<3, 3>::identity()
                + (1. - theta.cos()) / theta_sq * mat_omega
                + (theta - theta.sin()) / (theta_sq * theta) * mat_omega_sq
        }
    }

    fn mat_v_inverse(_params: &V<4>, tangent: &V<3>) -> M<3, 3> {
        let theta_sq = tangent.norm_squared();
        let mat_omega = Rotation3Impl::hat(tangent);
        if theta_sq < 1e-6 {
            M::<3, 3>::identity() - 0.5 * mat_omega + 1. / 12. * (mat_omega * mat_omega)
        } else {
            let theta = theta_sq.sqrt();
            let half_theta = 0.5 * theta;
            M::<3, 3>::identity() - 0.5 * mat_omega
                + (1. - 0.5 * theta * half_theta.cos() / half_theta.sin()) / (theta * theta)
                    * (mat_omega * mat_omega)
        }
    }

    fn adj_of_translation(params: &V<4>, point: &V<3>) -> M<3, 3> {
        Rotation3Impl::hat(point) * Rotation3Impl::matrix(params)
    }

    fn ad_of_translation(point: &V<3>) -> M<3, 3> {
        Rotation3Impl::hat(point)
    }
}

pub type Isometry3Impl =
    lie::semi_direct_product::SemiDirectProductImpl<6, 7, 3, 4, 3, 4, Rotation3Impl>;
pub type Isometry3 = lie::lie_group::LieGroup<6, 7, 3, 4, Isometry3Impl>;

mod tests {

    use super::*;

    #[test]
    fn rotation3_prop_tests() {
        Rotation3::test_suite();
    }

    #[test]
    fn isometry3_prop_tests() {
        Isometry3::test_suite();
    }
}
