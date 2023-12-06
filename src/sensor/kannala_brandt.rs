use std::marker::PhantomData;

use nalgebra::RowVector2;

use crate::calculus::types::params::ParamsImpl;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::M;
use crate::calculus::types::V;

use super::affine::AffineDistortionImpl;
use super::traits::IsCameraDistortionImpl;

#[derive(Debug, Clone, Copy)]
pub struct KannalaBrandtDistortionImpl<S: IsScalar> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar> ParamsImpl<S, 8> for KannalaBrandtDistortionImpl<S> {
    fn are_params_valid(params: &S::Vector<8>) -> bool {
        params.real()[0] != 0.0 && params.real()[1] != 0.0
    }

    fn params_examples() -> Vec<S::Vector<8>> {
        vec![S::Vector::<8>::from_c_array([
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])]
    }

    fn invalid_params_examples() -> Vec<S::Vector<8>> {
        vec![
            S::Vector::<8>::from_c_array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            S::Vector::<8>::from_c_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    }
}

impl<S: IsScalar> IsCameraDistortionImpl<S, 4, 8> for KannalaBrandtDistortionImpl<S> {
    fn distort(
        params: &S::Vector<8>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Vector<2> {
        let k0 = params.get(4);
        let k1 = params.get(5);
        let k2 = params.get(6);
        let k3 = params.get(7);

        let radius_sq = proj_point_in_camera_z1_plane.get(0) * proj_point_in_camera_z1_plane.get(0)
            + proj_point_in_camera_z1_plane.get(1) * proj_point_in_camera_z1_plane.get(1);

        if radius_sq.real() > 1e-8 {
            let radius = radius_sq.sqrt();
            let radius_inverse = S::c(1.0) / radius.clone();
            let theta = radius.atan2(1.0.into());
            let theta2 = theta.clone() * theta.clone();
            let theta4 = theta2.clone() * theta2.clone();
            let theta6 = theta2.clone() * theta4.clone();
            let theta8 = theta4.clone() * theta4.clone();

            let r_distorted =
                theta * (S::c(1.0) + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8);
            let scaling = r_distorted * radius_inverse;
            return S::Vector::<2>::from_array([
                scaling.clone() * proj_point_in_camera_z1_plane.get(0) * params.get(0)
                    + params.get(2),
                scaling * proj_point_in_camera_z1_plane.get(1) * params.get(1) + params.get(3),
            ]);
        }
        let pinhole_params = params.get_fixed_rows::<4>(0);
        AffineDistortionImpl::<S>::distort(&pinhole_params, proj_point_in_camera_z1_plane)
    }

    fn undistort(params: &S::Vector<8>, distorted_point: &S::Vector<2>) -> S::Vector<2> {
        let fu = params.get(0);
        let fv = params.get(1);
        let u0 = params.get(2);
        let v0 = params.get(3);

        let k0 = params.get(4);
        let k1 = params.get(5);
        let k2 = params.get(6);
        let k3 = params.get(7);

        let un = (distorted_point.get(0) - u0) / fu;
        let vn = (distorted_point.get(1) - v0) / fv;
        let rth2 = un.clone() * un.clone() + vn.clone() * vn.clone();

        if rth2.real() < 1e-8 {
            return S::Vector::<2>::from_array([un, vn]);
        }

        let rth = rth2.sqrt();

        let mut th = rth.clone().sqrt();

        let mut iters = 0;
        loop {
            let th2 = th.clone() * th.clone();
            let th4 = th2.clone() * th2.clone();
            let th6 = th2.clone() * th4.clone();
            let th8 = th4.clone() * th4.clone();

            let thd = th.clone()
                * (S::c(1.0)
                    + k0.clone() * th2.clone()
                    + k1.clone() * th4.clone()
                    + k2.clone() * th6.clone()
                    + k3.clone() * th8.clone());
            let d_thd_wtr_th = S::c(1.0)
                + S::c(3.0) * k0.clone() * th2
                + S::c(5.0) * k1.clone() * th4
                + S::c(7.0) * k2.clone() * th6
                + S::c(9.0) * k3.clone() * th8;

            let step = (thd - rth.clone()) / d_thd_wtr_th;
            th = th - step.clone();

            if step.real().abs() < 1e-8 {
                break;
            }

            iters += 1;

            const MAX_ITERS: usize = 20;
            const HIGH_ITERS: usize = MAX_ITERS / 2;

            if iters == HIGH_ITERS {
                // debug!(
                //     "undistort: did not converge in {} iterations, step: {}",
                //     iters, step
                // );
            }

            if iters > HIGH_ITERS {
                // trace!(
                //     "undistort: did not converge in {} iterations, step: {}",
                //     iters,
                //     step
                // );
            }

            if iters >= 20 {
                // warn!("undistort: max iters ({}) reached, step: {}", iters, step);
                break;
            }
        }

        let radius_undistorted = th.tan();

        if radius_undistorted.real() < 0.0 {
            S::Vector::<2>::from_array([
                -radius_undistorted.clone() * un / rth.clone(),
                -radius_undistorted * vn / rth,
            ])
        } else {
            S::Vector::<2>::from_array([
                radius_undistorted.clone() * un / rth.clone(),
                radius_undistorted * vn / rth,
            ])
        }
    }

    fn dx_distort_x(params: &V<8>, proj_point_in_camera_z1_plane: &V<2>) -> M<2, 2> {
        let a = proj_point_in_camera_z1_plane[0];
        let b = proj_point_in_camera_z1_plane[1];
        let fx = params[0];
        let fy = params[1];

        let k = params.fixed_rows::<4>(4).into_owned();

        let radius_sq = a * a + b * b;

        if radius_sq < 1e-8 {
            return M::<2, 2>::from_diagonal(&V::<2>::new(fx, fy));
        }

        let c0 = a.powi(2);
        let c1 = b.powi(2);
        let c2 = c0 + c1;
        let c3 = c2.powf(5.0 / 2.0);
        let c4 = c2 + 1.0;
        let c5 = c2.sqrt().atan();
        let c6 = c5.powi(2);
        let c7 = c6 * k[0];
        let c8 = c5.powi(4);
        let c9 = c8 * k[1];
        let c10 = c5.powi(6);
        let c11 = c10 * k[2];
        let c12 = c5.powi(8) * k[3];
        let c13 = 1.0 * c4 * c5 * (c11 + c12 + c7 + c9 + 1.0);
        let c14 = c13 * c3;
        let c15 = c2.powf(3.0 / 2.0);
        let c16 = c13 * c15;
        let c17 = 1.0 * c11
            + 1.0 * c12
            + 2.0 * c6 * (4.0 * c10 * k[3] + 2.0 * c6 * k[1] + 3.0 * c8 * k[2] + k[0])
            + 1.0 * c7
            + 1.0 * c9
            + 1.0;
        let c18 = c17 * c2.powi(2);
        let c19 = 1.0 / c4;
        let c20 = c19 / c2.powi(3);
        let c21 = a * b * c19 * (-c13 * c2 + c15 * c17) / c3;

        M::<2, 2>::from_rows(&[
            RowVector2::new(c20 * fx * (-c0 * c16 + c0 * c18 + c14), c21 * fx),
            RowVector2::new(c21 * fy, c20 * fy * (-c1 * c16 + c1 * c18 + c14)),
        ])
    }
}
