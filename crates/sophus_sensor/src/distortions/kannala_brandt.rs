use crate::prelude::*;
use crate::traits::IsCameraDistortionImpl;
use sophus_core::params::ParamsImpl;
use std::marker::PhantomData;

/// Kannala-Brandt distortion implementation
#[derive(Debug, Clone, Copy)]
pub struct KannalaBrandtDistortionImpl<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH>, const BATCH: usize> ParamsImpl<S, 8, BATCH>
    for KannalaBrandtDistortionImpl<S, BATCH>
{
    fn are_params_valid(_params: &S::Vector<8>) -> S::Mask {
        S::Mask::all_true()
    }

    fn params_examples() -> Vec<S::Vector<8>> {
        vec![S::Vector::<8>::from_f64_array([
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])]
    }

    fn invalid_params_examples() -> Vec<S::Vector<8>> {
        vec![
            S::Vector::<8>::from_f64_array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            S::Vector::<8>::from_f64_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> IsCameraDistortionImpl<S, 4, 8, BATCH>
    for KannalaBrandtDistortionImpl<S, BATCH>
{
    fn distort(
        params: &S::Vector<8>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Vector<2> {
        let k0 = params.get_elem(4);
        let k1 = params.get_elem(5);
        let k2 = params.get_elem(6);
        let k3 = params.get_elem(7);

        let radius_sq = proj_point_in_camera_z1_plane.get_elem(0)
            * proj_point_in_camera_z1_plane.get_elem(0)
            + proj_point_in_camera_z1_plane.get_elem(1) * proj_point_in_camera_z1_plane.get_elem(1);

        let radius = radius_sq.clone().sqrt();
        let radius_inverse = S::from_f64(1.0) / radius.clone();
        let theta = radius.atan2(S::from_f64(1.0));
        let theta2 = theta.clone() * theta.clone();
        let theta4 = theta2.clone() * theta2.clone();
        let theta6 = theta2.clone() * theta4.clone();
        let theta8 = theta4.clone() * theta4.clone();

        let r_distorted =
            theta * (S::from_f64(1.0) + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8);
        let scaling = r_distorted * radius_inverse;

        let near_zero = radius_sq.less_equal(&S::from_f64(1e-8));

        let scaling = S::ones().select(&near_zero, scaling);

        S::Vector::<2>::from_array([
            scaling.clone() * proj_point_in_camera_z1_plane.get_elem(0) * params.get_elem(0)
                + params.get_elem(2),
            scaling * proj_point_in_camera_z1_plane.get_elem(1) * params.get_elem(1)
                + params.get_elem(3),
        ])
    }

    fn undistort(params: &S::Vector<8>, distorted_point: &S::Vector<2>) -> S::Vector<2> {
        let fu = params.get_elem(0);
        let fv = params.get_elem(1);
        let u0 = params.get_elem(2);
        let v0 = params.get_elem(3);

        let k0 = params.get_elem(4);
        let k1 = params.get_elem(5);
        let k2 = params.get_elem(6);
        let k3 = params.get_elem(7);

        let un = (distorted_point.get_elem(0) - u0) / fu;
        let vn = (distorted_point.get_elem(1) - v0) / fv;
        let rth2 = un.clone() * un.clone() + vn.clone() * vn.clone();

        let rth2_near_zero = rth2.less_equal(&S::from_f64(1e-8));
        let point_z1_plane0 = S::Vector::<2>::from_array([un.clone(), vn.clone()]);

        let rth = rth2.sqrt();

        let mut th = rth.clone().sqrt();

        let mut iters = 0;
        loop {
            let th2 = th.clone() * th.clone();
            let th4 = th2.clone() * th2.clone();
            let th6 = th2.clone() * th4.clone();
            let th8 = th4.clone() * th4.clone();

            let thd = th.clone()
                * (S::from_f64(1.0)
                    + k0.clone() * th2.clone()
                    + k1.clone() * th4.clone()
                    + k2.clone() * th6.clone()
                    + k3.clone() * th8.clone());
            let d_thd_wtr_th = S::from_f64(1.0)
                + S::from_f64(3.0) * k0.clone() * th2
                + S::from_f64(5.0) * k1.clone() * th4
                + S::from_f64(7.0) * k2.clone() * th6
                + S::from_f64(9.0) * k3.clone() * th8;

            let step = (thd - rth.clone()) / d_thd_wtr_th;
            th -= step.clone();

            if (step
                .real_part()
                .abs()
                .less_equal(&S::RealScalar::from_f64(1e-8)))
            .all()
            {
                break;
            }

            iters += 1;

            if iters >= 20 {
                // warn!("undistort: max iters ({}) reached, step: {}", iters, step);
                break;
            }
        }

        let radius_undistorted = th.tan();

        let radius_undistorted_near_zero = radius_undistorted.less_equal(&S::from_f64(0.0));

        let sign = S::from_f64(-1.0).select(&radius_undistorted_near_zero, S::ones());

        point_z1_plane0.select(
            &rth2_near_zero,
            S::Vector::<2>::from_array([
                sign.clone() * radius_undistorted.clone() * un / rth.clone(),
                sign * radius_undistorted * vn / rth,
            ]),
        )
    }

    fn dx_distort_x(
        params: &S::Vector<8>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Matrix<2, 2> {
        let a = proj_point_in_camera_z1_plane.get_elem(0);
        let b = proj_point_in_camera_z1_plane.get_elem(1);
        let fx = params.get_elem(0);
        let fy = params.get_elem(1);

        let k = params.get_fixed_subvec::<4>(4);

        let radius_sq = a.clone() * a.clone() + b.clone() * b.clone();

        let near_zero = radius_sq.less_equal(&S::from_f64(1e-8));

        let dx0 =
            S::Matrix::<2, 2>::from_array2([[fx.clone(), S::zeros()], [S::zeros(), fy.clone()]]);

        let c0 = a.clone() * a.clone();
        let c1 = b.clone() * b.clone();
        let c2 = c0.clone() + c1.clone();
        let c2_sqrt = c2.clone().sqrt();
        let c3 =
            c2_sqrt.clone() * c2_sqrt.clone() * c2_sqrt.clone() * c2_sqrt.clone() * c2_sqrt.clone();
        let c4 = c2.clone() + S::from_f64(1.0);
        let c5 = c2_sqrt.clone().atan();
        let c6 = c5.clone() * c5.clone(); // c5^2
        let c7 = c6.clone() * k.get_elem(0);
        let c8 = c6.clone() * c6.clone(); // c5^4
        let c9 = c8.clone() * k.get_elem(1);
        let c10 = c8.clone() * c6.clone(); // c5^6
        let c11 = c10.clone() * k.get_elem(2);
        let c12 = c8.clone() * c8.clone() * k.get_elem(3); // c5^8 * k[3]
        let c13 = S::from_f64(1.0)
            * c4.clone()
            * c5
            * (c11.clone() + c12.clone() + c7.clone() + c9.clone() + S::from_f64(1.0));
        let c14 = c13.clone() * c3.clone();
        let c15 = c2_sqrt.clone() * c2_sqrt.clone() * c2_sqrt.clone();
        let c16 = c13.clone() * c15.clone();
        let c17 = S::from_f64(1.0) * c11
            + S::from_f64(1.0) * c12
            + S::from_f64(2.0)
                * c6.clone()
                * (S::from_f64(4.0) * c10 * k.get_elem(3)
                    + S::from_f64(2.0) * c6 * k.get_elem(1)
                    + S::from_f64(3.0) * c8 * k.get_elem(2)
                    + k.get_elem(0))
            + S::from_f64(1.0) * c7
            + S::from_f64(1.0) * c9
            + S::from_f64(1.0);
        let c18 = c17.clone() * c2.clone() * c2.clone();
        let c19 = S::from_f64(1.0) / c4;
        let c20 = c19.clone() / (c2.clone() * c2.clone() * c2.clone());
        let c21 = a * b * c19 * (-c13 * c2 + c15 * c17) / c3;

        let dx = S::Matrix::<2, 2>::from_array2([
            [
                c20.clone()
                    * fx.clone()
                    * (-c0.clone() * c16.clone() + c0 * c18.clone() + c14.clone()),
                c21.clone() * fx,
            ],
            [
                c21 * fy.clone(),
                c20 * fy * (-c1.clone() * c16 + c1 * c18 + c14),
            ],
        ]);

        dx0.select(&near_zero, dx)
    }
}
