use crate::prelude::*;
use crate::traits::IsCameraDistortionImpl;
use core::borrow::Borrow;
use core::marker::PhantomData;
use sophus_autodiff::linalg::EPS_F64;
use sophus_autodiff::params::IsParamsImpl;

extern crate alloc;

/// Kannala-Brandt distortion implementation
#[derive(Debug, Clone, Copy)]
pub struct KannalaBrandtDistortionImpl<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsParamsImpl<S, 8, BATCH, DM, DN> for KannalaBrandtDistortionImpl<S, BATCH, DM, DN>
{
    fn are_params_valid<P>(_params: P) -> S::Mask
    where
        P: Borrow<S::Vector<8>>,
    {
        S::Mask::all_true()
    }

    fn params_examples() -> alloc::vec::Vec<S::Vector<8>> {
        alloc::vec![S::Vector::<8>::from_f64_array([
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])]
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<8>> {
        alloc::vec![
            S::Vector::<8>::from_f64_array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            S::Vector::<8>::from_f64_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsCameraDistortionImpl<S, 4, 8, BATCH, DM, DN>
    for KannalaBrandtDistortionImpl<S, BATCH, DM, DN>
{
    fn distort<PA, PO>(params: PA, proj_point_in_camera_z1_plane: PO) -> S::Vector<2>
    where
        PA: Borrow<S::Vector<8>>,
        PO: Borrow<S::Vector<2>>,
    {
        let params = params.borrow();
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();

        let k0 = params.get_elem(4);
        let k1 = params.get_elem(5);
        let k2 = params.get_elem(6);
        let k3 = params.get_elem(7);

        let radius_sq = proj_point_in_camera_z1_plane.get_elem(0)
            * proj_point_in_camera_z1_plane.get_elem(0)
            + proj_point_in_camera_z1_plane.get_elem(1) * proj_point_in_camera_z1_plane.get_elem(1);

        let radius = radius_sq.clone().sqrt();
        let radius_inverse = S::from_f64(1.0) / radius;
        let theta = radius.atan2(S::from_f64(1.0));
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta2 * theta4;
        let theta8 = theta4 * theta4;

        let r_distorted =
            theta * (S::from_f64(1.0) + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8);
        let scaling = r_distorted * radius_inverse;

        let near_zero = radius_sq.less_equal(&S::from_f64(EPS_F64));

        let scaling = S::ones().select(&near_zero, scaling);

        S::Vector::<2>::from_array([
            scaling * proj_point_in_camera_z1_plane.get_elem(0) * params.get_elem(0)
                + params.get_elem(2),
            scaling * proj_point_in_camera_z1_plane.get_elem(1) * params.get_elem(1)
                + params.get_elem(3),
        ])
    }

    fn undistort<PA, PO>(params: PA, distorted_point: PO) -> S::Vector<2>
    where
        PA: Borrow<S::Vector<8>>,
        PO: Borrow<S::Vector<2>>,
    {
        let params = params.borrow();
        let distorted_point = distorted_point.borrow();
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
        let rth2 = un * un + vn * vn;

        let rth2_near_zero = rth2.less_equal(&S::from_f64(EPS_F64));
        let point_z1_plane0 = S::Vector::<2>::from_array([un, vn]);

        let rth = rth2.sqrt();

        let mut th = rth.clone().sqrt();

        let mut iters = 0;
        loop {
            let th2 = th * th;
            let th4 = th2 * th2;
            let th6 = th2 * th4;
            let th8 = th4 * th4;

            let thd = th * (S::from_f64(1.0) + k0 * th2 + k1 * th4 + k2 * th6 + k3 * th8);
            let d_thd_wtr_th = S::from_f64(1.0)
                + S::from_f64(3.0) * k0 * th2
                + S::from_f64(5.0) * k1 * th4
                + S::from_f64(7.0) * k2 * th6
                + S::from_f64(9.0) * k3 * th8;

            let step = (thd - rth) / d_thd_wtr_th;
            th -= step;

            if (step
                .real_part()
                .abs()
                .less_equal(&S::RealScalar::from_f64(EPS_F64)))
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
                sign * radius_undistorted * un / rth,
                sign * radius_undistorted * vn / rth,
            ]),
        )
    }

    fn dx_distort_x<PA, PO>(params: PA, proj_point_in_camera_z1_plane: PO) -> S::Matrix<2, 2>
    where
        PA: Borrow<S::Vector<8>>,
        PO: Borrow<S::Vector<2>>,
    {
        let params = params.borrow();
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();

        let a = proj_point_in_camera_z1_plane.get_elem(0);
        let b = proj_point_in_camera_z1_plane.get_elem(1);
        let fx = params.get_elem(0);
        let fy = params.get_elem(1);

        let k = params.get_fixed_subvec::<4>(4);

        let radius_sq = a * a + b * b;

        let near_zero = radius_sq.less_equal(&S::from_f64(EPS_F64));

        let dx0 = S::Matrix::<2, 2>::from_array2([[fx, S::zeros()], [S::zeros(), fy]]);

        let c0 = a * a;
        let c1 = b * b;
        let c2 = c0 + c1;
        let c2_sqrt = c2.clone().sqrt();
        let c3 = c2_sqrt * c2_sqrt * c2_sqrt * c2_sqrt * c2_sqrt;
        let c4 = c2 + S::from_f64(1.0);
        let c5 = c2_sqrt.clone().atan();
        let c6 = c5 * c5; // c5^2
        let c7 = c6 * k.get_elem(0);
        let c8 = c6 * c6; // c5^4
        let c9 = c8 * k.get_elem(1);
        let c10 = c8 * c6; // c5^6
        let c11 = c10 * k.get_elem(2);
        let c12 = c8 * c8 * k.get_elem(3); // c5^8 * k[3]
        let c13 = S::from_f64(1.0) * c4 * c5 * (c11 + c12 + c7 + c9 + S::from_f64(1.0));
        let c14 = c13 * c3;
        let c15 = c2_sqrt * c2_sqrt * c2_sqrt;
        let c16 = c13 * c15;
        let c17 = S::from_f64(1.0) * c11
            + S::from_f64(1.0) * c12
            + S::from_f64(2.0)
                * c6
                * (S::from_f64(4.0) * c10 * k.get_elem(3)
                    + S::from_f64(2.0) * c6 * k.get_elem(1)
                    + S::from_f64(3.0) * c8 * k.get_elem(2)
                    + k.get_elem(0))
            + S::from_f64(1.0) * c7
            + S::from_f64(1.0) * c9
            + S::from_f64(1.0);
        let c18 = c17 * c2 * c2;
        let c19 = S::from_f64(1.0) / c4;
        let c20 = c19 / (c2 * c2 * c2);
        let c21 = a * b * c19 * (-c13 * c2 + c15 * c17) / c3;

        let dx = S::Matrix::<2, 2>::from_array2([
            [c20 * fx * (-c0 * c16 + c0 * c18 + c14), c21 * fx],
            [c21 * fy, c20 * fy * (-c1 * c16 + c1 * c18 + c14)],
        ]);

        dx0.select(&near_zero, dx)
    }

    fn dx_distort_params<PA, PO>(params: PA, proj_point_in_camera_z1_plane: PO) -> S::Matrix<2, 8>
    where
        PA: Borrow<S::Vector<8>>,
        PO: Borrow<S::Vector<2>>,
    {
        let params = params.borrow();
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();

        let x = proj_point_in_camera_z1_plane.get_elem(0);
        let y = proj_point_in_camera_z1_plane.get_elem(1);
        let fx = params.get_elem(0);
        let fy = params.get_elem(1);
        let k0 = params.get_elem(4);
        let k1 = params.get_elem(5);
        let k2 = params.get_elem(6);
        let k3 = params.get_elem(7);

        let xx = x * x;
        let yy = y * y;
        let radius_sq = xx + yy;
        let radius = radius_sq.clone().sqrt();
        let theta = radius.clone().atan2(S::from_f64(1.0));
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta2 * theta4;
        let theta8 = theta4 * theta4;

        // alpha = x * atan(r^2) / r^2
        //
        // x==0 and y==0, then
        //
        // alpha = 1
        let alpha = theta / radius;
        let r_near_zero = radius_sq.clone().abs().less_equal(&S::from_f64(EPS_F64));
        let alpha = S::ones().select(&r_near_zero, alpha);

        let alpha_x = x * alpha;
        let alpha_y = y * alpha;

        let scaling_by_theta =
            S::from_f64(1.0) + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8;

        let dx_dfx = alpha_x * scaling_by_theta;
        let dy_dfy = alpha_y * scaling_by_theta;
        let dx_dcx = S::from_f64(1.0);
        let dy_dcy = S::from_f64(1.0);

        let dr_dk0_by_theta = theta2;
        let dr_dk1_by_theta = theta4;
        let dr_dk2_by_theta = theta6;
        let dr_dk3_by_theta = theta8;

        let dx_dk0 = fx * alpha_x * dr_dk0_by_theta;
        let dx_dk1 = fx * alpha_x * dr_dk1_by_theta;
        let dx_dk2 = fx * alpha_x * dr_dk2_by_theta;
        let dx_dk3 = fx * alpha_x * dr_dk3_by_theta;

        let dy_dk0 = fy * alpha_y * dr_dk0_by_theta;
        let dy_dk1 = fy * alpha_y * dr_dk1_by_theta;
        let dy_dk2 = fy * alpha_y * dr_dk2_by_theta;
        let dy_dk3 = fy * alpha_y * dr_dk3_by_theta;

        S::Matrix::<2, 8>::from_array2([
            [
                dx_dfx,
                S::zeros(),
                dx_dcx,
                S::zeros(),
                dx_dk0,
                dx_dk1,
                dx_dk2,
                dx_dk3,
            ],
            [
                S::zeros(),
                dy_dfy,
                S::zeros(),
                dy_dcy,
                dy_dk0,
                dy_dk1,
                dy_dk2,
                dy_dk3,
            ],
        ])
    }
}
