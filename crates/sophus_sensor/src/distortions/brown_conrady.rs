use crate::distortions::affine::AffineDistortionImpl;
use crate::prelude::*;
use crate::traits::IsCameraDistortionImpl;
use sophus_core::params::ParamsImpl;
use std::marker::PhantomData;

/// Kannala-Brandt distortion implementation
#[derive(Debug, Clone, Copy)]
pub struct BrownConradyDistortionImpl<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH>, const BATCH: usize> ParamsImpl<S, 12, BATCH>
    for BrownConradyDistortionImpl<S, BATCH>
{
    fn are_params_valid(_params: &S::Vector<12>) -> S::Mask {
        S::Mask::all_true()
    }

    fn params_examples() -> Vec<S::Vector<12>> {
        vec![S::Vector::<12>::from_f64_array([
            286.0,
            286.0,
            424.0,
            400.0,
            0.726405,
            -0.0148413,
            1.38447e-05,
            0.000419742,
            -0.00514224,
            1.06774,
            0.128429,
            -0.019901,
        ])]
    }

    fn invalid_params_examples() -> Vec<S::Vector<12>> {
        vec![]
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> BrownConradyDistortionImpl<S, BATCH> {
    fn distortion_impl(
        params: &S::Vector<8>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Vector<2> {
        let x = proj_point_in_camera_z1_plane.get_elem(0);
        let y = proj_point_in_camera_z1_plane.get_elem(1);
        let d0 = params.get_elem(0);
        let d1 = params.get_elem(1);
        let d2 = params.get_elem(2);
        let k3 = params.get_elem(3);
        let d4 = params.get_elem(4);
        let d5 = params.get_elem(5);
        let d6 = params.get_elem(6);
        let d7 = params.get_elem(7);

        // From:
        // https://github.com/opencv/opencv/blob/63bb2abadab875fc648a572faccafee134f06fc8/modules/calib3d/src/calibration.cpp#L791

        let one = S::from_f64(1.0);
        let two = S::from_f64(2.0);

        let r2 = proj_point_in_camera_z1_plane.squared_norm();
        let r4 = r2.clone() * r2.clone();
        let r6 = r4.clone() * r2.clone();
        let a1 = two.clone() * x.clone() * y.clone();
        let a2 = r2.clone() + two.clone() * x.clone() * x.clone();
        let a3 = r2.clone() + two.clone() * y.clone() * y.clone();
        let cdist = one.clone() + d0 * r2.clone() + d1 * r4.clone() + d4 * r6.clone();
        let icdist2 = one.clone() / (one + d5 * r2 + d6 * r4 + d7 * r6);
        let xd0 = x * cdist.clone() * icdist2.clone() + d2.clone() * a1.clone() + k3.clone() * a2;
        let yd0 = y * cdist * icdist2 + d2 * a3 + k3 * a1;

        S::Vector::<2>::from_array([xd0, yd0])
    }

    fn undistort_impl(
        distortion_params: &S::Vector<8>,
        distorted_point: &S::Vector<2>,
        dbg_info_distorted_point: &S::Vector<2>,
    ) -> S::Vector<2> {
        // from https://github.com/farm-ng/farm-ng-core/blob/main/cpp/sophus/sensor/camera_distortion/brown_conrady.h

        // We had no luck with OpenCV's undistort. It seems not to be accurate if
        // "icdist" is close to 0.
        // https://github.com/opencv/opencv/blob/63bb2abadab875fc648a572faccafee134f06fc8/modules/calib3d/src/undistort.dispatch.cpp#L365
        //
        // Hence, we derive the inverse approximation scheme from scratch.
        //
        //
        // Objective: find that xy such that proj_impl(xy) = uv
        //
        // Using multivariate Newton scheme, by defining f and find the root of it:
        //
        //  f: R^2 -> R^2
        //  f(xy) :=  proj_impl(xy) - uv
        //
        //  xy_i+1 = xy_i + J^{-1} * f(xy)   with J being the Jacobian of f.
        //
        // TODO(hauke): There is most likely a 1-dimensional embedding and one only
        // need to solve a less computational heavy newton iteration...

        // initial guess
        let mut xy = distorted_point.clone();

        let d0 = distortion_params.get_elem(0);
        let d1 = distortion_params.get_elem(1);
        let d2 = distortion_params.get_elem(2);
        let k3 = distortion_params.get_elem(3);
        let d4 = distortion_params.get_elem(4);
        let d5 = distortion_params.get_elem(5);
        let d6 = distortion_params.get_elem(6);
        let d7 = distortion_params.get_elem(7);

        for _i in 0..50 {
            let x = xy.get_elem(0);
            let y = xy.get_elem(1);

            let f_xy =
                Self::distortion_impl(distortion_params, &xy.clone()) - distorted_point.clone();

            let du_dx;
            let du_dy;
            let dv_dx;
            let dv_dy;

            {
                // Generated by brown_conrady_camera.py
                let a = x;
                let b = y;

                let c0 = a.clone() * a.clone(); // pow(a, 2);
                let c1 = b.clone() * b.clone(); // pow(b, 2);
                let c2 = c0.clone() + c1.clone();
                let c3 = c2.clone() * c2.clone(); // pow(c2, 2);
                let c4 = c3.clone() * c2.clone(); // pow(c2, 3);
                let c5 = c2.clone() * d5.clone()
                    + c3.clone() * d6.clone()
                    + c4.clone() * d7.clone()
                    + S::from_f64(1.0);
                let c6 = c5.clone() * c5.clone(); // pow(c5, 2);
                let c7 = S::from_f64(1.0) / c6.clone();
                let c8 = a.clone() * k3.clone();
                let c9 = S::from_f64(2.0) * d2.clone();
                let c10 = S::from_f64(2.0) * c2.clone();
                let c11 = S::from_f64(3.0) * c3.clone();
                let c12 = c2.clone() * d0.clone();
                let c13 = c3.clone() * d1.clone();
                let c14 = c4.clone() * d4.clone();
                let c15 = S::from_f64(2.0)
                    * (c10.clone() * d6.clone() + c11.clone() * d7.clone() + d5.clone())
                    * (c12.clone() + c13.clone() + c14.clone() + S::from_f64(1.0));
                let c16 = S::from_f64(2.0) * c10 * d1.clone()
                    + S::from_f64(2.0) * c11 * d4.clone()
                    + S::from_f64(2.0) * d0.clone();
                let c17 = c12 + c13 + c14 + S::from_f64(1.0);
                let c18 = b.clone() * k3.clone();
                let c19 = a.clone() * b.clone();
                let c20 = -c15.clone() * c19.clone() + c16.clone() * c19 * c5.clone();
                du_dx = c7.clone()
                    * (-c0.clone() * c15.clone()
                        + c5.clone() * (c0 * c16.clone() + c17.clone())
                        + c6.clone() * (b.clone() * c9.clone() + S::from_f64(6.0) * c8.clone()));
                du_dy = c7.clone()
                    * (c20.clone()
                        + c6.clone() * (a.clone() * c9.clone() + S::from_f64(2.0) * c18.clone()));
                dv_dx = c7.clone()
                    * (c20
                        + c6.clone()
                            * (S::from_f64(2.0) * a.clone() * d2.clone() + S::from_f64(2.0) * c18));
                dv_dy = c7
                    * (-c1.clone() * c15
                        + c5 * (c1 * c16 + c17)
                        + c6 * (S::from_f64(6.0) * b.clone() * d2.clone() + S::from_f64(2.0) * c8));
            }

            //     | du_dx  du_dy |      | a  b |
            // J = |              |  =:  |      |
            //     | dv_dx  dv_dy |      | c  d |

            let a = du_dx;
            let b = du_dy;
            let c = dv_dx;
            let d = dv_dy;

            // | a  b | -1       1   |  d  -b |
            // |      |     =  ----- |        |
            // | c  d |        ad-bc | -c   a |

            let m: S::Matrix<2, 2> =
                S::Matrix::from_array2([[d.clone(), -b.clone()], [-c.clone(), a.clone()]]);

            let j_inv: S::Matrix<2, 2> = m.scaled(S::from_f64(1.0) / (a * d - b * c));
            let step: S::Vector<2> = j_inv * f_xy;

            let eps = 1e-10;

            if step
                .norm()
                .real_part()
                .less_equal(&S::RealScalar::from_f64(eps))
                .all()
            {
                break;
            }

            xy.set_elem(0, xy.get_elem(0) - step.get_elem(0));
            xy.set_elem(1, xy.get_elem(1) - step.get_elem(1));
        }

        let f_xy = Self::distortion_impl(distortion_params, &xy.clone()) - distorted_point.clone();
        if !f_xy
            .norm()
            .real_part()
            .less_equal(&S::RealScalar::from_f64(1e-6))
            .any()
        {
            println!(
                "WARNING: Newton did not converge: f_xy: {:?}: pixel {:?}",
                f_xy, dbg_info_distorted_point
            );
        }
        xy.clone()
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> IsCameraDistortionImpl<S, 8, 12, BATCH>
    for BrownConradyDistortionImpl<S, BATCH>
{
    fn distort(
        params: &S::Vector<12>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Vector<2> {
        let distorted_point_in_camera_z1_plane =
            Self::distortion_impl(&params.get_fixed_subvec(4), proj_point_in_camera_z1_plane);

        AffineDistortionImpl::<S, BATCH>::distort(
            &params.get_fixed_subvec(0),
            &distorted_point_in_camera_z1_plane,
        )
    }

    fn undistort(params: &S::Vector<12>, distorted_point: &S::Vector<2>) -> S::Vector<2> {
        let undistorted_point_in_camera_z1_plane = AffineDistortionImpl::<S, BATCH>::undistort(
            &params.get_fixed_subvec(0),
            distorted_point,
        );

        Self::undistort_impl(
            &params.get_fixed_subvec(4),
            &undistorted_point_in_camera_z1_plane,
            distorted_point,
        )
    }

    fn dx_distort_x(
        params: &S::Vector<12>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Matrix<2, 2> {
        const OFFSET: usize = 4;

        let fx = params.get_elem(0);
        let fy = params.get_elem(1);
        let d0 = params.get_elem(OFFSET);
        let d1 = params.get_elem(1 + OFFSET);
        let d2 = params.get_elem(2 + OFFSET);
        let d3 = params.get_elem(3 + OFFSET);
        let d4 = params.get_elem(4 + OFFSET);
        let d5 = params.get_elem(5 + OFFSET);
        let d6 = params.get_elem(6 + OFFSET);
        let d7 = params.get_elem(7 + OFFSET);

        let one = S::from_f64(1.0);
        let two = S::from_f64(2.0);

        let a = proj_point_in_camera_z1_plane.get_elem(0);
        let b = proj_point_in_camera_z1_plane.get_elem(1);

        // Generated by brown_conrady_camera.py
        let c0 = a.clone() * d3.clone();
        let c1 = two.clone() * d2.clone();
        let c2 = a.clone() * a.clone(); // pow(a, 2);
        let c3 = b.clone() * b.clone(); // pow(b, 2);
        let c4 = c2.clone() + c3.clone();
        let c5 = c4.clone() * c4.clone(); // pow(c4, 2);
        let c6 = c5.clone() * c4.clone(); // pow(c4, 3);
        let c7 = c4.clone() * d5.clone()
            + c5.clone() * d6.clone()
            + c6.clone() * d7.clone()
            + one.clone();
        let c8 = c7.clone() * c7.clone(); // pow(c7, 2);
        let c9 = two.clone() * c4.clone();
        let c10 = S::from_f64(3.0) * c5.clone();
        let c11 = c4.clone() * d0.clone();
        let c12 = c5.clone() * d1.clone();
        let c13 = c6.clone() * d4.clone();
        let c14 = two.clone()
            * (c10.clone() * d7.clone() + c9.clone() * d6.clone() + d5.clone())
            * (c11.clone() + c12.clone() + c13.clone() + one.clone());
        let c15 = two.clone() * c10.clone() * d4.clone()
            + two.clone() * c9.clone() * d1.clone()
            + two.clone() * d0.clone();
        let c16 = one.clone() * c11 + c12 + c13 + one.clone();
        let c17 = one.clone() / c8.clone();
        let c18 = c17.clone() * fx;
        let c19 = b.clone() * d3.clone();
        let c20 = a.clone() * b.clone();
        let c21 = -c14.clone() * c20.clone() + c15.clone() * c20.clone() * c7.clone();
        let c22 = c17.clone() * fy;
        S::Matrix::from_array2([
            [
                c18.clone()
                    * (-c14.clone() * c2.clone()
                        + c7.clone() * (c15.clone() * c2.clone() + c16.clone())
                        + c8.clone() * (b.clone() * c1.clone() + S::from_f64(6.0) * c0.clone())),
                c22.clone()
                    * (c21.clone()
                        + c8.clone()
                            * (two.clone() * a.clone() * d2.clone() + two.clone() * c19.clone())),
            ],
            [
                c18 * (c21 + c8.clone() * (a * c1 + two.clone() * c19)),
                c22 * (-c14 * c3.clone()
                    + c7 * (c15 * c3 + c16)
                    + c8 * (S::from_f64(6.0) * b * d2.clone() + two.clone() * c0)),
            ],
        ])
    }
}
