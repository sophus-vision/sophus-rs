use crate::prelude::*;
use crate::traits::IsCameraDistortionImpl;
use sophus_core::params::ParamsImpl;
use sophus_core::prelude::IsMatrix;
use sophus_core::prelude::IsScalar;
use sophus_core::prelude::IsVector;
use std::marker::PhantomData;

/// Unified Extended distortion implementation
#[derive(Debug, Clone, Copy)]
pub struct UnifiedDistortionImpl<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH>, const BATCH: usize> ParamsImpl<S, 6, BATCH>
    for UnifiedDistortionImpl<S, BATCH>
{
    fn are_params_valid(_params: &S::Vector<6>) -> S::Mask {
        S::Mask::all_true()
    }

    fn params_examples() -> Vec<S::Vector<6>> {
        vec![S::Vector::<6>::from_f64_array([
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ])]
    }

    fn invalid_params_examples() -> Vec<S::Vector<6>> {
        vec![
            S::Vector::<6>::from_f64_array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            S::Vector::<6>::from_f64_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    }
}

/// unified extended for z==1
impl<S: IsScalar<BATCH>, const BATCH: usize> IsCameraDistortionImpl<S, 2, 6, BATCH>
    for UnifiedDistortionImpl<S, BATCH>
{
    fn distort(
        params: &<S as IsScalar<BATCH>>::Vector<6>,
        proj_point_in_camera_z1_plane: &<S as IsScalar<BATCH>>::Vector<2>,
    ) -> <S as IsScalar<BATCH>>::Vector<2> {
        // https://gitlab.com/VladyslavUsenko/basalt-headers/-/blob/master/include/basalt/camera/extended_camera.hpp?ref_type=heads#L125
        let fx = params.get_elem(0);
        let fy = params.get_elem(1);
        let cx = params.get_elem(2);
        let cy = params.get_elem(3);
        let alpha = params.get_elem(4);
        let beta = params.get_elem(5);

        let x = proj_point_in_camera_z1_plane.get_elem(0);
        let y = proj_point_in_camera_z1_plane.get_elem(1);

        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let rho2 = beta * r2 + S::from_f64(1.0);
        let rho = rho2.sqrt();

        let norm = alpha.clone() * rho.clone() + (S::from_f64(1.0) - alpha.clone());

        let mx = x / norm.clone();
        let my = y / norm.clone();

        S::Vector::<2>::from_array([fx * mx + cx, fy * my + cy])
    }

    fn undistort(
        params: &<S as IsScalar<BATCH>>::Vector<6>,
        distorted_point: &<S as IsScalar<BATCH>>::Vector<2>,
    ) -> <S as IsScalar<BATCH>>::Vector<2> {
        let fx = params.get_elem(0);
        let fy = params.get_elem(1);
        let cx = params.get_elem(2);
        let cy = params.get_elem(3);
        let alpha = params.get_elem(4);
        let beta = params.get_elem(5);

        let mx = (distorted_point.get_elem(0) - cx.clone()) / fx.clone();
        let my = (distorted_point.get_elem(1) - cy.clone()) / fy.clone();

        let r2 = mx.clone() * mx.clone() + my.clone() * my.clone();
        let gamma = S::from_f64(1.0) - alpha.clone();

        let tmp1 = S::from_f64(1.0) - alpha.clone() * alpha.clone() * beta.clone() * r2.clone();
        let tmp_sqrt =
            (S::from_f64(1.0) - (alpha.clone() - gamma.clone()) * beta.clone() * r2.clone()).sqrt();
        let tmp2 = alpha.clone() * tmp_sqrt.clone() + gamma.clone();

        let k = tmp1.clone() / tmp2.clone();

        S::Vector::<2>::from_array([mx / k.clone(), my / k.clone()])
    }

    fn dx_distort_x(
        params: &<S as IsScalar<BATCH>>::Vector<6>,
        proj_point_in_camera_z1_plane: &<S as IsScalar<BATCH>>::Vector<2>,
    ) -> <S as IsScalar<BATCH>>::Matrix<2, 2> {
        let fx = params.get_elem(0);
        let fy = params.get_elem(1);
        let alpha = params.get_elem(4);
        let beta = params.get_elem(5);

        let x = proj_point_in_camera_z1_plane.get_elem(0);
        let y = proj_point_in_camera_z1_plane.get_elem(1);

        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let rho2 = beta.clone() * r2.clone() + S::from_f64(1.0);
        let rho = rho2.sqrt();

        let norm = alpha.clone() * rho.clone() + (S::from_f64(1.0) - alpha.clone());

        // Compute partial derivatives
        let drho2_dx = S::from_f64(2.0) * beta.clone() * x.clone();
        let drho2_dy = S::from_f64(2.0) * beta.clone() * y.clone();

        let drho_dx = drho2_dx.clone() / (S::from_f64(2.0) * rho.clone());
        let drho_dy = drho2_dy.clone() / (S::from_f64(2.0) * rho.clone());

        let dnorm_dx = alpha.clone() * drho_dx.clone();
        let dnorm_dy = alpha.clone() * drho_dy.clone();

        let dmx_dx = (norm.clone() - x.clone() * dnorm_dx.clone()) / (norm.clone() * norm.clone());
        let dmx_dy = -x.clone() * dnorm_dy.clone() / (norm.clone() * norm.clone());

        let dmy_dx = -y.clone() * dnorm_dx.clone() / (norm.clone() * norm.clone());
        let dmy_dy = (norm.clone() - y.clone() * dnorm_dy.clone()) / (norm.clone() * norm.clone());

        S::Matrix::<2, 2>::from_array2([
            [fx.clone() * dmx_dx, fx.clone() * dmx_dy],
            [fy.clone() * dmy_dx, fy.clone() * dmy_dy],
        ])
    }

    fn dx_distort_params(
        params: &<S as IsScalar<BATCH>>::Vector<6>,
        proj_point_in_camera_z1_plane: &<S as IsScalar<BATCH>>::Vector<2>,
    ) -> <S as IsScalar<BATCH>>::Matrix<2, 6> {
        let fx = params.get_elem(0);
        let fy = params.get_elem(1);
        let alpha = params.get_elem(4);
        let beta = params.get_elem(5);

        let x = proj_point_in_camera_z1_plane.get_elem(0);
        let y = proj_point_in_camera_z1_plane.get_elem(1);

        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let rho2 = beta.clone() * r2.clone() + S::from_f64(1.0);
        let rho = rho2.sqrt();

        let norm = alpha.clone() * rho.clone() + (S::from_f64(1.0) - alpha.clone());

        // Compute partial derivatives
        let drho2_dbeta = r2.clone();
        let drho_dbeta = drho2_dbeta.clone() / (S::from_f64(2.0) * rho.clone());

        let dnorm_dalpha = rho.clone() - S::from_f64(1.0);
        let dnorm_dbeta = alpha.clone() * drho_dbeta.clone();

        let mx = x.clone() / norm.clone();
        let my = y.clone() / norm.clone();

        let dmx_dalpha = -x.clone() * dnorm_dalpha.clone() / (norm.clone() * norm.clone());
        let dmy_dalpha = -y.clone() * dnorm_dalpha.clone() / (norm.clone() * norm.clone());

        let dmx_dbeta = -x.clone() * dnorm_dbeta.clone() / (norm.clone() * norm.clone());
        let dmy_dbeta = -y.clone() * dnorm_dbeta.clone() / (norm.clone() * norm.clone());

        S::Matrix::<2, 6>::from_array2([
            [
                mx,
                S::zero(),
                S::ones(),
                S::zero(),
                fx.clone() * dmx_dalpha,
                fx * dmx_dbeta,
            ],
            [
                S::zero(),
                my,
                S::zero(),
                S::ones(),
                fy.clone() * dmy_dalpha,
                fy * dmy_dbeta,
            ],
        ])
    }

    fn identity_params() -> S::Vector<6> {
        let mut params = S::Vector::<6>::zeros();
        params.set_elem(0, <S>::ones());
        params.set_elem(1, <S>::ones());
        params
    }
}
