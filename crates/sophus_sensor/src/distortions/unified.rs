use crate::prelude::*;
use crate::traits::IsCameraDistortionImpl;
use core::borrow::Borrow;
use core::marker::PhantomData;
use sophus_autodiff::params::IsParamsImpl;
use sophus_autodiff::prelude::IsMatrix;
use sophus_autodiff::prelude::IsScalar;
use sophus_autodiff::prelude::IsVector;

extern crate alloc;

/// Unified Extended distortion implementation
#[derive(Debug, Clone, Copy)]
pub struct UnifiedDistortionImpl<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsParamsImpl<S, 6, BATCH, DM, DN> for UnifiedDistortionImpl<S, BATCH, DM, DN>
{
    fn are_params_valid<P>(_params: P) -> S::Mask
    where
        P: Borrow<S::Vector<6>>,
    {
        S::Mask::all_true()
    }

    fn params_examples() -> alloc::vec::Vec<S::Vector<6>> {
        alloc::vec![S::Vector::<6>::from_f64_array([
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ])]
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<6>> {
        alloc::vec![
            S::Vector::<6>::from_f64_array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            S::Vector::<6>::from_f64_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    }
}

/// unified extended for z==1
impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsCameraDistortionImpl<S, 2, 6, BATCH, DM, DN> for UnifiedDistortionImpl<S, BATCH, DM, DN>
{
    fn distort<PA, PO>(params: PA, proj_point_in_camera_z1_plane: PO) -> S::Vector<2>
    where
        PA: Borrow<S::Vector<6>>,
        PO: Borrow<S::Vector<2>>,
    {
        let params = params.borrow();
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();

        // https://gitlab.com/VladyslavUsenko/basalt-headers/-/blob/master/include/basalt/camera/extended_camera.hpp?ref_type=heads#L125
        let fx = params.get_elem(0);
        let fy = params.get_elem(1);
        let cx = params.get_elem(2);
        let cy = params.get_elem(3);
        let alpha = params.get_elem(4);
        let beta = params.get_elem(5);

        let x = proj_point_in_camera_z1_plane.get_elem(0);
        let y = proj_point_in_camera_z1_plane.get_elem(1);

        let r2 = x * x + y * y;
        let rho2 = beta * r2 + S::from_f64(1.0);
        let rho = rho2.sqrt();

        let norm = alpha * rho + (S::from_f64(1.0) - alpha);

        let mx = x / norm;
        let my = y / norm;

        S::Vector::<2>::from_array([fx * mx + cx, fy * my + cy])
    }

    fn undistort<PA, PO>(params: PA, distorted_point: PO) -> S::Vector<2>
    where
        PA: Borrow<S::Vector<6>>,
        PO: Borrow<S::Vector<2>>,
    {
        let params = params.borrow();
        let distorted_point = distorted_point.borrow();
        let fx = params.get_elem(0);
        let fy = params.get_elem(1);
        let cx = params.get_elem(2);
        let cy = params.get_elem(3);
        let alpha = params.get_elem(4);
        let beta = params.get_elem(5);

        let mx = (distorted_point.get_elem(0) - cx) / fx;
        let my = (distorted_point.get_elem(1) - cy) / fy;

        let r2 = mx * mx + my * my;
        let gamma = S::from_f64(1.0) - alpha;

        let nominator = S::from_f64(1.0) - alpha * alpha * beta * r2;
        let denominator = alpha * (S::from_f64(1.0) - (alpha - gamma) * beta * r2).sqrt() + gamma;

        let k = nominator / denominator;

        S::Vector::<2>::from_array([mx / k, my / k])
    }
    fn dx_distort_x<PA, PO>(params: PA, proj_point_in_camera_z1_plane: PO) -> S::Matrix<2, 2>
    where
        PA: Borrow<S::Vector<6>>,
        PO: Borrow<S::Vector<2>>,
    {
        let params = params.borrow();
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();

        let fx = params.get_elem(0);
        let fy = params.get_elem(1);
        let alpha = params.get_elem(4);
        let beta = params.get_elem(5);

        let x = proj_point_in_camera_z1_plane.get_elem(0);
        let y = proj_point_in_camera_z1_plane.get_elem(1);

        let r2 = x * x + y * y;
        let rho2 = beta * r2 + S::from_f64(1.0);
        let rho = rho2.sqrt();

        let norm = alpha * rho + (S::from_f64(1.0) - alpha);

        // Compute partial derivatives
        let drho2_dx = S::from_f64(2.0) * beta * x;
        let drho2_dy = S::from_f64(2.0) * beta * y;

        let drho_dx = drho2_dx / (S::from_f64(2.0) * rho);
        let drho_dy = drho2_dy / (S::from_f64(2.0) * rho);

        let dnorm_dx = alpha * drho_dx;
        let dnorm_dy = alpha * drho_dy;

        let dmx_dx = (norm - x * dnorm_dx) / (norm * norm);
        let dmx_dy = -x * dnorm_dy / (norm * norm);

        let dmy_dx = -y * dnorm_dx / (norm * norm);
        let dmy_dy = (norm - y * dnorm_dy) / (norm * norm);

        S::Matrix::<2, 2>::from_array2([[fx * dmx_dx, fx * dmx_dy], [fy * dmy_dx, fy * dmy_dy]])
    }

    fn dx_distort_params<PA, PO>(params: PA, proj_point_in_camera_z1_plane: PO) -> S::Matrix<2, 6>
    where
        PA: Borrow<S::Vector<6>>,
        PO: Borrow<S::Vector<2>>,
    {
        let params = params.borrow();
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();

        let fx = params.get_elem(0);
        let fy = params.get_elem(1);
        let alpha = params.get_elem(4);
        let beta = params.get_elem(5);

        let x = proj_point_in_camera_z1_plane.get_elem(0);
        let y = proj_point_in_camera_z1_plane.get_elem(1);

        let r2 = x * x + y * y;
        let rho2 = beta * r2 + S::from_f64(1.0);
        let rho = rho2.sqrt();

        let norm = alpha * rho + (S::from_f64(1.0) - alpha);

        // Compute partial derivatives
        let drho2_dbeta = r2;
        let drho_dbeta = drho2_dbeta / (S::from_f64(2.0) * rho);

        let dnorm_dalpha = rho - S::from_f64(1.0);
        let dnorm_dbeta = alpha * drho_dbeta;

        let mx = x / norm;
        let my = y / norm;

        let dmx_dalpha = -x * dnorm_dalpha / (norm * norm);
        let dmy_dalpha = -y * dnorm_dalpha / (norm * norm);

        let dmx_dbeta = -x * dnorm_dbeta / (norm * norm);
        let dmy_dbeta = -y * dnorm_dbeta / (norm * norm);

        S::Matrix::<2, 6>::from_array2([
            [
                mx,
                S::zero(),
                S::ones(),
                S::zero(),
                fx * dmx_dalpha,
                fx * dmx_dbeta,
            ],
            [
                S::zero(),
                my,
                S::zero(),
                S::ones(),
                fy * dmy_dalpha,
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
