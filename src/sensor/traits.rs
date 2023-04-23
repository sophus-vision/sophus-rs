use nalgebra::{SMatrix, SVector};

use crate::calculus;
type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

pub trait CameraDistortionImpl<const DISTORT: usize, const PARAMS: usize>:
    calculus::traits::ParamsImpl<PARAMS>
{
    fn identity_params() -> V<PARAMS> {
        let mut params = V::<PARAMS>::zeros();
        params[0] = 1.0;
        params[1] = 1.0;
        params
    }

    fn distort(params: &V<PARAMS>, proj_point_in_camera_z1_plane: &V<2>) -> V<2>;

    fn undistort(params: &V<PARAMS>, distorted_point: &V<2>) -> V<2>;

    fn dx_distort_x(params: &V<PARAMS>, proj_point_in_camera_z1_plane: &V<2>) -> M<2, 2>;
}

pub trait Projection {
    fn proj(point_in_camera: &V<3>) -> V<2>;

    fn unproj(point_in_camera: &V<2>, extension: f64) -> V<3>;

    fn dx_proj_x(point_in_camera: &V<3>) -> M<2, 3>;
}
