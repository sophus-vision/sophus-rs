use nalgebra::SMatrix;
use nalgebra::SVector;

use crate::image::view::ImageSize;
use crate::image::mut_image::MutImage2F32;
use crate::manifold::traits::ParamsImpl;

type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

pub trait CameraDistortionImpl<const DISTORT: usize, const PARAMS: usize>:
    ParamsImpl<f64, PARAMS>
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

#[derive(Debug, Copy, Clone)]
pub struct WrongParamsDim;

pub trait CameraEnum {
    fn new_pinhole(params: &V<4>, image_size: ImageSize) -> Self;
    fn new_kannala_brandt(params: &V<8>, image_size: ImageSize) -> Self;

    fn cam_proj(&self, point_in_camera: &V<3>) -> V<2>;
    fn cam_unproj_with_z(&self, point_in_camera: &V<2>, z: f64) -> V<3>;
    fn distort(&self, point_in_camera: &V<2>) -> V<2>;
    fn undistort(&self, point_in_camera: &V<2>) -> V<2>;
    fn undistort_table(&self) -> MutImage2F32;

    fn dx_distort_x(&self, point_in_camera: &V<2>) -> M<2, 2>;

    fn try_set_params(&mut self, params: &nalgebra::DVector<f64>) -> Result<(), WrongParamsDim>;
}
