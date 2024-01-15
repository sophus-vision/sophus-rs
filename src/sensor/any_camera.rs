use crate::image::mut_image::MutImage2F32;
use crate::image::view::ImageSize;

use super::ortho_camera::OrthoCamera;
use super::perspective_camera::PerspectiveCameraType;
use super::traits::DistortTable;
use super::traits::IsCameraEnum;

type V<const N: usize> = nalgebra::SVector<f64, N>;
type M<const N: usize, const O: usize> = nalgebra::SMatrix<f64, N, O>;

#[derive(Debug, Clone)]
pub enum AnyProjCameraType {
    Perspective(PerspectiveCameraType),
    Ortho(OrthoCamera<f64>),
}

impl AnyProjCameraType {
    pub fn new_perspective(model: PerspectiveCameraType) -> Self {
        Self::Perspective(model)
    }
}

impl IsCameraEnum for AnyProjCameraType {
    fn new_pinhole(params: &V<4>, image_size: ImageSize) -> Self {
        Self::Perspective(PerspectiveCameraType::new_pinhole(params, image_size))
    }

    fn new_kannala_brandt(params: &V<8>, image_size: ImageSize) -> Self {
        Self::Perspective(PerspectiveCameraType::new_kannala_brandt(
            params, image_size,
        ))
    }

    fn cam_proj(&self, point_in_camera: &V<3>) -> V<2> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.cam_proj(point_in_camera),
            AnyProjCameraType::Ortho(camera) => camera.cam_proj(point_in_camera),
        }
    }

    fn cam_unproj_with_z(&self, point_in_camera: &V<2>, z: f64) -> V<3> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.cam_unproj_with_z(point_in_camera, z),
            AnyProjCameraType::Ortho(camera) => camera.cam_unproj_with_z(point_in_camera, z),
        }
    }

    fn distort(&self, point_in_camera: &V<2>) -> V<2> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.distort(point_in_camera),
            AnyProjCameraType::Ortho(camera) => camera.distort(point_in_camera),
        }
    }

    fn undistort(&self, point_in_camera: &V<2>) -> V<2> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.undistort(point_in_camera),
            AnyProjCameraType::Ortho(camera) => camera.undistort(point_in_camera),
        }
    }

    fn dx_distort_x(&self, point_in_camera: &V<2>) -> M<2, 2> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.dx_distort_x(point_in_camera),
            AnyProjCameraType::Ortho(camera) => camera.dx_distort_x(point_in_camera),
        }
    }

    fn undistort_table(&self) -> MutImage2F32 {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.undistort_table(),
            AnyProjCameraType::Ortho(camera) => camera.undistort_table(),
        }
    }

    fn image_size(&self) -> ImageSize {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.image_size(),
            AnyProjCameraType::Ortho(camera) => camera.image_size(),
        }
    }

    fn distort_table(&self) -> DistortTable {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.distort_table(),
            AnyProjCameraType::Ortho(camera) => camera.distort_table(),
        }
    }
}
