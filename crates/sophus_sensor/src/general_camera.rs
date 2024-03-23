use crate::distortion_table::DistortTable;
use crate::ortho_camera::OrthoCamera;
use crate::perspective_camera::PerspectiveCameraEnum;
use crate::traits::IsCameraEnum;

use sophus_calculus::types::MatF64;
use sophus_calculus::types::VecF64;
use sophus_image::image_view::ImageSize;
use sophus_image::mut_image::MutImage2F32;

/// Generalized camera enum
#[derive(Debug, Clone)]
pub enum GeneralCameraEnum {
    /// Perspective camera enum
    Perspective(PerspectiveCameraEnum),
    /// Orthographic camera
    Ortho(OrthoCamera<f64>),
}

impl GeneralCameraEnum {
    /// Create a new perspective camera instance
    pub fn new_perspective(model: PerspectiveCameraEnum) -> Self {
        Self::Perspective(model)
    }
}

impl IsCameraEnum for GeneralCameraEnum {
    fn new_pinhole(params: &VecF64<4>, image_size: ImageSize) -> Self {
        Self::Perspective(PerspectiveCameraEnum::new_pinhole(params, image_size))
    }

    fn new_kannala_brandt(params: &VecF64<8>, image_size: ImageSize) -> Self {
        Self::Perspective(PerspectiveCameraEnum::new_kannala_brandt(
            params, image_size,
        ))
    }

    fn cam_proj(&self, point_in_camera: &VecF64<3>) -> VecF64<2> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.cam_proj(point_in_camera),
            GeneralCameraEnum::Ortho(camera) => camera.cam_proj(point_in_camera),
        }
    }

    fn cam_unproj_with_z(&self, point_in_camera: &VecF64<2>, z: f64) -> VecF64<3> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.cam_unproj_with_z(point_in_camera, z),
            GeneralCameraEnum::Ortho(camera) => camera.cam_unproj_with_z(point_in_camera, z),
        }
    }

    fn distort(&self, point_in_camera: &VecF64<2>) -> VecF64<2> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.distort(point_in_camera),
            GeneralCameraEnum::Ortho(camera) => camera.distort(point_in_camera),
        }
    }

    fn undistort(&self, point_in_camera: &VecF64<2>) -> VecF64<2> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.undistort(point_in_camera),
            GeneralCameraEnum::Ortho(camera) => camera.undistort(point_in_camera),
        }
    }

    fn dx_distort_x(&self, point_in_camera: &VecF64<2>) -> MatF64<2, 2> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.dx_distort_x(point_in_camera),
            GeneralCameraEnum::Ortho(camera) => camera.dx_distort_x(point_in_camera),
        }
    }

    fn undistort_table(&self) -> MutImage2F32 {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.undistort_table(),
            GeneralCameraEnum::Ortho(camera) => camera.undistort_table(),
        }
    }

    fn image_size(&self) -> ImageSize {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.image_size(),
            GeneralCameraEnum::Ortho(camera) => camera.image_size(),
        }
    }

    fn distort_table(&self) -> DistortTable {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.distort_table(),
            GeneralCameraEnum::Ortho(camera) => camera.distort_table(),
        }
    }
}
