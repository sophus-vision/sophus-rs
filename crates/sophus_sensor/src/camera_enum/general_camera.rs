use crate::camera_enum::perspective_camera::PerspectiveCameraEnum;
use crate::prelude::*;
use crate::projections::orthographic::OrthographicCamera;
use sophus_image::ImageSize;

/// Generalized camera enum
#[derive(Debug, Clone)]
pub enum GeneralCameraEnum<S: IsScalar<BATCH>, const BATCH: usize> {
    /// Perspective camera enum
    Perspective(PerspectiveCameraEnum<S, BATCH>),
    /// Orthographic camera
    Orthographic(OrthographicCamera<S, BATCH>),
}

impl<S: IsScalar<BATCH>, const BATCH: usize> GeneralCameraEnum<S, BATCH> {
    /// Create a new perspective camera instance
    pub fn new_perspective(model: PerspectiveCameraEnum<S, BATCH>) -> Self {
        Self::Perspective(model)
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> IsCameraEnum<S, BATCH>
    for GeneralCameraEnum<S, BATCH>
{
    fn new_pinhole(params: &S::Vector<4>, image_size: ImageSize) -> Self {
        Self::Perspective(PerspectiveCameraEnum::new_pinhole(params, image_size))
    }

    fn new_kannala_brandt(params: &S::Vector<8>, image_size: ImageSize) -> Self {
        Self::Perspective(PerspectiveCameraEnum::new_kannala_brandt(
            params, image_size,
        ))
    }

    fn new_brown_conrady(params: &S::Vector<12>, image_size: ImageSize) -> Self {
        Self::Perspective(PerspectiveCameraEnum::new_brown_conrady(params, image_size))
    }

    fn cam_proj(&self, point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.cam_proj(point_in_camera),
            GeneralCameraEnum::Orthographic(camera) => camera.cam_proj(point_in_camera),
        }
    }

    fn cam_unproj_with_z(&self, point_in_camera: &S::Vector<2>, z: S) -> S::Vector<3> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.cam_unproj_with_z(point_in_camera, z),
            GeneralCameraEnum::Orthographic(camera) => camera.cam_unproj_with_z(point_in_camera, z),
        }
    }

    fn distort(&self, point_in_camera: &S::Vector<2>) -> S::Vector<2> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.distort(point_in_camera),
            GeneralCameraEnum::Orthographic(camera) => camera.distort(point_in_camera),
        }
    }

    fn undistort(&self, point_in_camera: &S::Vector<2>) -> S::Vector<2> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.undistort(point_in_camera),
            GeneralCameraEnum::Orthographic(camera) => camera.undistort(point_in_camera),
        }
    }

    fn dx_distort_x(&self, point_in_camera: &S::Vector<2>) -> S::Matrix<2, 2> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.dx_distort_x(point_in_camera),
            GeneralCameraEnum::Orthographic(camera) => camera.dx_distort_x(point_in_camera),
        }
    }

    fn image_size(&self) -> ImageSize {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.image_size(),
            GeneralCameraEnum::Orthographic(camera) => camera.image_size(),
        }
    }

    fn try_get_brown_conrady(self) -> Option<crate::BrownConradyCamera<S, BATCH>> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.try_get_brown_conrady(),
            GeneralCameraEnum::Orthographic(_) => None,
        }
    }

    fn try_get_kannala_brandt(self) -> Option<crate::KannalaBrandtCamera<S, BATCH>> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.try_get_kannala_brandt(),
            GeneralCameraEnum::Orthographic(_) => None,
        }
    }

    fn try_get_pinhole(self) -> Option<crate::PinholeCamera<S, BATCH>> {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.try_get_pinhole(),
            GeneralCameraEnum::Orthographic(_) => None,
        }
    }
}
