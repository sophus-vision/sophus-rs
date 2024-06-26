use crate::distortions::affine::AffineDistortionImpl;
use crate::distortions::kannala_brandt::KannalaBrandtDistortionImpl;
use crate::prelude::*;
use crate::projections::perspective::PerspectiveProjectionImpl;
use crate::Camera;
use sophus_image::ImageSize;

/// Pinhole camera
pub type PinholeCamera<S, const BATCH: usize> =
    Camera<S, 0, 4, BATCH, AffineDistortionImpl<S, BATCH>, PerspectiveProjectionImpl>;
/// Kannala-Brandt camera
pub type KannalaBrandtCamera<S, const BATCH: usize> =
    Camera<S, 4, 8, BATCH, KannalaBrandtDistortionImpl<S, BATCH>, PerspectiveProjectionImpl>;

/// Perspective camera enum
#[derive(Debug, Clone)]
pub enum PerspectiveCameraEnum<S: IsScalar<BATCH>, const BATCH: usize> {
    /// Pinhole camera
    Pinhole(PinholeCamera<S, BATCH>),
    /// Kannala-Brandt camera
    KannalaBrandt(KannalaBrandtCamera<S, BATCH>),
}

impl<S: IsScalar<BATCH>, const BATCH: usize> IsCameraEnum<S, BATCH>
    for PerspectiveCameraEnum<S, BATCH>
{
    fn new_pinhole(params: &S::Vector<4>, image_size: ImageSize) -> Self {
        Self::Pinhole(PinholeCamera::from_params_and_size(params, image_size))
    }
    fn image_size(&self) -> ImageSize {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.image_size(),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.image_size(),
        }
    }

    fn new_kannala_brandt(params: &S::Vector<8>, image_size: ImageSize) -> Self {
        Self::KannalaBrandt(KannalaBrandtCamera::from_params_and_size(
            params, image_size,
        ))
    }

    fn cam_proj(&self, point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.cam_proj(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.cam_proj(point_in_camera),
        }
    }

    fn cam_unproj_with_z(&self, point_in_camera: &S::Vector<2>, z: S) -> S::Vector<3> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.cam_unproj_with_z(point_in_camera, z),
            PerspectiveCameraEnum::KannalaBrandt(camera) => {
                camera.cam_unproj_with_z(point_in_camera, z)
            }
        }
    }

    fn distort(&self, point_in_camera: &S::Vector<2>) -> S::Vector<2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.distort(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.distort(point_in_camera),
        }
    }

    fn undistort(&self, point_in_camera: &S::Vector<2>) -> S::Vector<2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.undistort(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.undistort(point_in_camera),
        }
    }

    fn dx_distort_x(&self, point_in_camera: &S::Vector<2>) -> S::Matrix<2, 2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.dx_distort_x(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.dx_distort_x(point_in_camera),
        }
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> IsPerspectiveCameraEnum<S, BATCH>
    for PerspectiveCameraEnum<S, BATCH>
{
    fn pinhole_params(&self) -> S::Vector<4> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.params().clone(),
            PerspectiveCameraEnum::KannalaBrandt(camera) => {
                camera.params().get_fixed_subvec::<4>(0)
            }
        }
    }
}
