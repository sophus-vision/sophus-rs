use crate::affine::AffineDistortionImpl;
use crate::distortion_table::DistortTable;
use crate::generic_camera::Camera;
use crate::kannala_brandt::KannalaBrandtDistortionImpl;
use crate::traits::IsCameraEnum;
use crate::traits::IsProjection;

use sophus_calculus::types::matrix::IsMatrix;
use sophus_calculus::types::scalar::IsScalar;
use sophus_calculus::types::vector::IsVector;
use sophus_calculus::types::MatF64;
use sophus_calculus::types::VecF64;
use sophus_image::image_view::ImageSize;
use sophus_image::mut_image::MutImage2F32;

/// Perspective camera projection - using z=1 plane
///
/// Projects a 3D point in the camera frame to a 2D point in the z=1 plane
#[derive(Debug, Clone, Copy)]
pub struct ProjectionZ1;

impl<S: IsScalar<1>> IsProjection<S> for ProjectionZ1 {
    fn proj(point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        S::Vector::<2>::from_array([
            point_in_camera.get(0) / point_in_camera.get(2),
            point_in_camera.get(1) / point_in_camera.get(2),
        ])
    }

    fn unproj(point_in_camera: &S::Vector<2>, extension: S) -> S::Vector<3> {
        S::Vector::<3>::from_array([
            point_in_camera.get(0) * extension.clone(),
            point_in_camera.get(1) * extension.clone(),
            extension,
        ])
    }

    fn dx_proj_x(point_in_camera: &VecF64<3>) -> MatF64<2, 3> {
        MatF64::<2, 3>::from_array2([
            [
                1.0 / point_in_camera[2],
                0.0,
                -point_in_camera[0] / (point_in_camera[2] * point_in_camera[2]),
            ],
            [
                0.0,
                1.0 / point_in_camera[2],
                -point_in_camera[1] / (point_in_camera[2] * point_in_camera[2]),
            ],
        ])
    }
}

/// Pinhole camera
pub type PinholeCamera<S> = Camera<S, 0, 4, AffineDistortionImpl<S>, ProjectionZ1>;
/// Kannala-Brandt camera
pub type KannalaBrandtCamera<S> = Camera<S, 4, 8, KannalaBrandtDistortionImpl<S>, ProjectionZ1>;

/// Perspective camera enum
#[derive(Debug, Clone, Copy)]
pub enum PerspectiveCameraEnum {
    /// Pinhole camera
    Pinhole(PinholeCamera<f64>),
    /// Kannala-Brandt camera
    KannalaBrandt(KannalaBrandtCamera<f64>),
}

impl IsCameraEnum for PerspectiveCameraEnum {
    fn new_pinhole(params: &VecF64<4>, image_size: ImageSize) -> Self {
        Self::Pinhole(PinholeCamera::from_params_and_size(params, image_size))
    }

    fn new_kannala_brandt(params: &VecF64<8>, image_size: ImageSize) -> Self {
        Self::KannalaBrandt(KannalaBrandtCamera::from_params_and_size(
            params, image_size,
        ))
    }

    fn cam_proj(&self, point_in_camera: &VecF64<3>) -> VecF64<2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.cam_proj(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.cam_proj(point_in_camera),
        }
    }

    fn cam_unproj_with_z(&self, point_in_camera: &VecF64<2>, z: f64) -> VecF64<3> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.cam_unproj_with_z(point_in_camera, z),
            PerspectiveCameraEnum::KannalaBrandt(camera) => {
                camera.cam_unproj_with_z(point_in_camera, z)
            }
        }
    }

    fn distort(&self, point_in_camera: &VecF64<2>) -> VecF64<2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.distort(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.distort(point_in_camera),
        }
    }

    fn undistort(&self, point_in_camera: &VecF64<2>) -> VecF64<2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.undistort(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.undistort(point_in_camera),
        }
    }

    fn dx_distort_x(&self, point_in_camera: &VecF64<2>) -> MatF64<2, 2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.dx_distort_x(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.dx_distort_x(point_in_camera),
        }
    }

    fn undistort_table(&self) -> MutImage2F32 {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.undistort_table(),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.undistort_table(),
        }
    }

    fn image_size(&self) -> ImageSize {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.image_size(),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.image_size(),
        }
    }

    fn distort_table(&self) -> DistortTable {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.distort_table(),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.distort_table(),
        }
    }
}
