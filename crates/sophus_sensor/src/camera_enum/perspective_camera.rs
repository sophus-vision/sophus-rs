use core::borrow::Borrow;

use crate::distortions::affine::AffineDistortionImpl;
use crate::distortions::brown_conrady::BrownConradyDistortionImpl;
use crate::distortions::kannala_brandt::KannalaBrandtDistortionImpl;
use crate::distortions::unified::UnifiedDistortionImpl;
use crate::prelude::*;
use crate::projections::perspective::PerspectiveProjectionImpl;
use crate::Camera;
use sophus_image::ImageSize;

/// Pinhole camera
pub type PinholeCamera<S, const BATCH: usize> =
    Camera<S, 0, 4, BATCH, AffineDistortionImpl<S, BATCH>, PerspectiveProjectionImpl<S, BATCH>>;
/// Kannala-Brandt camera
pub type KannalaBrandtCamera<S, const BATCH: usize> = Camera<
    S,
    4,
    8,
    BATCH,
    KannalaBrandtDistortionImpl<S, BATCH>,
    PerspectiveProjectionImpl<S, BATCH>,
>;
/// Brown-Conrady camera
pub type BrownConradyCamera<S, const BATCH: usize> = Camera<
    S,
    8,
    12,
    BATCH,
    BrownConradyDistortionImpl<S, BATCH>,
    PerspectiveProjectionImpl<S, BATCH>,
>;
/// unified camera
pub type UnifiedCamera<S, const BATCH: usize> =
    Camera<S, 2, 6, BATCH, UnifiedDistortionImpl<S, BATCH>, PerspectiveProjectionImpl<S, BATCH>>;

/// Pinhole camera with f64 scalar type
pub type PinholeCameraF64 = PinholeCamera<f64, 1>;
/// Kannala-Brandt camera with f64 scalar type
pub type KannalaBrandtCameraF64 = KannalaBrandtCamera<f64, 1>;
/// Brown-Conrady camera with f64 scalar type
pub type BrownConradyCameraF64 = BrownConradyCamera<f64, 1>;
/// Unified camera with f64 scalar type
pub type UnifiedCameraF64 = UnifiedCamera<f64, 1>;

/// Perspective camera enum
#[derive(Debug, Clone)]
pub enum PerspectiveCameraEnum<S: IsScalar<BATCH>, const BATCH: usize> {
    /// Pinhole camera
    Pinhole(PinholeCamera<S, BATCH>),
    /// Kannala-Brandt camera
    KannalaBrandt(KannalaBrandtCamera<S, BATCH>),
    /// Brown-Conrady camera
    BrownConrady(BrownConradyCamera<S, BATCH>),
    /// Unified Extended camera
    UnifiedExtended(UnifiedCamera<S, BATCH>),
}

impl<S: IsScalar<BATCH>, const BATCH: usize> IsCamera<S, BATCH>
    for PerspectiveCameraEnum<S, BATCH>
{
    fn new_pinhole<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<4>>,
    {
        let params = params.borrow();
        PerspectiveCameraEnum::Pinhole(PinholeCamera::new(params, image_size))
    }

    fn new_kannala_brandt<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<8>>,
    {
        let params = params.borrow();
        PerspectiveCameraEnum::KannalaBrandt(KannalaBrandtCamera::new(params, image_size))
    }

    fn new_brown_conrady<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<12>>,
    {
        let params = params.borrow();
        PerspectiveCameraEnum::BrownConrady(BrownConradyCamera::new(params, image_size))
    }

    fn new_unified<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<6>>,
    {
        let params = params.borrow();
        PerspectiveCameraEnum::UnifiedExtended(UnifiedCamera::new(params, image_size))
    }

    fn image_size(&self) -> ImageSize {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.image_size(),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.image_size(),
            PerspectiveCameraEnum::BrownConrady(camera) => camera.image_size(),
            PerspectiveCameraEnum::UnifiedExtended(camera) => camera.image_size(),
        }
    }

    fn cam_proj<P>(&self, point_in_camera: P) -> S::Vector<2>
    where
        P: Borrow<S::Vector<3>>,
    {
        let point_in_camera = point_in_camera.borrow();

        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.cam_proj(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.cam_proj(point_in_camera),
            PerspectiveCameraEnum::BrownConrady(camera) => camera.cam_proj(point_in_camera),
            PerspectiveCameraEnum::UnifiedExtended(camera) => camera.cam_proj(point_in_camera),
        }
    }

    fn cam_unproj_with_z<P>(&self, pixel: P, z: S) -> S::Vector<3>
    where
        P: Borrow<S::Vector<2>>,
    {
        let pixel = pixel.borrow();
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.cam_unproj_with_z(pixel, z),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.cam_unproj_with_z(pixel, z),
            PerspectiveCameraEnum::BrownConrady(camera) => camera.cam_unproj_with_z(pixel, z),
            PerspectiveCameraEnum::UnifiedExtended(camera) => camera.cam_unproj_with_z(pixel, z),
        }
    }

    fn distort<P>(&self, proj_point_in_camera_z1_plane: P) -> S::Vector<2>
    where
        P: Borrow<S::Vector<2>>,
    {
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.distort(proj_point_in_camera_z1_plane),
            PerspectiveCameraEnum::KannalaBrandt(camera) => {
                camera.distort(proj_point_in_camera_z1_plane)
            }
            PerspectiveCameraEnum::BrownConrady(camera) => {
                camera.distort(proj_point_in_camera_z1_plane)
            }
            PerspectiveCameraEnum::UnifiedExtended(camera) => {
                camera.distort(proj_point_in_camera_z1_plane)
            }
        }
    }

    fn undistort<P>(&self, pixel: P) -> S::Vector<2>
    where
        P: Borrow<S::Vector<2>>,
    {
        let pixel = pixel.borrow();
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.undistort(pixel),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.undistort(pixel),
            PerspectiveCameraEnum::BrownConrady(camera) => camera.undistort(pixel),
            PerspectiveCameraEnum::UnifiedExtended(camera) => camera.undistort(pixel),
        }
    }

    fn dx_distort_x<P>(&self, proj_point_in_camera_z1_plane: P) -> S::Matrix<2, 2>
    where
        P: Borrow<S::Vector<2>>,
    {
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => {
                camera.dx_distort_x(proj_point_in_camera_z1_plane)
            }
            PerspectiveCameraEnum::KannalaBrandt(camera) => {
                camera.dx_distort_x(proj_point_in_camera_z1_plane)
            }
            PerspectiveCameraEnum::BrownConrady(camera) => {
                camera.dx_distort_x(proj_point_in_camera_z1_plane)
            }
            PerspectiveCameraEnum::UnifiedExtended(camera) => {
                camera.dx_distort_x(proj_point_in_camera_z1_plane)
            }
        }
    }

    fn try_get_brown_conrady(self) -> Option<BrownConradyCamera<S, BATCH>> {
        if let PerspectiveCameraEnum::BrownConrady(camera) = self {
            Some(camera)
        } else {
            None
        }
    }

    fn try_get_kannala_brandt(self) -> Option<KannalaBrandtCamera<S, BATCH>> {
        if let PerspectiveCameraEnum::KannalaBrandt(camera) = self {
            Some(camera)
        } else {
            None
        }
    }

    fn try_get_pinhole(self) -> Option<PinholeCamera<S, BATCH>> {
        if let PerspectiveCameraEnum::Pinhole(camera) = self {
            Some(camera)
        } else {
            None
        }
    }

    fn try_get_unified_extended(self) -> Option<UnifiedCamera<S, BATCH>> {
        if let PerspectiveCameraEnum::UnifiedExtended(camera) = self {
            Some(camera)
        } else {
            None
        }
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> IsPerspectiveCamera<S, BATCH>
    for PerspectiveCameraEnum<S, BATCH>
{
    fn pinhole_params(&self) -> S::Vector<4> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.params().clone(),
            PerspectiveCameraEnum::KannalaBrandt(camera) => {
                camera.params().clone().get_fixed_subvec(0)
            }
            PerspectiveCameraEnum::BrownConrady(camera) => {
                camera.params().clone().get_fixed_subvec(0)
            }
            PerspectiveCameraEnum::UnifiedExtended(camera) => {
                camera.params().clone().get_fixed_subvec(0)
            }
        }
    }
}
