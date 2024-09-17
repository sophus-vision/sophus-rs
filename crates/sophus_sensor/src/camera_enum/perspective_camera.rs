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

impl<S: IsScalar<BATCH>, const BATCH: usize> IsCameraEnum<S, BATCH>
    for PerspectiveCameraEnum<S, BATCH>
{
    fn new_pinhole(params: &S::Vector<4>, image_size: ImageSize) -> Self {
        Self::Pinhole(PinholeCamera::from_params_and_size(params, image_size))
    }

    fn new_kannala_brandt(params: &S::Vector<8>, image_size: ImageSize) -> Self {
        Self::KannalaBrandt(KannalaBrandtCamera::from_params_and_size(
            params, image_size,
        ))
    }

    fn new_brown_conrady(params: &S::Vector<12>, image_size: ImageSize) -> Self {
        Self::BrownConrady(BrownConradyCamera::from_params_and_size(params, image_size))
    }

    fn new_unified(params: &<S as IsScalar<BATCH>>::Vector<6>, image_size: ImageSize) -> Self {
        Self::UnifiedExtended(UnifiedCamera::from_params_and_size(params, image_size))
    }

    fn image_size(&self) -> ImageSize {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.image_size(),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.image_size(),
            PerspectiveCameraEnum::BrownConrady(camera) => camera.image_size(),
            PerspectiveCameraEnum::UnifiedExtended(camera) => camera.image_size(),
        }
    }

    fn cam_proj(&self, point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.cam_proj(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.cam_proj(point_in_camera),
            PerspectiveCameraEnum::BrownConrady(camera) => camera.cam_proj(point_in_camera),
            PerspectiveCameraEnum::UnifiedExtended(camera) => camera.cam_proj(point_in_camera),
        }
    }

    fn cam_unproj_with_z(&self, point_in_camera: &S::Vector<2>, z: S) -> S::Vector<3> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.cam_unproj_with_z(point_in_camera, z),
            PerspectiveCameraEnum::KannalaBrandt(camera) => {
                camera.cam_unproj_with_z(point_in_camera, z)
            }
            PerspectiveCameraEnum::BrownConrady(camera) => {
                camera.cam_unproj_with_z(point_in_camera, z)
            }
            PerspectiveCameraEnum::UnifiedExtended(camera) => {
                camera.cam_unproj_with_z(point_in_camera, z)
            }
        }
    }

    fn distort(&self, point_in_camera: &S::Vector<2>) -> S::Vector<2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.distort(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.distort(point_in_camera),
            PerspectiveCameraEnum::BrownConrady(camera) => camera.distort(point_in_camera),
            PerspectiveCameraEnum::UnifiedExtended(camera) => camera.distort(point_in_camera),
        }
    }

    fn undistort(&self, point_in_camera: &S::Vector<2>) -> S::Vector<2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.undistort(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.undistort(point_in_camera),
            PerspectiveCameraEnum::BrownConrady(camera) => camera.undistort(point_in_camera),
            PerspectiveCameraEnum::UnifiedExtended(camera) => camera.undistort(point_in_camera),
        }
    }

    fn dx_distort_x(&self, point_in_camera: &S::Vector<2>) -> S::Matrix<2, 2> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.dx_distort_x(point_in_camera),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.dx_distort_x(point_in_camera),
            PerspectiveCameraEnum::BrownConrady(camera) => camera.dx_distort_x(point_in_camera),
            PerspectiveCameraEnum::UnifiedExtended(camera) => camera.dx_distort_x(point_in_camera),
        }
    }

    fn try_get_brown_conrady(self) -> Option<BrownConradyCamera<S, BATCH>> {
        match self {
            PerspectiveCameraEnum::Pinhole(_) => None,
            PerspectiveCameraEnum::KannalaBrandt(_) => None,
            PerspectiveCameraEnum::BrownConrady(camera) => Some(camera),
            PerspectiveCameraEnum::UnifiedExtended(_) => None,
        }
    }

    fn try_get_kannala_brandt(self) -> Option<KannalaBrandtCamera<S, BATCH>> {
        match self {
            PerspectiveCameraEnum::Pinhole(_) => None,
            PerspectiveCameraEnum::KannalaBrandt(camera) => Some(camera),
            PerspectiveCameraEnum::BrownConrady(_) => None,
            PerspectiveCameraEnum::UnifiedExtended(_) => None,
        }
    }

    fn try_get_pinhole(self) -> Option<PinholeCamera<S, BATCH>> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => Some(camera),
            PerspectiveCameraEnum::KannalaBrandt(_) => None,
            PerspectiveCameraEnum::BrownConrady(_) => None,
            PerspectiveCameraEnum::UnifiedExtended(_) => None,
        }
    }

    fn try_get_unified_extended(self) -> Option<UnifiedCamera<S, BATCH>> {
        match self {
            PerspectiveCameraEnum::Pinhole(_) => None,
            PerspectiveCameraEnum::KannalaBrandt(_) => None,
            PerspectiveCameraEnum::BrownConrady(_) => None,
            PerspectiveCameraEnum::UnifiedExtended(camera) => Some(camera),
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
            PerspectiveCameraEnum::BrownConrady(camera) => camera.params().get_fixed_subvec::<4>(0),
            PerspectiveCameraEnum::UnifiedExtended(camera) => {
                camera.params().get_fixed_subvec::<4>(0)
            }
        }
    }
}
