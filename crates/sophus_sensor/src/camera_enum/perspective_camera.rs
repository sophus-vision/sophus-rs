use core::borrow::Borrow;

use sophus_autodiff::manifold::IsVariable;
use sophus_image::ImageSize;

use crate::{
    Camera,
    distortions::{
        AffineDistortionImpl,
        BrownConradyDistortionImpl,
        EnhancedUnifiedDistortionImpl,
        KannalaBrandtDistortionImpl,
    },
    prelude::*,
    projections::PerspectiveProjectionImpl,
    traits::IsCameraDistortionImpl,
};

/// Pinhole camera
pub type PinholeCamera<S, const BATCH: usize, const DM: usize, const DN: usize> = Camera<
    S,
    0,
    4,
    BATCH,
    DM,
    DN,
    AffineDistortionImpl<S, BATCH, DM, DN>,
    PerspectiveProjectionImpl<S, BATCH, DM, DN>,
>;
/// Kannala-Brandt camera
pub type KannalaBrandtCamera<S, const BATCH: usize, const DM: usize, const DN: usize> = Camera<
    S,
    4,
    8,
    BATCH,
    DM,
    DN,
    KannalaBrandtDistortionImpl<S, BATCH, DM, DN>,
    PerspectiveProjectionImpl<S, BATCH, DM, DN>,
>;
/// Brown-Conrady camera
pub type BrownConradyCamera<S, const BATCH: usize, const DM: usize, const DN: usize> = Camera<
    S,
    8,
    12,
    BATCH,
    DM,
    DN,
    BrownConradyDistortionImpl<S, BATCH, DM, DN>,
    PerspectiveProjectionImpl<S, BATCH, DM, DN>,
>;
/// enhanced unified camera
pub type EnhancedUnifiedCamera<S, const BATCH: usize, const DM: usize, const DN: usize> = Camera<
    S,
    2,
    6,
    BATCH,
    DM,
    DN,
    EnhancedUnifiedDistortionImpl<S, BATCH, DM, DN>,
    PerspectiveProjectionImpl<S, BATCH, DM, DN>,
>;

/// Pinhole camera with f64 scalar type
pub type PinholeCameraF64 = PinholeCamera<f64, 1, 0, 0>;
/// Kannala-Brandt camera with f64 scalar type
pub type KannalaBrandtCameraF64 = KannalaBrandtCamera<f64, 1, 0, 0>;
/// Brown-Conrady camera with f64 scalar type
pub type BrownConradyCameraF64 = BrownConradyCamera<f64, 1, 0, 0>;
/// Enhanced unified camera model (EUCM) with f64 scalar type
pub type EnhancedUnifiedCameraF64 = EnhancedUnifiedCamera<f64, 1, 0, 0>;

impl<
    const DISTORT: usize,
    const PARAMS: usize,
    Distort: IsCameraDistortionImpl<f64, DISTORT, PARAMS, 1, 0, 0>,
    Proj: IsProjection<f64, 1, 0, 0>,
> IsVariable for Camera<f64, DISTORT, PARAMS, 1, 0, 0, Distort, Proj>
{
    const NUM_DOF: usize = PARAMS;

    fn update(&mut self, delta: nalgebra::DVectorView<f64>) {
        let new_params = *self.params() + delta;
        self.set_params(new_params);
    }
}

/// Perspective camera enum
#[derive(Debug, Clone)]
pub enum PerspectiveCameraEnum<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    /// Pinhole camera
    Pinhole(PinholeCamera<S, BATCH, DM, DN>),
    /// Kannala-Brandt camera
    KannalaBrandt(KannalaBrandtCamera<S, BATCH, DM, DN>),
    /// Brown-Conrady camera
    BrownConrady(BrownConradyCamera<S, BATCH, DM, DN>),
    /// Unified Extended camera
    EnhancedUnified(EnhancedUnifiedCamera<S, BATCH, DM, DN>),
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsCamera<S, BATCH, DM, DN> for PerspectiveCameraEnum<S, BATCH, DM, DN>
{
    fn new_pinhole(params: S::Vector<4>, image_size: ImageSize) -> Self {
        PerspectiveCameraEnum::Pinhole(PinholeCamera::new(params, image_size))
    }

    fn new_kannala_brandt(params: S::Vector<8>, image_size: ImageSize) -> Self {
        PerspectiveCameraEnum::KannalaBrandt(KannalaBrandtCamera::new(params, image_size))
    }

    fn new_brown_conrady(params: S::Vector<12>, image_size: ImageSize) -> Self {
        PerspectiveCameraEnum::BrownConrady(BrownConradyCamera::new(params, image_size))
    }

    fn new_enhanced_unified(params: S::Vector<6>, image_size: ImageSize) -> Self {
        PerspectiveCameraEnum::EnhancedUnified(EnhancedUnifiedCamera::new(params, image_size))
    }

    fn image_size(&self) -> ImageSize {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => camera.image_size(),
            PerspectiveCameraEnum::KannalaBrandt(camera) => camera.image_size(),
            PerspectiveCameraEnum::BrownConrady(camera) => camera.image_size(),
            PerspectiveCameraEnum::EnhancedUnified(camera) => camera.image_size(),
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
            PerspectiveCameraEnum::EnhancedUnified(camera) => camera.cam_proj(point_in_camera),
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
            PerspectiveCameraEnum::EnhancedUnified(camera) => camera.cam_unproj_with_z(pixel, z),
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
            PerspectiveCameraEnum::EnhancedUnified(camera) => {
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
            PerspectiveCameraEnum::EnhancedUnified(camera) => camera.undistort(pixel),
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
            PerspectiveCameraEnum::EnhancedUnified(camera) => {
                camera.dx_distort_x(proj_point_in_camera_z1_plane)
            }
        }
    }

    fn try_get_brown_conrady(self) -> Option<BrownConradyCamera<S, BATCH, DM, DN>> {
        if let PerspectiveCameraEnum::BrownConrady(camera) = self {
            Some(camera)
        } else {
            None
        }
    }

    fn try_get_kannala_brandt(self) -> Option<KannalaBrandtCamera<S, BATCH, DM, DN>> {
        if let PerspectiveCameraEnum::KannalaBrandt(camera) = self {
            Some(camera)
        } else {
            None
        }
    }

    fn try_get_pinhole(self) -> Option<PinholeCamera<S, BATCH, DM, DN>> {
        if let PerspectiveCameraEnum::Pinhole(camera) = self {
            Some(camera)
        } else {
            None
        }
    }

    fn try_get_enhanced_unified(self) -> Option<EnhancedUnifiedCamera<S, BATCH, DM, DN>> {
        if let PerspectiveCameraEnum::EnhancedUnified(camera) = self {
            Some(camera)
        } else {
            None
        }
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsPerspectiveCamera<S, BATCH, DM, DN> for PerspectiveCameraEnum<S, BATCH, DM, DN>
{
    fn pinhole_params(&self) -> S::Vector<4> {
        match self {
            PerspectiveCameraEnum::Pinhole(camera) => *camera.params(),
            PerspectiveCameraEnum::KannalaBrandt(camera) => {
                camera.params().clone().get_fixed_subvec(0)
            }
            PerspectiveCameraEnum::BrownConrady(camera) => {
                camera.params().clone().get_fixed_subvec(0)
            }
            PerspectiveCameraEnum::EnhancedUnified(camera) => {
                camera.params().clone().get_fixed_subvec(0)
            }
        }
    }
}
