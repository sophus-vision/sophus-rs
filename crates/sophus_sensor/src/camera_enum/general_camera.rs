use core::borrow::Borrow;

use crate::camera_enum::perspective_camera::PerspectiveCameraEnum;
use crate::camera_enum::perspective_camera::UnifiedCamera;
use crate::prelude::*;
use crate::projections::orthographic::OrthographicCamera;
use crate::BrownConradyCamera;
use crate::KannalaBrandtCamera;
use crate::PinholeCamera;
use sophus_image::ImageSize;

/// Generalized camera enum
#[derive(Debug, Clone)]
pub enum GeneralCameraEnum<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    /// Perspective camera enum
    Perspective(PerspectiveCameraEnum<S, BATCH, DM, DN>),
    /// Orthographic camera
    Orthographic(OrthographicCamera<S, BATCH, DM, DN>),
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    GeneralCameraEnum<S, BATCH, DM, DN>
{
    /// Create a new perspective camera instance
    pub fn new_perspective(model: PerspectiveCameraEnum<S, BATCH, DM, DN>) -> Self {
        Self::Perspective(model)
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsCamera<S, BATCH, DM, DN> for GeneralCameraEnum<S, BATCH, DM, DN>
{
    fn new_pinhole<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<4>>,
    {
        Self::Perspective(PerspectiveCameraEnum::new_pinhole(params, image_size))
    }

    fn new_kannala_brandt<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<8>>,
    {
        Self::Perspective(PerspectiveCameraEnum::new_kannala_brandt(
            params, image_size,
        ))
    }

    fn new_brown_conrady<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<12>>,
    {
        Self::Perspective(PerspectiveCameraEnum::new_brown_conrady(params, image_size))
    }

    fn new_unified<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<6>>,
    {
        Self::Perspective(PerspectiveCameraEnum::new_unified(params, image_size))
    }

    fn image_size(&self) -> ImageSize {
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.image_size(),
            GeneralCameraEnum::Orthographic(camera) => camera.image_size(),
        }
    }

    fn cam_proj<P>(&self, point_in_camera: P) -> S::Vector<2>
    where
        P: Borrow<S::Vector<3>>,
    {
        let point_in_camera = point_in_camera.borrow();
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.cam_proj(point_in_camera),
            GeneralCameraEnum::Orthographic(camera) => camera.cam_proj(point_in_camera),
        }
    }

    fn cam_unproj_with_z<P>(&self, pixel: P, z: S) -> S::Vector<3>
    where
        P: Borrow<S::Vector<2>>,
    {
        let pixel = pixel.borrow();
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.cam_unproj_with_z(pixel, z),
            GeneralCameraEnum::Orthographic(camera) => camera.cam_unproj_with_z(pixel, z),
        }
    }

    fn distort<P>(&self, proj_point_in_camera_z1_plane: P) -> S::Vector<2>
    where
        P: Borrow<S::Vector<2>>,
    {
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();
        match self {
            GeneralCameraEnum::Perspective(camera) => camera.distort(proj_point_in_camera_z1_plane),
            GeneralCameraEnum::Orthographic(camera) => {
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
            GeneralCameraEnum::Perspective(camera) => camera.undistort(pixel),
            GeneralCameraEnum::Orthographic(camera) => camera.undistort(pixel),
        }
    }

    fn dx_distort_x<P>(&self, proj_point_in_camera_z1_plane: P) -> S::Matrix<2, 2>
    where
        P: Borrow<S::Vector<2>>,
    {
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();
        match self {
            GeneralCameraEnum::Perspective(camera) => {
                camera.dx_distort_x(proj_point_in_camera_z1_plane)
            }
            GeneralCameraEnum::Orthographic(camera) => {
                camera.dx_distort_x(proj_point_in_camera_z1_plane)
            }
        }
    }

    fn try_get_brown_conrady(self) -> Option<BrownConradyCamera<S, BATCH, DM, DN>> {
        if let GeneralCameraEnum::Perspective(camera) = self {
            camera.try_get_brown_conrady()
        } else {
            None
        }
    }

    fn try_get_kannala_brandt(self) -> Option<KannalaBrandtCamera<S, BATCH, DM, DN>> {
        if let GeneralCameraEnum::Perspective(camera) = self {
            camera.try_get_kannala_brandt()
        } else {
            None
        }
    }

    fn try_get_pinhole(self) -> Option<PinholeCamera<S, BATCH, DM, DN>> {
        if let GeneralCameraEnum::Perspective(camera) = self {
            camera.try_get_pinhole()
        } else {
            None
        }
    }

    fn try_get_unified_extended(self) -> Option<UnifiedCamera<S, BATCH, DM, DN>> {
        if let GeneralCameraEnum::Perspective(camera) = self {
            camera.try_get_unified_extended()
        } else {
            None
        }
    }
}
