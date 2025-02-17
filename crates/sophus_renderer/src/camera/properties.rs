use sophus_image::ImageSize;
use sophus_sensor::DynCameraF64;

use crate::{
    camera::{
        clipping_planes::{
            ClippingPlanes,
            ClippingPlanesF64,
        },
        intrinsics::RenderIntrinsics,
    },
    uniform_buffers::CameraPropertiesUniform,
};

/// Camera properties
#[derive(Clone, Debug)]
pub struct RenderCameraProperties {
    /// Camera intrinsics
    pub intrinsics: RenderIntrinsics,
    /// Clipping planes
    pub clipping_planes: ClippingPlanesF64,
}

impl Default for RenderCameraProperties {
    fn default() -> Self {
        RenderCameraProperties::default_from(ImageSize::new(639, 479))
    }
}

impl RenderCameraProperties {
    /// Create a new RenderCameraProperties from a DynCameraF64
    pub fn new(intrinsics: DynCameraF64, clipping_planes: ClippingPlanesF64) -> Self {
        RenderCameraProperties {
            intrinsics: RenderIntrinsics::new(&intrinsics),
            clipping_planes,
        }
    }

    /// Create default viewer camera from image size
    pub fn default_from(image_size: ImageSize) -> RenderCameraProperties {
        RenderCameraProperties {
            intrinsics: RenderIntrinsics::Pinhole(
                DynCameraF64::default_pinhole(image_size)
                    .try_get_pinhole()
                    .unwrap(),
            ),
            clipping_planes: ClippingPlanes::default(),
        }
    }

    /// intrinsics
    pub fn from_intrinsics(intrinsics: &DynCameraF64) -> Self {
        RenderCameraProperties {
            intrinsics: RenderIntrinsics::new(intrinsics),
            clipping_planes: ClippingPlanes::default(),
        }
    }

    pub(crate) fn to_uniform(&self) -> CameraPropertiesUniform {
        match self.intrinsics {
            RenderIntrinsics::Pinhole(pinhole) => CameraPropertiesUniform {
                camera_image_width: pinhole.image_size().width as f32,
                camera_image_height: pinhole.image_size().height as f32,
                near: self.clipping_planes.near as f32,
                far: self.clipping_planes.far as f32,
                fx: pinhole.params()[0] as f32,
                fy: pinhole.params()[1] as f32,
                px: pinhole.params()[2] as f32,
                py: pinhole.params()[3] as f32,
                alpha: 0.0,
                beta: 0.0,
            },
            RenderIntrinsics::UnifiedExtended(unified) => CameraPropertiesUniform {
                camera_image_width: unified.image_size().width as f32,
                camera_image_height: unified.image_size().height as f32,
                near: self.clipping_planes.near as f32,
                far: self.clipping_planes.far as f32,
                fx: unified.params()[0] as f32,
                fy: unified.params()[1] as f32,
                px: unified.params()[2] as f32,
                py: unified.params()[3] as f32,
                alpha: unified.params()[4] as f32,
                beta: unified.params()[5] as f32,
            },
        }
    }
}
