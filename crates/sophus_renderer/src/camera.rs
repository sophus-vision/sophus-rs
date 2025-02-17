use sophus_autodiff::linalg::VecF64;
use sophus_image::ImageSize;
use sophus_lie::{
    Isometry3,
    Isometry3F64,
};

mod clipping_planes;
mod intrinsics;
mod properties;

pub use clipping_planes::*;
pub use intrinsics::*;
pub use properties::*;

/// Render camera configuration.
#[derive(Clone, Debug)]
pub struct RenderCamera {
    /// Scene from camera pose
    pub scene_from_camera: Isometry3F64,
    /// Camera properties
    pub properties: RenderCameraProperties,
}

impl Default for RenderCamera {
    fn default() -> Self {
        RenderCamera::default_from(ImageSize::new(639, 479))
    }
}

impl RenderCamera {
    /// Create default viewer camera from image size
    pub fn default_from(image_size: ImageSize) -> RenderCamera {
        RenderCamera {
            properties: RenderCameraProperties::default_from(image_size),
            scene_from_camera: Isometry3::from_translation(VecF64::<3>::new(0.0, 0.0, -5.0)),
        }
    }
}
