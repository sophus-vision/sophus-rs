use eframe::egui;
use sophus_core::linalg::VecF64;
use sophus_image::arc_image::ArcImageF32;
use sophus_image::ImageSize;
use sophus_lie::Isometry3;
use sophus_lie::Isometry3F64;
use sophus_sensor::DynCamera;

use crate::renderer::types::ClippingPlanes;
use crate::renderer::types::ClippingPlanesF64;
use crate::viewer::interactions::ViewportScale;

/// Cancel request.
pub struct CancelRequest {}

/// Viewer camera configuration.
#[derive(Clone, Debug)]
pub struct ViewerCamera {
    /// Camera intrinsics
    pub intrinsics: DynCamera<f64, 1>,
    /// Clipping planes
    pub clipping_planes: ClippingPlanesF64,
    /// Scene from camera pose
    pub scene_from_camera: Isometry3F64,
}

impl Default for ViewerCamera {
    fn default() -> Self {
        ViewerCamera::default_from(ImageSize::new(639, 479))
    }
}

impl ViewerCamera {
    /// Create default viewer camera from image size
    pub fn default_from(image_size: ImageSize) -> ViewerCamera {
        ViewerCamera {
            intrinsics: DynCamera::default_pinhole(image_size),
            clipping_planes: ClippingPlanes::default(),
            scene_from_camera: Isometry3::from_translation(&VecF64::<3>::new(0.0, 0.0, -5.0)),
        }
    }
}

pub(crate) struct ResponseStruct {
    pub(crate) ui_response: egui::Response,
    pub(crate) z_image: ArcImageF32,
    pub(crate) scales: ViewportScale,
    pub(crate) view_port_size: ImageSize,
}
