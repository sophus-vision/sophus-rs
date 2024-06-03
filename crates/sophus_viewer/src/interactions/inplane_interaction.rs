use eframe::egui;
use sophus_core::linalg::VecF32;
use sophus_image::arc_image::ArcImageF32;
use sophus_lie::Isometry3;
use sophus_sensor::DynCamera;

use crate::interactions::InteractionPointerState;
use crate::interactions::SceneFocus;
use crate::interactions::ScrollState;
use crate::interactions::WgpuClippingPlanes;

#[derive(Clone, Copy)]
/// Interaction state
pub struct InplaneInteraction {
    pub(crate) maybe_pointer_state: Option<InteractionPointerState>,
    pub(crate) maybe_scroll_state: Option<ScrollState>,
    pub(crate) maybe_scene_focus: Option<SceneFocus>,
    pub(crate) _clipping_planes: WgpuClippingPlanes,
    pub(crate) scene_from_camera: Isometry3<f64, 1>,
}

impl InplaneInteraction {
    /// Process event
    pub fn process_event(
        &mut self,
        _cam: &DynCamera<f64, 1>,
        _response: &egui::Response,
        _scales: &VecF32<2>,
        _z_buffer: &ArcImageF32,
    ) {
        todo!()
    }
}
