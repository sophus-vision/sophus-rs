/// in-plane interaction
pub mod inplane_interaction;
/// orbit interaction
pub mod orbit_interaction;

use eframe::egui;
use sophus_core::linalg::VecF32;
use sophus_core::linalg::VecF64;
use sophus_image::arc_image::ArcImageF32;
use sophus_lie::Isometry3;
use sophus_sensor::dyn_camera::DynCamera;

use crate::interactions::inplane_interaction::InplaneInteraction;
use crate::interactions::orbit_interaction::OrbitalInteraction;

/// Clipping planes for the Wgpu renderer
#[derive(Clone, Copy, Debug)]
pub struct WgpuClippingPlanes {
    /// Near clipping plane
    pub near: f64,
    /// Far clipping plane
    pub far: f64,
}

impl Default for WgpuClippingPlanes {
    fn default() -> Self {
        WgpuClippingPlanes {
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl WgpuClippingPlanes {
    fn z_from_ndc(&self, ndc: f64) -> f64 {
        -(self.far * self.near) / (-self.far + ndc * self.far - ndc * self.near)
    }

    pub(crate) fn _ndc_from_z(&self, z: f64) -> f64 {
        (self.far * (z - self.near)) / (z * (self.far - self.near))
    }
}

/// Interaction state for pointer
#[derive(Clone, Copy)]
pub struct InteractionPointerState {
    /// Start uv position
    pub start_uv: VecF64<2>,
}

/// Scene focus
#[derive(Clone, Copy)]
pub struct SceneFocus {
    /// Depth
    pub depth: f64,
    /// UV position
    pub uv_in_virtual_camera: VecF64<2>,
}

#[derive(Clone, Copy)]
/// Scroll state
pub struct ScrollState {}

/// Interaction state
pub enum InteractionEnum {
    /// orbit interaction state
    OrbitalInteraction(OrbitalInteraction),
    /// in-plane interaction state
    InplaneInteraction(InplaneInteraction),
}

impl InteractionEnum {
    /// Get scene_from_camera isometry
    pub fn scene_from_camera(&self) -> Isometry3<f64, 1> {
        match self {
            InteractionEnum::OrbitalInteraction(orbit) => orbit.scene_from_camera,
            InteractionEnum::InplaneInteraction(inplane) => inplane.scene_from_camera,
        }
    }

    /// process event
    pub fn process_event(
        &mut self,
        cam: &DynCamera<f64, 1>,
        response: &egui::Response,
        scales: &VecF32<2>,
        z_buffer: &ArcImageF32,
    ) {
        match self {
            InteractionEnum::OrbitalInteraction(orbit) => {
                orbit.process_event(cam, response, scales, z_buffer)
            }
            InteractionEnum::InplaneInteraction(inplane) => {
                inplane.process_event(cam, response, scales, z_buffer)
            }
        }
    }
}
