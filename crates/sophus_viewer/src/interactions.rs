/// in-plane interaction
pub mod inplane_interaction;
/// orbit interaction
pub mod orbit_interaction;

use crate::interactions::inplane_interaction::InplaneInteraction;
use crate::interactions::orbit_interaction::OrbitalInteraction;
use crate::prelude::*;
use crate::views::ViewportSize;
use eframe::egui;
use sophus_autodiff::linalg::VecF64;
use sophus_image::arc_image::ArcImageF32;
use sophus_image::ImageSize;
use sophus_lie::Isometry3F64;
use sophus_renderer::camera::clipping_planes::ClippingPlanesF64;
use sophus_renderer::camera::intrinsics::RenderIntrinsics;
use sophus_renderer::renderables::color::Color;
use sophus_renderer::textures::depth_image::ndc_z_to_color;
use sophus_renderer::types::SceneFocusMarker;
use sophus_renderer::types::TranslationAndScaling;

/// Viewport scale
pub struct ViewportScale {
    /// the scale
    pub scale: VecF64<2>,
}

impl ViewportScale {
    pub(crate) fn from_image_size_and_viewport_size(
        image_size: ImageSize,
        view_port_size: ViewportSize,
    ) -> ViewportScale {
        let scale = VecF64::<2>::new(
            image_size.width as f64 / view_port_size.width as f64,
            image_size.height as f64 / view_port_size.height as f64,
        );
        ViewportScale { scale }
    }

    pub(crate) fn apply(&self, uv_viewport: egui::Pos2) -> VecF64<2> {
        VecF64::<2>::new(
            (uv_viewport.x as f64 + 0.5) * self.scale[0] - 0.5,
            (uv_viewport.y as f64 + 0.5) * self.scale[1] - 0.5,
        )
    }
}

/// Interaction state
pub enum InteractionEnum {
    /// orbit interaction state
    Orbital(OrbitalInteraction),
    /// in-plane interaction state
    InPlane(InplaneInteraction),
    /// no interaction
    No,
}

impl InteractionEnum {
    /// Get scene_from_camera isometry
    pub fn scene_from_camera(&self) -> Isometry3F64 {
        match self {
            InteractionEnum::Orbital(orbit) => orbit.scene_from_camera,
            InteractionEnum::InPlane(inplane) => inplane.scene_from_camera(),
            InteractionEnum::No => Isometry3F64::identity(),
        }
    }

    /// Get zoom
    pub fn zoom2d(&self) -> TranslationAndScaling {
        match self {
            InteractionEnum::Orbital(orbit) => orbit.zoom2d(),
            InteractionEnum::InPlane(inplane) => inplane.zoom2d(),
            InteractionEnum::No => TranslationAndScaling::identity(),
        }
    }

    /// Get scene focus point
    pub fn maybe_scene_focus(&self) -> Option<SceneFocus> {
        match self {
            InteractionEnum::Orbital(orbital) => orbital.maybe_scene_focus,
            InteractionEnum::InPlane(inplane) => inplane.maybe_scene_focus,
            InteractionEnum::No => None,
        }
    }

    /// Is there a current interaction?
    pub fn is_active(&self) -> bool {
        if self.maybe_scene_focus().is_none() {
            return false;
        }
        match self {
            InteractionEnum::Orbital(orbital) => {
                orbital.maybe_pointer_state.is_some() || orbital.maybe_scroll_state.is_some()
            }
            InteractionEnum::InPlane(plane) => plane.maybe_scroll_state.is_some(),
            InteractionEnum::No => false,
        }
    }

    /// Process event
    #[allow(clippy::too_many_arguments)]
    pub fn process_event(
        &mut self,
        active_view: &mut String,
        cam: &RenderIntrinsics,
        locked_to_birds_eye_orientation: bool,
        response: &egui::Response,
        scales: &ViewportScale,
        view_port_size: ImageSize,
        z_buffer: &ArcImageF32,
    ) {
        match self {
            InteractionEnum::Orbital(orbit) => orbit.process_event(
                active_view,
                cam,
                locked_to_birds_eye_orientation,
                response,
                scales,
                view_port_size,
                z_buffer,
            ),
            InteractionEnum::InPlane(inplane) => {
                inplane.process_event(active_view, cam, response, scales, view_port_size)
            }
            InteractionEnum::No => {}
        }
    }

    /// get marker
    pub fn marker(&self) -> Option<SceneFocusMarker> {
        match self.is_active() {
            true => {
                let scene_focus = self.maybe_scene_focus().unwrap();

                let color = ndc_z_to_color(scene_focus.ndc_z);

                Some(SceneFocusMarker {
                    color: Color {
                        r: color[0] as f32 / 255.0,
                        g: color[1] as f32 / 255.0,
                        b: color[2] as f32 / 255.0,
                        a: 1.0,
                    },
                    u: scene_focus.uv_in_virtual_camera[0] as f32,
                    v: scene_focus.uv_in_virtual_camera[1] as f32,
                    ndc_z: scene_focus.ndc_z,
                })
            }
            false => None,
        }
    }
}

/// Scene focus
#[derive(Clone, Copy, Debug)]
pub struct SceneFocus {
    /// NDC z
    pub ndc_z: f32,
    /// UV position
    pub uv_in_virtual_camera: VecF64<2>,
}

impl SceneFocus {
    /// metric depth
    pub fn metric_depth(&self, clipping_planes: &ClippingPlanesF64) -> f64 {
        clipping_planes.metric_z_from_ndc_z(self.ndc_z as f64)
    }
}
