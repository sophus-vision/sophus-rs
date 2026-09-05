pub(crate) mod inplane_interaction;
pub(crate) mod orbit_interaction;

use eframe::egui;
use sophus_autodiff::linalg::VecF64;
use sophus_image::{
    ArcImageF32,
    ImageSize,
};
use sophus_lie::Isometry3F64;
use sophus_renderer::{
    ScenePivotMarker,
    TranslationAndScaling,
    camera::RenderIntrinsics,
    renderables::Color,
    textures::{
        inverse_distance_to_color,
        normalized_to_color,
    },
};

use crate::{
    interactions::{
        inplane_interaction::InplaneInteraction,
        orbit_interaction::OrbitalInteraction,
    },
    prelude::*,
};

/// Viewport scale
pub struct ViewportScale {
    /// the scale
    pub scale: VecF64<2>,
}

impl ViewportScale {
    /// Scale from the view port to the image.
    ///
    /// Note: `view_port_rect` is the rect of the rendered image widget itself - not the rect of
    /// the surrounding window, which also contains the border and the (optional) title bar.
    pub(crate) fn from_image_size_and_viewport_rect(
        image_size: ImageSize,
        view_port_rect: egui::Rect,
    ) -> ViewportScale {
        let scale = VecF64::<2>::new(
            image_size.width as f64 / view_port_rect.width() as f64,
            image_size.height as f64 / view_port_rect.height() as f64,
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

    /// Turn a scene view to look straight down on the scene, if it is one which can be turned.
    pub fn look_straight_down(&mut self) {
        match self {
            InteractionEnum::Orbital(orbit) => orbit.look_straight_down(),
            InteractionEnum::InPlane(_) | InteractionEnum::No => {}
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

    /// Get the point the interaction turns about
    pub fn maybe_pivot(&self) -> Option<ScenePivot> {
        match self {
            InteractionEnum::Orbital(orbital) => orbital.maybe_pivot,
            InteractionEnum::InPlane(inplane) => inplane.maybe_pivot,
            InteractionEnum::No => None,
        }
    }

    /// Is there a current interaction?
    pub fn is_active(&self) -> bool {
        if self.maybe_pivot().is_none() {
            return false;
        }
        match self {
            InteractionEnum::Orbital(orbital) => {
                orbital.maybe_pointer_state.is_some() || orbital.maybe_scroll_state.is_some()
            }
            InteractionEnum::InPlane(plane) => {
                plane.maybe_pointer_state.is_some() || plane.maybe_scroll_state.is_some()
            }
            InteractionEnum::No => false,
        }
    }

    /// Process event
    ///
    /// Precondition: z_buffer must not be None if self is [InteractionEnum::Orbital].
    #[allow(clippy::too_many_arguments)]
    pub fn process_event(
        &mut self,
        active_view: &mut String,
        cam: &RenderIntrinsics,
        locked_to_birds_eye_orientation: bool,
        response: &egui::Response,
        scales: &ViewportScale,
        view_port_size: ImageSize,
        z_buffer: Option<ArcImageF32>,
    ) {
        match self {
            InteractionEnum::Orbital(orbit) => orbit.process_event(
                active_view,
                cam,
                locked_to_birds_eye_orientation,
                response,
                scales,
                view_port_size,
                &z_buffer.unwrap(),
            ),
            InteractionEnum::InPlane(inplane) => {
                inplane.process_event(active_view, cam, response, scales)
            }
            InteractionEnum::No => {}
        }
    }

    /// get marker
    pub fn marker(&self) -> Option<ScenePivotMarker> {
        match self.is_active() {
            true => {
                let pivot = self.maybe_pivot().unwrap();

                let color = match self {
                    InteractionEnum::Orbital(orbit) => inverse_distance_to_color(
                        1.0 / pivot.distance as f32,
                        orbit.clipping_planes.cast(),
                    ),
                    // an image view has no scene and hence no distance to show; the marker is
                    // only a handle on the point being dragged
                    _ => normalized_to_color(0.5),
                };

                Some(ScenePivotMarker {
                    color: Color {
                        r: color[0] as f32 / 255.0,
                        g: color[1] as f32 / 255.0,
                        b: color[2] as f32 / 255.0,
                        a: 1.0,
                    },
                    u: pivot.pixel[0] as f32,
                    v: pivot.pixel[1] as f32,
                    distance: pivot.distance as f32,
                })
            }
            false => None,
        }
    }
}

/// Scene pivot
#[derive(Clone, Copy, Debug)]
pub struct ScenePivot {
    /// the pixel of the image the pivot was picked in
    pub pixel: VecF64<2>,
    /// metric distance from the camera, along the ray through `pixel`
    pub distance: f64,
}

impl ScenePivot {
    /// Where the pivot is, in the camera frame.
    pub fn point_in_camera(&self, intrinsics: &RenderIntrinsics) -> VecF64<3> {
        intrinsics.cam_unproj_to_unit_vector(&self.pixel) * self.distance
    }
}
