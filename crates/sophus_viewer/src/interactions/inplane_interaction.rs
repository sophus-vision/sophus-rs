use eframe::egui;
use sophus_autodiff::linalg::VecF64;
use sophus_lie::{
    Isometry3,
    Isometry3F64,
};
use sophus_renderer::{
    TranslationAndScaling,
    camera::RenderIntrinsics,
};

use crate::{
    interactions::{
        ScenePivot,
        ViewportScale,
    },
    prelude::*,
};

#[derive(Clone, Copy)]
pub(crate) struct InplaneScrollState {}

#[derive(Clone)]
/// Interaction state
pub struct InplaneInteraction {
    pub(crate) view_name: String,
    pub(crate) maybe_scroll_state: Option<InplaneScrollState>,
    pub(crate) maybe_pivot: Option<ScenePivot>,
    pub(crate) zoom2d: TranslationAndScaling,
}

impl InplaneInteraction {
    pub(crate) fn new(view_name: &str) -> Self {
        InplaneInteraction {
            view_name: view_name.to_string(),
            maybe_scroll_state: None,
            maybe_pivot: None,
            zoom2d: TranslationAndScaling::identity(),
        }
    }

    /// Process "scroll" events
    ///
    /// Scroll up/down: zoom in/out
    pub fn process_scrolls(
        &mut self,
        active_view: &mut String,
        cam: &RenderIntrinsics,
        response: &egui::Response,
        scales: &ViewportScale,
    ) {
        let Some(last_pointer_pos) = response.ctx.input(|i| i.pointer.latest_pos()) else {
            return;
        };
        if !response.rect.contains(last_pointer_pos) {
            return;
        }
        let uv_view_port = egui::Pos2::new(
            (last_pointer_pos - response.rect.min)[0],
            (last_pointer_pos - response.rect.min)[1],
        );

        let smooth_scroll_delta = response.ctx.input(|i| i.smooth_scroll_delta);

        let is_scroll_zero = smooth_scroll_delta.x == 0.0 && smooth_scroll_delta.y == 0.0;

        let scroll_started = self.maybe_scroll_state.is_none() && !is_scroll_zero;
        let scroll_stopped = self.maybe_scroll_state.is_some() && is_scroll_zero;

        if scroll_started {
            self.maybe_scroll_state = Some(InplaneScrollState {});
            *active_view = self.view_name.clone();
        } else if scroll_stopped {
            self.maybe_scroll_state = None;
        }

        if smooth_scroll_delta.y != 0.0 {
            let zoom2d = self.zoom2d();
            let zoomed_width = cam.image_size().width as f32 / zoom2d.scaling[0] as f32;

            let uv_in_virtual_camera = zoom2d.apply(scales.apply(uv_view_port));

            // The pivot marker is a 2d renderable given in image coordinates, hence it sticks to
            // the image point under the pointer. An image view has no scene behind it, so there
            // is no distance to that point - only the pixel matters.
            self.maybe_pivot = Some(ScenePivot {
                pixel: uv_in_virtual_camera,
                distance: f64::INFINITY,
            });

            let zoomed_width = (zoomed_width * (smooth_scroll_delta.y * 0.001).exp())
                .clamp(1.0, cam.image_size().width as f32);

            let scale = cam.image_size().width as f32 / zoomed_width;
            self.zoom2d = TranslationAndScaling {
                translation: VecF64::<2>::new(
                    (1.0 - scale as f64) * uv_in_virtual_camera[0],
                    (1.0 - scale as f64) * uv_in_virtual_camera[1],
                ),
                scaling: VecF64::<2>::new(scale as f64, scale as f64),
            };
        }
    }

    /// scene_from_camera - always the identity
    pub fn scene_from_camera(&self) -> Isometry3F64 {
        Isometry3::from_translation(VecF64::<3>::new(0.0, 0.0, 0.0))
    }

    /// zoom
    pub fn zoom2d(&self) -> TranslationAndScaling {
        self.zoom2d
    }

    /// Process event
    pub fn process_event(
        &mut self,
        active_view: &mut String,
        cam: &RenderIntrinsics,
        response: &egui::Response,
        scales: &ViewportScale,
    ) {
        self.process_scrolls(active_view, cam, response, scales);
    }
}
