use eframe::egui;
use sophus_autodiff::linalg::VecF64;
use sophus_image::ImageSize;
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

#[derive(Clone, Copy)]
pub(crate) struct InplanePointerState {}

#[derive(Clone)]
/// Interaction state
pub struct InplaneInteraction {
    pub(crate) view_name: String,
    pub(crate) maybe_scroll_state: Option<InplaneScrollState>,
    pub(crate) maybe_pointer_state: Option<InplanePointerState>,
    pub(crate) maybe_pivot: Option<ScenePivot>,
    pub(crate) zoom2d: TranslationAndScaling,
}

impl InplaneInteraction {
    pub(crate) fn new(view_name: &str) -> Self {
        InplaneInteraction {
            view_name: view_name.to_string(),
            maybe_scroll_state: None,
            maybe_pointer_state: None,
            maybe_pivot: None,
            zoom2d: TranslationAndScaling::identity(),
        }
    }

    /// The view port shows [0, width] x [0, height] of the zoomed image. Restrict the translation
    /// such that the zoomed image cannot be dragged out of the view port. (Since `scaling >= 1`,
    /// the lower bounds are <= 0.)
    fn clamped_translation(
        translation: VecF64<2>,
        scaling: VecF64<2>,
        image_size: ImageSize,
    ) -> VecF64<2> {
        let width = image_size.width as f64;
        let height = image_size.height as f64;
        VecF64::<2>::new(
            translation[0].clamp(width * (1.0 - scaling[0]), 0.0),
            translation[1].clamp(height * (1.0 - scaling[1]), 0.0),
        )
    }

    /// Position of `pointer` relative to the view port, in units of image pixels.
    fn uv_view_port_in_image_units(
        pointer: egui::Pos2,
        response: &egui::Response,
        scales: &ViewportScale,
    ) -> VecF64<2> {
        scales.apply(egui::Pos2::new(
            (pointer - response.rect.min)[0],
            (pointer - response.rect.min)[1],
        ))
    }

    /// Process pointer events
    ///
    /// primary button: drag the zoomed image
    ///
    /// double click: reset the zoom, so that the whole image is shown again
    pub fn process_pointer(
        &mut self,
        active_view: &mut String,
        cam: &RenderIntrinsics,
        response: &egui::Response,
        scales: &ViewportScale,
    ) {
        if response.double_clicked() {
            *active_view = self.view_name.clone();
            self.zoom2d = TranslationAndScaling::identity();
            // the early return skips the drag handling below, so end any drag right here
            self.maybe_pointer_state = None;
            return;
        }

        if response.drag_started() {
            let Some(pointer) = response.interact_pointer_pos() else {
                self.maybe_pointer_state = None;
                return;
            };
            *active_view = self.view_name.clone();

            // The pivot marker is a 2d renderable given in image coordinates, hence it sticks to
            // the image point which was grabbed. An image view has no scene behind it, so there
            // is no distance to the point - only the pixel matters.
            self.maybe_pivot = Some(ScenePivot {
                pixel: self
                    .zoom2d
                    .apply_inverse(Self::uv_view_port_in_image_units(pointer, response, scales)),
                distance: f64::INFINITY,
            });
            self.maybe_pointer_state = Some(InplanePointerState {});
        }

        if response.dragged() {
            // The drag delta is in view-port points, the zoom translation in image pixel units.
            let delta = response.drag_delta();
            let translation = self.zoom2d.translation
                + VecF64::<2>::new(
                    delta.x as f64 * scales.scale[0],
                    delta.y as f64 * scales.scale[1],
                );
            // When not zoomed in, the clamp pins the translation to zero - so there is nothing to
            // drag, as intended.
            self.zoom2d.translation =
                Self::clamped_translation(translation, self.zoom2d.scaling, cam.image_size());
        }

        if response.drag_stopped() {
            self.maybe_pointer_state = None;
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
        let smooth_scroll_delta = response.ctx.input(|i| i.smooth_scroll_delta);
        let is_scroll_zero = smooth_scroll_delta.x == 0.0 && smooth_scroll_delta.y == 0.0;

        // A scroll has to be ended even when the pointer has meanwhile left the view, which the
        // checks below return on - otherwise the interaction stays `is_active` forever, and the
        // pivot marker with it.
        if is_scroll_zero {
            self.maybe_scroll_state = None;
        }

        let Some(last_pointer_pos) = response.ctx.input(|i| i.pointer.latest_pos()) else {
            return;
        };
        if !response.rect.contains(last_pointer_pos) {
            return;
        }

        if !is_scroll_zero && self.maybe_scroll_state.is_none() {
            self.maybe_scroll_state = Some(InplaneScrollState {});
            *active_view = self.view_name.clone();
        }

        if smooth_scroll_delta.y != 0.0 {
            let width = cam.image_size().width as f64;

            // Position of the pointer in the view port, in units of image pixels.
            let uv_view_port_in_image_units =
                Self::uv_view_port_in_image_units(last_pointer_pos, response, scales);
            // The image point which is currently displayed under the pointer. Note that the zoom
            // maps image coordinates to view-port coordinates, hence the inverse is needed here.
            let uv_in_image = self.zoom2d.apply_inverse(uv_view_port_in_image_units);

            // The pivot marker is a 2d renderable, so it is given in image coordinates and the
            // zoom will place it back under the pointer.
            self.maybe_pivot = Some(ScenePivot {
                pixel: uv_in_image,
                distance: f64::INFINITY,
            });

            // Scroll up (positive delta) zooms in - as for the orbital interaction.
            let zoomed_width = width / self.zoom2d.scaling[0];
            let zoomed_width =
                (zoomed_width * (-smooth_scroll_delta.y as f64 * 0.001).exp()).clamp(1.0, width);
            let scale = width / zoomed_width;

            // Keep the image point under the pointer fixed: `scale * uv_in_image + t` shall be
            // the view-port position of the pointer.
            let translation = uv_view_port_in_image_units - scale * uv_in_image;
            let scaling = VecF64::<2>::new(scale, scale);

            self.zoom2d = TranslationAndScaling {
                translation: Self::clamped_translation(translation, scaling, cam.image_size()),
                scaling,
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
        self.process_pointer(active_view, cam, response, scales);
        self.process_scrolls(active_view, cam, response, scales);
    }
}
