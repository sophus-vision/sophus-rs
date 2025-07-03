use eframe::egui;
use sophus_autodiff::linalg::VecF64;
use sophus_image::{
    ArcImageF32,
    ImageSize,
};
use sophus_lie::{
    IsAffineGroup,
    Isometry3,
    Isometry3F64,
};
use sophus_renderer::{
    TranslationAndScaling,
    camera::{
        ClippingPlanesF64,
        RenderIntrinsics,
    },
};

use crate::{
    interactions::{
        SceneFocus,
        ViewportScale,
    },
    prelude::*,
};

#[derive(Clone, Copy)]
pub(crate) struct OrbitalPointerState {
    pub(crate) start_uv_virtual_camera: VecF64<2>,
}

#[derive(Clone, Copy)]
pub(crate) struct OrbitalScrollState {}

#[derive(Clone)]
/// Interaction state
pub struct OrbitalInteraction {
    pub(crate) view_name: String,
    pub(crate) maybe_pointer_state: Option<OrbitalPointerState>,
    pub(crate) maybe_scroll_state: Option<OrbitalScrollState>,
    pub(crate) maybe_scene_focus: Option<SceneFocus>,
    pub(crate) clipping_planes: ClippingPlanesF64,
    pub(crate) scene_from_camera: Isometry3F64,
}

impl OrbitalInteraction {
    pub(crate) fn new(
        view_name: &str,
        scene_from_camera: Isometry3F64,
        clipping_planes: ClippingPlanesF64,
    ) -> OrbitalInteraction {
        OrbitalInteraction {
            view_name: view_name.to_string(),
            maybe_pointer_state: None,
            maybe_scroll_state: None,
            maybe_scene_focus: None,
            clipping_planes,
            scene_from_camera,
        }
    }
}

impl OrbitalInteraction {
    fn median_scene_ndc_z(&self, z_buffer: &ArcImageF32) -> f64 {
        // to median ndc z
        let scalar_view = z_buffer.tensor.scalar_view();
        let mut valid_z_values = scalar_view
            .as_slice()
            .unwrap()
            .iter()
            .filter(|x| **x < 1.0)
            .collect::<Vec<_>>();
        valid_z_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if !valid_z_values.is_empty() {
            let idx = (valid_z_values.len() as f64 * 0.5) as usize;
            *valid_z_values[idx] as f64
        } else {
            0.5
        }
    }

    /// Process "scroll" events
    ///
    /// Scroll up/down: zoom in/out
    ///
    /// Scroll left/right: rotate about scene focus
    pub fn process_scrolls(
        &mut self,
        active_view: &mut String,
        cam: &RenderIntrinsics,
        response: &egui::Response,
        scales: &ViewportScale,
        viewport_size: ImageSize,
        z_buffer: &ArcImageF32,
    ) {
        let last_pointer_pos = response.ctx.input(|i| i.pointer.latest_pos());
        if last_pointer_pos.is_none() {
            return;
        }

        let last_pointer_pos = last_pointer_pos.unwrap();
        let uv_viewport = egui::Pos2::new(
            (last_pointer_pos - response.rect.min)[0],
            (last_pointer_pos - response.rect.min)[1],
        );

        if uv_viewport.x < 0.0
            || uv_viewport.y < 0.0
            || uv_viewport.x >= viewport_size.width as f32
            || uv_viewport.y >= viewport_size.height as f32
        {
            return;
        }

        let smooth_scroll_delta = response.ctx.input(|i| i.smooth_scroll_delta);

        let is_scroll_zero = smooth_scroll_delta.x == 0.0 && smooth_scroll_delta.y == 0.0;

        let scroll_started = self.maybe_scroll_state.is_none() && !is_scroll_zero;
        let scroll_stopped = self.maybe_scroll_state.is_some() && is_scroll_zero;
        if scroll_started {
            *active_view = self.view_name.clone();

            let uv_in_virtual_camera = scales.apply(uv_viewport);
            let ndc_z = self.get_ndc_z(uv_viewport, z_buffer);

            self.maybe_scene_focus = Some(SceneFocus {
                ndc_z: ndc_z as f32,
                uv_in_virtual_camera,
            });
            self.maybe_scroll_state = Some(OrbitalScrollState {});
        } else if scroll_stopped {
            self.maybe_scroll_state = None;
        }

        if self.maybe_scene_focus.is_none() {
            return;
        }

        let scene_focus = self.maybe_scene_focus.unwrap();
        let pixel = scene_focus.uv_in_virtual_camera;
        let depth = scene_focus.metric_depth(&self.clipping_planes);
        let focus_point_in_camera = cam.cam_unproj_with_z(&pixel, depth);

        if smooth_scroll_delta.y != 0.0 {
            let scene_from_camera = self.scene_from_camera;
            let camera_in_scene = scene_from_camera.translation();
            let zoom: f64 = (0.002 * smooth_scroll_delta.y) as f64;
            let focus_point_in_scene = scene_from_camera.transform(focus_point_in_camera);
            let camera_to_focus_point_vec_in_scene = focus_point_in_scene - camera_in_scene;

            let new_camera_in_scene = camera_in_scene + camera_to_focus_point_vec_in_scene * zoom;
            let mut new_scene_from_camera = self.scene_from_camera;
            new_scene_from_camera.set_translation(new_camera_in_scene);

            let focus_point_in_camera = new_scene_from_camera
                .inverse()
                .transform(focus_point_in_scene);

            let z_in_camera = focus_point_in_camera[2];
            let ndc_z = self.clipping_planes.ndc_z_from_metric_z(z_in_camera);

            if ndc_z > 0.03 && ndc_z < 0.9999 {
                self.scene_from_camera = new_scene_from_camera;

                self.maybe_scene_focus = Some(SceneFocus {
                    ndc_z: ndc_z as f32,
                    uv_in_virtual_camera: pixel,
                });
            }
        }

        if smooth_scroll_delta.x != 0.0 {
            let delta_z: f64 = (smooth_scroll_delta.x) as f64;
            let delta = 0.002 * VecF64::<6>::new(0.0, 0.0, delta_z, 0.0, 0.0, 0.0);
            let camera_from_scene_point = Isometry3::from_translation(focus_point_in_camera);

            self.scene_from_camera = self.scene_from_camera
                * camera_from_scene_point
                * Isometry3::exp(delta)
                * camera_from_scene_point.inverse();
        }
    }

    fn get_ndc_z(&self, viewport_pixel: egui::Pos2, z_buffer: &ArcImageF32) -> f64 {
        let ndc_z = z_buffer.pixel(viewport_pixel.x as usize, viewport_pixel.y as usize) as f64;
        if ndc_z > 0.99 {
            match self.maybe_scene_focus {
                Some(scene_focus) => scene_focus.ndc_z as f64,
                None => self.median_scene_ndc_z(z_buffer),
            }
        } else {
            ndc_z.clamp(0.01, 0.99)
        }
    }

    /// Process pointer events
    ///
    /// primary button: in-plane translate
    ///
    /// secondary button: rotate about scene focus
    pub fn process_pointer(
        &mut self,
        active_view: &mut String,
        cam: &RenderIntrinsics,
        locked_to_birds_eye_orientation: bool,
        response: &egui::Response,
        scales: &ViewportScale,
        z_buffer: &ArcImageF32,
    ) {
        let delta_x = response.drag_delta().x;
        let delta_y = response.drag_delta().y;

        if response.drag_started() {
            // A drag event started - select new scene focus

            *active_view = self.view_name.clone();

            let pointer = response.interact_pointer_pos().unwrap();

            let uv_viewport = egui::Pos2::new(
                (pointer - response.rect.min)[0],
                (pointer - response.rect.min)[1],
            );

            let uv_in_virtual_camera = scales.apply(uv_viewport);

            let ndc_z = self.get_ndc_z(uv_viewport, z_buffer);

            self.maybe_scene_focus = Some(SceneFocus {
                ndc_z: ndc_z as f32,
                uv_in_virtual_camera,
            });
            self.maybe_pointer_state = Some(OrbitalPointerState {
                start_uv_virtual_camera: uv_in_virtual_camera,
            });
        } else if response.drag_stopped() {
            // A drag event finished
            self.maybe_pointer_state = None;
        };

        if !locked_to_birds_eye_orientation
            && (response.dragged_by(egui::PointerButton::Secondary)
                || (response.dragged_by(egui::PointerButton::Primary)
                    && response.ctx.input(|i| i.modifiers.shift)))
        {
            // rotate about scene focus
            let scene_focus = self.maybe_scene_focus.unwrap();
            let pixel = scene_focus.uv_in_virtual_camera;
            let depth = scene_focus.metric_depth(&self.clipping_planes);
            let delta =
                0.01 * VecF64::<6>::new(-delta_y as f64, delta_x as f64, 0.0, 0.0, 0.0, 0.0);
            let camera_from_scene_point =
                Isometry3::from_translation(cam.cam_unproj_with_z(&pixel, depth));
            self.scene_from_camera = self.scene_from_camera
                * camera_from_scene_point
                * Isometry3::exp(delta)
                * camera_from_scene_point.inverse();
        } else if response.dragged_by(egui::PointerButton::Primary) {
            // translate scene

            let uv_viewport = response.interact_pointer_pos().unwrap() - response.rect.min;
            let current_pixel = scales.apply(uv_viewport.to_pos2()).cast::<f32>();
            let scene_focus = self.maybe_scene_focus.unwrap();
            let start_pixel = self.maybe_pointer_state.unwrap().start_uv_virtual_camera;
            let depth = scene_focus.metric_depth(&self.clipping_planes);
            let p0 = cam.cam_unproj_with_z(&start_pixel, depth);
            let p1 = cam.cam_unproj_with_z(
                &VecF64::<2>::new(
                    start_pixel.x + (delta_x as f64 * scales.scale.x),
                    start_pixel.y + (delta_y as f64 * scales.scale.y),
                ),
                depth,
            );
            let mut scene_from_camera = self.scene_from_camera;
            let delta = p0 - p1;
            let translation_update = scene_from_camera.factor().transform(delta);
            scene_from_camera.set_translation(scene_from_camera.translation() + translation_update);
            self.scene_from_camera = scene_from_camera;

            if let Some(focus) = &mut self.maybe_scene_focus {
                focus.uv_in_virtual_camera =
                    VecF64::<2>::new(current_pixel.x as f64, current_pixel.y as f64);
            }
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
        self.process_pointer(
            active_view,
            cam,
            locked_to_birds_eye_orientation,
            response,
            scales,
            z_buffer,
        );
        self.process_scrolls(active_view, cam, response, scales, view_port_size, z_buffer);
    }

    /// Get zoom
    pub fn zoom2d(&self) -> TranslationAndScaling {
        TranslationAndScaling::identity()
    }
}
