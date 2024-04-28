use eframe::egui;
use sophus_core::linalg::VecF64;
use sophus_image::arc_image::ArcImageF32;
use sophus_image::prelude::*;
use sophus_lie::prelude::*;
use sophus_lie::Isometry3;
use sophus_sensor::dyn_camera::DynCamera;

/// Clipping planes for the Wgpu renderer
#[derive(Clone, Copy)]
pub struct WgpuClippingPlanes {
    /// Near clipping plane
    pub near: f64,
    /// Far clipping plane
    pub far: f64,
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
    /// Current uv position
    pub current_uv: VecF64<2>,
}

/// Scene focus
#[derive(Clone, Copy)]
pub struct SceneFocus {
    /// Depth
    pub depth: f64,
    /// UV position
    pub uv: VecF64<2>,
}

#[derive(Clone, Copy)]
/// Scroll state
pub struct ScrollState {}

#[derive(Clone, Copy)]
/// Interaction state
pub struct Interaction {
    pub(crate) maybe_pointer_state: Option<InteractionPointerState>,
    pub(crate) maybe_scroll_state: Option<ScrollState>,
    pub(crate) maybe_scene_focus: Option<SceneFocus>,
    pub(crate) clipping_planes: WgpuClippingPlanes,
    pub(crate) scene_from_camera: Isometry3<f64, 1>,
}

impl Interaction {
    fn median_scene_depth(&self, z_buffer: &ArcImageF32) -> f64 {
        // to median ndc z
        let scalar_view = z_buffer.tensor.scalar_view();
        let mut valid_z_values = scalar_view
            .as_slice()
            .unwrap()
            .iter()
            .filter(|x| **x < 1.0)
            .collect::<Vec<_>>();
        valid_z_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let ndc_z = if !valid_z_values.is_empty() {
            let idx = (valid_z_values.len() as f64 * 0.5) as usize;
            *valid_z_values[idx] as f64
        } else {
            0.5
        };

        self.clipping_planes.z_from_ndc(ndc_z)
    }

    /// Process scroll events
    ///
    /// Scroll up/down: zoom in/out
    ///
    /// Scroll left/right: rotate about scene focus
    pub fn process_scrolls(
        &mut self,
        cam: &DynCamera<f64, 1>,
        response: &egui::Response,
        z_buffer: &ArcImageF32,
    ) {
        let last_pointer_pos = response.ctx.input(|i| i.pointer.latest_pos());
        if last_pointer_pos.is_none() {
            return;
        }
        let last_pointer_pos = last_pointer_pos.unwrap();

        let smooth_scroll_delta = response.ctx.input(|i| i.smooth_scroll_delta);

        let is_scroll_zero = smooth_scroll_delta.x == 0.0 && smooth_scroll_delta.y == 0.0;

        let scroll_started = self.maybe_scroll_state.is_none() && !is_scroll_zero;
        let scroll_stopped = self.maybe_scroll_state.is_some() && is_scroll_zero;

        if scroll_started {
            let uv = last_pointer_pos - response.rect.min;

            if self.maybe_scene_focus.is_none() {
                // Typically, the scene focus shall only be set by the pointer interaction event. But
                // it was never set, we set it here.
                let mut z = self
                    .clipping_planes
                    .z_from_ndc(z_buffer.pixel(uv.x as usize, uv.y as usize) as f64);
                if z >= self.clipping_planes.far {
                    z = self.median_scene_depth(z_buffer);
                }
                self.maybe_scene_focus = Some(SceneFocus {
                    depth: z,
                    uv: VecF64::<2>::new(uv.x as f64, uv.y as f64),
                });
            }

            let mut z = self
                .clipping_planes
                .z_from_ndc(z_buffer.pixel(uv.x as usize, uv.y as usize) as f64);
            if z >= self.clipping_planes.far {
                z = self.median_scene_depth(z_buffer);
            }
            self.maybe_scroll_state = Some(ScrollState {});
        } else if scroll_stopped {
            self.maybe_scroll_state = None;
        }

        if smooth_scroll_delta.y != 0.0 {
            let zoom: f64 = (1.0 + 0.01 * smooth_scroll_delta.y) as f64;

            // let current_pixel = response.interact_pointer_pos().unwrap() - response.rect.min;
            // let drag_state = self.maybe_state.unwrap();
            // let pixel = drag_state.drag_start_uv;
            // let depth = drag_state.depth;
            // let p0 = cam.cam_unproj_with_z(&pixel, depth);
            // let p1 = cam.cam_unproj_with_z(
            //     &VecF64::<2>::new(pixel.x + delta_x as f64, pixel.y + delta_y as f64),
            //     depth,
            // );
            let mut scene_from_camera = self.scene_from_camera;

            scene_from_camera.set_translation(&(scene_from_camera.translation() * zoom));
            self.scene_from_camera = scene_from_camera;
            // self.maybe_state.as_mut().unwrap().current_uv =
            //     VecF64::<2>::new(current_pixel[0] as f64, current_pixel[1] as f64);
        }
        println!("{:?}", response);

        if smooth_scroll_delta.x != 0.0 {
            let delta_z: f64 = (smooth_scroll_delta.x) as f64;

            let scene_focus = self.maybe_scene_focus.unwrap();
            let pixel = scene_focus.uv;
            let depth = scene_focus.depth;
            let delta = 0.01 * VecF64::<6>::new(0.0, 0.0, 0.0, 0.0, 0.0, delta_z);
            println!("delta: {:?}", delta);
            let camera_from_scene_point = Isometry3::from_t(&cam.cam_unproj_with_z(&pixel, depth));
            self.scene_from_camera =
                self.scene_from_camera
                    .group_mul(&camera_from_scene_point.group_mul(
                        &Isometry3::exp(&delta).group_mul(&camera_from_scene_point.inverse()),
                    ));
        }
    }

    /// Process pointer events
    ///
    /// primary button: rotate about scene focus
    ///
    /// secondary button: in-plane translate
    pub fn process_pointer(
        &mut self,
        cam: &DynCamera<f64, 1>,
        response: &egui::Response,
        z_buffer: &ArcImageF32,
    ) {
        let delta_x = response.drag_delta().x;
        let delta_y = response.drag_delta().y;

        if let Some(touch) = response.ctx.multi_touch() {
            println!("touch: {:?}", touch);
        } else {
            println!("no touch");
        }

        let maybe_uv = response.interact_pointer_pos();

        if response.drag_started() {
            // A drag event started - select new scene focus

            let pointer = response.interact_pointer_pos().unwrap();

            let pixel = pointer - response.rect.min;
            let uv = maybe_uv.unwrap();

            let mut z = self
                .clipping_planes
                .z_from_ndc(z_buffer.pixel(uv.x as usize, uv.y as usize) as f64);
            if z >= self.clipping_planes.far {
                z = self.median_scene_depth(z_buffer);
            }
            self.maybe_scene_focus = Some(SceneFocus {
                depth: z,
                uv: VecF64::<2>::new(pixel.x as f64, pixel.y as f64),
            });
            self.maybe_pointer_state = Some(InteractionPointerState {
                current_uv: VecF64::<2>::new(pixel.x as f64, pixel.y as f64),
            });
        } else if response.drag_stopped() {
            // A drag event finished
            self.maybe_pointer_state = None;
        };

        if response.dragged_by(egui::PointerButton::Secondary) {
            // translate scene

            let current_pixel = response.interact_pointer_pos().unwrap() - response.rect.min;
            let scene_focus = self.maybe_scene_focus.unwrap();
            let pixel = scene_focus.uv;
            let depth = scene_focus.depth;
            let p0 = cam.cam_unproj_with_z(&pixel, depth);
            let p1 = cam.cam_unproj_with_z(
                &VecF64::<2>::new(pixel.x + delta_x as f64, pixel.y + delta_y as f64),
                depth,
            );
            let mut scene_from_camera = self.scene_from_camera;
            let delta = p0 - p1;
            let translation_update = scene_from_camera.factor().transform(&delta);
            scene_from_camera
                .set_translation(&(scene_from_camera.translation() + translation_update));
            self.scene_from_camera = scene_from_camera;
            self.maybe_pointer_state.as_mut().unwrap().current_uv =
                VecF64::<2>::new(current_pixel[0] as f64, current_pixel[1] as f64);
        } else if response.dragged_by(egui::PointerButton::Primary) {
            // rotate around scene focus

            let scene_focus = self.maybe_scene_focus.unwrap();
            let pixel = scene_focus.uv;
            let depth = scene_focus.depth;
            let delta =
                0.01 * VecF64::<6>::new(0.0, 0.0, 0.0, -delta_y as f64, delta_x as f64, 0.0);
            let camera_from_scene_point = Isometry3::from_t(&cam.cam_unproj_with_z(&pixel, depth));
            self.scene_from_camera =
                self.scene_from_camera
                    .group_mul(&camera_from_scene_point.group_mul(
                        &Isometry3::exp(&delta).group_mul(&camera_from_scene_point.inverse()),
                    ));
        }
    }

    /// Process event
    pub fn process_event(
        &mut self,
        cam: &DynCamera<f64, 1>,
        response: &egui::Response,
        z_buffer: &ArcImageF32,
    ) {
        self.process_pointer(cam, response, z_buffer);
        self.process_scrolls(cam, response, z_buffer);
    }
}
