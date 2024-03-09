use eframe::egui;

use crate::calculus::types::V;
use crate::image::arc_image::ArcImageF32;
use crate::image::view::IsImageView;
use crate::lie::rotation3::Isometry3;
use crate::lie::traits::IsTranslationProductGroup;
use crate::sensor::perspective_camera::KannalaBrandtCamera;
use crate::tensor::view::IsTensorLike;

#[derive(Clone, Copy)]
pub struct WgpuClippingPlanes {
    pub near: f64,
    pub far: f64,
}

impl WgpuClippingPlanes {
    fn z_from_ndc(&self, ndc: f64) -> f64 {
        -(self.far * self.near) / (-self.far + ndc * self.far - ndc * self.near)
    }

    pub fn ndc_from_z(&self, z: f64) -> f64 {
        (self.far * (z - self.near)) / (z * (self.far - self.near))
    }
}

#[derive(Clone, Copy)]
pub struct InteractionState {
    pub depth: f64,
    pub drag_start_uv: V<2>,
    pub current_uv: V<2>,
}

#[derive(Clone, Copy)]
pub struct Interaction {
    pub maybe_state: Option<InteractionState>,
    pub clipping_planes: WgpuClippingPlanes,
    pub scene_from_camera: Isometry3<f64>,
}

impl Interaction {
    fn median_scene_depth(&self, z_buffer: ArcImageF32) -> f64 {
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

    pub fn process_event(
        &mut self,
        cam: &KannalaBrandtCamera<f64>,
        response: &egui::Response,
        z_buffer: ArcImageF32,
    ) {
        let delta_x = response.drag_delta().x;
        let delta_y = response.drag_delta().y;

        let maybe_uv = response.interact_pointer_pos();

        if response.drag_started() {
            // A drag event started - select new scene focus
            let pixel = response.interact_pointer_pos().unwrap() - response.rect.min;
            let uv = maybe_uv.unwrap();
            let mut z = self
                .clipping_planes
                .z_from_ndc(z_buffer.pixel(uv.x as usize, uv.y as usize) as f64);
            if z >= self.clipping_planes.far {
                z = self.median_scene_depth(z_buffer);
            }
            self.maybe_state = Some(InteractionState {
                depth: z,
                drag_start_uv: V::<2>::new(pixel.x as f64, pixel.y as f64),
                current_uv: V::<2>::new(pixel.x as f64, pixel.y as f64),
            });
        } else if response.drag_released() {
            // A drag event finished
            self.maybe_state = None;
        };

        if response.dragged_by(egui::PointerButton::Secondary) {
            // rotate around scene focus

            let current_pixel = response.interact_pointer_pos().unwrap() - response.rect.min;
            let drag_state = self.maybe_state.unwrap();
            let pixel = drag_state.drag_start_uv;
            let depth = drag_state.depth;
            let p0 = cam.cam_unproj_with_z(&pixel, depth);
            let p1 = cam.cam_unproj_with_z(
                &V::<2>::new(pixel.x + delta_x as f64, pixel.y + delta_y as f64),
                depth,
            );
            let mut scene_from_camera = self.scene_from_camera;
            let delta = p0 - p1;
            let translation_update = scene_from_camera.factor().transform(&delta);
            scene_from_camera
                .set_translation(&(scene_from_camera.translation() + translation_update));
            self.scene_from_camera = scene_from_camera;
            self.maybe_state.as_mut().unwrap().current_uv =
                V::<2>::new(current_pixel[0] as f64, current_pixel[1] as f64);
        } else if response.dragged_by(egui::PointerButton::Primary) {
            // translate scene

            let drag_state = self.maybe_state.unwrap();
            let pixel = drag_state.drag_start_uv;
            let depth = drag_state.depth;
            let delta = 0.01 * V::<6>::new(0.0, 0.0, 0.0, -delta_y as f64, delta_x as f64, 0.0);
            let camera_from_scene_point = Isometry3::from_t(&cam.cam_unproj_with_z(&pixel, depth));
            self.scene_from_camera =
                self.scene_from_camera
                    .group_mul(&camera_from_scene_point.group_mul(
                        &Isometry3::exp(&delta).group_mul(&camera_from_scene_point.inverse()),
                    ));
        }
    }
}
