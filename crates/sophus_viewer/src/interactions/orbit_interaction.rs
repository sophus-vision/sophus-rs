use eframe::egui;
use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};
use sophus_image::{
    ArcImageF32,
    ImageSize,
};
use sophus_lie::{
    IsAffineGroup,
    Isometry3,
    Isometry3F64,
    Rotation3,
};
use sophus_renderer::{
    PivotGesture,
    TranslationAndScaling,
    camera::{
        ClippingPlanesF64,
        RenderIntrinsics,
    },
    textures::is_background,
};

use crate::{
    interactions::{
        ScenePivot,
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
    pub(crate) maybe_pivot: Option<ScenePivot>,
    pub(crate) clipping_planes: ClippingPlanesF64,
    pub(crate) scene_from_camera: Isometry3F64,
    /// where the view started, which a double click comes back to
    initial_scene_from_camera: Isometry3F64,
    /// when a drag last moved the camera, in seconds of egui's clock
    last_moved_at: f64,
    /// what the interaction is doing right now, if anything
    pub(crate) maybe_gesture: Option<PivotGesture>,
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
            maybe_pivot: None,
            clipping_planes,
            scene_from_camera,
            initial_scene_from_camera: scene_from_camera,
            last_moved_at: f64::NEG_INFINITY,
            maybe_gesture: None,
        }
    }

    /// Turn the camera to look straight down on the scene, keeping where it is.
    ///
    /// The scene's up is +z, so this points the camera's own z down the world's -z. A camera's
    /// frame is x right, y down, z forward, so its y then runs along the world's -y - which puts
    /// x to the right of the image and y up it.
    pub fn look_straight_down(&mut self) {
        self.scene_from_camera.set_rotation(
            Rotation3::try_from_mat(MatF64::<3, 3>::from_columns(&[
                VecF64::<3>::new(1.0, 0.0, 0.0),
                VecF64::<3>::new(0.0, -1.0, 0.0),
                VecF64::<3>::new(0.0, 0.0, -1.0),
            ]))
            .expect("the three axes are orthonormal"),
        );
    }
}

/// How far from the pivot the camera ends up, having been asked to close a fraction `zoom` of
/// the way to it.
///
/// Zooming scales the distance to the pivot, so what it needs is a limit on that distance: do not
/// zoom through the pivot, nor so far out that it leaves the frustum. That limit is a clamp rather
/// than a refusal, and it only bites against the direction the zoom is going - refusing the whole
/// step whenever the new distance fell outside the planes locked the zoom up in *both* directions
/// as soon as the pivot was nearer than the near plane, which is easily reached, since the pivot
/// is whatever the pointer happens to be over.
fn zoomed_distance(distance: f64, zoom: f64, clipping_planes: ClippingPlanesF64) -> f64 {
    let target = distance * (1.0 - zoom);
    match zoom > 0.0 {
        true => target.max(clipping_planes.near.min(distance)),
        false => target.min(clipping_planes.far.max(distance)),
    }
}

impl OrbitalInteraction {
    /// The median distance of everything in view - where to put the pivot when the pointer is
    /// over nothing at all.
    fn median_scene_distance(&self, inverse_distance_image: &ArcImageF32) -> f64 {
        let scalar_view = inverse_distance_image.tensor.scalar_view();
        let mut inverse_distances = scalar_view
            .as_slice()
            .unwrap()
            .iter()
            .filter(|rho| !is_background(**rho))
            .collect::<Vec<_>>();
        inverse_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        match inverse_distances.is_empty() {
            false => 1.0 / *inverse_distances[inverse_distances.len() / 2] as f64,
            // nothing in view at all: half way down the frustum, as good a guess as any
            true => 0.5 * (self.clipping_planes.near + self.clipping_planes.far),
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
        inverse_distance_image: &ArcImageF32,
    ) {
        let smooth_scroll_delta = response.ctx.input(|i| i.smooth_scroll_delta);
        let is_scroll_zero = smooth_scroll_delta.x == 0.0 && smooth_scroll_delta.y == 0.0;

        // A scroll has to be ended even when the pointer has meanwhile left the view, which the
        // checks below return on - otherwise the interaction stays `is_active` forever, and the
        // focus marker with it.
        if is_scroll_zero {
            self.maybe_scroll_state = None;
            if self.maybe_pointer_state.is_none() {
                self.maybe_gesture = None;
            }
        }

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

        let scroll_started = self.maybe_scroll_state.is_none() && !is_scroll_zero;
        if scroll_started {
            *active_view = self.view_name.clone();

            self.maybe_pivot = Some(ScenePivot {
                pixel: scales.apply(uv_viewport),
                distance: self.distance_under(uv_viewport, inverse_distance_image),
            });
            self.maybe_scroll_state = Some(OrbitalScrollState {});
        }

        if self.maybe_pivot.is_none() {
            return;
        }

        let pivot = self.maybe_pivot.unwrap();
        let pixel = pivot.pixel;
        let pivot_in_camera = pivot.point_in_camera(cam);

        if smooth_scroll_delta.y != 0.0 {
            self.maybe_gesture = Some(PivotGesture::Zoom);

            let scene_from_camera = self.scene_from_camera;
            let camera_in_scene = scene_from_camera.translation();
            let zoom: f64 = (0.002 * smooth_scroll_delta.y) as f64;
            let pivot_in_scene = scene_from_camera.transform(pivot_in_camera);
            let camera_to_pivot_in_scene = pivot_in_scene - camera_in_scene;
            let distance = camera_to_pivot_in_scene.norm();
            if distance < 1e-9 {
                return;
            }

            let clamped = zoomed_distance(distance, zoom, self.clipping_planes);

            let mut new_scene_from_camera = self.scene_from_camera;
            new_scene_from_camera
                .set_translation(pivot_in_scene - camera_to_pivot_in_scene * (clamped / distance));
            self.scene_from_camera = new_scene_from_camera;
            self.maybe_pivot = Some(ScenePivot {
                pixel,
                distance: clamped,
            });
            self.last_moved_at = response.ctx.input(|i| i.time);
        }

        if smooth_scroll_delta.x != 0.0 {
            self.maybe_gesture = Some(PivotGesture::Roll);
            let delta_z: f64 = (smooth_scroll_delta.x) as f64;
            let delta = 0.002 * VecF64::<6>::new(0.0, 0.0, delta_z, 0.0, 0.0, 0.0);
            let camera_from_scene_point = Isometry3::from_translation(pivot_in_camera);

            self.scene_from_camera = self.scene_from_camera
                * camera_from_scene_point
                * Isometry3::exp(delta)
                * camera_from_scene_point.inverse();
        }
    }

    /// The distance to whatever is under the pointer - or, where that is nothing, the distance
    /// the pivot already had, and failing that the median of the scene.
    fn distance_under(
        &self,
        viewport_pixel: egui::Pos2,
        inverse_distance_image: &ArcImageF32,
    ) -> f64 {
        let inverse_distance =
            inverse_distance_image.pixel(viewport_pixel.x as usize, viewport_pixel.y as usize);
        if !is_background(inverse_distance) {
            return 1.0 / inverse_distance as f64;
        }
        match self.maybe_pivot {
            Some(pivot) => pivot.distance,
            None => self.median_scene_distance(inverse_distance_image),
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
        inverse_distance_image: &ArcImageF32,
    ) {
        // Back to the pose the view was created with. An orbit has no home to find its own way
        // back to - pan far enough and the scene is off the screen, with nothing left to pick a
        // pivot off, and so no way to drag back to it.
        //
        // Not while the view is being orbited, though. A short quick drag is a click as far as
        // egui is concerned, and two of them in a row - which is how anyone turns a view a little
        // at a time - is a double click, which would throw the view away just as it was being
        // aimed. So the gesture only counts when the camera has been still for a moment.
        let now = response.ctx.input(|i| i.time);
        if response.double_clicked() && now - self.last_moved_at > 0.4 {
            *active_view = self.view_name.clone();
            self.scene_from_camera = self.initial_scene_from_camera;
            self.maybe_pivot = None;
            self.maybe_pointer_state = None;
            return;
        }

        let delta_x = response.drag_delta().x;
        let delta_y = response.drag_delta().y;

        if response.drag_started() {
            // A drag event started - select new scene focus

            *active_view = self.view_name.clone();

            let Some(pointer) = response.interact_pointer_pos() else {
                self.maybe_pointer_state = None;
                return;
            };

            let uv_viewport = egui::Pos2::new(
                (pointer - response.rect.min)[0],
                (pointer - response.rect.min)[1],
            );

            let pixel = scales.apply(uv_viewport);

            self.maybe_pivot = Some(ScenePivot {
                pixel,
                distance: self.distance_under(uv_viewport, inverse_distance_image),
            });
            self.maybe_pointer_state = Some(OrbitalPointerState {
                start_uv_virtual_camera: pixel,
            });
        } else if response.drag_stopped() {
            // A drag event finished
            self.maybe_pointer_state = None;
            self.maybe_gesture = None;
        };

        if !locked_to_birds_eye_orientation
            && (response.dragged_by(egui::PointerButton::Secondary)
                || (response.dragged_by(egui::PointerButton::Primary)
                    && response.ctx.input(|i| i.modifiers.shift)))
        {
            // rotate about scene focus
            let Some(pivot) = self.maybe_pivot else {
                return;
            };
            let delta =
                0.01 * VecF64::<6>::new(-delta_y as f64, delta_x as f64, 0.0, 0.0, 0.0, 0.0);
            self.maybe_gesture = Some(PivotGesture::Orbit);
            let camera_from_scene_point = Isometry3::from_translation(pivot.point_in_camera(cam));
            self.scene_from_camera = self.scene_from_camera
                * camera_from_scene_point
                * Isometry3::exp(delta)
                * camera_from_scene_point.inverse();
            self.last_moved_at = now;
        } else if response.dragged_by(egui::PointerButton::Primary) {
            // translate scene

            let Some(pointer) = response.interact_pointer_pos() else {
                return;
            };
            let uv_viewport = pointer - response.rect.min;
            let current_pixel = scales.apply(uv_viewport.to_pos2()).cast::<f32>();
            let Some(pivot) = self.maybe_pivot else {
                return;
            };
            let Some(pointer_state) = self.maybe_pointer_state else {
                return;
            };
            self.maybe_gesture = Some(PivotGesture::Pan);
            let start_pixel = pointer_state.start_uv_virtual_camera;
            // both ends of the drag at the distance of the pivot, so that the scene keeps up
            // with the pointer at the depth being dragged
            let p0 = ScenePivot {
                pixel: start_pixel,
                distance: pivot.distance,
            }
            .point_in_camera(cam);
            let p1 = ScenePivot {
                pixel: VecF64::<2>::new(
                    start_pixel.x + (delta_x as f64 * scales.scale.x),
                    start_pixel.y + (delta_y as f64 * scales.scale.y),
                ),
                distance: pivot.distance,
            }
            .point_in_camera(cam);
            let mut scene_from_camera = self.scene_from_camera;
            let delta = p0 - p1;
            let translation_update = scene_from_camera.factor().transform(delta);
            scene_from_camera.set_translation(scene_from_camera.translation() + translation_update);
            self.scene_from_camera = scene_from_camera;
            self.last_moved_at = now;

            if let Some(pivot) = &mut self.maybe_pivot {
                pivot.pixel = VecF64::<2>::new(current_pixel.x as f64, current_pixel.y as f64);
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
        inverse_distance_image: &ArcImageF32,
    ) {
        self.process_pointer(
            active_view,
            cam,
            locked_to_birds_eye_orientation,
            response,
            scales,
            inverse_distance_image,
        );
        self.process_scrolls(
            active_view,
            cam,
            response,
            scales,
            view_port_size,
            inverse_distance_image,
        );
    }

    /// Get zoom
    pub fn zoom2d(&self) -> TranslationAndScaling {
        TranslationAndScaling::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_stops_at_the_planes_without_locking_up() {
        let planes = ClippingPlanesF64 {
            near: 1.0,
            far: 1000.0,
        };
        let closer = 0.1;
        let further = -0.1;

        // in and out, well inside the planes
        approx::assert_abs_diff_eq!(zoomed_distance(10.0, closer, planes), 9.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(
            zoomed_distance(10.0, further, planes),
            11.0,
            epsilon = 1e-12
        );

        // up against them
        approx::assert_abs_diff_eq!(zoomed_distance(1.05, closer, planes), 1.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(
            zoomed_distance(995.0, further, planes),
            1000.0,
            epsilon = 1e-12
        );

        // And the case which used to lock up: the pivot is already nearer than the near plane,
        // because it is whatever the pointer is over. Closing in is refused, since it would only
        // make that worse - but backing away has to work, or there is no way out of it.
        approx::assert_abs_diff_eq!(zoomed_distance(0.5, closer, planes), 0.5, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(zoomed_distance(0.5, further, planes), 0.55, epsilon = 1e-12);
        // likewise beyond the far plane
        approx::assert_abs_diff_eq!(
            zoomed_distance(2000.0, further, planes),
            2000.0,
            epsilon = 1e-9
        );
        approx::assert_abs_diff_eq!(
            zoomed_distance(2000.0, closer, planes),
            1800.0,
            epsilon = 1e-9
        );
    }
}
