mod active_view_info;
mod image_view;
mod plot_view;
mod scene_view;

use alloc::{
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

pub use active_view_info::*;
pub(crate) use image_view::*;
use linked_hash_map::LinkedHashMap;
pub use plot_view::*;
pub(crate) use scene_view::*;
use sophus_image::ImageSize;
use sophus_renderer::{
    HasAspectRatio,
    camera::RenderCameraProperties,
};

use crate::interactions::InteractionEnum;

extern crate alloc;

/// The view enum.
pub(crate) enum View {
    /// view of a 3d scene - optionally locked to 2d birds eye view
    Scene(SceneView),
    /// view of a 2d image or empty image canvas - with potential 2d pixel and 3d scene overlay
    Image(ImageView),
    /// graph view
    Plot(PlotView),
}

impl HasAspectRatio for View {
    fn aspect_ratio(&self) -> f32 {
        match self {
            View::Scene(view) => view.aspect_ratio(),
            View::Image(view) => view.aspect_ratio(),
            View::Plot(view) => view.aspect_ratio(),
        }
    }
}

impl View {
    pub(crate) fn enabled_mut(&mut self) -> &mut bool {
        match self {
            View::Scene(view) => &mut view.enabled,
            View::Image(view) => &mut view.enabled,
            View::Plot(view) => &mut view.enabled,
        }
    }

    pub(crate) fn view_type(&mut self) -> String {
        match self {
            View::Scene(_) => "Scene".to_string(),
            View::Image(_) => "Image".to_string(),
            View::Plot(_) => "Plot".to_string(),
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        match self {
            View::Scene(view) => view.enabled,
            View::Image(view) => view.enabled,
            View::Plot(view) => view.enabled,
        }
    }

    pub(crate) fn interaction(&self) -> &InteractionEnum {
        match self {
            View::Scene(view) => &view.interaction,
            View::Image(view) => &view.interaction,
            View::Plot(view) => &view.interaction,
        }
    }

    pub(crate) fn locked_to_birds_eye_orientation(&self) -> bool {
        match self {
            View::Scene(view) => view.locked_to_birds_eye_orientation,
            View::Image(_) => true,
            View::Plot(_) => true,
        }
    }

    pub(crate) fn camera_propterties(&self) -> RenderCameraProperties {
        match self {
            View::Scene(view) => view.renderer.camera_properties(),
            View::Image(view) => view.renderer.camera_properties(),
            // not really meaningful, so we just return the default
            View::Plot(_) => RenderCameraProperties::default(),
        }
    }
}

pub(crate) fn get_median_aspect_ratio_and_num(
    views: &LinkedHashMap<String, View>,
) -> Option<(f32, usize)> {
    let mut aspect_ratios = Vec::with_capacity(views.len());
    for (_label, widget) in views.iter() {
        if widget.enabled() {
            aspect_ratios.push(widget.aspect_ratio());
        }
    }
    aspect_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n = aspect_ratios.len();
    if n == 0 {
        return None;
    }
    if n % 2 == 1 {
        Some((aspect_ratios[n / 2], n))
    } else {
        Some((
            0.5 * aspect_ratios[n / 2] + 0.5 * aspect_ratios[n / 2 - 1],
            n,
        ))
    }
}

pub(crate) fn get_max_size(
    views: &LinkedHashMap<String, View>,
    available_width: f32,
    available_height: f32,
) -> Option<(f32, f32)> {
    if let Some((median_aspect_ratio, n)) = get_median_aspect_ratio_and_num(views) {
        let mut max_width = 0.0;
        let mut max_height = 0.0;

        for num_cols in 1..=n {
            let num_rows: f32 = ((n as f32) / (num_cols as f32)).ceil();

            let w: f32 = available_width / (num_cols as f32);
            let h = (w / median_aspect_ratio).min(available_height / num_rows);
            let w = median_aspect_ratio * h;
            if w > max_width {
                max_width = w;
                max_height = h;
            }
        }

        return Some((max_width, max_height));
    }
    None
}

#[derive(Clone, Copy)]
pub(crate) struct ViewportSize {
    pub(crate) width: f32,
    pub(crate) height: f32,
}

impl ViewportSize {
    pub(crate) fn image_size(&self) -> ImageSize {
        ImageSize {
            width: self.width.ceil() as usize,
            height: self.height.ceil() as usize,
        }
    }
}

pub(crate) fn get_adjusted_view_size(
    view_aspect_ratio: f32,
    max_width: f32,
    max_height: f32,
) -> ViewportSize {
    let width = 1.0_f32.max(max_width.min(max_height * view_aspect_ratio));
    let height = 1.0_f32.max(max_height.min(max_width / view_aspect_ratio));
    ViewportSize { width, height }
}
