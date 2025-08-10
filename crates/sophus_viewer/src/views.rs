mod active_view_info;
mod image_view;
/// View layout algorithms.
pub mod layout;
mod plot_view;
mod scene_view;

use alloc::string::{
    String,
    ToString,
};

pub use active_view_info::*;
pub(crate) use image_view::*;
pub use plot_view::*;
pub(crate) use scene_view::*;
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
