/// 2d view
pub mod view2d;
/// 3d view
pub mod view3d;

use crate::viewer::aspect_ratio::HasAspectRatio;
use crate::viewer::views::view2d::View2d;
use crate::viewer::views::view3d::View3d;

/// The view enum.
pub(crate) enum View {
    /// 3D view
    View3d(View3d),
    /// Image view
    View2d(View2d),
}

impl HasAspectRatio for View {
    fn aspect_ratio(&self) -> f32 {
        match self {
            View::View3d(view) => view.aspect_ratio(),
            View::View2d(view) => view.aspect_ratio(),
        }
    }
}

impl View {
    pub(crate) fn enabled_mut(&mut self) -> &mut bool {
        match self {
            View::View3d(view) => &mut view.enabled,
            View::View2d(view) => &mut view.enabled,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        match self {
            View::View3d(view) => view.enabled,
            View::View2d(view) => view.enabled,
        }
    }
}
