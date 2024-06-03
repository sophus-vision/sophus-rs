/// aspect ratio
pub mod aspect_ratio;
/// 2d view
pub mod view2d;
/// 3d view
pub mod view3d;

use sophus_image::ImageSize;
use sophus_sensor::DynCamera;

use crate::views::aspect_ratio::HasAspectRatio;
use crate::views::view2d::View2d;
use crate::views::view3d::View3d;

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

    fn intrinsics(&self) -> &DynCamera<f64, 1> {
        match self {
            View::View3d(view) => view.intrinsics(),
            View::View2d(view) => view.intrinsics(),
        }
    }

    fn view_size(&self) -> ImageSize {
        self.intrinsics().image_size()
    }
}
