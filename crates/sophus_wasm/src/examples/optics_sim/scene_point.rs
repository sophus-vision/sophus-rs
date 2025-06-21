use sophus_autodiff::linalg::VecF64;
use sophus_renderer::renderables::{
    Color,
    SceneRenderable,
    make_point3,
};

use crate::examples::optics_sim::{
    element::Element,
    light_path::unproj2,
};

/// A struct representing a pair of points in 2D space.
pub struct ScenePoints {
    /// Point pair.
    pub p: [VecF64<2>; 2],
}

impl Element for ScenePoints {
    fn to_renderable3(&self) -> SceneRenderable {
        make_point3(
            "points",
            &[unproj2(self.p[0].cast()), unproj2(self.p[1].cast())],
            &Color::blue(),
            5.0,
        )
    }
}
