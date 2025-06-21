use sophus_autodiff::linalg::VecF64;
use sophus_geo::LineF64;
use sophus_renderer::renderables::{
    Color,
    SceneRenderable,
    make_line3,
};

use crate::examples::optics_sim::element::Element;

/// Detector in an optical system, i.e., the image plane.
#[derive(Clone)]
pub struct Detector {
    /// The x-coordinate of the detector.
    pub x: f64,
}

impl Detector {
    const HALF_WIDTH: f64 = 0.300;

    /// Creates a new Detector instance.
    pub fn image_plane(&self) -> LineF64 {
        let p0 = VecF64::<2>::new(self.x, -Self::HALF_WIDTH);
        let p1 = VecF64::<2>::new(self.x, Self::HALF_WIDTH);

        LineF64::from_point_pair(p0, p1)
    }
}

impl Element for Detector {
    fn to_renderable3(&self) -> SceneRenderable {
        let p0 = VecF64::<3>::new(self.x, -Self::HALF_WIDTH, 0.0);
        let p1 = VecF64::<3>::new(self.x, Self::HALF_WIDTH, 0.0);

        make_line3(
            "image-plane",
            &[[p0.cast(), p1.cast()]],
            &Color::blue(),
            1.5,
        )
    }
}
