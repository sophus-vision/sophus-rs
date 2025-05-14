use sophus_autodiff::linalg::VecF64;
use sophus_geo::LineF64;
use sophus_renderer::renderables::{
    Color,
    SceneRenderable,
    make_line3,
};

use crate::examples::optics_sim::element::Element;

/// A struct representing the aperture stop in an optical system.
#[derive(Clone)]
pub struct ApertureStop {
    /// The x-coordinate of the aperture stop.
    pub x: f64,
    /// The radius of the aperture stop.
    pub radius: f64,
}

impl ApertureStop {
    /// Plane through the aperture stop, perpendicular to the optical axis.
    pub fn aperture_plane(&self) -> LineF64 {
        let p0 = VecF64::<2>::new(self.x, -0.200);
        let p1 = VecF64::<2>::new(self.x, 0.200);
        LineF64::from_point_pair(p0, p1)
    }
}

impl Element for ApertureStop {
    fn to_renderable3(&self) -> SceneRenderable {
        let up = 0.5;
        make_line3(
            "aperture",
            &[
                [
                    VecF64::<3>::new(self.x, up, 0.0).cast(),
                    VecF64::<3>::new(self.x, self.radius, 0.0).cast(),
                ],
                [
                    VecF64::<3>::new(self.x, -up, 0.0).cast(),
                    VecF64::<3>::new(self.x, -self.radius, 0.0).cast(),
                ],
            ],
            &Color::black(1.0),
            5.0,
        )
    }
}
