use sophus_renderer::renderables::{
    Color,
    SceneRenderable,
};

/// A trait for an optical element.
pub trait Element {
    /// Converts the element to a renderable 3D object.
    fn to_renderable3(&self) -> SceneRenderable;
}

/// Creates a gray color with RGB values of 0.3.
pub fn gray_color() -> Color {
    Color {
        r: 0.3,
        g: 0.3,
        b: 0.3,
        a: 1.0,
    }
}
