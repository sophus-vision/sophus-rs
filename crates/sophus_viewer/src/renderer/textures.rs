use crate::renderer::textures::depth::DepthTextures;
use crate::renderer::textures::rgba::RgbdTexture;
use crate::RenderContext;
use sophus_image::ImageSize;

/// Depth textures.
pub mod depth;
/// depth image
pub mod depth_image;
/// Main render z buffer texture.
pub mod main_render_z_buffer;
/// RGBA texture.
pub mod rgba;
/// Visual depth texture.
pub mod visual_depth;

#[derive(Debug)]
pub(crate) struct Textures {
    pub(crate) view_port_size: ImageSize,
    pub(crate) rgbd: RgbdTexture,
    pub depth: DepthTextures,
}

impl Textures {
    pub(crate) fn new(render_state: &RenderContext, view_port_size: &ImageSize) -> Self {
        Self {
            view_port_size: *view_port_size,
            rgbd: RgbdTexture::new(render_state, view_port_size),
            depth: DepthTextures::new(render_state, view_port_size),
        }
    }
}
