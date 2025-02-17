mod depth;
mod depth_image;
mod ndc_z_buffer;
mod rgba;
mod visual_depth;

pub(crate) use depth::DepthTextures;
pub use depth_image::*;
pub use rgba::*;
use sophus_image::ImageSize;

use crate::RenderContext;

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
