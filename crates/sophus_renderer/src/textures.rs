use sophus_image::ImageSize;

use crate::{
    textures::{
        depth::DepthTextures,
        rgba::RgbdTexture,
    },
    RenderContext,
};

/// depth textures
pub mod depth;
/// depth image
pub mod depth_image;
/// NDC z buffer textures
pub mod ndc_z_buffer;
/// RGBA textures
pub mod rgba;
/// Visual depth texture
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
