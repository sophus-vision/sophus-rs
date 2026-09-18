mod depth;
mod faces;
mod inverse_distance_image;
mod ndc_z_buffer;
mod rgba;
mod visual_depth;

pub use depth::{
    DepthTextures,
    download_depth,
};
pub(crate) use faces::FaceTextures;
pub use inverse_distance_image::*;
pub use rgba::*;
use sophus_image::ImageSize;

use crate::RenderContext;

#[derive(Debug)]
pub(crate) struct Textures {
    pub(crate) view_port_size: ImageSize,
    pub(crate) rgbd: RgbdTexture,
    pub depth: DepthTextures,
    /// Only allocated for a view too wide for a single plane - see `needs_frusta`.
    pub(crate) faces: Option<FaceTextures>,
}

impl Textures {
    pub(crate) fn new(render_state: &RenderContext, view_port_size: &ImageSize) -> Self {
        Self {
            view_port_size: *view_port_size,
            rgbd: RgbdTexture::new(render_state, view_port_size),
            depth: DepthTextures::new(render_state, view_port_size),
            faces: None,
        }
    }

    /// Makes sure the multi-frustum targets exist at `face_size`.
    pub(crate) fn ensure_faces(&mut self, render_state: &RenderContext, face_size: u32) {
        let matches = self
            .faces
            .as_ref()
            .is_some_and(|faces| faces.face_size == face_size);
        if !matches {
            self.faces = Some(FaceTextures::new(render_state, face_size));
        }
    }
}
