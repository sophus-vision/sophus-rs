use eframe::{
    egui,
    wgpu,
};
use sophus_autodiff::linalg::VecF64;
use sophus_image::ArcImage4U8;

use crate::{
    offscreen_renderer::OffscreenRenderer,
    renderables::Color,
    textures::DepthImage,
};

/// The intermediate render result.
///
/// Depth textures are still on the GPU
#[derive(Debug, Clone)]
pub struct RenderResult {
    /// rgba image
    pub rgba_image: Option<ArcImage4U8>,

    /// depth image - on the GPU
    pub depth_texture: wgpu::Texture,
    /// visual depth texture - on the GPU
    pub visual_depth_texture: wgpu::Texture,
    /// depth staging buffer - to download from the GPU
    pub depth_staging_buffer: wgpu::Buffer,

    /// rgba egui texture id
    pub rgba_egui_tex_id: egui::TextureId,
    /// depth egui texture id
    pub depth_egui_tex_id: egui::TextureId,
}

/// The final render result.
///
/// Depth images are downloaded to the CPU / registered as egui textures.
#[derive(Clone)]
pub struct FinalRenderResult {
    /// rgba image
    pub rgba_image: Option<ArcImage4U8>,

    /// rgba egui texture id
    pub rgba_egui_tex_id: egui::TextureId,

    /// depth egui texture id
    pub depth_egui_tex_id: egui::TextureId,

    /// depth image - on the CPU
    pub depth_image: DepthImage,
}

/// aspect ratio
pub trait HasAspectRatio {
    /// return aspect ratio
    fn aspect_ratio(&self) -> f32;
}

impl HasAspectRatio for OffscreenRenderer {
    fn aspect_ratio(&self) -> f32 {
        self.camera_properties
            .intrinsics
            .image_size()
            .aspect_ratio()
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Zoom2dPod {
    pub(crate) translation_x: f32,
    pub(crate) translation_y: f32,
    pub(crate) scaling_x: f32,
    pub(crate) scaling_y: f32,
}

impl Default for Zoom2dPod {
    fn default() -> Self {
        Zoom2dPod {
            translation_x: 0.0,
            translation_y: 0.0,
            scaling_x: 1.0,
            scaling_y: 1.0,
        }
    }
}

/// Translation and scaling
///
/// todo: move to sophus_lie
#[derive(Clone, Copy, Debug)]
pub struct TranslationAndScaling {
    /// translation
    pub translation: VecF64<2>,
    /// scaling
    pub scaling: VecF64<2>,
}

impl TranslationAndScaling {
    /// identity
    pub fn identity() -> Self {
        TranslationAndScaling {
            translation: VecF64::<2>::zeros(),
            scaling: VecF64::<2>::new(1.0, 1.0),
        }
    }

    /// apply translation and scaling
    pub fn apply(&self, xy: VecF64<2>) -> VecF64<2> {
        VecF64::<2>::new(
            xy[0] * self.scaling[0] + self.translation[0],
            xy[1] * self.scaling[1] + self.translation[1],
        )
    }
}

/// focus point to overlay
pub struct SceneFocusMarker {
    /// color
    pub color: Color,
    /// u viewport pixel
    pub u: f32,
    /// v viewport pixel
    pub v: f32,
    /// ndc_z
    pub ndc_z: f32,
}

/// multisample count
pub const SOPHUS_RENDER_MULTISAMPLE_COUNT: u32 = 4;
