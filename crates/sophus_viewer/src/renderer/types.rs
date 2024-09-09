use eframe::egui;
use sophus_core::linalg::VecF64;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::arc_image::ArcImageF32;

use crate::renderer::Renderer;
use crate::viewer::aspect_ratio::HasAspectRatio;

/// Clipping planes for the Wgpu renderer
#[derive(Clone, Copy, Debug)]
pub struct ClippingPlanes {
    /// Near clipping plane
    pub near: f64,
    /// Far clipping plane
    pub far: f64,
}

impl ClippingPlanes {
    /// default near clipping plabe
    pub const DEFAULT_NEAR: f64 = 1.0;
    /// default far clipping plabe
    pub const DEFAULT_FAR: f64 = 1000.0;
}

impl Default for ClippingPlanes {
    fn default() -> Self {
        ClippingPlanes {
            near: ClippingPlanes::DEFAULT_NEAR,
            far: ClippingPlanes::DEFAULT_FAR,
        }
    }
}

impl ClippingPlanes {
    pub(crate) fn z_from_ndc(&self, ndc: f64) -> f64 {
        -(self.far * self.near) / (-self.far + ndc * self.far - ndc * self.near)
    }

    pub(crate) fn _ndc_from_z(&self, z: f64) -> f64 {
        (self.far * (z - self.near)) / (z * (self.far - self.near))
    }
}

/// Render result
pub struct RenderResult {
    /// rgba image
    pub image_4u8: Option<ArcImage4U8>,

    /// rgba egui texture id
    pub rgba_egui_tex_id: egui::TextureId,

    /// depth image - might have a greater width than the requested width
    pub depth: ArcImageF32,

    /// depth egui texture id
    pub depth_egui_tex_id: egui::TextureId,
}

impl HasAspectRatio for Renderer {
    fn aspect_ratio(&self) -> f32 {
        self.intrinsics.image_size().aspect_ratio()
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Zoom2d {
    pub(crate) translation_x: f32,
    pub(crate) translation_y: f32,
    pub(crate) scaling_x: f32,
    pub(crate) scaling_y: f32,
}

impl Default for Zoom2d {
    fn default() -> Self {
        Zoom2d {
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
