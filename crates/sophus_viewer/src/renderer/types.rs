use eframe::egui;
use num_traits;
use num_traits::cast;
use sophus_core::linalg::VecF64;
use sophus_image::arc_image::ArcImage4U8;

use crate::renderer::textures::depth_image::DepthImage;
use crate::renderer::OffscreenRenderer;
use crate::viewer::aspect_ratio::HasAspectRatio;

/// floating point: f32 or f64
pub trait FloatingPointNumber:
    num_traits::Float + num_traits::FromPrimitive + num_traits::NumCast
{
}
impl FloatingPointNumber for f32 {}
impl FloatingPointNumber for f64 {}

/// Clipping planes for the Wgpu renderer
#[derive(Clone, Copy, Debug)]
pub struct ClippingPlanes<S: FloatingPointNumber> {
    /// Near clipping plane
    pub near: S,
    /// Far clipping plane
    pub far: S,
}

/// f32 clipping planes
pub type ClippingPlanesF32 = ClippingPlanes<f32>;
/// f64 clipping planes
pub type ClippingPlanesF64 = ClippingPlanes<f64>;

impl ClippingPlanesF64 {
    /// default near clipping plane
    pub const DEFAULT_NEAR: f64 = 1.0;
    /// default far clipping plane
    pub const DEFAULT_FAR: f64 = 1000.0;
}

impl Default for ClippingPlanesF64 {
    fn default() -> Self {
        ClippingPlanes {
            near: ClippingPlanes::DEFAULT_NEAR,
            far: ClippingPlanes::DEFAULT_FAR,
        }
    }
}

impl<S: FloatingPointNumber> ClippingPlanes<S> {
    /// metric z from ndc z
    pub fn metric_z_from_ndc_z(&self, ndc_z: S) -> S {
        -(self.far * self.near) / (-self.far + ndc_z * self.far - ndc_z * self.near)
    }

    /// ndc z from metric z
    pub fn ndc_z_from_metric_z(&self, z: S) -> S {
        (self.far * (z - self.near)) / (z * (self.far - self.near))
    }

    /// cast
    pub fn cast<Other: FloatingPointNumber>(self) -> ClippingPlanes<Other> {
        ClippingPlanes {
            near: cast(self.near).unwrap(),
            far: cast(self.far).unwrap(),
        }
    }
}

/// Render result
pub struct RenderResult {
    /// rgba image
    pub rgba_image: Option<ArcImage4U8>,

    /// rgba egui texture id
    pub rgba_egui_tex_id: egui::TextureId,

    /// depth image
    pub depth_image: DepthImage,

    /// depth egui texture id
    pub depth_egui_tex_id: egui::TextureId,
}

impl HasAspectRatio for OffscreenRenderer {
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

#[test]
fn clipping_plane_tests() {
    for near in [0.1, 0.5, 1.0, 7.0] {
        for far in [10.0, 5.0, 100.0, 1000.0] {
            for ndc_z in [0.0, 0.1, 0.5, 0.7, 0.99, 1.0] {
                let clipping_planes = ClippingPlanesF64 { near, far };

                let metric_z = clipping_planes.metric_z_from_ndc_z(ndc_z);
                let roundtrip_ndc_z = clipping_planes.ndc_z_from_metric_z(metric_z);

                approx::assert_abs_diff_eq!(roundtrip_ndc_z, ndc_z, epsilon = 0.0001);
            }
        }
    }
}
