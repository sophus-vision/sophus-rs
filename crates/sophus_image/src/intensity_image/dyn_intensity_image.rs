use crate::arc_image::ArcImage2F32;
use crate::arc_image::ArcImage2U16;
use crate::arc_image::ArcImage2U8;
use crate::arc_image::ArcImage3F32;
use crate::arc_image::ArcImage3U16;
use crate::arc_image::ArcImage3U8;
use crate::arc_image::ArcImage4F32;
use crate::arc_image::ArcImage4U16;
use crate::arc_image::ArcImage4U8;
use crate::arc_image::ArcImageF32;
use crate::arc_image::ArcImageR;
use crate::arc_image::ArcImageU16;
use crate::arc_image::ArcImageU8;
use crate::image_view::ImageView2F32;
use crate::image_view::ImageView2U16;
use crate::image_view::ImageView2U8;
use crate::image_view::ImageView3F32;
use crate::image_view::ImageView3U16;
use crate::image_view::ImageView3U8;
use crate::image_view::ImageView4F32;
use crate::image_view::ImageView4U16;
use crate::image_view::ImageView4U8;
use crate::image_view::ImageViewF32;
use crate::image_view::ImageViewU16;
use crate::image_view::ImageViewU8;
use crate::intensity_image::intensity_arc_image::IsIntensityArcImage;
use crate::intensity_image::intensity_scalar::IsIntensityScalar;
use crate::mut_image::MutImage2F32;
use crate::mut_image::MutImage2U16;
use crate::mut_image::MutImage2U8;
use crate::mut_image::MutImage3F32;
use crate::mut_image::MutImage3U16;
use crate::mut_image::MutImage3U8;
use crate::mut_image::MutImage4F32;
use crate::mut_image::MutImage4U16;
use crate::mut_image::MutImage4U8;
use crate::mut_image::MutImageF32;
use crate::mut_image::MutImageU16;
use crate::mut_image::MutImageU8;
use crate::ArcImage;

/// dynamic mutable intensity image of unsigned integer values
pub enum DynIntensityMutImageU {
    /// mutable u8 grayscale image
    GrayscaleU8(MutImageU8),
    /// mutable u8 grayscale+alpha image
    GrayscaleAlphaU8(MutImage2U8),
    /// mutable u8 RGB image
    RgbU8(MutImage3U8),
    /// mutable u8 RGBA image
    RgbaU8(MutImage4U8),
    /// mutable u16 grayscale image
    GrayscaleU16(MutImageU16),
    /// mutable u16 grayscale+alpha image
    GrayscaleAlphaU16(MutImage2U16),
    /// mutable u16 RGB image
    RgbU16(MutImage3U16),
    /// mutable u16 RGBA image
    RgbaU16(MutImage4U16),
}

/// dynamic mutable intensity image of unsigned integer values
pub enum DynIntensityArcImageU {
    /// shared u8 grayscale image
    GrayscaleU8(ArcImageU8),
    /// shared u8 grayscale+alpha image
    GrayscaleAlphaU8(ArcImage2U8),
    /// shared u8 RGB image
    RgbU8(ArcImage3U8),
    /// shared u8 RGBA image
    RgbaU8(ArcImage4U8),
    /// shared u16 grayscale image
    GrayscaleU16(ArcImageU16),
    /// shared u16 grayscale+alpha image
    GrayscaleAlphaU16(ArcImage2U16),
    /// shared u16 RGB image
    RgbU16(ArcImage3U16),
    /// shared u16 RGBA image
    RgbaU16(ArcImage4U16),
}

impl From<DynIntensityMutImageU> for DynIntensityArcImageU {
    fn from(image: DynIntensityMutImageU) -> Self {
        match image {
            DynIntensityMutImageU::GrayscaleU8(image) => {
                DynIntensityArcImageU::GrayscaleU8(image.into())
            }
            DynIntensityMutImageU::GrayscaleAlphaU8(image) => {
                DynIntensityArcImageU::GrayscaleAlphaU8(image.into())
            }
            DynIntensityMutImageU::RgbU8(image) => DynIntensityArcImageU::RgbU8(image.into()),
            DynIntensityMutImageU::RgbaU8(image) => DynIntensityArcImageU::RgbaU8(image.into()),
            DynIntensityMutImageU::GrayscaleU16(image) => {
                DynIntensityArcImageU::GrayscaleU16(image.into())
            }
            DynIntensityMutImageU::GrayscaleAlphaU16(image) => {
                DynIntensityArcImageU::GrayscaleAlphaU16(image.into())
            }
            DynIntensityMutImageU::RgbU16(image) => DynIntensityArcImageU::RgbU16(image.into()),
            DynIntensityMutImageU::RgbaU16(image) => DynIntensityArcImageU::RgbaU16(image.into()),
        }
    }
}

/// dynamic intensity image view of unsigned integer values
pub enum DynIntensityImageViewU<'a> {
    /// u8 grayscale image view
    GrayscaleU8(ImageViewU8<'a>),
    /// u8 grayscale+alpha image view
    GrayscaleAlphaU8(ImageView2U8<'a>),
    /// u8 RGB image view
    RgbU8(ImageView3U8<'a>),
    /// u8 RGBA image view
    RgbaU8(ImageView4U8<'a>),
    /// u16 grayscale image view
    GrayscaleU16(ImageViewU16<'a>),
    /// u16 grayscale+alpha image view
    GrayscaleAlphaU16(ImageView2U16<'a>),
    /// u16 RGB image view
    RgbU16(ImageView3U16<'a>),
    /// u16 RGBA image view
    RgbaU16(ImageView4U16<'a>),
}

/// dynamic mutable intensity image
pub enum DynIntensityMutImage {
    /// mutable u8 grayscale image
    GrayscaleU8(MutImageU8),
    /// mutable u8 grayscale+alpha image
    GrayscaleAlphaU8(MutImage2U8),
    /// mutable u8 RGB image
    RgbU8(MutImage3U8),
    /// mutable u8 RGBA image
    RgbaU8(MutImage4U8),
    /// mutable u16 grayscale image
    GrayscaleU16(MutImageU16),
    /// mutable u16 grayscale+alpha image
    GrayscaleAlphaU16(MutImage2U16),
    /// mutable u16 RGB image
    RgbU16(MutImage3U16),
    /// mutable u16 RGBA image
    RgbaU16(MutImage4U16),
    /// mutable f32 grayscale image
    GrayscaleF32(MutImageF32),
    /// mutable f32 grayscale+alpha image
    GrayscaleAlphaF32(MutImage2F32),
    /// mutable f32 RGB image
    RgbF32(MutImage3F32),
    /// mutable f32 RGBA image
    RgbaF32(MutImage4F32),
}

/// dynamic intensity image view
#[derive(Clone, Debug)]
pub enum DynIntensityArcImage {
    /// shared u8 grayscale image
    GrayscaleU8(ArcImageU8),
    /// shared u8 grayscale+alpha image
    GrayscaleAlphaU8(ArcImage2U8),
    /// shared u8 RGB image
    RgbU8(ArcImage3U8),
    /// shared u8 RGBA image
    RgbaU8(ArcImage4U8),
    /// shared u16 grayscale image
    GrayscaleU16(ArcImageU16),
    /// shared u16 grayscale+alpha image
    GrayscaleAlphaU16(ArcImage2U16),
    /// shared u16 RGB image
    RgbU16(ArcImage3U16),
    /// shared u16 RGBA image
    RgbaU16(ArcImage4U16),
    /// shared f32 grayscale image
    GrayscaleF32(ArcImageF32),
    /// shared f32 grayscale+alpha image
    GrayscaleAlphaF32(ArcImage2F32),
    /// shared f32 RGB image
    RgbF32(ArcImage3F32),
    /// shared f32 RGBA image
    RgbaF32(ArcImage4F32),
}

/// Convert a GenMutImage to an GenArcImage
///
impl From<DynIntensityMutImage> for DynIntensityArcImage {
    fn from(image: DynIntensityMutImage) -> Self {
        match image {
            DynIntensityMutImage::GrayscaleU8(image) => {
                DynIntensityArcImage::GrayscaleU8(image.into())
            }
            DynIntensityMutImage::GrayscaleAlphaU8(image) => {
                DynIntensityArcImage::GrayscaleAlphaU8(image.into())
            }
            DynIntensityMutImage::RgbU8(image) => DynIntensityArcImage::RgbU8(image.into()),
            DynIntensityMutImage::RgbaU8(image) => DynIntensityArcImage::RgbaU8(image.into()),
            DynIntensityMutImage::GrayscaleU16(image) => {
                DynIntensityArcImage::GrayscaleU16(image.into())
            }
            DynIntensityMutImage::GrayscaleAlphaU16(image) => {
                DynIntensityArcImage::GrayscaleAlphaU16(image.into())
            }
            DynIntensityMutImage::RgbU16(image) => DynIntensityArcImage::RgbU16(image.into()),
            DynIntensityMutImage::RgbaU16(image) => DynIntensityArcImage::RgbaU16(image.into()),
            DynIntensityMutImage::GrayscaleF32(image) => {
                DynIntensityArcImage::GrayscaleF32(image.into())
            }
            DynIntensityMutImage::GrayscaleAlphaF32(image) => {
                DynIntensityArcImage::GrayscaleAlphaF32(image.into())
            }
            DynIntensityMutImage::RgbF32(image) => DynIntensityArcImage::RgbF32(image.into()),
            DynIntensityMutImage::RgbaF32(image) => DynIntensityArcImage::RgbaF32(image.into()),
        }
    }
}

impl DynIntensityArcImage {
    /// Converts to grayscale image of specified scalar type
    pub fn to_grayscale<OtherScalar: IsIntensityScalar>(self) -> ArcImage<OtherScalar> {
        match self {
            DynIntensityArcImage::GrayscaleU8(image) => image.to_grayscale(),
            DynIntensityArcImage::GrayscaleAlphaU8(image) => image.to_grayscale(),
            DynIntensityArcImage::RgbU8(image) => image.to_grayscale(),
            DynIntensityArcImage::RgbaU8(image) => image.to_grayscale(),
            DynIntensityArcImage::GrayscaleU16(image) => image.to_grayscale(),
            DynIntensityArcImage::GrayscaleAlphaU16(image) => image.to_grayscale(),
            DynIntensityArcImage::RgbU16(image) => image.to_grayscale(),
            DynIntensityArcImage::RgbaU16(image) => image.to_grayscale(),
            DynIntensityArcImage::GrayscaleF32(image) => image.to_grayscale(),
            DynIntensityArcImage::GrayscaleAlphaF32(image) => image.to_grayscale(),
            DynIntensityArcImage::RgbF32(image) => image.to_grayscale(),
            DynIntensityArcImage::RgbaF32(image) => image.to_grayscale(),
        }
    }

    /// Converts to rgba image of specified scalar type
    pub fn to_rgba<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 4> {
        match self {
            DynIntensityArcImage::GrayscaleU8(image) => image.to_rgba(),
            DynIntensityArcImage::GrayscaleAlphaU8(image) => image.to_rgba(),
            DynIntensityArcImage::RgbU8(image) => image.to_rgba(),
            DynIntensityArcImage::RgbaU8(image) => image.to_rgba(),
            DynIntensityArcImage::GrayscaleU16(image) => image.to_rgba(),
            DynIntensityArcImage::GrayscaleAlphaU16(image) => image.to_rgba(),
            DynIntensityArcImage::RgbU16(image) => image.to_rgba(),
            DynIntensityArcImage::RgbaU16(image) => image.to_rgba(),
            DynIntensityArcImage::GrayscaleF32(image) => image.to_rgba(),
            DynIntensityArcImage::GrayscaleAlphaF32(image) => image.to_rgba(),
            DynIntensityArcImage::RgbF32(image) => image.to_rgba(),
            DynIntensityArcImage::RgbaF32(image) => image.to_rgba(),
        }
    }
}

/// dynamic intensity image view
pub enum DynIntensityImageView<'a> {
    /// u8 grayscale image view
    GrayscaleU8(ImageViewU8<'a>),
    /// u8 grayscale+alpha image view
    GrayscaleAlphaU8(ImageView2U8<'a>),
    /// u8 RGB image view
    RgbU8(ImageView3U8<'a>),
    /// u8 RGBA image view
    RgbaU8(ImageView4U8<'a>),
    /// u16 grayscale image view
    GrayscaleU16(ImageViewU16<'a>),
    /// u16 grayscale+alpha image view
    GrayscaleAlphaU16(ImageView2U16<'a>),
    /// u16 RGB image view
    RgbU16(ImageView3U16<'a>),
    /// u16 RGBA image view
    RgbaU16(ImageView4U16<'a>),
    /// f32 grayscale image view
    GrayscaleF32(ImageViewF32<'a>),
    /// f32 grayscale+alpha image view
    GrayscaleAlphaF32(ImageView2F32<'a>),
    /// f32 RGB image view
    RgbF32(ImageView3F32<'a>),
    /// f32 RGBA image view
    RgbaF32(ImageView4F32<'a>),
}
