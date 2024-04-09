use crate::arc_image::ArcImage;
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
use crate::image_view::ImageSize;
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
use crate::image_view::IsImageView;
use crate::mut_image::MutImage;
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
use crate::mut_image::MutImageR;
use crate::mut_image::MutImageU16;
use crate::mut_image::MutImageU8;
use sophus_core::linalg::scalar::IsCoreScalar;

use sophus_core::linalg::SVec;
use sophus_core::tensor::element::IsStaticTensor;

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
    /// Converts to u8 grayscale image
    pub fn to_grayscale_u8(&self) -> ArcImageU8 {
        match self {
            DynIntensityArcImage::GrayscaleU8(image) => image.clone(),
            DynIntensityArcImage::GrayscaleAlphaU8(image) => IntensityArcImage::to_grayscale(image),
            DynIntensityArcImage::RgbU8(image) => IntensityArcImage::to_grayscale(image),
            DynIntensityArcImage::RgbaU8(image) => IntensityArcImage::to_grayscale(image),
            DynIntensityArcImage::GrayscaleU16(image) => {
                IntensityArcImage::cast_u8(&IntensityArcImage::to_grayscale(image))
            }
            DynIntensityArcImage::GrayscaleAlphaU16(image) => {
                IntensityArcImage::cast_u8(&IntensityArcImage::to_grayscale(image))
            }
            DynIntensityArcImage::RgbU16(image) => {
                IntensityArcImage::cast_u8(&IntensityArcImage::to_grayscale(image))
            }
            DynIntensityArcImage::RgbaU16(image) => {
                IntensityArcImage::cast_u8(&IntensityArcImage::to_grayscale(image))
            }
            DynIntensityArcImage::GrayscaleF32(image) => {
                IntensityArcImage::cast_u8(&IntensityArcImage::to_grayscale(image))
            }
            DynIntensityArcImage::GrayscaleAlphaF32(image) => {
                IntensityArcImage::cast_u8(&IntensityArcImage::to_grayscale(image))
            }
            DynIntensityArcImage::RgbF32(image) => {
                IntensityArcImage::cast_u8(&IntensityArcImage::to_grayscale(image))
            }
            DynIntensityArcImage::RgbaF32(image) => {
                IntensityArcImage::cast_u8(&IntensityArcImage::to_grayscale(image))
            }
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

/// Trait for "intensity" images (grayscale, grayscale+alpha, RGB, RGBA).
///
/// Hence it s a trait for grayscale (1-channel), grayscale+alpha (2-channel), RGB (3-channel), and
/// RGBA images (4-channel).
///
/// This trait provides methods for converting between different image type. As of now, three
/// scalar type are supported: `u8`, `u16`, and `f32`:
///
///  - u8 images are in the range [0, 255], i.e. 100% intensity corresponds to 255.
///
///  - u16 images are in the range [0, 65535], i.e. 100% intensity corresponds to 65535.
///
///  - f32 images shall be in the range [0.0, 1.0] and 100% intensity corresponds to 1.0.
///    If the f32 is outside this range, conversion results may be surprising.
///
/// These are image type which typically used for computer vision and graphics applications.
pub trait IntensityMutImage<
    const TOTAL_RANK: usize,
    const SRANK: usize,
    Scalar: IsCoreScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS> + 'static,
    const ROWS: usize,
    const COLS: usize,
>
{
    /// Shared tensor type
    type GenArcImage<S: IsCoreScalar>;
    /// Mutable tensor type
    type GenMutImage<S: IsCoreScalar>;

    /// Pixel type
    type Pixel<S: IsCoreScalar>;

    /// Converts a pixel to a grayscale value.
    fn pixel_to_grayscale(pixel: &STensor) -> Scalar;

    /// Converts the image to a grayscale image.
    fn to_grayscale(img: Self) -> MutImage<Scalar>;

    /// Converts the image to a u8 image.
    fn cast_pixel_u8(p: &STensor) -> Self::Pixel<u8>;

    /// Converts the image to a u16 image.
    fn cast_pixel_u16(p: &STensor) -> Self::Pixel<u16>;

    /// Converts the image to a f32 image.
    fn cast_pixel_f32(p: &STensor) -> Self::Pixel<f32>;

    /// Converts the image to a u8 image.
    fn cast_u8(img: Self) -> Self::GenMutImage<u8>;

    /// Converts the image to a u16 image.
    fn cast_u16(img: Self) -> Self::GenMutImage<u16>;

    /// Converts the image to a f32 image.
    fn cast_f32(img: Self) -> Self::GenMutImage<f32>;

    /// Returns a dynamic image view.
    fn into_dyn_image_view(img: Self) -> DynIntensityMutImage;

    /// Tries to return a dynamic image view of unsigned values.
    ///
    /// If the image is not of unsigned type, it returns None.
    fn try_into_dyn_image_view_u(img: Self) -> Option<DynIntensityMutImageU>;
}

impl<'a> IntensityMutImage<2, 0, u8, u8, 1, 1> for MutImageU8 {
    type Pixel<S: IsCoreScalar> = S;

    fn pixel_to_grayscale(pixel: &u8) -> u8 {
        *pixel
    }

    fn to_grayscale(img: Self) -> MutImageU8 {
        img
    }

    fn into_dyn_image_view(image: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::GrayscaleU8(image)
    }

    fn try_into_dyn_image_view_u(image: Self) -> Option<DynIntensityMutImageU> {
        Some(DynIntensityMutImageU::GrayscaleU8(image))
    }

    fn cast_pixel_u8(p: &u8) -> u8 {
        *p
    }

    fn cast_pixel_u16(p: &u8) -> u16 {
        *p as u16
    }

    fn cast_pixel_f32(p: &u8) -> f32 {
        *p as f32 / 255.0
    }

    type GenArcImage<S: IsCoreScalar> = ArcImage<S>;

    type GenMutImage<S: IsCoreScalar> = MutImage<S>;

    fn cast_u8(img: Self) -> Self::GenMutImage<u8> {
        img
    }

    fn cast_u16(img: Self) -> Self::GenMutImage<u16> {
        Self::GenMutImage::<u16>::from_map(&img.image_view(), |rgb: &u8| -> u16 {
            Self::cast_pixel_u16(rgb)
        })
    }

    fn cast_f32(img: Self) -> Self::GenMutImage<f32> {
        Self::GenMutImage::<f32>::from_map(&img.image_view(), |rgb: &u8| -> f32 {
            Self::cast_pixel_f32(rgb)
        })
    }
}

impl IntensityMutImage<2, 0, u16, u16, 1, 1> for MutImageU16 {
    type Pixel<S: IsCoreScalar> = S;

    fn pixel_to_grayscale(pixel: &u16) -> u16 {
        *pixel
    }

    fn to_grayscale(img: Self) -> MutImageU16 {
        img
    }

    fn into_dyn_image_view(image: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::GrayscaleU16(image)
    }

    fn try_into_dyn_image_view_u(image: Self) -> Option<DynIntensityMutImageU> {
        Some(DynIntensityMutImageU::GrayscaleU16(image))
    }

    fn cast_pixel_u8(p: &u16) -> u8 {
        (p / 255).clamp(0, 255) as u8
    }

    fn cast_pixel_u16(p: &u16) -> u16 {
        *p
    }

    fn cast_pixel_f32(p: &u16) -> f32 {
        *p as f32 / 65535.0
    }

    type GenArcImage<S: IsCoreScalar> = ArcImage<S>;

    type GenMutImage<S: IsCoreScalar> = MutImage<S>;

    fn cast_u8(img: Self) -> Self::GenMutImage<u8> {
        Self::GenMutImage::<u8>::from_map(&img.image_view(), |rgb: &u16| -> u8 {
            Self::cast_pixel_u8(rgb)
        })
    }

    fn cast_u16(img: Self) -> Self::GenMutImage<u16> {
        img
    }

    fn cast_f32(img: Self) -> Self::GenMutImage<f32> {
        Self::GenMutImage::<f32>::from_map(&img.image_view(), |rgb: &u16| -> f32 {
            Self::cast_pixel_f32(rgb)
        })
    }
}

impl IntensityMutImage<2, 0, f32, f32, 1, 1> for MutImageF32 {
    type Pixel<S: IsCoreScalar> = S;

    fn pixel_to_grayscale(pixel: &f32) -> f32 {
        *pixel
    }

    fn to_grayscale(img: Self) -> MutImageF32 {
        img
    }

    fn into_dyn_image_view(image: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::GrayscaleF32(image)
    }

    fn try_into_dyn_image_view_u(_image: Self) -> Option<DynIntensityMutImageU> {
        None
    }

    fn cast_pixel_u8(p: &f32) -> u8 {
        (p * 255.0).clamp(0.0, 255.0) as u8
    }

    fn cast_pixel_u16(p: &f32) -> u16 {
        (p * 65535.0).clamp(0.0, 65535.0) as u16
    }

    fn cast_pixel_f32(p: &f32) -> f32 {
        *p
    }

    type GenArcImage<S: IsCoreScalar> = ArcImage<S>;

    type GenMutImage<S: IsCoreScalar> = MutImage<S>;

    fn cast_u8(img: Self) -> Self::GenMutImage<u8> {
        Self::GenMutImage::<u8>::from_map(&img.image_view(), |rgb: &f32| -> u8 {
            Self::cast_pixel_u8(rgb)
        })
    }

    fn cast_u16(img: Self) -> Self::GenMutImage<u16> {
        Self::GenMutImage::<u16>::from_map(&img.image_view(), |rgb: &f32| -> u16 {
            Self::cast_pixel_u16(rgb)
        })
    }

    fn cast_f32(img: Self) -> Self::GenMutImage<f32> {
        img
    }
}

impl IntensityMutImage<3, 1, u8, SVec<u8, 4>, 4, 1> for MutImage4U8 {
    type Pixel<S: IsCoreScalar> = SVec<S, 4>;

    fn pixel_to_grayscale(pixel: &SVec<u8, 4>) -> u8 {
        pixel[0]
    }

    fn to_grayscale(img: Self) -> MutImageU8 {
        MutImageU8::from_map(&img.image_view(), |rgb: &SVec<u8, 4>| -> u8 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::RgbaU8(image)
    }

    fn try_into_dyn_image_view_u(image: Self) -> Option<DynIntensityMutImageU> {
        Some(DynIntensityMutImageU::RgbaU8(image))
    }

    fn cast_pixel_u8(p: &SVec<u8, 4>) -> SVec<u8, 4> {
        *p
    }

    fn cast_pixel_u16(p: &SVec<u8, 4>) -> SVec<u16, 4> {
        SVec::<u16, 4>::new(p[0] as u16, p[1] as u16, p[2] as u16, p[3] as u16)
    }

    fn cast_pixel_f32(p: &SVec<u8, 4>) -> SVec<f32, 4> {
        SVec::<f32, 4>::new(
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
            p[3] as f32 / 255.0,
        )
    }

    type GenArcImage<S: IsCoreScalar> = ArcImageR<S, 4>;

    type GenMutImage<S: IsCoreScalar> = MutImageR<S, 4>;

    fn cast_u8(img: Self) -> Self::GenMutImage<u8> {
        img
    }

    fn cast_u16(img: Self) -> Self::GenMutImage<u16> {
        Self::GenMutImage::<u16>::from_map(&img.image_view(), |rgb: &SVec<u8, 4>| -> SVec<u16, 4> {
            Self::cast_pixel_u16(rgb)
        })
    }

    fn cast_f32(img: Self) -> Self::GenMutImage<f32> {
        Self::GenMutImage::<f32>::from_map(&img.image_view(), |rgb: &SVec<u8, 4>| -> SVec<f32, 4> {
            Self::cast_pixel_f32(rgb)
        })
    }
}

/// Trait for "intensity" images with shared ownership.
pub trait IntensityArcImage<
    const TOTAL_RANK: usize,
    const SRANK: usize,
    Scalar: IsCoreScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS> + 'static,
    const ROWS: usize,
    const COLS: usize,
>
{
    /// Shared tensor type
    type GenArcImage<S: IsCoreScalar>;
    /// Mutable tensor type
    type GenMutImage<S: IsCoreScalar>;
    /// Pixel type
    type Pixel<S: IsCoreScalar>;

    /// Converts a pixel to a grayscale value.
    fn pixel_to_grayscale(pixel: &STensor) -> Scalar;

    /// Converts the image to a grayscale image.
    fn to_grayscale(img: &Self) -> ArcImage<Scalar>;

    /// Converts the image to a u8 image.
    fn cast_pixel_u8(p: &STensor) -> Self::Pixel<u8>;

    /// Converts the image to a u16 image.
    fn cast_pixel_u16(p: &STensor) -> Self::Pixel<u16>;

    /// Converts the image to a f32 image.
    fn cast_pixel_f32(p: &STensor) -> Self::Pixel<f32>;

    /// Converts the image to a u8 image.
    fn cast_u8(img: &Self) -> Self::GenArcImage<u8>;

    /// Converts the image to a u16 image.
    fn cast_u16(img: &Self) -> Self::GenArcImage<u16>;
    //     Self::to_map(&img, |rgb: &STensor| -> Scalar {
    //         Self::cast_pixel_u16(rgb)
    //     })
    // }

    /// Converts the image to a f32 image.
    fn cast_f32(img: &Self) -> Self::GenArcImage<f32>;

    /// Returns a dynamic image view.
    fn into_dyn_image_view(img: &Self) -> DynIntensityArcImage;

    /// Tries to return a dynamic image view of unsigned values.
    ///
    /// If the image is not of unsigned type, it returns None.
    fn try_into_dyn_image_view_u(img: &Self) -> Option<DynIntensityArcImageU>;
}

impl IntensityArcImage<2, 0, u8, u8, 1, 1> for ArcImageU8 {
    type Pixel<S: IsCoreScalar> = S;

    fn pixel_to_grayscale(pixel: &u8) -> u8 {
        *pixel
    }

    fn to_grayscale(img: &Self) -> ArcImageU8 {
        img.clone()
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleU8(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::GrayscaleU8(image.clone()))
    }

    fn cast_pixel_u8(p: &u8) -> u8 {
        *p
    }

    fn cast_pixel_u16(p: &u8) -> u16 {
        *p as u16
    }

    fn cast_pixel_f32(p: &u8) -> f32 {
        *p as f32 / 255.0
    }

    type GenArcImage<S: IsCoreScalar> = ArcImage<S>;

    type GenMutImage<S: IsCoreScalar> = MutImage<S>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        img.clone()
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        Self::GenArcImage::<u16>::from_map(&img.image_view(), |rgb: &u8| -> u16 {
            Self::cast_pixel_u16(rgb)
        })
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        Self::GenArcImage::<f32>::from_map(&img.image_view(), |rgb: &u8| -> f32 {
            Self::cast_pixel_f32(rgb)
        })
    }
}

impl IntensityArcImage<2, 0, u16, u16, 1, 1> for ArcImageU16 {
    type Pixel<S: IsCoreScalar> = S;

    fn pixel_to_grayscale(pixel: &u16) -> u16 {
        *pixel
    }

    fn to_grayscale(img: &Self) -> ArcImageU16 {
        ArcImageU16::from_map(&img.image_view(), |rgb: &u16| -> u16 {
            Self::cast_pixel_u16(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleU16(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::GrayscaleU16(image.clone()))
    }

    fn cast_pixel_u8(p: &u16) -> u8 {
        (p / 255).clamp(0, 255) as u8
    }

    fn cast_pixel_u16(p: &u16) -> u16 {
        *p
    }

    fn cast_pixel_f32(p: &u16) -> f32 {
        *p as f32 / 65535.0
    }

    type GenArcImage<S: IsCoreScalar> = ArcImage<S>;

    type GenMutImage<S: IsCoreScalar> = MutImage<S>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        Self::GenArcImage::<u8>::from_map(&img.image_view(), |rgb: &u16| -> u8 {
            Self::cast_pixel_u8(rgb)
        })
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        img.clone()
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        Self::GenArcImage::<f32>::from_map(&img.image_view(), |rgb: &u16| -> f32 {
            Self::cast_pixel_f32(rgb)
        })
    }
}

impl IntensityArcImage<2, 0, f32, f32, 1, 1> for ArcImageF32 {
    type Pixel<S: IsCoreScalar> = S;

    fn pixel_to_grayscale(pixel: &f32) -> f32 {
        *pixel
    }

    fn to_grayscale(img: &Self) -> ArcImageF32 {
        img.clone()
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleF32(image.clone())
    }

    fn try_into_dyn_image_view_u(_image: &Self) -> Option<DynIntensityArcImageU> {
        None
    }

    fn cast_pixel_u8(p: &f32) -> u8 {
        (p * 255.0).clamp(0.0, 255.0) as u8
    }

    fn cast_pixel_u16(p: &f32) -> u16 {
        (p * 65535.0).clamp(0.0, 65535.0) as u16
    }

    fn cast_pixel_f32(p: &f32) -> f32 {
        *p
    }

    type GenArcImage<S: IsCoreScalar> = ArcImage<S>;

    type GenMutImage<S: IsCoreScalar> = MutImage<S>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        Self::GenArcImage::<u8>::from_map(&img.image_view(), |rgb: &f32| -> u8 {
            Self::cast_pixel_u8(rgb)
        })
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        Self::GenArcImage::<u16>::from_map(&img.image_view(), |rgb: &f32| -> u16 {
            Self::cast_pixel_u16(rgb)
        })
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        img.clone()
    }
}

impl IntensityArcImage<3, 1, u8, SVec<u8, 2>, 2, 1> for ArcImage2U8 {
    type Pixel<S: IsCoreScalar> = SVec<S, 2>;

    fn pixel_to_grayscale(pixel: &SVec<u8, 2>) -> u8 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU8 {
        ArcImageU8::from_map(&img.image_view(), |rgb: &SVec<u8, 2>| -> u8 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleAlphaU8(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::GrayscaleAlphaU8(image.clone()))
    }

    fn cast_pixel_u8(p: &SVec<u8, 2>) -> SVec<u8, 2> {
        *p
    }

    fn cast_pixel_u16(p: &SVec<u8, 2>) -> SVec<u16, 2> {
        SVec::<u16, 2>::new(p[0] as u16, p[1] as u16)
    }

    fn cast_pixel_f32(p: &SVec<u8, 2>) -> SVec<f32, 2> {
        SVec::<f32, 2>::new(p[0] as f32 / 255.0, p[1] as f32 / 255.0)
    }

    type GenArcImage<S: IsCoreScalar> = ArcImageR<S, 2>;

    type GenMutImage<S: IsCoreScalar> = MutImageR<S, 2>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        img.clone()
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        Self::GenArcImage::<u16>::from_map(&img.image_view(), |rgb: &SVec<u8, 2>| -> SVec<u16, 2> {
            Self::cast_pixel_u16(rgb)
        })
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        Self::GenArcImage::<f32>::from_map(&img.image_view(), |rgb: &SVec<u8, 2>| -> SVec<f32, 2> {
            Self::cast_pixel_f32(rgb)
        })
    }
}

impl IntensityArcImage<3, 1, u8, SVec<u8, 3>, 3, 1> for ArcImage3U8 {
    type Pixel<S: IsCoreScalar> = SVec<S, 3>;

    fn pixel_to_grayscale(pixel: &SVec<u8, 3>) -> u8 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU8 {
        ArcImageU8::from_map(&img.image_view(), |rgb: &SVec<u8, 3>| -> u8 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbU8(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::RgbU8(image.clone()))
    }

    fn cast_pixel_u8(p: &SVec<u8, 3>) -> SVec<u8, 3> {
        *p
    }

    fn cast_pixel_u16(p: &SVec<u8, 3>) -> SVec<u16, 3> {
        SVec::<u16, 3>::new(p[0] as u16, p[1] as u16, p[2] as u16)
    }

    fn cast_pixel_f32(p: &SVec<u8, 3>) -> SVec<f32, 3> {
        SVec::<f32, 3>::new(
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
        )
    }

    type GenArcImage<S: IsCoreScalar> = ArcImageR<S, 3>;

    type GenMutImage<S: IsCoreScalar> = MutImageR<S, 3>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        img.clone()
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        Self::GenArcImage::<u16>::from_map(&img.image_view(), |rgb: &SVec<u8, 3>| -> SVec<u16, 3> {
            Self::cast_pixel_u16(rgb)
        })
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        Self::GenArcImage::<f32>::from_map(&img.image_view(), |rgb: &SVec<u8, 3>| -> SVec<f32, 3> {
            Self::cast_pixel_f32(rgb)
        })
    }
}

impl IntensityArcImage<3, 1, u8, SVec<u8, 4>, 4, 1> for ArcImage4U8 {
    type Pixel<S: IsCoreScalar> = SVec<S, 4>;

    fn pixel_to_grayscale(pixel: &SVec<u8, 4>) -> u8 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU8 {
        ArcImageU8::from_map(&img.image_view(), |rgb: &SVec<u8, 4>| -> u8 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbaU8(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::RgbaU8(image.clone()))
    }

    fn cast_pixel_u8(p: &SVec<u8, 4>) -> SVec<u8, 4> {
        *p
    }

    fn cast_pixel_u16(p: &SVec<u8, 4>) -> SVec<u16, 4> {
        SVec::<u16, 4>::new(p[0] as u16, p[1] as u16, p[2] as u16, p[3] as u16)
    }

    fn cast_pixel_f32(p: &SVec<u8, 4>) -> SVec<f32, 4> {
        SVec::<f32, 4>::new(
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
            p[3] as f32 / 255.0,
        )
    }

    type GenArcImage<S: IsCoreScalar> = ArcImageR<S, 4>;

    type GenMutImage<S: IsCoreScalar> = MutImageR<S, 4>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        img.clone()
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        Self::GenArcImage::<u16>::from_map(&img.image_view(), |rgb: &SVec<u8, 4>| -> SVec<u16, 4> {
            Self::cast_pixel_u16(rgb)
        })
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        Self::GenArcImage::<f32>::from_map(&img.image_view(), |rgb: &SVec<u8, 4>| -> SVec<f32, 4> {
            Self::cast_pixel_f32(rgb)
        })
    }
}

impl IntensityArcImage<3, 1, u16, SVec<u16, 2>, 2, 1> for ArcImage2U16 {
    type Pixel<S: IsCoreScalar> = SVec<S, 2>;

    fn pixel_to_grayscale(pixel: &SVec<u16, 2>) -> u16 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU16 {
        ArcImageU16::from_map(&img.image_view(), |rgb: &SVec<u16, 2>| -> u16 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleAlphaU16(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::GrayscaleAlphaU16(image.clone()))
    }

    fn cast_pixel_u8(p: &SVec<u16, 2>) -> SVec<u8, 2> {
        SVec::<u8, 2>::new(
            (p[0] / 255).clamp(0, 255) as u8,
            (p[1] / 255).clamp(0, 255) as u8,
        )
    }

    fn cast_pixel_u16(p: &SVec<u16, 2>) -> SVec<u16, 2> {
        *p
    }

    fn cast_pixel_f32(p: &SVec<u16, 2>) -> SVec<f32, 2> {
        SVec::<f32, 2>::new(p[0] as f32 / 65535.0, p[1] as f32 / 65535.0)
    }

    type GenArcImage<S: IsCoreScalar> = ArcImageR<S, 2>;

    type GenMutImage<S: IsCoreScalar> = MutImageR<S, 2>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        Self::GenArcImage::<u8>::from_map(&img.image_view(), |rgb: &SVec<u16, 2>| -> SVec<u8, 2> {
            Self::cast_pixel_u8(rgb)
        })
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        img.clone()
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        Self::GenArcImage::<f32>::from_map(
            &img.image_view(),
            |rgb: &SVec<u16, 2>| -> SVec<f32, 2> { Self::cast_pixel_f32(rgb) },
        )
    }
}

impl IntensityArcImage<3, 1, u16, SVec<u16, 3>, 3, 1> for ArcImage3U16 {
    type Pixel<S: IsCoreScalar> = SVec<S, 3>;

    fn pixel_to_grayscale(pixel: &SVec<u16, 3>) -> u16 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU16 {
        ArcImageU16::from_map(&img.image_view(), |rgb: &SVec<u16, 3>| -> u16 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbU16(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::RgbU16(image.clone()))
    }

    fn cast_pixel_u8(p: &SVec<u16, 3>) -> SVec<u8, 3> {
        SVec::<u8, 3>::new(
            (p[0] / 255).clamp(0, 255) as u8,
            (p[1] / 255).clamp(0, 255) as u8,
            (p[2] / 255).clamp(0, 255) as u8,
        )
    }

    fn cast_pixel_u16(p: &SVec<u16, 3>) -> SVec<u16, 3> {
        *p
    }

    fn cast_pixel_f32(p: &SVec<u16, 3>) -> SVec<f32, 3> {
        SVec::<f32, 3>::new(
            p[0] as f32 / 65535.0,
            p[1] as f32 / 65535.0,
            p[2] as f32 / 65535.0,
        )
    }

    type GenArcImage<S: IsCoreScalar> = ArcImageR<S, 3>;

    type GenMutImage<S: IsCoreScalar> = MutImageR<S, 3>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        Self::GenArcImage::<u8>::from_map(&img.image_view(), |rgb: &SVec<u16, 3>| -> SVec<u8, 3> {
            Self::cast_pixel_u8(rgb)
        })
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        img.clone()
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        Self::GenArcImage::<f32>::from_map(
            &img.image_view(),
            |rgb: &SVec<u16, 3>| -> SVec<f32, 3> { Self::cast_pixel_f32(rgb) },
        )
    }
}

impl IntensityArcImage<3, 1, u16, SVec<u16, 4>, 4, 1> for ArcImage4U16 {
    type Pixel<S: IsCoreScalar> = SVec<S, 4>;

    fn pixel_to_grayscale(pixel: &SVec<u16, 4>) -> u16 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU16 {
        ArcImageU16::from_map(&img.image_view(), |rgb: &SVec<u16, 4>| -> u16 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbaU16(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::RgbaU16(image.clone()))
    }

    fn cast_pixel_u8(p: &SVec<u16, 4>) -> SVec<u8, 4> {
        SVec::<u8, 4>::new(
            (p[0] / 255).clamp(0, 255) as u8,
            (p[1] / 255).clamp(0, 255) as u8,
            (p[2] / 255).clamp(0, 255) as u8,
            (p[3] / 255).clamp(0, 255) as u8,
        )
    }

    fn cast_pixel_u16(p: &SVec<u16, 4>) -> SVec<u16, 4> {
        *p
    }

    fn cast_pixel_f32(p: &SVec<u16, 4>) -> SVec<f32, 4> {
        SVec::<f32, 4>::new(
            p[0] as f32 / 65535.0,
            p[1] as f32 / 65535.0,
            p[2] as f32 / 65535.0,
            p[3] as f32 / 65535.0,
        )
    }

    type GenArcImage<S: IsCoreScalar> = ArcImageR<S, 4>;

    type GenMutImage<S: IsCoreScalar> = MutImageR<S, 4>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        Self::GenArcImage::<u8>::from_map(&img.image_view(), |rgb: &SVec<u16, 4>| -> SVec<u8, 4> {
            Self::cast_pixel_u8(rgb)
        })
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        img.clone()
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        Self::GenArcImage::<f32>::from_map(
            &img.image_view(),
            |rgb: &SVec<u16, 4>| -> SVec<f32, 4> { Self::cast_pixel_f32(rgb) },
        )
    }
}

impl IntensityArcImage<3, 1, f32, SVec<f32, 2>, 2, 1> for ArcImage2F32 {
    type Pixel<S: IsCoreScalar> = SVec<S, 2>;

    fn pixel_to_grayscale(pixel: &SVec<f32, 2>) -> f32 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageF32 {
        ArcImageF32::from_map(&img.image_view(), |rgb: &SVec<f32, 2>| -> f32 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleAlphaF32(image.clone())
    }

    fn try_into_dyn_image_view_u(_image: &Self) -> Option<DynIntensityArcImageU> {
        None
    }

    fn cast_pixel_u8(p: &SVec<f32, 2>) -> SVec<u8, 2> {
        SVec::<u8, 2>::new(
            (p[0] * 255.0).clamp(0.0, 255.0) as u8,
            (p[1] * 255.0).clamp(0.0, 255.0) as u8,
        )
    }

    fn cast_pixel_u16(p: &SVec<f32, 2>) -> SVec<u16, 2> {
        SVec::<u16, 2>::new(
            (p[0] * 65535.0).clamp(0.0, 65535.0) as u16,
            (p[1] * 65535.0).clamp(0.0, 65535.0) as u16,
        )
    }

    fn cast_pixel_f32(p: &SVec<f32, 2>) -> SVec<f32, 2> {
        *p
    }

    type GenArcImage<S: IsCoreScalar> = ArcImageR<S, 2>;

    type GenMutImage<S: IsCoreScalar> = MutImageR<S, 2>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        Self::GenArcImage::<u8>::from_map(&img.image_view(), |rgb: &SVec<f32, 2>| -> SVec<u8, 2> {
            Self::cast_pixel_u8(rgb)
        })
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        Self::GenArcImage::<u16>::from_map(
            &img.image_view(),
            |rgb: &SVec<f32, 2>| -> SVec<u16, 2> { Self::cast_pixel_u16(rgb) },
        )
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        img.clone()
    }
}

impl IntensityArcImage<3, 1, f32, SVec<f32, 3>, 3, 1> for ArcImage3F32 {
    type Pixel<S: IsCoreScalar> = SVec<S, 3>;

    fn pixel_to_grayscale(pixel: &SVec<f32, 3>) -> f32 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageF32 {
        ArcImageF32::from_map(&img.image_view(), |rgb: &SVec<f32, 3>| -> f32 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbF32(image.clone())
    }

    fn try_into_dyn_image_view_u(_image: &Self) -> Option<DynIntensityArcImageU> {
        None
    }

    fn cast_pixel_u8(p: &SVec<f32, 3>) -> SVec<u8, 3> {
        SVec::<u8, 3>::new(
            (p[0] * 255.0).clamp(0.0, 255.0) as u8,
            (p[1] * 255.0).clamp(0.0, 255.0) as u8,
            (p[2] * 255.0).clamp(0.0, 255.0) as u8,
        )
    }

    fn cast_pixel_u16(p: &SVec<f32, 3>) -> SVec<u16, 3> {
        SVec::<u16, 3>::new(
            (p[0] * 65535.0).clamp(0.0, 65535.0) as u16,
            (p[1] * 65535.0).clamp(0.0, 65535.0) as u16,
            (p[2] * 65535.0).clamp(0.0, 65535.0) as u16,
        )
    }

    fn cast_pixel_f32(p: &SVec<f32, 3>) -> SVec<f32, 3> {
        *p
    }

    type GenArcImage<S: IsCoreScalar> = ArcImageR<S, 3>;

    type GenMutImage<S: IsCoreScalar> = MutImageR<S, 3>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        Self::GenArcImage::<u8>::from_map(&img.image_view(), |rgb: &SVec<f32, 3>| -> SVec<u8, 3> {
            Self::cast_pixel_u8(rgb)
        })
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        Self::GenArcImage::<u16>::from_map(
            &img.image_view(),
            |rgb: &SVec<f32, 3>| -> SVec<u16, 3> { Self::cast_pixel_u16(rgb) },
        )
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        img.clone()
    }
}

impl IntensityArcImage<3, 1, f32, SVec<f32, 4>, 4, 1> for ArcImage4F32 {
    type Pixel<S: IsCoreScalar> = SVec<S, 4>;

    fn pixel_to_grayscale(pixel: &SVec<f32, 4>) -> f32 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageF32 {
        ArcImageF32::from_map(&img.image_view(), |rgb: &SVec<f32, 4>| -> f32 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbaF32(image.clone())
    }

    fn try_into_dyn_image_view_u(_image: &Self) -> Option<DynIntensityArcImageU> {
        None
    }

    fn cast_pixel_u8(p: &SVec<f32, 4>) -> SVec<u8, 4> {
        SVec::<u8, 4>::new(
            (p[0] * 255.0).clamp(0.0, 255.0) as u8,
            (p[1] * 255.0).clamp(0.0, 255.0) as u8,
            (p[2] * 255.0).clamp(0.0, 255.0) as u8,
            (p[3] * 255.0).clamp(0.0, 255.0) as u8,
        )
    }

    fn cast_pixel_u16(p: &SVec<f32, 4>) -> SVec<u16, 4> {
        SVec::<u16, 4>::new(
            (p[0] * 65535.0).clamp(0.0, 65535.0) as u16,
            (p[1] * 65535.0).clamp(0.0, 65535.0) as u16,
            (p[2] * 65535.0).clamp(0.0, 65535.0) as u16,
            (p[3] * 65535.0).clamp(0.0, 65535.0) as u16,
        )
    }

    fn cast_pixel_f32(p: &SVec<f32, 4>) -> SVec<f32, 4> {
        *p
    }

    type GenArcImage<S: IsCoreScalar> = ArcImageR<S, 4>;

    type GenMutImage<S: IsCoreScalar> = MutImageR<S, 4>;

    fn cast_u8(img: &Self) -> Self::GenArcImage<u8> {
        Self::GenArcImage::<u8>::from_map(&img.image_view(), |rgb: &SVec<f32, 4>| -> SVec<u8, 4> {
            Self::cast_pixel_u8(rgb)
        })
    }

    fn cast_u16(img: &Self) -> Self::GenArcImage<u16> {
        Self::GenArcImage::<u16>::from_map(
            &img.image_view(),
            |rgb: &SVec<f32, 4>| -> SVec<u16, 4> { Self::cast_pixel_u16(rgb) },
        )
    }

    fn cast_f32(img: &Self) -> Self::GenArcImage<f32> {
        img.clone()
    }
}

/// Intensity image view of unsigned integer values.
pub trait IntensityViewImageU<'a> {
    /// Color type of the image
    const COLOR_TYPE: png::ColorType;
    /// Bit depth of the image
    const BIT_DEPTH: png::BitDepth;

    /// Size of the image
    fn size(&'a self) -> ImageSize;

    /// raw u8 slice of the image
    fn raw_u8_slice(&self) -> &[u8];
}

impl<'a> IntensityViewImageU<'a> for ImageViewU8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Grayscale;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IntensityViewImageU<'a> for ImageView2U8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::GrayscaleAlpha;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IntensityViewImageU<'a> for ImageView3U8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgb;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IntensityViewImageU<'a> for ImageView4U8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgba;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IntensityViewImageU<'a> for ImageViewU16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Grayscale;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> IntensityViewImageU<'a> for ImageView2U16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::GrayscaleAlpha;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> IntensityViewImageU<'a> for ImageView3U16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgb;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> IntensityViewImageU<'a> for ImageView4U16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgba;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}
