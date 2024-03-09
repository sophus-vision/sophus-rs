use crate::tensor::element::IsStaticTensor;
use crate::tensor::element::IsTensorScalar;
use crate::tensor::element::SVec;

use super::arc_image::ArcImage;
use super::arc_image::ArcImage2F32;
use super::arc_image::ArcImage2U16;
use super::arc_image::ArcImage2U8;
use super::arc_image::ArcImage3F32;
use super::arc_image::ArcImage3U16;
use super::arc_image::ArcImage3U8;
use super::arc_image::ArcImage4F32;
use super::arc_image::ArcImage4U16;
use super::arc_image::ArcImage4U8;
use super::arc_image::ArcImageF32;
use super::arc_image::ArcImageR;
use super::arc_image::ArcImageU16;
use super::arc_image::ArcImageU8;
use super::mut_image::MutImage;
use super::mut_image::MutImage2F32;
use super::mut_image::MutImage2U16;
use super::mut_image::MutImage2U8;
use super::mut_image::MutImage3F32;
use super::mut_image::MutImage3U16;
use super::mut_image::MutImage3U8;
use super::mut_image::MutImage4F32;
use super::mut_image::MutImage4U16;
use super::mut_image::MutImage4U8;
use super::mut_image::MutImageF32;
use super::mut_image::MutImageR;
use super::mut_image::MutImageU16;
use super::mut_image::MutImageU8;
use super::view::ImageSize;
use super::view::ImageView2F32;
use super::view::ImageView2U16;
use super::view::ImageView2U8;
use super::view::ImageView3F32;
use super::view::ImageView3U16;
use super::view::ImageView3U8;
use super::view::ImageView4F32;
use super::view::ImageView4U16;
use super::view::ImageView4U8;
use super::view::ImageViewF32;
use super::view::ImageViewU16;
use super::view::ImageViewU8;
use super::view::IsImageView;

pub enum DynPercentageMutImageU {
    GrayscaleU8(MutImageU8),
    GrayscaleAlphaU8(MutImage2U8),
    RgbU8(MutImage3U8),
    RgbaU8(MutImage4U8),
    GrayscaleU16(MutImageU16),
    GrayscaleAlphaU16(MutImage2U16),
    RgbU16(MutImage3U16),
    RgbaU16(MutImage4U16),
}

pub enum DynPercentageArcImageU {
    GrayscaleU8(ArcImageU8),
    GrayscaleAlphaU8(ArcImage2U8),
    RgbU8(ArcImage3U8),
    RgbaU8(ArcImage4U8),
    GrayscaleU16(ArcImageU16),
    GrayscaleAlphaU16(ArcImage2U16),
    RgbU16(ArcImage3U16),
    RgbaU16(ArcImage4U16),
}

impl From<DynPercentageMutImageU> for DynPercentageArcImageU {
    fn from(image: DynPercentageMutImageU) -> Self {
        match image {
            DynPercentageMutImageU::GrayscaleU8(image) => {
                DynPercentageArcImageU::GrayscaleU8(image.into())
            }
            DynPercentageMutImageU::GrayscaleAlphaU8(image) => {
                DynPercentageArcImageU::GrayscaleAlphaU8(image.into())
            }
            DynPercentageMutImageU::RgbU8(image) => DynPercentageArcImageU::RgbU8(image.into()),
            DynPercentageMutImageU::RgbaU8(image) => DynPercentageArcImageU::RgbaU8(image.into()),
            DynPercentageMutImageU::GrayscaleU16(image) => {
                DynPercentageArcImageU::GrayscaleU16(image.into())
            }
            DynPercentageMutImageU::GrayscaleAlphaU16(image) => {
                DynPercentageArcImageU::GrayscaleAlphaU16(image.into())
            }
            DynPercentageMutImageU::RgbU16(image) => DynPercentageArcImageU::RgbU16(image.into()),
            DynPercentageMutImageU::RgbaU16(image) => DynPercentageArcImageU::RgbaU16(image.into()),
        }
    }
}

pub enum DynPercentageImageViewU<'a> {
    GrayscaleU8(ImageViewU8<'a>),
    GrayscaleAlphaU8(ImageView2U8<'a>),
    RgbU8(ImageView3U8<'a>),
    RgbaU8(ImageView4U8<'a>),
    GrayscaleU16(ImageViewU16<'a>),
    GrayscaleAlphaU16(ImageView2U16<'a>),
    RgbU16(ImageView3U16<'a>),
    RgbaU16(ImageView4U16<'a>),
}

pub enum DynPercentageMutImage {
    GrayscaleU8(MutImageU8),
    GrayscaleAlphaU8(MutImage2U8),
    RgbU8(MutImage3U8),
    RgbaU8(MutImage4U8),
    GrayscaleU16(MutImageU16),
    GrayscaleAlphaU16(MutImage2U16),
    RgbU16(MutImage3U16),
    RgbaU16(MutImage4U16),
    GrayscaleF32(MutImageF32),
    GrayscaleAlphaF32(MutImage2F32),
    RgbF32(MutImage3F32),
    RgbaF32(MutImage4F32),
}

pub enum DynPercentageArcImage {
    GrayscaleU8(ArcImageU8),
    GrayscaleAlphaU8(ArcImage2U8),
    RgbU8(ArcImage3U8),
    RgbaU8(ArcImage4U8),
    GrayscaleU16(ArcImageU16),
    GrayscaleAlphaU16(ArcImage2U16),
    RgbU16(ArcImage3U16),
    RgbaU16(ArcImage4U16),
    GrayscaleF32(ArcImageF32),
    GrayscaleAlphaF32(ArcImage2F32),
    RgbF32(ArcImage3F32),
    RgbaF32(ArcImage4F32),
}

/// Convert a GenMutImage to an GenArcImage  
///
impl From<DynPercentageMutImage> for DynPercentageArcImage {
    fn from(image: DynPercentageMutImage) -> Self {
        match image {
            DynPercentageMutImage::GrayscaleU8(image) => {
                DynPercentageArcImage::GrayscaleU8(image.into())
            }
            DynPercentageMutImage::GrayscaleAlphaU8(image) => {
                DynPercentageArcImage::GrayscaleAlphaU8(image.into())
            }
            DynPercentageMutImage::RgbU8(image) => DynPercentageArcImage::RgbU8(image.into()),
            DynPercentageMutImage::RgbaU8(image) => DynPercentageArcImage::RgbaU8(image.into()),
            DynPercentageMutImage::GrayscaleU16(image) => {
                DynPercentageArcImage::GrayscaleU16(image.into())
            }
            DynPercentageMutImage::GrayscaleAlphaU16(image) => {
                DynPercentageArcImage::GrayscaleAlphaU16(image.into())
            }
            DynPercentageMutImage::RgbU16(image) => DynPercentageArcImage::RgbU16(image.into()),
            DynPercentageMutImage::RgbaU16(image) => DynPercentageArcImage::RgbaU16(image.into()),
            DynPercentageMutImage::GrayscaleF32(image) => {
                DynPercentageArcImage::GrayscaleF32(image.into())
            }
            DynPercentageMutImage::GrayscaleAlphaF32(image) => {
                DynPercentageArcImage::GrayscaleAlphaF32(image.into())
            }
            DynPercentageMutImage::RgbF32(image) => DynPercentageArcImage::RgbF32(image.into()),
            DynPercentageMutImage::RgbaF32(image) => DynPercentageArcImage::RgbaF32(image.into()),
        }
    }
}

impl DynPercentageArcImage {
    pub fn to_grayscale_u8(&self) -> ArcImageU8 {
        match self {
            DynPercentageArcImage::GrayscaleU8(image) => image.clone(),
            DynPercentageArcImage::GrayscaleAlphaU8(image) => {
                PercentageArcImage::to_grayscale(image)
            }
            DynPercentageArcImage::RgbU8(image) => PercentageArcImage::to_grayscale(image),
            DynPercentageArcImage::RgbaU8(image) => PercentageArcImage::to_grayscale(image),
            DynPercentageArcImage::GrayscaleU16(image) => {
                PercentageArcImage::cast_u8(&PercentageArcImage::to_grayscale(image))
            }
            DynPercentageArcImage::GrayscaleAlphaU16(image) => {
                PercentageArcImage::cast_u8(&PercentageArcImage::to_grayscale(image))
            }
            DynPercentageArcImage::RgbU16(image) => {
                PercentageArcImage::cast_u8(&PercentageArcImage::to_grayscale(image))
            }
            DynPercentageArcImage::RgbaU16(image) => {
                PercentageArcImage::cast_u8(&PercentageArcImage::to_grayscale(image))
            }
            DynPercentageArcImage::GrayscaleF32(image) => {
                PercentageArcImage::cast_u8(&PercentageArcImage::to_grayscale(image))
            }
            DynPercentageArcImage::GrayscaleAlphaF32(image) => {
                PercentageArcImage::cast_u8(&PercentageArcImage::to_grayscale(image))
            }
            DynPercentageArcImage::RgbF32(image) => {
                PercentageArcImage::cast_u8(&PercentageArcImage::to_grayscale(image))
            }
            DynPercentageArcImage::RgbaF32(image) => {
                PercentageArcImage::cast_u8(&PercentageArcImage::to_grayscale(image))
            }
        }
    }
}

pub enum DynPercentageImageView<'a> {
    GrayscaleU8(ImageViewU8<'a>),
    GrayscaleAlphaU8(ImageView2U8<'a>),
    RgbU8(ImageView3U8<'a>),
    RgbaU8(ImageView4U8<'a>),
    GrayscaleU16(ImageViewU16<'a>),
    GrayscaleAlphaU16(ImageView2U16<'a>),
    RgbU16(ImageView3U16<'a>),
    RgbaU16(ImageView4U16<'a>),
    GrayscaleF32(ImageViewF32<'a>),
    GrayscaleAlphaF32(ImageView2F32<'a>),
    RgbF32(ImageView3F32<'a>),
    RgbaF32(ImageView4F32<'a>),
}

/// Trait for "percentage" images (grayscale, RGB, RGBA).
///
/// Hence it s a trait for grayscale (1-channel), RGB (3-channel), and RGBA images (4-channel).
///
/// This trait provides methods for converting between different image types. Three scalar types
/// are supported: `u8`, `u16`, and `f32`:
///
///  - u8 images are in the range [0, 255], i.e. 100% corresponds to 255.
///
///  - u16 images are in the range [0, 65535], i.e. 100% corresponds to 65535.
///
///  - f32 images shall be in the range [0.0, 1.0] and 100% corresponds to 1.0.
///    If the f32 is outside this range, conversion results may be surprising.
///
/// These are image types which typically used for computer vision and graphics applications.
pub trait PercentageMutImage<
    const SCALAR_RANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS, BATCHES> + 'static,
    const ROWS: usize,
    const COLS: usize,
    const BATCHES: usize,
>
{
    /// The type of the image with the same number of channels, but with a u8 scalar type.
    type GenArcImage<S: IsTensorScalar>;
    type GenMutImage<S: IsTensorScalar>;

    type Pixel<S: IsTensorScalar>;

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

    // {
    //     Self::to_map(&img, |rgb: &STensor| -> Scalar {
    //         Self::cast_pixel_u8(rgb)
    //     })
    // }

    /// Converts the image to a u16 image.
    fn cast_u16(img: Self) -> Self::GenMutImage<u16>;
    //     Self::to_map(&img, |rgb: &STensor| -> Scalar {
    //         Self::cast_pixel_u16(rgb)
    //     })
    // }

    /// Converts the image to a f32 image.
    fn cast_f32(img: Self) -> Self::GenMutImage<f32>;

    // {
    //     Self::to_map(&img, |rgb: &STensor| -> Scalar {
    //         Self::cast_pixel_f32(rgb)
    //     })
    // }

    fn into_dyn_image_view(img: Self) -> DynPercentageMutImage;

    fn try_into_dyn_image_view_u(img: Self) -> Option<DynPercentageMutImageU>;
}

impl<'a> PercentageMutImage<2, 0, u8, u8, 1, 1, 1> for MutImageU8 {
    type Pixel<S: IsTensorScalar> = S;

    fn pixel_to_grayscale(pixel: &u8) -> u8 {
        *pixel
    }

    fn to_grayscale(img: Self) -> MutImageU8 {
        img
    }

    fn into_dyn_image_view(image: Self) -> DynPercentageMutImage {
        DynPercentageMutImage::GrayscaleU8(image)
    }

    fn try_into_dyn_image_view_u(image: Self) -> Option<DynPercentageMutImageU> {
        Some(DynPercentageMutImageU::GrayscaleU8(image))
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

    type GenArcImage<S: IsTensorScalar> = ArcImage<S>;

    type GenMutImage<S: IsTensorScalar> = MutImage<S>;

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

impl PercentageMutImage<2, 0, u16, u16, 1, 1, 1> for MutImageU16 {
    type Pixel<S: IsTensorScalar> = S;

    fn pixel_to_grayscale(pixel: &u16) -> u16 {
        *pixel
    }

    fn to_grayscale(img: Self) -> MutImageU16 {
        img
    }

    fn into_dyn_image_view(image: Self) -> DynPercentageMutImage {
        DynPercentageMutImage::GrayscaleU16(image)
    }

    fn try_into_dyn_image_view_u(image: Self) -> Option<DynPercentageMutImageU> {
        Some(DynPercentageMutImageU::GrayscaleU16(image))
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

    type GenArcImage<S: IsTensorScalar> = ArcImage<S>;

    type GenMutImage<S: IsTensorScalar> = MutImage<S>;

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

impl PercentageMutImage<2, 0, f32, f32, 1, 1, 1> for MutImageF32 {
    type Pixel<S: IsTensorScalar> = S;

    fn pixel_to_grayscale(pixel: &f32) -> f32 {
        *pixel
    }

    fn to_grayscale(img: Self) -> MutImageF32 {
        img
    }

    fn into_dyn_image_view(image: Self) -> DynPercentageMutImage {
        DynPercentageMutImage::GrayscaleF32(image)
    }

    fn try_into_dyn_image_view_u(image: Self) -> Option<DynPercentageMutImageU> {
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

    type GenArcImage<S: IsTensorScalar> = ArcImage<S>;

    type GenMutImage<S: IsTensorScalar> = MutImage<S>;

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

impl PercentageMutImage<3, 1, u8, SVec<u8, 4>, 4, 1, 1> for MutImage4U8 {
    type Pixel<S: IsTensorScalar> = SVec<S, 4>;

    fn pixel_to_grayscale(pixel: &SVec<u8, 4>) -> u8 {
        pixel[0]
    }

    fn to_grayscale(img: Self) -> MutImageU8 {
        MutImageU8::from_map(&img.image_view(), |rgb: &SVec<u8, 4>| -> u8 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: Self) -> DynPercentageMutImage {
        DynPercentageMutImage::RgbaU8(image)
    }

    fn try_into_dyn_image_view_u(image: Self) -> Option<DynPercentageMutImageU> {
        Some(DynPercentageMutImageU::RgbaU8(image))
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

    type GenArcImage<S: IsTensorScalar> = ArcImageR<S, 4>;

    type GenMutImage<S: IsTensorScalar> = MutImageR<S, 4>;

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

pub trait PercentageArcImage<
    const SCALAR_RANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS, BATCHES> + 'static,
    const ROWS: usize,
    const COLS: usize,
    const BATCHES: usize,
>
{
    /// The type of the image with the same number of channels, but with a u8 scalar type.
    type GenArcImage<S: IsTensorScalar>;
    type GenMutImage<S: IsTensorScalar>;

    type Pixel<S: IsTensorScalar>;

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

    // {
    //     Self::to_map(&img, |rgb: &STensor| -> Scalar {
    //         Self::cast_pixel_u8(rgb)
    //     })
    // }

    /// Converts the image to a u16 image.
    fn cast_u16(img: &Self) -> Self::GenArcImage<u16>;
    //     Self::to_map(&img, |rgb: &STensor| -> Scalar {
    //         Self::cast_pixel_u16(rgb)
    //     })
    // }

    /// Converts the image to a f32 image.
    fn cast_f32(img: &Self) -> Self::GenArcImage<f32>;

    // {
    //     Self::to_map(&img, |rgb: &STensor| -> Scalar {
    //         Self::cast_pixel_f32(rgb)
    //     })
    // }

    fn into_dyn_image_view(img: &Self) -> DynPercentageArcImage;

    fn try_into_dyn_image_view_u(img: &Self) -> Option<DynPercentageArcImageU>;
}

impl PercentageArcImage<2, 0, u8, u8, 1, 1, 1> for ArcImageU8 {
    type Pixel<S: IsTensorScalar> = S;

    fn pixel_to_grayscale(pixel: &u8) -> u8 {
        *pixel
    }

    fn to_grayscale(img: &Self) -> ArcImageU8 {
        img.clone()
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::GrayscaleU8(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
        Some(DynPercentageArcImageU::GrayscaleU8(image.clone()))
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

    type GenArcImage<S: IsTensorScalar> = ArcImage<S>;

    type GenMutImage<S: IsTensorScalar> = MutImage<S>;

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

impl PercentageArcImage<2, 0, u16, u16, 1, 1, 1> for ArcImageU16 {
    type Pixel<S: IsTensorScalar> = S;

    fn pixel_to_grayscale(pixel: &u16) -> u16 {
        *pixel
    }

    fn to_grayscale(img: &Self) -> ArcImageU16 {
        ArcImageU16::from_map(&img.image_view(), |rgb: &u16| -> u16 {
            Self::cast_pixel_u16(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::GrayscaleU16(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
        Some(DynPercentageArcImageU::GrayscaleU16(image.clone()))
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

    type GenArcImage<S: IsTensorScalar> = ArcImage<S>;

    type GenMutImage<S: IsTensorScalar> = MutImage<S>;

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

impl PercentageArcImage<2, 0, f32, f32, 1, 1, 1> for ArcImageF32 {
    type Pixel<S: IsTensorScalar> = S;

    fn pixel_to_grayscale(pixel: &f32) -> f32 {
        *pixel
    }

    fn to_grayscale(img: &Self) -> ArcImageF32 {
        img.clone()
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::GrayscaleF32(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
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

    type GenArcImage<S: IsTensorScalar> = ArcImage<S>;

    type GenMutImage<S: IsTensorScalar> = MutImage<S>;

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

impl PercentageArcImage<3, 1, u8, SVec<u8, 2>, 2, 1, 1> for ArcImage2U8 {
    type Pixel<S: IsTensorScalar> = SVec<S, 2>;

    fn pixel_to_grayscale(pixel: &SVec<u8, 2>) -> u8 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU8 {
        ArcImageU8::from_map(&img.image_view(), |rgb: &SVec<u8, 2>| -> u8 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::GrayscaleAlphaU8(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
        Some(DynPercentageArcImageU::GrayscaleAlphaU8(image.clone()))
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

    type GenArcImage<S: IsTensorScalar> = ArcImageR<S, 2>;

    type GenMutImage<S: IsTensorScalar> = MutImageR<S, 2>;

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

impl PercentageArcImage<3, 1, u8, SVec<u8, 3>, 3, 1, 1> for ArcImage3U8 {
    type Pixel<S: IsTensorScalar> = SVec<S, 3>;

    fn pixel_to_grayscale(pixel: &SVec<u8, 3>) -> u8 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU8 {
        ArcImageU8::from_map(&img.image_view(), |rgb: &SVec<u8, 3>| -> u8 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::RgbU8(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
        Some(DynPercentageArcImageU::RgbU8(image.clone()))
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

    type GenArcImage<S: IsTensorScalar> = ArcImageR<S, 3>;

    type GenMutImage<S: IsTensorScalar> = MutImageR<S, 3>;

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

impl PercentageArcImage<3, 1, u8, SVec<u8, 4>, 4, 1, 1> for ArcImage4U8 {
    type Pixel<S: IsTensorScalar> = SVec<S, 4>;

    fn pixel_to_grayscale(pixel: &SVec<u8, 4>) -> u8 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU8 {
        ArcImageU8::from_map(&img.image_view(), |rgb: &SVec<u8, 4>| -> u8 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::RgbaU8(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
        Some(DynPercentageArcImageU::RgbaU8(image.clone()))
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

    type GenArcImage<S: IsTensorScalar> = ArcImageR<S, 4>;

    type GenMutImage<S: IsTensorScalar> = MutImageR<S, 4>;

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

impl PercentageArcImage<3, 1, u16, SVec<u16, 2>, 2, 1, 1> for ArcImage2U16 {
    type Pixel<S: IsTensorScalar> = SVec<S, 2>;

    fn pixel_to_grayscale(pixel: &SVec<u16, 2>) -> u16 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU16 {
        ArcImageU16::from_map(&img.image_view(), |rgb: &SVec<u16, 2>| -> u16 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::GrayscaleAlphaU16(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
        Some(DynPercentageArcImageU::GrayscaleAlphaU16(image.clone()))
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

    type GenArcImage<S: IsTensorScalar> = ArcImageR<S, 2>;

    type GenMutImage<S: IsTensorScalar> = MutImageR<S, 2>;

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

impl PercentageArcImage<3, 1, u16, SVec<u16, 3>, 3, 1, 1> for ArcImage3U16 {
    type Pixel<S: IsTensorScalar> = SVec<S, 3>;

    fn pixel_to_grayscale(pixel: &SVec<u16, 3>) -> u16 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU16 {
        ArcImageU16::from_map(&img.image_view(), |rgb: &SVec<u16, 3>| -> u16 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::RgbU16(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
        Some(DynPercentageArcImageU::RgbU16(image.clone()))
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

    type GenArcImage<S: IsTensorScalar> = ArcImageR<S, 3>;

    type GenMutImage<S: IsTensorScalar> = MutImageR<S, 3>;

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

impl PercentageArcImage<3, 1, u16, SVec<u16, 4>, 4, 1, 1> for ArcImage4U16 {
    type Pixel<S: IsTensorScalar> = SVec<S, 4>;

    fn pixel_to_grayscale(pixel: &SVec<u16, 4>) -> u16 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageU16 {
        ArcImageU16::from_map(&img.image_view(), |rgb: &SVec<u16, 4>| -> u16 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::RgbaU16(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
        Some(DynPercentageArcImageU::RgbaU16(image.clone()))
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

    type GenArcImage<S: IsTensorScalar> = ArcImageR<S, 4>;

    type GenMutImage<S: IsTensorScalar> = MutImageR<S, 4>;

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

impl PercentageArcImage<3, 1, f32, SVec<f32, 2>, 2, 1, 1> for ArcImage2F32 {
    type Pixel<S: IsTensorScalar> = SVec<S, 2>;

    fn pixel_to_grayscale(pixel: &SVec<f32, 2>) -> f32 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageF32 {
        ArcImageF32::from_map(&img.image_view(), |rgb: &SVec<f32, 2>| -> f32 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::GrayscaleAlphaF32(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
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

    type GenArcImage<S: IsTensorScalar> = ArcImageR<S, 2>;

    type GenMutImage<S: IsTensorScalar> = MutImageR<S, 2>;

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

impl PercentageArcImage<3, 1, f32, SVec<f32, 3>, 3, 1, 1> for ArcImage3F32 {
    type Pixel<S: IsTensorScalar> = SVec<S, 3>;

    fn pixel_to_grayscale(pixel: &SVec<f32, 3>) -> f32 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageF32 {
        ArcImageF32::from_map(&img.image_view(), |rgb: &SVec<f32, 3>| -> f32 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::RgbF32(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
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

    type GenArcImage<S: IsTensorScalar> = ArcImageR<S, 3>;

    type GenMutImage<S: IsTensorScalar> = MutImageR<S, 3>;

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

impl PercentageArcImage<3, 1, f32, SVec<f32, 4>, 4, 1, 1> for ArcImage4F32 {
    type Pixel<S: IsTensorScalar> = SVec<S, 4>;

    fn pixel_to_grayscale(pixel: &SVec<f32, 4>) -> f32 {
        pixel[0]
    }

    fn to_grayscale(img: &Self) -> ArcImageF32 {
        ArcImageF32::from_map(&img.image_view(), |rgb: &SVec<f32, 4>| -> f32 {
            Self::pixel_to_grayscale(rgb)
        })
    }

    fn into_dyn_image_view(image: &Self) -> DynPercentageArcImage {
        DynPercentageArcImage::RgbaF32(image.clone())
    }

    fn try_into_dyn_image_view_u(image: &Self) -> Option<DynPercentageArcImageU> {
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

    type GenArcImage<S: IsTensorScalar> = ArcImageR<S, 4>;

    type GenMutImage<S: IsTensorScalar> = MutImageR<S, 4>;

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

pub trait PercentageViewImageU<'a> {
    const COLOR_TYPE: png::ColorType;
    const BIT_DEPTH: png::BitDepth;

    fn size(&'a self) -> ImageSize;

    fn u8_slice(&self) -> &[u8];
}

impl<'a> PercentageViewImageU<'a> for ImageViewU8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Grayscale;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> PercentageViewImageU<'a> for ImageView2U8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::GrayscaleAlpha;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> PercentageViewImageU<'a> for ImageView3U8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgb;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> PercentageViewImageU<'a> for ImageView4U8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgba;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> PercentageViewImageU<'a> for ImageViewU16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Grayscale;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> PercentageViewImageU<'a> for ImageView2U16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::GrayscaleAlpha;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> PercentageViewImageU<'a> for ImageView3U16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgb;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> PercentageViewImageU<'a> for ImageView4U16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgba;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}
