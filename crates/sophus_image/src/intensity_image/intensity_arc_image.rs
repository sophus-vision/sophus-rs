use core::any::TypeId;

use crate::arc_image::ArcImageR;
use crate::intensity_image::dyn_intensity_image::DynIntensityArcImage;
use crate::intensity_image::dyn_intensity_image::DynIntensityArcImageU;
use crate::intensity_image::intensity_pixel::IntensityPixel;
use crate::intensity_image::intensity_scalar::IsIntensityScalar;
use crate::prelude::*;
use crate::ArcImage;
use sophus_autodiff::linalg::SVec;

/// Trait for "intensity" images (grayscale, grayscale+alpha, RGB, RGBA).
///
/// Hence it s a trait for grayscale (1-channel), grayscale+alpha (2-channel), RGB (3-channel), and
/// RGBA images (4-channel).
///
/// This trait provides methods for converting between di   erent image type. As of now, three
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
pub trait IsIntensityArcImage<
    Scalar: IsIntensityScalar + 'static,
    Pixel: IntensityPixel<Scalar> + 'static,
>: core::marker::Sized
{
    /// Pixel type
    type Pixel<OtherScalar: IsIntensityScalar>: IntensityPixel<OtherScalar>;

    /// Casted mutable image type
    type CastedArcImage<OtherScalar: IsIntensityScalar>;

    /// Image type with single channel
    type ArcImage<OtherScalar: IsIntensityScalar>;

    /// Image type with multiple channels
    type ArcImageR<OtherScalar: IsIntensityScalar, const N: usize>;

    /// Converts the image to a grayscale image.
    fn to_grayscale<OtherScalar: IsIntensityScalar>(self) -> Self::ArcImage<OtherScalar>;

    /// Converts the image to a grayscale_alpha image.
    fn to_grayscale_alpha<OtherScalar: IsIntensityScalar>(self) -> Self::ArcImageR<OtherScalar, 2>;

    /// Converts the image to a RGB image.
    fn to_rgb<OtherScalar: IsIntensityScalar>(self) -> Self::ArcImageR<OtherScalar, 3>;

    /// Converts the image to a RGBA image.
    fn to_rgba<OtherScalar: IsIntensityScalar>(self) -> Self::ArcImageR<OtherScalar, 4>;

    /// Converts the image to a u8 image.
    fn convert_to<OtherScalar: IsIntensityScalar>(self) -> Self::CastedArcImage<OtherScalar>;
}

impl<Scalar: IsIntensityScalar + 'static> IsIntensityArcImage<Scalar, Scalar> for ArcImage<Scalar> {
    type Pixel<S: IsIntensityScalar> = S;

    type CastedArcImage<OtherScalar: IsIntensityScalar> = ArcImage<OtherScalar>;

    type ArcImage<OtherScalar: IsIntensityScalar> = ArcImage<OtherScalar>;

    type ArcImageR<OtherScalar: IsIntensityScalar, const N: usize> = ArcImageR<OtherScalar, N>;

    fn to_grayscale<OtherScalar: IsIntensityScalar>(self) -> ArcImage<OtherScalar> {
        self.convert_to::<OtherScalar>()
    }

    fn to_grayscale_alpha<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 2> {
        ArcImageR::<OtherScalar, 2>::from_map(
            &self.image_view(),
            |gray: &Scalar| -> SVec<OtherScalar, 2> { gray.to_grayscale_alpha().convert_to() },
        )
    }

    fn to_rgb<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 3> {
        ArcImageR::<OtherScalar, 3>::from_map(
            &self.image_view(),
            |gray: &Scalar| -> SVec<OtherScalar, 3> { gray.to_rgb().convert_to() },
        )
    }

    fn to_rgba<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 4> {
        ArcImageR::<OtherScalar, 4>::from_map(
            &self.image_view(),
            |gray: &Scalar| -> SVec<OtherScalar, 4> { gray.to_rgba().convert_to() },
        )
    }

    fn convert_to<OtherScalar: IsIntensityScalar>(self) -> ArcImage<OtherScalar> {
        if TypeId::of::<Scalar>() == TypeId::of::<OtherScalar>() {
            // If the scalar type is the same, just return the image.

            // Safety: This is safe because the types are actually the same.
            return unsafe {
                core::mem::transmute::<ArcImage<Scalar>, ArcImage<OtherScalar>>(self)
            };
        }
        ArcImage::<OtherScalar>::from_map(&self.image_view(), |rgb: &Scalar| -> OtherScalar {
            OtherScalar::from(*rgb)
        })
    }
}

impl<Scalar: IsIntensityScalar + 'static> IsIntensityArcImage<Scalar, SVec<Scalar, 2>>
    for ArcImageR<Scalar, 2>
{
    type Pixel<S: IsIntensityScalar> = SVec<S, 2>;

    type CastedArcImage<OtherScalar: IsIntensityScalar> = ArcImageR<OtherScalar, 2>;

    type ArcImage<OtherScalar: IsIntensityScalar> = ArcImage<OtherScalar>;

    type ArcImageR<OtherScalar: IsIntensityScalar, const N: usize> = ArcImageR<OtherScalar, N>;

    fn to_grayscale<OtherScalar: IsIntensityScalar>(self) -> ArcImage<OtherScalar> {
        ArcImage::<OtherScalar>::from_map(
            &self.image_view(),
            |gray_alpha: &SVec<Scalar, 2>| -> OtherScalar {
                gray_alpha.to_grayscale().convert_to()
            },
        )
    }

    fn to_grayscale_alpha<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 2> {
        self.convert_to::<OtherScalar>()
    }

    fn to_rgb<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 3> {
        ArcImageR::<OtherScalar, 3>::from_map(
            &self.image_view(),
            |gray_alpha: &SVec<Scalar, 2>| -> SVec<OtherScalar, 3> {
                gray_alpha.to_rgb().convert_to()
            },
        )
    }

    fn to_rgba<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 4> {
        ArcImageR::<OtherScalar, 4>::from_map(
            &self.image_view(),
            |gray_alpha: &SVec<Scalar, 2>| -> SVec<OtherScalar, 4> {
                gray_alpha.to_rgba().convert_to()
            },
        )
    }

    fn convert_to<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 2> {
        if TypeId::of::<Scalar>() == TypeId::of::<OtherScalar>() {
            // If the scalar type is the same, just return the image.

            // Safety: This is safe because the types are actually the same.
            return unsafe {
                core::mem::transmute::<ArcImageR<Scalar, 2>, ArcImageR<OtherScalar, 2>>(self)
            };
        }
        ArcImageR::<OtherScalar, 2>::from_map(
            &self.image_view(),
            |rgba: &SVec<Scalar, 2>| -> SVec<OtherScalar, 2> {
                SVec::<OtherScalar, 2>::new(rgba[0].convert_to(), rgba[1].convert_to())
            },
        )
    }
}

impl<Scalar: IsIntensityScalar + 'static> IsIntensityArcImage<Scalar, SVec<Scalar, 3>>
    for ArcImageR<Scalar, 3>
{
    type Pixel<S: IsIntensityScalar> = SVec<S, 3>;

    type CastedArcImage<OtherScalar: IsIntensityScalar> = ArcImageR<OtherScalar, 3>;

    type ArcImage<OtherScalar: IsIntensityScalar> = ArcImage<OtherScalar>;

    type ArcImageR<OtherScalar: IsIntensityScalar, const N: usize> = ArcImageR<OtherScalar, N>;

    fn to_grayscale<OtherScalar: IsIntensityScalar>(self) -> ArcImage<OtherScalar> {
        ArcImage::<OtherScalar>::from_map(
            &self.image_view(),
            |rgb: &SVec<Scalar, 3>| -> OtherScalar { rgb.to_grayscale().convert_to() },
        )
    }

    fn to_grayscale_alpha<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 2> {
        ArcImageR::<OtherScalar, 2>::from_map(
            &self.image_view(),
            |rgb: &SVec<Scalar, 3>| -> SVec<OtherScalar, 2> {
                rgb.to_grayscale_alpha().convert_to()
            },
        )
    }

    fn to_rgb<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 3> {
        self.convert_to::<OtherScalar>()
    }

    fn to_rgba<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 4> {
        ArcImageR::<OtherScalar, 4>::from_map(
            &self.image_view(),
            |rgb: &SVec<Scalar, 3>| -> SVec<OtherScalar, 4> { rgb.to_rgba().convert_to() },
        )
    }

    fn convert_to<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 3> {
        if TypeId::of::<Scalar>() == TypeId::of::<OtherScalar>() {
            // If the scalar type is the same, just return the image.

            // Safety: This is safe because the types are actually the same.
            return unsafe {
                core::mem::transmute::<ArcImageR<Scalar, 3>, ArcImageR<OtherScalar, 3>>(self)
            };
        }
        ArcImageR::<OtherScalar, 3>::from_map(
            &self.image_view(),
            |rgb: &SVec<Scalar, 3>| -> SVec<OtherScalar, 3> {
                SVec::<OtherScalar, 3>::new(
                    rgb[0].convert_to(),
                    rgb[1].convert_to(),
                    rgb[2].convert_to(),
                )
            },
        )
    }
}

impl<Scalar: IsIntensityScalar + 'static> IsIntensityArcImage<Scalar, SVec<Scalar, 4>>
    for ArcImageR<Scalar, 4>
{
    type Pixel<OtherScalar: IsIntensityScalar> = SVec<OtherScalar, 4>;

    type CastedArcImage<OtherScalar: IsIntensityScalar> = ArcImageR<OtherScalar, 4>;

    type ArcImage<OtherScalar: IsIntensityScalar> = ArcImage<OtherScalar>;

    type ArcImageR<OtherScalar: IsIntensityScalar, const N: usize> = ArcImageR<OtherScalar, N>;

    fn to_grayscale<OtherScalar: IsIntensityScalar>(self) -> ArcImage<OtherScalar> {
        ArcImage::<OtherScalar>::from_map(
            &self.image_view(),
            |rgba: &SVec<Scalar, 4>| -> OtherScalar { rgba.to_grayscale().convert_to() },
        )
    }

    fn to_grayscale_alpha<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 2> {
        ArcImageR::<OtherScalar, 2>::from_map(
            &self.image_view(),
            |rgba: &SVec<Scalar, 4>| -> SVec<OtherScalar, 2> {
                rgba.to_grayscale_alpha().convert_to()
            },
        )
    }

    fn to_rgb<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 3> {
        ArcImageR::<OtherScalar, 3>::from_map(
            &self.image_view(),
            |rgba: &SVec<Scalar, 4>| -> SVec<OtherScalar, 3> { rgba.to_rgb().convert_to() },
        )
    }

    fn to_rgba<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 4> {
        self.convert_to::<OtherScalar>()
    }

    fn convert_to<OtherScalar: IsIntensityScalar>(self) -> ArcImageR<OtherScalar, 4> {
        if TypeId::of::<Scalar>() == TypeId::of::<OtherScalar>() {
            // If the scalar type is the same, just return the image.

            // Safety: This is safe because the types are actually the same.
            return unsafe {
                core::mem::transmute::<ArcImageR<Scalar, 4>, ArcImageR<OtherScalar, 4>>(self)
            };
        }
        ArcImageR::<OtherScalar, 4>::from_map(
            &self.image_view(),
            |rgba: &SVec<Scalar, 4>| -> SVec<OtherScalar, 4> {
                SVec::<OtherScalar, 4>::new(
                    rgba[0].convert_to(),
                    rgba[1].convert_to(),
                    rgba[2].convert_to(),
                    rgba[3].convert_to(),
                )
            },
        )
    }
}

/// Trait for "intensity" images (grayscale, grayscale+alpha, RGB, RGBA).
pub trait HasIntoDynIntensityArcImage<
    Scalar: IsIntensityScalar + 'static,
    Pixel: IntensityPixel<Scalar> + 'static,
>: IsIntensityArcImage<Scalar, Pixel>
{
    /// Returns a dynamic representation of the image.
    fn into_dyn(img: Self) -> DynIntensityArcImage;

    /// Tries to return a dynamic image view of unsigned values.
    ///
    /// If the image is not of unsigned type, it returns None.
    fn try_into_dyn_u(img: Self) -> Option<DynIntensityArcImageU>;
}

impl HasIntoDynIntensityArcImage<u8, u8> for ArcImage<u8> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleU8(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::GrayscaleU8(img))
    }
}

impl HasIntoDynIntensityArcImage<u16, u16> for ArcImage<u16> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleU16(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::GrayscaleU16(img))
    }
}

impl HasIntoDynIntensityArcImage<f32, f32> for ArcImage<f32> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleF32(img)
    }

    fn try_into_dyn_u(_img: Self) -> Option<DynIntensityArcImageU> {
        None
    }
}

impl HasIntoDynIntensityArcImage<u8, SVec<u8, 2>> for ArcImageR<u8, 2> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleAlphaU8(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::GrayscaleAlphaU8(img))
    }
}

impl HasIntoDynIntensityArcImage<u16, SVec<u16, 2>> for ArcImageR<u16, 2> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleAlphaU16(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::GrayscaleAlphaU16(img))
    }
}

impl HasIntoDynIntensityArcImage<f32, SVec<f32, 2>> for ArcImageR<f32, 2> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::GrayscaleAlphaF32(img)
    }

    fn try_into_dyn_u(_img: Self) -> Option<DynIntensityArcImageU> {
        None
    }
}

impl HasIntoDynIntensityArcImage<u8, SVec<u8, 3>> for ArcImageR<u8, 3> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbU8(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::RgbU8(img))
    }
}

impl HasIntoDynIntensityArcImage<u16, SVec<u16, 3>> for ArcImageR<u16, 3> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbU16(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::RgbU16(img))
    }
}

impl HasIntoDynIntensityArcImage<f32, SVec<f32, 3>> for ArcImageR<f32, 3> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbF32(img)
    }

    fn try_into_dyn_u(_img: Self) -> Option<DynIntensityArcImageU> {
        None
    }
}

impl HasIntoDynIntensityArcImage<u8, SVec<u8, 4>> for ArcImageR<u8, 4> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbaU8(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::RgbaU8(img))
    }
}

impl HasIntoDynIntensityArcImage<u16, SVec<u16, 4>> for ArcImageR<u16, 4> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbaU16(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityArcImageU> {
        Some(DynIntensityArcImageU::RgbaU16(img))
    }
}

impl HasIntoDynIntensityArcImage<f32, SVec<f32, 4>> for ArcImageR<f32, 4> {
    fn into_dyn(img: Self) -> DynIntensityArcImage {
        DynIntensityArcImage::RgbaF32(img)
    }

    fn try_into_dyn_u(_img: Self) -> Option<DynIntensityArcImageU> {
        None
    }
}
