use sophus_autodiff::linalg::SVec;

use crate::{
    intensity_image::{
        dyn_intensity_image::DynIntensityMutImageU,
        intensity_pixel::IntensityPixel,
        intensity_scalar::IsIntensityScalar,
    },
    mut_image::MutImageR,
    prelude::{
        DynIntensityMutImage,
        *,
    },
    MutImage,
};

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
///  - f32 images shall be in the range [0.0, 1.0] and 100% intensity corresponds to 1.0. If the f32
///    is outside this range, conversion results may be surprising.
///
/// These are image type which typically used for computer vision and graphics applications.
pub trait IsIntensityMutImage<
    Scalar: IsIntensityScalar + 'static,
    Pixel: IntensityPixel<Scalar> + 'static,
>: core::marker::Sized
{
    /// Pixel type
    type Pixel<OtherScalar: IsIntensityScalar>: IntensityPixel<OtherScalar>;

    /// Casted mutable image type
    type CastedMutImage<OtherScalar: IsIntensityScalar>;

    /// Converts the image to a grayscale image.
    fn to_grayscale(img: Self) -> MutImage<Scalar>;

    /// Converts the image to a grayscale_alpha image.
    fn to_grayscale_alpha(img: Self) -> MutImageR<Scalar, 2>;

    /// Converts the image to a RGB image.
    fn to_rgb(img: Self) -> MutImageR<Scalar, 3>;

    /// Converts the image to a RGBA image.
    fn to_rgba(img: Self) -> MutImageR<Scalar, 4>;

    /// Converts the image to a u8 image.
    fn convert_to<OtherScalar: IsIntensityScalar>(img: Self) -> Self::CastedMutImage<OtherScalar>;
}

impl<Scalar: IsIntensityScalar + 'static> IsIntensityMutImage<Scalar, Scalar> for MutImage<Scalar> {
    type Pixel<S: IsIntensityScalar> = S;

    type CastedMutImage<OtherScalar: IsIntensityScalar> = MutImage<OtherScalar>;

    fn to_grayscale(img: Self) -> MutImage<Scalar> {
        img
    }

    fn to_grayscale_alpha(img: Self) -> MutImageR<Scalar, 2> {
        MutImageR::<Scalar, 2>::from_map(&img.image_view(), |rgb: &Scalar| -> SVec<Scalar, 2> {
            SVec::<Scalar, 2>::new(*rgb, Scalar::MAX)
        })
    }

    fn to_rgb(img: Self) -> MutImageR<Scalar, 3> {
        MutImageR::<Scalar, 3>::from_map(&img.image_view(), |rgb: &Scalar| -> SVec<Scalar, 3> {
            SVec::<Scalar, 3>::new(*rgb, *rgb, *rgb)
        })
    }

    fn to_rgba(img: Self) -> MutImageR<Scalar, 4> {
        MutImageR::<Scalar, 4>::from_map(&img.image_view(), |rgb: &Scalar| -> SVec<Scalar, 4> {
            SVec::<Scalar, 4>::new(*rgb, *rgb, *rgb, Scalar::MAX)
        })
    }

    fn convert_to<OtherScalar: IsIntensityScalar>(img: Self) -> MutImage<OtherScalar> {
        MutImage::<OtherScalar>::from_map(&img.image_view(), |rgb: &Scalar| -> OtherScalar {
            OtherScalar::from(*rgb)
        })
    }
}

impl<Scalar: IsIntensityScalar + 'static> IsIntensityMutImage<Scalar, SVec<Scalar, 2>>
    for MutImageR<Scalar, 2>
{
    type Pixel<S: IsIntensityScalar> = SVec<S, 2>;

    type CastedMutImage<OtherScalar: IsIntensityScalar> = MutImageR<OtherScalar, 2>;

    fn to_grayscale(img: Self) -> MutImage<Scalar> {
        MutImage::<Scalar>::from_map(&img.image_view(), |rgba: &SVec<Scalar, 2>| -> Scalar {
            rgba[0]
        })
    }

    fn to_grayscale_alpha(img: Self) -> MutImageR<Scalar, 2> {
        img
    }

    fn to_rgb(img: Self) -> MutImageR<Scalar, 3> {
        MutImageR::<Scalar, 3>::from_map(
            &img.image_view(),
            |rgba: &SVec<Scalar, 2>| -> SVec<Scalar, 3> {
                SVec::<Scalar, 3>::new(rgba[0], rgba[0], rgba[0])
            },
        )
    }

    fn to_rgba(img: Self) -> MutImageR<Scalar, 4> {
        MutImageR::<Scalar, 4>::from_map(
            &img.image_view(),
            |rgba: &SVec<Scalar, 2>| -> SVec<Scalar, 4> {
                SVec::<Scalar, 4>::new(rgba[0], rgba[0], rgba[0], rgba[1])
            },
        )
    }

    fn convert_to<OtherScalar: IsIntensityScalar>(img: Self) -> MutImageR<OtherScalar, 2> {
        MutImageR::<OtherScalar, 2>::from_map(
            &img.image_view(),
            |rgba: &SVec<Scalar, 2>| -> SVec<OtherScalar, 2> {
                SVec::<OtherScalar, 2>::new(rgba[0].cast_to(), rgba[1].cast_to())
            },
        )
    }
}

impl<Scalar: IsIntensityScalar + 'static> IsIntensityMutImage<Scalar, SVec<Scalar, 3>>
    for MutImageR<Scalar, 3>
{
    type Pixel<S: IsIntensityScalar> = SVec<S, 3>;

    type CastedMutImage<OtherScalar: IsIntensityScalar> = MutImageR<OtherScalar, 3>;

    fn to_grayscale(img: Self) -> MutImage<Scalar> {
        MutImage::<Scalar>::from_map(&img.image_view(), |rgb: &SVec<Scalar, 3>| -> Scalar {
            rgb[0]
        })
    }

    fn to_grayscale_alpha(img: Self) -> MutImageR<Scalar, 2> {
        MutImageR::<Scalar, 2>::from_map(
            &img.image_view(),
            |rgb: &SVec<Scalar, 3>| -> SVec<Scalar, 2> {
                SVec::<Scalar, 2>::new(rgb[0], Scalar::MAX)
            },
        )
    }

    fn to_rgb(img: Self) -> MutImageR<Scalar, 3> {
        img
    }

    fn to_rgba(img: Self) -> MutImageR<Scalar, 4> {
        MutImageR::<Scalar, 4>::from_map(
            &img.image_view(),
            |rgb: &SVec<Scalar, 3>| -> SVec<Scalar, 4> {
                SVec::<Scalar, 4>::new(rgb[0], rgb[1], rgb[2], Scalar::MAX)
            },
        )
    }

    fn convert_to<OtherScalar: IsIntensityScalar>(img: Self) -> MutImageR<OtherScalar, 3> {
        MutImageR::<OtherScalar, 3>::from_map(
            &img.image_view(),
            |rgb: &SVec<Scalar, 3>| -> SVec<OtherScalar, 3> {
                SVec::<OtherScalar, 3>::new(rgb[0].cast_to(), rgb[1].cast_to(), rgb[2].cast_to())
            },
        )
    }
}

impl<Scalar: IsIntensityScalar + 'static> IsIntensityMutImage<Scalar, SVec<Scalar, 4>>
    for MutImageR<Scalar, 4>
{
    type Pixel<S: IsIntensityScalar> = SVec<S, 4>;

    type CastedMutImage<OtherScalar: IsIntensityScalar> = MutImageR<OtherScalar, 4>;

    fn to_grayscale(img: Self) -> MutImage<Scalar> {
        MutImage::<Scalar>::from_map(&img.image_view(), |rgba: &SVec<Scalar, 4>| -> Scalar {
            rgba[0]
        })
    }

    fn to_grayscale_alpha(img: Self) -> MutImageR<Scalar, 2> {
        MutImageR::<Scalar, 2>::from_map(
            &img.image_view(),
            |rgba: &SVec<Scalar, 4>| -> SVec<Scalar, 2> {
                SVec::<Scalar, 2>::new(rgba[0], rgba[3])
            },
        )
    }

    fn to_rgb(img: Self) -> MutImageR<Scalar, 3> {
        MutImageR::<Scalar, 3>::from_map(
            &img.image_view(),
            |rgba: &SVec<Scalar, 4>| -> SVec<Scalar, 3> {
                SVec::<Scalar, 3>::new(rgba[0], rgba[1], rgba[2])
            },
        )
    }

    fn to_rgba(img: Self) -> MutImageR<Scalar, 4> {
        img
    }

    fn convert_to<OtherScalar: IsIntensityScalar>(img: Self) -> MutImageR<OtherScalar, 4> {
        MutImageR::<OtherScalar, 4>::from_map(
            &img.image_view(),
            |rgba: &SVec<Scalar, 4>| -> SVec<OtherScalar, 4> {
                SVec::<OtherScalar, 4>::new(
                    rgba[0].cast_to(),
                    rgba[1].cast_to(),
                    rgba[2].cast_to(),
                    rgba[3].cast_to(),
                )
            },
        )
    }
}

/// Trait for "intensity" images (grayscale, grayscale+alpha, RGB, RGBA).
pub trait HasIntoDynIntensityMutImage<
    Scalar: IsIntensityScalar + 'static,
    Pixel: IntensityPixel<Scalar> + 'static,
>: IsIntensityMutImage<Scalar, Pixel>
{
    /// Returns a dynamic representation of the image.
    fn into_dyn(img: Self) -> DynIntensityMutImage;

    /// Tries to return a dynamic image view of unsigned values.
    ///
    /// If the image is not of unsigned type, it returns None.
    fn try_into_dyn_u(img: Self) -> Option<DynIntensityMutImageU>;
}

impl HasIntoDynIntensityMutImage<u8, u8> for MutImage<u8> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::GrayscaleU8(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityMutImageU> {
        Some(DynIntensityMutImageU::GrayscaleU8(img))
    }
}

impl HasIntoDynIntensityMutImage<u16, u16> for MutImage<u16> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::GrayscaleU16(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityMutImageU> {
        Some(DynIntensityMutImageU::GrayscaleU16(img))
    }
}

impl HasIntoDynIntensityMutImage<f32, f32> for MutImage<f32> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::GrayscaleF32(img)
    }

    fn try_into_dyn_u(_img: Self) -> Option<DynIntensityMutImageU> {
        None
    }
}

impl HasIntoDynIntensityMutImage<u8, SVec<u8, 2>> for MutImageR<u8, 2> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::GrayscaleAlphaU8(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityMutImageU> {
        Some(DynIntensityMutImageU::GrayscaleAlphaU8(img))
    }
}

impl HasIntoDynIntensityMutImage<u16, SVec<u16, 2>> for MutImageR<u16, 2> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::GrayscaleAlphaU16(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityMutImageU> {
        Some(DynIntensityMutImageU::GrayscaleAlphaU16(img))
    }
}

impl HasIntoDynIntensityMutImage<f32, SVec<f32, 2>> for MutImageR<f32, 2> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::GrayscaleAlphaF32(img)
    }

    fn try_into_dyn_u(_img: Self) -> Option<DynIntensityMutImageU> {
        None
    }
}

impl HasIntoDynIntensityMutImage<u8, SVec<u8, 3>> for MutImageR<u8, 3> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::RgbU8(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityMutImageU> {
        Some(DynIntensityMutImageU::RgbU8(img))
    }
}

impl HasIntoDynIntensityMutImage<u16, SVec<u16, 3>> for MutImageR<u16, 3> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::RgbU16(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityMutImageU> {
        Some(DynIntensityMutImageU::RgbU16(img))
    }
}

impl HasIntoDynIntensityMutImage<f32, SVec<f32, 3>> for MutImageR<f32, 3> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::RgbF32(img)
    }

    fn try_into_dyn_u(_img: Self) -> Option<DynIntensityMutImageU> {
        None
    }
}

impl HasIntoDynIntensityMutImage<u8, SVec<u8, 4>> for MutImageR<u8, 4> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::RgbaU8(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityMutImageU> {
        Some(DynIntensityMutImageU::RgbaU8(img))
    }
}

impl HasIntoDynIntensityMutImage<u16, SVec<u16, 4>> for MutImageR<u16, 4> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::RgbaU16(img)
    }

    fn try_into_dyn_u(img: Self) -> Option<DynIntensityMutImageU> {
        Some(DynIntensityMutImageU::RgbaU16(img))
    }
}

impl HasIntoDynIntensityMutImage<f32, SVec<f32, 4>> for MutImageR<f32, 4> {
    fn into_dyn(img: Self) -> DynIntensityMutImage {
        DynIntensityMutImage::RgbaF32(img)
    }

    fn try_into_dyn_u(_img: Self) -> Option<DynIntensityMutImageU> {
        None
    }
}
