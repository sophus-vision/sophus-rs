use sophus_core::linalg::SVec;

use crate::intensity_image::intensity_scalar::IsIntensityScalar;

/// A pixel that can be converted to grayscale, grayscale alpha, rgb, and rgba.
pub trait IntensityPixel<Scalar: IsIntensityScalar>: Clone + Sized {
    /// The pixel type for a different scalar type.
    type OtherPixel<OtherScalar: IsIntensityScalar>: IntensityPixel<OtherScalar>;

    /// Converts the pixel to a pixel of a different scalar type.
    fn convert_to<OtherScalar: IsIntensityScalar>(&self) -> Self::OtherPixel<OtherScalar>;

    /// Converts the pixel to a grayscale pixel.
    fn to_grayscale(&self) -> Scalar;

    /// Converts the pixel to a grayscale alpha pixel.
    fn to_grayscale_alpha(&self) -> SVec<Scalar, 2>;

    /// Converts the pixel to a rgb pixel.
    fn to_rgb(&self) -> SVec<Scalar, 3>;

    /// Converts the pixel to a rgba pixel.
    fn to_rgba(&self) -> SVec<Scalar, 4>;
}

impl<Scalar: IsIntensityScalar> IntensityPixel<Scalar> for Scalar {
    type OtherPixel<OtherScalar: IsIntensityScalar> = OtherScalar;

    fn convert_to<OtherScalar: IsIntensityScalar>(&self) -> Self::OtherPixel<OtherScalar> {
        OtherScalar::from(*self)
    }

    fn to_grayscale(&self) -> Scalar {
        *self
    }

    fn to_grayscale_alpha(&self) -> SVec<Scalar, 2> {
        SVec::<Scalar, 2>::new(*self, Scalar::MAX)
    }

    fn to_rgb(&self) -> SVec<Scalar, 3> {
        SVec::<Scalar, 3>::new(*self, *self, *self)
    }

    fn to_rgba(&self) -> SVec<Scalar, 4> {
        SVec::<Scalar, 4>::new(*self, *self, *self, Scalar::MAX)
    }
}

impl<Scalar: IsIntensityScalar> IntensityPixel<Scalar> for SVec<Scalar, 2> {
    type OtherPixel<OtherScalar: IsIntensityScalar> = SVec<OtherScalar, 2>;

    fn convert_to<OtherScalar: IsIntensityScalar>(&self) -> Self::OtherPixel<OtherScalar> {
        SVec::<OtherScalar, 2>::new(self[0].cast_to(), self[1].cast_to())
    }

    fn to_grayscale(&self) -> Scalar {
        self[0]
    }

    fn to_grayscale_alpha(&self) -> SVec<Scalar, 2> {
        *self
    }

    fn to_rgb(&self) -> SVec<Scalar, 3> {
        SVec::<Scalar, 3>::new(self[0], self[0], self[0])
    }

    fn to_rgba(&self) -> SVec<Scalar, 4> {
        SVec::<Scalar, 4>::new(self[0], self[0], self[0], self[1])
    }
}

impl<Scalar: IsIntensityScalar> IntensityPixel<Scalar> for SVec<Scalar, 3> {
    type OtherPixel<OtherScalar: IsIntensityScalar> = SVec<OtherScalar, 3>;

    fn convert_to<OtherScalar: IsIntensityScalar>(&self) -> Self::OtherPixel<OtherScalar> {
        SVec::<OtherScalar, 3>::new(
            self[0].convert_to(),
            self[1].convert_to(),
            self[2].convert_to(),
        )
    }

    fn to_grayscale(&self) -> Scalar {
        ((self[0].cast_to_f32() + self[1].cast_to_f32() + self[2].cast_to_f32()) / 3.0).cast_to()
    }

    fn to_grayscale_alpha(&self) -> SVec<Scalar, 2> {
        SVec::<Scalar, 2>::new(
            ((self[0].cast_to_f32() + self[1].cast_to_f32() + self[2].cast_to_f32()) / 3.0)
                .cast_to(),
            Scalar::MAX,
        )
    }

    fn to_rgb(&self) -> SVec<Scalar, 3> {
        *self
    }

    fn to_rgba(&self) -> SVec<Scalar, 4> {
        SVec::<Scalar, 4>::new(self[0], self[1], self[2], Scalar::MAX)
    }
}

impl<Scalar: IsIntensityScalar> IntensityPixel<Scalar> for SVec<Scalar, 4> {
    type OtherPixel<OtherScalar: IsIntensityScalar> = SVec<OtherScalar, 4>;

    fn convert_to<OtherScalar: IsIntensityScalar>(&self) -> Self::OtherPixel<OtherScalar> {
        SVec::<OtherScalar, 4>::new(
            self[0].convert_to(),
            self[1].convert_to(),
            self[2].convert_to(),
            self[3].convert_to(),
        )
    }

    fn to_grayscale(&self) -> Scalar {
        ((self[0].cast_to_f32() + self[1].cast_to_f32() + self[2].cast_to_f32()) / 3.0).cast_to()
    }

    fn to_grayscale_alpha(&self) -> SVec<Scalar, 2> {
        SVec::<Scalar, 2>::new(
            ((self[0].cast_to_f32() + self[1].cast_to_f32() + self[2].cast_to_f32()) / 3.0)
                .cast_to(),
            self[3],
        )
    }

    fn to_rgb(&self) -> SVec<Scalar, 3> {
        SVec::<Scalar, 3>::new(self[0], self[1], self[2])
    }

    fn to_rgba(&self) -> SVec<Scalar, 4> {
        *self
    }
}
