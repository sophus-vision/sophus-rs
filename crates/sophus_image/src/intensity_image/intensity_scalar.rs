use core::any::TypeId;
use sophus_autodiff::prelude::*;

/// either u8, u16, or f32
pub trait IsIntensityScalar: IsCoreScalar + Sized + Copy {
    /// Maximum value for the scalar type.
    ///
    /// It should be 1.0 for floating point numbers and the maximum value for unsigned integers.
    const MAX: Self;
    /// Maximum value for the scalar type as a f32.
    const MAX_F32: f32;

    /// Creates a scalar from a u8 value.
    fn from_u8(value: u8) -> Self;

    /// Creates a scalar from a u16 value.
    fn from_u16(value: u16) -> Self;

    /// Creates a scalar from a f32 value.
    fn from_f32(value: f32) -> Self;

    /// Creates a scalar from a value of any type that implements `IsIntensityScalar`.
    fn from<T: IsIntensityScalar>(value: T) -> Self {
        if TypeId::of::<T>() == TypeId::of::<u8>() {
            Self::from_u8(value.convert_to_u8())
        } else if TypeId::of::<T>() == TypeId::of::<u16>() {
            Self::from_u16(value.convert_to_u16())
        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
            Self::from_f32(value.convert_to_f32())
        } else {
            panic!("Unsupported type for IntensityScalar");
        }
    }

    /// Casts the scalar to a u8 value.
    fn cast_to_u8(&self) -> u8;
    /// Casts the scalar to a u16 value.
    fn cast_to_u16(&self) -> u16;
    /// Casts the scalar to a f32 value.
    fn cast_to_f32(&self) -> f32;

    /// Casts the scalar to a value of any type that implements `IsIntensityScalar`.
    fn cast_to<Other: IsIntensityScalar>(&self) -> Other {
        if TypeId::of::<Other>() == TypeId::of::<u8>() {
            Other::from_u8(self.cast_to_u8())
        } else if TypeId::of::<Other>() == TypeId::of::<u16>() {
            Other::from_u16(self.cast_to_u16())
        } else if TypeId::of::<Other>() == TypeId::of::<f32>() {
            Other::from_f32(self.cast_to_f32())
        } else {
            panic!("Unsupported type for IntensityScalar");
        }
    }

    /// Converts the scalar to a u8 value.
    fn convert_to_u8(&self) -> u8;
    /// Converts the scalar to a u16 value.
    fn convert_to_u16(&self) -> u16;
    /// Converts the scalar to a f32 value.
    fn convert_to_f32(&self) -> f32;

    /// Converts the scalar to a u8 value.
    fn convert_to<Other: IsIntensityScalar>(&self) -> Other {
        if TypeId::of::<Other>() == TypeId::of::<u8>() {
            Other::from_u8(self.convert_to_u8())
        } else if TypeId::of::<Other>() == TypeId::of::<u16>() {
            Other::from_u16(self.convert_to_u16())
        } else if TypeId::of::<Other>() == TypeId::of::<f32>() {
            Other::from_f32(self.convert_to_f32())
        } else {
            panic!("Unsupported type for IntensityScalar");
        }
    }
}

impl IsIntensityScalar for u8 {
    const MAX: Self = u8::MAX;
    const MAX_F32: f32 = u8::MAX as f32;

    fn from_u8(value: u8) -> Self {
        value
    }

    fn from_u16(value: u16) -> Self {
        (value >> 8) as u8
    }

    fn from_f32(value: f32) -> Self {
        (value * Self::MAX_F32) as u8
    }

    fn convert_to_u8(&self) -> u8 {
        *self
    }

    fn convert_to_u16(&self) -> u16 {
        (*self as u16) << 8
    }

    fn convert_to_f32(&self) -> f32 {
        *self as f32 / Self::MAX_F32
    }

    fn cast_to_u8(&self) -> u8 {
        *self
    }

    fn cast_to_u16(&self) -> u16 {
        *self as u16
    }

    fn cast_to_f32(&self) -> f32 {
        *self as f32
    }
}

impl IsIntensityScalar for u16 {
    const MAX: Self = u16::MAX;
    const MAX_F32: f32 = u16::MAX as f32;

    fn from_u8(value: u8) -> Self {
        (value as u16) << 8
    }

    fn from_u16(value: u16) -> Self {
        value
    }

    fn from_f32(value: f32) -> Self {
        (value * Self::MAX_F32) as u16
    }

    fn convert_to_u8(&self) -> u8 {
        (*self >> 8) as u8
    }

    fn convert_to_u16(&self) -> u16 {
        *self
    }

    fn convert_to_f32(&self) -> f32 {
        *self as f32 / Self::MAX_F32
    }

    fn cast_to_u8(&self) -> u8 {
        *self as u8
    }

    fn cast_to_u16(&self) -> u16 {
        *self
    }

    fn cast_to_f32(&self) -> f32 {
        *self as f32
    }
}

impl IsIntensityScalar for f32 {
    const MAX: Self = 1.0;
    const MAX_F32: f32 = 1.0;

    fn from_u8(value: u8) -> Self {
        value as f32 / u8::MAX_F32
    }

    fn from_u16(value: u16) -> Self {
        value as f32 / u16::MAX_F32
    }

    fn from_f32(value: f32) -> Self {
        value
    }

    fn convert_to_u8(&self) -> u8 {
        (*self * u8::MAX_F32) as u8
    }

    fn convert_to_u16(&self) -> u16 {
        (*self * u16::MAX_F32) as u16
    }

    fn convert_to_f32(&self) -> f32 {
        *self
    }

    fn cast_to_u8(&self) -> u8 {
        *self as u8
    }

    fn cast_to_u16(&self) -> u16 {
        *self as u16
    }

    fn cast_to_f32(&self) -> f32 {
        *self
    }
}
