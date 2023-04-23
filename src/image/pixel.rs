#[cfg(not(target_arch = "wasm32"))]
use pyo3::pyclass;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NumberCategory {
    Real,
    Unsigned,
}

#[derive(Debug, Copy, Clone)]
#[repr(packed(64))]
pub struct RawDataChunk([u8; 64]);

impl RawDataChunk {
    pub fn u8_ptr(&self) -> *const u8 {
        &self.0[0]
    }

    pub fn mut_u8_ptr(&mut self) -> *mut u8 {
        &mut self.0[0]
    }
}

impl Default for RawDataChunk {
    fn default() -> Self {
        RawDataChunk([0; 64])
    }
}

pub trait ScalarTrait:
    num_traits::Num + std::fmt::Debug + std::default::Default + std::clone::Clone + std::marker::Copy
{
    const NUMBER_CATEGORY: NumberCategory;
}

impl ScalarTrait for u8 {
    const NUMBER_CATEGORY: NumberCategory = NumberCategory::Unsigned;
}

impl ScalarTrait for u16 {
    const NUMBER_CATEGORY: NumberCategory = NumberCategory::Unsigned;
}

impl ScalarTrait for f32 {
    const NUMBER_CATEGORY: NumberCategory = NumberCategory::Real;
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Pixel<const N: usize, S: ScalarTrait>(pub [S; N]);

impl<const N: usize, S: ScalarTrait> std::default::Default for Pixel<N, S> {
    fn default() -> Self {
        Self([S::default(); N])
    }
}

pub trait PixelTrait: std::marker::Copy + num_traits::Zero + PartialEq {
    type Scalar: ScalarTrait;
    const NUM_CHANNELS: usize;
    const NUMBER_CATEGORY: NumberCategory;

    fn cast(raw: *const RawDataChunk) -> *const Self {
        raw as *const Self
    }

    fn scalar_cast(raw: *const RawDataChunk) -> *const Self::Scalar {
        raw as *const Self::Scalar
    }

    fn mut_cast(raw: *mut RawDataChunk) -> *mut Self {
        raw as *mut Self
    }

    fn mut_scalar_cast(raw: *mut RawDataChunk) -> *mut Self::Scalar {
        raw as *mut Self::Scalar
    }

    fn u8_cast(raw: *const Self) -> *const u8 {
        raw as *const u8
    }

    fn u8_mut_cast(raw: *mut Self) -> *mut u8 {
        raw as *mut u8
    }
}

impl<const N: usize, S: ScalarTrait> core::ops::Add for Pixel<N, S> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut output = Self::default();
        for t in output.0.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *t.0 = *t.1 .0 + *t.1 .1;
        }
        output
    }
}

impl<const N: usize, S: ScalarTrait> num_traits::Zero for Pixel<N, S> {
    fn zero() -> Self {
        Pixel([S::zero(); N])
    }

    fn is_zero(&self) -> bool {
        let o = Self::zero();
        o == *self
    }
}

impl PixelTrait for u8 {
    type Scalar = u8;
    const NUM_CHANNELS: usize = 1;
    const NUMBER_CATEGORY: NumberCategory = NumberCategory::Unsigned;
}

impl PixelTrait for u16 {
    type Scalar = u16;
    const NUM_CHANNELS: usize = 1;
    const NUMBER_CATEGORY: NumberCategory = NumberCategory::Unsigned;
}

impl PixelTrait for f32 {
    type Scalar = f32;
    const NUM_CHANNELS: usize = 1;
    const NUMBER_CATEGORY: NumberCategory = NumberCategory::Real;
}

impl<const N: usize, S: ScalarTrait> PixelTrait for Pixel<N, S> {
    type Scalar = S;

    const NUM_CHANNELS: usize = N;

    const NUMBER_CATEGORY: NumberCategory = S::NUMBER_CATEGORY;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PixelFormat {
    pub number_category: NumberCategory, // unsigned otherwise
    pub num_scalars: usize,
    pub num_bytes_per_scalar: usize,
}

impl PixelFormat {
    pub fn new<T: PixelTrait>() -> Self {
        PixelFormat {
            number_category: T::NUMBER_CATEGORY,
            num_scalars: T::NUM_CHANNELS,
            num_bytes_per_scalar: std::mem::size_of::<T::Scalar>(),
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), pyclass)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PixelTag {
    PU8,
    PU16,
    PF32,
    P3U8,
    P3U16,
    P3F32,
    P4U8,
    P4U16,
    P4F32,
}

impl PixelFormat {
    pub fn num_bytes(&self) -> usize {
        self.num_scalars * self.num_bytes_per_scalar
    }

    pub fn from(tag: PixelTag) -> Self {
        match tag {
            PixelTag::PU8 => PixelFormat::new::<u8>(),
            PixelTag::PU16 => todo!(),
            PixelTag::PF32 => todo!(),
            PixelTag::P3U8 => todo!(),
            PixelTag::P3U16 => todo!(),
            PixelTag::P3F32 => todo!(),
            PixelTag::P4U8 => todo!(),
            PixelTag::P4U16 => todo!(),
            PixelTag::P4F32 => todo!(),
        }
    }
}

pub trait IntensityPixelTrait: PixelTrait {}

impl IntensityPixelTrait for u8 {}
impl IntensityPixelTrait for u16 {}
impl IntensityPixelTrait for f32 {}
impl IntensityPixelTrait for Pixel<3, u8> {}
impl IntensityPixelTrait for Pixel<3, u16> {}
impl IntensityPixelTrait for Pixel<3, f32> {}
impl IntensityPixelTrait for Pixel<4, u8> {}
impl IntensityPixelTrait for Pixel<4, u16> {}
impl IntensityPixelTrait for Pixel<4, f32> {}
