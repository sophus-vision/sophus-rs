use std::ops::{Index, Mul, IndexMut};

use num_traits::NumCast;
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
    num_traits::Num
    + std::fmt::Debug
    + std::default::Default
    + std::clone::Clone
    + std::marker::Copy
    + std::cmp::PartialEq
    + std::ops::AddAssign
    + num_traits::Zero
    + NumCast
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

pub type P<const R: usize, Scalar> = nalgebra::SVector<Scalar, R>;

pub trait PixelTrait<const CHANNELS: usize, Scalar: ScalarTrait+'static>:
    std::marker::Copy
    + num_traits::Zero
    + PartialEq
    + Index<usize, Output = Scalar>
    + IndexMut<usize, Output = Scalar>
    + num_traits::Zero
{
    fn cast_from_raw(raw: *const RawDataChunk) -> *const Self {
        raw as *const Self
    }

    fn scalar_cast(raw: *const RawDataChunk) -> *const Scalar {
        raw as *const Scalar
    }

    fn mut_cast(raw: *mut RawDataChunk) -> *mut Self {
        raw as *mut Self
    }

    fn mut_scalar_cast(raw: *mut RawDataChunk) -> *mut Scalar {
        raw as *mut Scalar
    }

    fn u8_cast(raw: *const Self) -> *const u8 {
        raw as *const u8
    }

    fn u8_mut_cast(raw: *mut Self) -> *mut u8 {
        raw as *mut u8
    }

    fn scale(&self, factor: f32) ->  Self{
        let mut result = *self;
        for i in 0..CHANNELS {
            let v:f32 = NumCast::from(self[i]).unwrap();
            result[i] = NumCast::from(v * factor).unwrap();
        }
        result
    }
}

impl<const NUM: usize, S: ScalarTrait + 'static> PixelTrait<NUM, S> for P<NUM, S> {}

pub type P1U8 = P<1, u8>;
pub type P1U16 = P<1, u16>;
pub type P1F32 = P<1, f32>;
pub type P3U8 = P<3, u8>;
pub type P3U16 = P<3, u16>;
pub type P3F32 = P<3, f32>;
pub type P4U8 = P<4, u8>;
pub type P4U16 = P<4, u16>;
pub type P4F32 = P<4, f32>;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PixelFormat {
    pub number_category: NumberCategory, // unsigned otherwise
    pub num_scalars: usize,
    pub num_bytes_per_scalar: usize,
}

impl PixelFormat {
    pub fn new<const N: usize, S: ScalarTrait+'static>() -> Self {
        PixelFormat {
            number_category: S::NUMBER_CATEGORY,
            num_scalars: N,
            num_bytes_per_scalar: std::mem::size_of::<S>(),
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
            PixelTag::PU8 => PixelFormat::new::<1, u8>(),
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

pub trait IntensityPixelTrait<const N: usize, S: ScalarTrait+'static>: PixelTrait<N, S> {}

impl IntensityPixelTrait<1, u8> for P<1, u8> {}
impl IntensityPixelTrait<1, u16> for P<1, u16> {}
impl IntensityPixelTrait<1, f32> for P<1, f32> {}
impl IntensityPixelTrait<3, u8> for P<3, u8> {}
impl IntensityPixelTrait<3, u16> for P<3, u16> {}
impl IntensityPixelTrait<3, f32> for P<3, f32> {}
impl IntensityPixelTrait<4, u8> for P<4, u8> {}
impl IntensityPixelTrait<4, u16> for P<4, u16> {}
impl IntensityPixelTrait<4, f32> for P<4, f32> {}
