use crate::prelude::IsBoolMask;
use std::simd::LaneCount;
use std::simd::Mask;
use std::simd::MaskElement;
use std::simd::SupportedLaneCount;

impl<const BATCH: usize, T> IsBoolMask for Mask<T, BATCH>
where
    T: MaskElement,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn all_true() -> Self {
        Mask::from_array([true; BATCH])
    }

    fn all_false() -> Self {
        Mask::from_array([false; BATCH])
    }

    fn all(&self) -> bool {
        Mask::all(*self)
    }

    fn any(&self) -> bool {
        Mask::any(*self)
    }

    fn count(&self) -> usize {
        self.to_array().iter().filter(|x| **x).count()
    }

    fn lanes(&self) -> usize {
        BATCH
    }

    fn and(&self, other: &Self) -> Self {
        *self & *other
    }

    fn or(&self, other: &Self) -> Self {
        *self | *other
    }

    fn not(&self) -> Self {
        !*self
    }

    fn xor(&self, other: &Self) -> Self {
        *self ^ *other
    }

    fn dbg_str(&self) -> String {
        format!("{:?}", self.to_array())
    }
}
