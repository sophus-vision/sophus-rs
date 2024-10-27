use crate::prelude::IsBoolMask;
use core::simd::LaneCount;
use core::simd::Mask;
use core::simd::MaskElement;
use core::simd::SupportedLaneCount;

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
}
