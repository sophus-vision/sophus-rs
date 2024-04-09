use std::simd::LaneCount;
use std::simd::Mask;
use std::simd::MaskElement;
use std::simd::SupportedLaneCount;

pub trait BoolMask {
    fn all_true() -> Self;
    fn all_false() -> Self;
    fn all(&self) -> bool;
    fn any(&self) -> bool;
    fn count(&self) -> usize;
    fn lanes(&self) -> usize;
}

impl BoolMask for bool {
    fn all_true() -> bool {
        true
    }
    fn all_false() -> bool {
        false
    }

    fn all(&self) -> bool {
        *self
    }

    fn any(&self) -> bool {
        *self
    }

    fn count(&self) -> usize {
        match *self {
            true => 1,
            false => 0,
        }
    }

    fn lanes(&self) -> usize {
        1
    }
}

impl<const BATCH: usize, T> BoolMask for Mask<T, BATCH>
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
