use std::simd::LaneCount;
use std::simd::Mask;
use std::simd::MaskElement;
use std::simd::SupportedLaneCount;

/// Boolean mask - generalization of boolean comparison to SIMDs
pub trait IsBoolMask {
    /// Mask with all lanes set to true
    fn all_true() -> Self;

    /// Mask with all lanes set to false
    fn all_false() -> Self;

    /// Returns true if all lanes are true
    fn all(&self) -> bool;

    /// Returns true if any lane is true
    fn any(&self) -> bool;

    /// Returns the number of lanes that are true
    fn count(&self) -> usize;

    /// Returns the number of lanes
    fn lanes(&self) -> usize;
}

impl IsBoolMask for bool {
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
