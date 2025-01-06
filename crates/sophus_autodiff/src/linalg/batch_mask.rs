use crate::prelude::IsBoolMask;
use core::fmt;
use core::fmt::Debug;
use core::simd::LaneCount;
use core::simd::Mask;
use core::simd::SupportedLaneCount;

/// Boolean mask - generalization of boolean comparison to SIMDs
pub struct BatchMask<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub(crate) inner: Mask<i64, N>,
}

impl<const N: usize> Debug for BatchMask<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BatchMask({:?})", self.inner)
    }
}

impl<const BATCH: usize> IsBoolMask for BatchMask<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn all_true() -> Self {
        BatchMask {
            inner: Mask::from_array([true; BATCH]),
        }
    }

    fn all_false() -> Self {
        BatchMask {
            inner: Mask::from_array([false; BATCH]),
        }
    }

    fn all(&self) -> bool {
        Mask::all(self.inner)
    }

    fn any(&self) -> bool {
        Mask::any(self.inner)
    }

    fn count(&self) -> usize {
        self.inner.to_array().iter().filter(|x| **x).count()
    }

    fn lanes(&self) -> usize {
        BATCH
    }
}
