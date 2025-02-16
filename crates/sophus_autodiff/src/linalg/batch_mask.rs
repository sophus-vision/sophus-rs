use core::{
    fmt,
    fmt::Debug,
    simd::{
        LaneCount,
        Mask,
        SupportedLaneCount,
    },
};

use crate::prelude::IsBoolMask;

/// A lane-wise boolean mask for batch (SIMD) operations.
///
/// This struct wraps a [`Mask<i64, N>`][core::simd::Mask], storing a boolean value
/// for each of `N` lanes. It implements [`IsBoolMask`] for:
///
/// - Checking whether all or any lanes are `true`.
/// - Counting the number of `true` lanes.
/// - Creating a fully `true` or fully `false` mask.
///
/// # Generic Parameters
/// - `N`: The number of lanes in the SIMD mask. Must implement [`SupportedLaneCount`].
///
/// # Feature
/// This type is only available when the `"simd"` feature is enabled.
#[cfg(feature = "simd")]
pub struct BatchMask<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    // The underlying lane-wise mask storing `true` or `false` for each lane.
    pub(crate) inner: Mask<i64, N>,
}

impl<const N: usize> Debug for BatchMask<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Displays the mask in a user-friendly way.
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
