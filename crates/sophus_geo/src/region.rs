/// Box region - axis aligned bounding box
mod box_region;
/// Interval
mod interval;

use sophus_autodiff::points::IsPoint;

pub use crate::region::{
    box_region::{
        BoxRegion,
        NonEmptyBoxRegion,
    },
    interval::{
        Interval,
        NonEmptyInterval,
    },
};

/// Base trait for regions.
///
/// A region is either an interval [a, b] in ℝ, or its generalization to ℝⁿ.
/// Concrete implementations include [Interval], [NonEmptyInterval], [BoxRegion] and
/// [NonEmptyBoxRegion].
///
/// Note that the region bounds are considered being part of the region and hence implementations
/// of [IsRegionBase] are closed sets from a mathematical point of view.
pub trait IsRegionBase<const D: usize, P: IsPoint<D>>: core::marker::Sized {
    /// The general region type.
    type Region: IsRegion<D, P>;
    /// The non-empty flavor of the region type.
    type NonEmptyRegion: IsNonEmptyRegion<D, P>;

    /// Create unbounded region.
    fn unbounded() -> Self;

    /// Create region from upper and lower bounds
    ///
    /// Creates interval
    ///
    ///  * [a, b]  if a <= b
    ///  * [b, a]  if a > b
    fn from_bounds(bound_a: P, bound_b: P) -> Self;

    /// Create region from single point.
    ///
    /// Note that this region is considered degenerated but not empty.
    fn from_point(point: P) -> Self {
        Self::from_bounds(point, point)
    }

    /// Is region degenerated?
    ///
    /// A region is degenerated if it is not empty and min == max.
    fn is_degenerated(&self) -> bool;

    /// Is region unbounded?
    ///
    /// A region is unbound if it includes all points of the . For example,
    /// an [Interval] is unbound if it includes all point x with
    /// [f64::NEG_INFINITY] <= x <=  [f64::INFINITY].
    fn is_unbounded(&self) -> bool;

    /// Extend region to include the point.
    fn extend(&mut self, point: P);

    /// Clamp point to the region.
    fn clamp_point(&self, p: P) -> P;

    /// Intersect with other region.
    fn intersect(self, other: Self) -> Self::Region;

    /// Check if the region contains the point.
    fn contains(&self, point: P) -> bool;

    /// Return range of the region.
    ///
    /// For an interval [a, b], the range is b - a.
    fn range(&self) -> P;

    /// Convert self to the [Self::Region] type.
    fn to_region(self) -> Self::Region;

    /// Convert self to the [Self::NonEmptyRegion] type.
    ///
    /// Return [None] if the region is empty.
    fn to_non_empty_region(self) -> Option<Self::NonEmptyRegion>;
}

/// Trait for a possibly empty region.
pub trait IsRegion<const D: usize, P: IsPoint<D>>: IsRegionBase<D, P> {
    /// Create empty region.
    fn empty() -> Self;

    /// Return lower bound of the region or [None] if the region is empty.
    fn try_lower(&self) -> Option<P>;

    /// Return upper bound of the region or [None] if the region is empty.
    fn try_upper(&self) -> Option<P>;

    /// Center of the region, or [None] if the region is empty.
    fn try_center(&self) -> Option<P>;

    /// Is region empty?
    fn is_empty(&self) -> bool;

    /// Is region proper?
    ///
    /// A region is proper if it is not empty and not degenerated.
    fn is_proper(&self) -> bool {
        !self.is_empty() && !self.is_degenerated()
    }
}

/// Trait for a non-empty region.
pub trait IsNonEmptyRegion<const D: usize, P: IsPoint<D>>: IsRegionBase<D, P> {
    /// Return center of the region.
    ///
    /// For an interval [a, b], the center is (a + b) / 2.
    fn center(&self) -> P;
    /// Return lower bound.
    fn lower(&self) -> P;
    /// Return upper bound.
    fn upper(&self) -> P;
    /// Set lower bound.
    fn set_lower(&mut self, l: P);
    /// Set upper bound.
    fn set_upper(&mut self, u: P);
}
