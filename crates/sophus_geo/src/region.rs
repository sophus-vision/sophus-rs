/// Box region - axis aligned bounding box
pub mod box_region;
/// Interval
pub mod interval;

use sophus_autodiff::IsPoint;

/// Traits for regions
pub trait IsRegionBase<const D: usize, P: IsPoint<D>>: core::marker::Sized {
    /// A region - might be empty
    type Region: IsRegion<D, P>;
    /// A non-empty region
    type NonEmptyRegion: IsNonEmptyRegion<D, P>;

    /// create unbounded region
    fn unbounded() -> Self;

    /// create region from upper and lower bounds
    ///
    /// Creates interval
    ///
    ///   [a, b]  if a <= b
    ///   [b, a]  if a > b
    fn from_bounds(bound_a: P, bound_b: P) -> Self;

    /// creates region from single point
    fn from_point(point: P) -> Self {
        Self::from_bounds(point, point)
    }

    /// is region degenerated
    ///
    /// A region is degenerated if it is not empty and min == max
    fn is_degenerated(&self) -> bool;

    /// is region unbounded
    fn is_unbounded(&self) -> bool;

    /// extend region to include point
    fn extend(&mut self, point: P);

    /// clamp point to the region
    fn clamp(&self, p: P) -> P;

    /// intersect
    fn intersect(self, other: Self) -> Self::Region;

    /// check if the region contains a point
    fn contains(&self, p: P) -> bool;

    /// range of the region
    fn range(&self) -> P;

    /// convert to region
    fn to_region(self) -> Self::Region;

    /// convert to non-empty-region
    fn to_non_empty_region(self) -> Option<Self::NonEmptyRegion>;
}

/// Traits for regions - might be empty
pub trait IsRegion<const D: usize, P: IsPoint<D>>: IsRegionBase<D, P> {
    /// create empty region
    fn empty() -> Self;

    /// return lower bound of the region or None if the region is empty
    fn try_lower(&self) -> Option<P>;

    /// return upper bound of the region or None if the region is empty
    fn try_upper(&self) -> Option<P>;

    /// center of the region
    fn try_center(&self) -> Option<P>;

    /// is region empty
    fn is_empty(&self) -> bool;

    /// is region proper
    ///
    /// A region is proper if it is not empty and not degenerated
    fn is_proper(&self) -> bool {
        !self.is_empty() && !self.is_degenerated()
    }
}

/// Traits for a non-empty region
pub trait IsNonEmptyRegion<const D: usize, P: IsPoint<D>>: IsRegionBase<D, P> {
    /// center of the region
    fn center(&self) -> P;

    /// lower bound
    fn lower(&self) -> P;
    /// upper bound
    fn upper(&self) -> P;
    /// set lower bound
    fn set_lower(&mut self, l: P);
    /// set upper bound
    fn set_upper(&mut self, u: P);
}
