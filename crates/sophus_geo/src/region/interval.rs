use sophus_autodiff::linalg::VecF64;

use super::{
    box_region::{BoxRegion, NonEmptyBoxRegion},
    IsNonEmptyRegion, IsRegion, IsRegionBase,
};

/// Floating-point interval
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct NonEmptyInterval {
    pub(crate) region1: NonEmptyBoxRegion<1>,
}

impl NonEmptyInterval {
    /// Convert to interval
    pub fn to_box_region(self) -> NonEmptyBoxRegion<1> {
        self.region1
    }
}

impl IsRegionBase<1, f64> for NonEmptyInterval {
    type Region = Interval;

    type NonEmptyRegion = NonEmptyInterval;

    fn unbounded() -> Self {
        NonEmptyBoxRegion::unbounded().to_interval()
    }

    fn from_bounds(bound_a: f64, bound_b: f64) -> Self {
        if bound_a < bound_b {
            return NonEmptyInterval {
                region1: NonEmptyBoxRegion {
                    lower: VecF64::<1>::new(bound_a),
                    upper: VecF64::<1>::new(bound_b),
                },
            };
        }
        NonEmptyInterval {
            region1: NonEmptyBoxRegion {
                lower: VecF64::<1>::new(bound_b),
                upper: VecF64::<1>::new(bound_a),
            },
        }
    }

    fn is_degenerated(&self) -> bool {
        self.region1.is_degenerated()
    }

    fn is_unbounded(&self) -> bool {
        self.region1.is_unbounded()
    }

    fn extend(&mut self, point: f64) {
        self.region1.extend(VecF64::<1>::new(point));
    }

    fn clamp(&self, p: f64) -> f64 {
        self.region1.clamp(VecF64::<1>::new(p))[0]
    }

    fn intersect(self, other: Self) -> Self::Region {
        self.region1.intersect(other.region1).to_interval()
    }

    fn contains(&self, p: f64) -> bool {
        self.region1.contains(VecF64::<1>::new(p))
    }

    fn range(&self) -> f64 {
        self.region1.range()[0]
    }

    fn to_region(self) -> Self::Region {
        self.region1.to_region().to_interval()
    }

    fn to_non_empty_region(self) -> Option<Self::NonEmptyRegion> {
        Some(self)
    }
}

impl IsNonEmptyRegion<1, f64> for NonEmptyInterval {
    fn center(&self) -> f64 {
        self.region1.center()[0]
    }

    fn lower(&self) -> f64 {
        self.region1.lower()[0]
    }

    fn upper(&self) -> f64 {
        self.region1.upper()[0]
    }

    fn set_lower(&mut self, l: f64) {
        self.region1.lower[0] = l;
    }

    fn set_upper(&mut self, u: f64) {
        self.region1.upper[0] = u;
    }
}

/// Floating-point interval - might be empty
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Interval {
    pub(crate) region1: BoxRegion<1>,
}

impl Interval {
    /// Convert to interval
    pub fn to_box_region(self) -> BoxRegion<1> {
        self.region1
    }
}

impl IsRegionBase<1, f64> for Interval {
    type Region = Interval;

    type NonEmptyRegion = NonEmptyInterval;

    fn unbounded() -> Self {
        BoxRegion::unbounded().to_interval()
    }

    fn from_bounds(bound_a: f64, bound_b: f64) -> Self {
        NonEmptyInterval::from_bounds(bound_a, bound_b).to_region()
    }

    fn is_degenerated(&self) -> bool {
        self.region1.is_degenerated()
    }

    fn is_unbounded(&self) -> bool {
        self.region1.is_unbounded()
    }

    fn extend(&mut self, point: f64) {
        self.region1.extend(VecF64::<1>::new(point));
    }

    fn clamp(&self, p: f64) -> f64 {
        self.region1.clamp(VecF64::<1>::new(p))[0]
    }

    fn intersect(self, other: Self) -> Self::Region {
        self.region1.intersect(other.region1).to_interval()
    }

    fn contains(&self, p: f64) -> bool {
        self.region1.contains(VecF64::<1>::new(p))
    }

    fn range(&self) -> f64 {
        self.region1.range()[0]
    }

    fn to_region(self) -> Self::Region {
        self.region1.to_region().to_interval()
    }

    fn to_non_empty_region(self) -> Option<Self::NonEmptyRegion> {
        if let Some(r) = self.region1.to_non_empty_region() {
            return Some(r.to_interval());
        }
        None
    }
}

impl IsRegion<1, f64> for Interval {
    fn empty() -> Self {
        BoxRegion::empty().to_interval()
    }

    fn try_lower(&self) -> Option<f64> {
        if let Some(l) = self.region1.try_lower() {
            return Some(l[0]);
        }
        None
    }

    fn try_upper(&self) -> Option<f64> {
        if let Some(l) = self.region1.try_upper() {
            return Some(l[0]);
        }
        None
    }

    fn try_center(&self) -> Option<f64> {
        if let Some(l) = self.region1.try_center() {
            return Some(l[0]);
        }
        None
    }

    fn is_empty(&self) -> bool {
        self.region1.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic properties: empty, unbounded, degenerate, etc.
    #[test]
    fn interval_basics() {
        // 1) Empty interval
        let empty = Interval::empty();
        assert!(empty.is_empty(), "Empty interval should be empty");
        assert!(!empty.is_degenerated(), "Empty interval is not degenerate");
        assert!(!empty.is_proper(), "Empty interval is not 'proper'");
        assert!(!empty.is_unbounded(), "Empty interval is not unbounded");

        // 2) Unbounded interval
        let unbounded = Interval::unbounded();
        assert!(!unbounded.is_empty(), "Unbounded interval is not empty");
        assert!(
            !unbounded.is_degenerated(),
            "Unbounded interval is not degenerate"
        );
        assert!(
            unbounded.is_proper(),
            "Unbounded interval is typically 'proper'"
        );
        assert!(unbounded.is_unbounded(), "Should be unbounded indeed");

        // 3) Single-point (degenerate) interval
        let single_pt = Interval::from_bounds(2.0, 2.0);
        assert!(!single_pt.is_empty(), "Single-point interval is not empty");
        assert!(single_pt.is_degenerated(), "Single-point is degenerate");
        assert!(
            !single_pt.is_proper(),
            "Degenerate interval is not 'proper'"
        );
        assert!(!single_pt.is_unbounded(), "Single-point is not unbounded");
    }

    /// Test extending intervals
    #[test]
    fn interval_extend() {
        // Extend an empty interval
        let mut interval = Interval::empty();
        assert!(interval.is_empty());
        interval.extend(3.0);

        // Now should be degenerate [3..3]
        assert!(!interval.is_empty());
        assert!(interval.is_degenerated());
        assert_eq!(interval.try_lower().unwrap(), 3.0);
        assert_eq!(interval.try_upper().unwrap(), 3.0);

        // Extend further
        interval.extend(5.0);
        assert!(!interval.is_empty());
        assert!(!interval.is_degenerated());
        assert_eq!(interval.try_lower().unwrap(), 3.0);
        assert_eq!(interval.try_upper().unwrap(), 5.0);
    }

    /// Test intersection logic
    #[test]
    fn interval_intersect() {
        // intersecting empty => empty
        let empty = Interval::empty();
        let a = Interval::from_bounds(0.0, 5.0);
        let i1 = empty.intersect(a);
        assert!(i1.is_empty(), "Empty ∩ anything = empty");

        // overlapping intervals
        let b = Interval::from_bounds(3.0, 10.0);
        let i2 = a.intersect(b);
        assert!(!i2.is_empty(), "Should have valid intersection");
        assert_eq!(i2.try_lower().unwrap(), 3.0);
        assert_eq!(i2.try_upper().unwrap(), 5.0);
        assert!(i2.is_proper(), "Should be a proper interval");

        // disjoint intervals => empty
        let c = Interval::from_bounds(6.0, 7.0);
        let i3 = a.intersect(c);
        assert!(i3.is_empty(), "Disjoint => empty");
    }

    /// Test clamping values
    #[test]
    fn interval_clamp() {
        let interval = Interval::from_bounds(0.0, 10.0);

        // inside => remain
        assert_eq!(interval.clamp(5.0), 5.0);

        // below => clamp to lower
        assert_eq!(interval.clamp(-2.0), 0.0);

        // above => clamp to upper
        assert_eq!(interval.clamp(30.0), 10.0);

        // empty => return input as-is
        let empty = Interval::empty();
        assert_eq!(empty.clamp(42.0), 42.0);
    }

    /// Test that from_bounds reorders if a > b
    #[test]
    fn test_from_bounds_reordering() {
        let interval = Interval::from_bounds(10.0, 3.0);
        assert_eq!(interval.try_lower().unwrap(), 3.0);
        assert_eq!(interval.try_upper().unwrap(), 10.0);
        assert!(!interval.is_empty());
        assert!(!interval.is_degenerated());
    }

    /// Overlapping exactly at boundary => degenerate
    #[test]
    fn test_partial_overlap_creates_degenerate() {
        // [0..5] ∩ [5..10] => [5..5], degenerate
        let a = Interval::from_bounds(0.0, 5.0);
        let b = Interval::from_bounds(5.0, 10.0);
        let inter = a.intersect(b);
        assert!(
            !inter.is_empty(),
            "Boundary-only overlap yields degenerate, not empty"
        );
        assert!(inter.is_degenerated());
        assert_eq!(inter.try_lower().unwrap(), 5.0);
        assert_eq!(inter.try_upper().unwrap(), 5.0);
    }

    /// Test `NonEmptyInterval` setters (lower, upper)
    #[test]
    fn test_nonemptyinterval_setters() {
        let mut nonempty = NonEmptyInterval::from_bounds(2.0, 6.0);
        assert_eq!(nonempty.lower(), 2.0);
        assert_eq!(nonempty.upper(), 6.0);

        nonempty.set_lower(1.0);
        nonempty.set_upper(10.0);
        assert_eq!(nonempty.lower(), 1.0);
        assert_eq!(nonempty.upper(), 10.0);

        assert!(!nonempty.is_degenerated());
    }

    /// Test center calculation in degenerate vs. normal intervals
    #[test]
    fn test_center_calc() {
        let normal = NonEmptyInterval::from_bounds(2.0, 6.0);
        assert_eq!(normal.center(), 4.0);

        let deg = NonEmptyInterval::from_bounds(5.0, 5.0);
        assert_eq!(deg.center(), 5.0);
    }

    /// Test converting between Interval and NonEmptyInterval
    #[test]
    fn test_interval_conversions() {
        // NonEmpty -> Interval
        let nonempty = NonEmptyInterval::from_bounds(1.0, 3.0);
        let interval = nonempty.to_region(); // yields Interval
        assert!(!interval.is_empty());
        assert_eq!(interval.try_lower().unwrap(), 1.0);
        assert_eq!(interval.try_upper().unwrap(), 3.0);

        // Interval -> Option<NonEmptyInterval>
        let maybe_ner = interval.to_non_empty_region();
        assert!(maybe_ner.is_some());
        let unwrapped = maybe_ner.unwrap();
        assert_eq!(unwrapped.lower(), 1.0);
        assert_eq!(unwrapped.upper(), 3.0);

        // Empty interval => returns None
        let empty = Interval::empty();
        assert!(empty.to_non_empty_region().is_none());
    }

    /// Test is_proper logic
    #[test]
    fn test_is_proper_logic() {
        // empty
        let e = Interval::empty();
        assert!(!e.is_proper());

        // degenerate
        let d = Interval::from_bounds(2.0, 2.0);
        assert!(!d.is_empty());
        assert!(d.is_degenerated());
        assert!(!d.is_proper());

        // unbounded => typically 'proper'
        let u = Interval::unbounded();
        assert!(!u.is_empty());
        assert!(!u.is_degenerated());
        assert!(u.is_proper());

        // normal
        let n = Interval::from_bounds(0.0, 5.0);
        assert!(!n.is_empty());
        assert!(!n.is_degenerated());
        assert!(n.is_proper());
    }

    /// Test intersection with self
    #[test]
    fn test_intersection_with_self() {
        let iv = Interval::from_bounds(1.0, 4.0);
        let inter = iv.intersect(iv);
        assert!(!inter.is_empty());
        assert_eq!(inter.try_lower(), Some(1.0));
        assert_eq!(inter.try_upper(), Some(4.0));
        assert_eq!(iv.is_degenerated(), inter.is_degenerated());
        assert_eq!(iv.is_empty(), inter.is_empty());
    }
}
