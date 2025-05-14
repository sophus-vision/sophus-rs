use sophus_autodiff::{
    linalg::{
        SVec,
        VecF64,
    },
    points::{
        IsPoint,
        IsUnboundedPoint,
    },
};

use super::{
    IsNonEmptyRegion,
    IsRegion,
    IsRegionBase,
    interval::{
        Interval,
        NonEmptyInterval,
    },
};

/// A non-empty n-dimensional "box" interval.
///
/// A box region is a Cartesian product of non-empty intervals or, in other words, an axis-aligned
/// bounding box.
///
/// See [IsRegionBase] and [IsNonEmptyRegion] for details about the (non-empty) region concept.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct NonEmptyBoxRegion<const N: usize> {
    pub(crate) lower: SVec<f64, N>,
    pub(crate) upper: SVec<f64, N>,
}

impl NonEmptyBoxRegion<1> {
    /// Convert one-dimensional region to the [Interval] type.
    pub fn to_interval(self) -> NonEmptyInterval {
        NonEmptyInterval { region1: self }
    }
}

impl<const N: usize> IsRegionBase<N, SVec<f64, N>> for NonEmptyBoxRegion<N> {
    type Region = BoxRegion<N>;
    type NonEmptyRegion = NonEmptyBoxRegion<N>;

    fn intersect(self, other: Self) -> BoxRegion<N> {
        // Step 1: compute the intersection bounds
        let mut new_lower = VecF64::zeros();
        let mut new_upper = VecF64::zeros();

        for i in 0..N {
            new_lower[i] = self.lower[i].max(other.lower[i]);
            new_upper[i] = self.upper[i].min(other.upper[i]);
        }

        // Step 2: check if valid intersection
        for i in 0..N {
            // if any dimension does not overlap, return an empty Region
            if new_lower[i] > new_upper[i] {
                return BoxRegion {
                    non_empty_region: None,
                };
            }
        }

        // Step 3: otherwise, it’s a valid NonEmptyRegion
        NonEmptyBoxRegion {
            lower: new_lower,
            upper: new_upper,
        }
        .to_region()
    }

    fn contains(&self, p: SVec<f64, N>) -> bool {
        self.lower.is_less_equal(p) && p.is_less_equal(self.upper)
    }

    fn unbounded() -> Self {
        let l: SVec<f64, N> = SVec::<f64, N>::neg_infinity();
        let u: SVec<f64, N> = SVec::<f64, N>::infinity();
        Self::from_bounds(l, u)
    }

    fn is_degenerated(&self) -> bool {
        self.lower == self.upper
    }

    fn is_unbounded(&self) -> bool {
        self.lower == SVec::<f64, N>::neg_infinity() && self.upper == SVec::<f64, N>::infinity()
    }

    fn from_bounds(bound_a: SVec<f64, N>, bound_b: SVec<f64, N>) -> Self {
        let mut lower = VecF64::zeros();
        let mut upper = VecF64::zeros();

        for i in 0..N {
            lower[i] = bound_a[i].min(bound_b[i]);
            upper[i] = bound_a[i].max(bound_b[i]);
        }

        Self { lower, upper }
    }

    fn clamp_point(&self, p: SVec<f64, N>) -> SVec<f64, N> {
        p.clamp(self.lower, self.upper)
    }

    fn range(&self) -> SVec<f64, N> {
        self.upper - self.lower
    }

    fn extend(&mut self, point: SVec<f64, N>) {
        for i in 0..N {
            self.lower[i] = self.lower[i].min(point[i]);
            self.upper[i] = self.upper[i].max(point[i]);
        }
    }

    fn to_region(self) -> Self::Region {
        BoxRegion {
            non_empty_region: Some(self),
        }
    }

    fn to_non_empty_region(self) -> Option<Self::NonEmptyRegion> {
        Some(self)
    }

    fn from_point(point: SVec<f64, N>) -> Self {
        Self::from_bounds(point, point)
    }
}

impl<const N: usize> IsNonEmptyRegion<N, SVec<f64, N>> for NonEmptyBoxRegion<N> {
    fn center(&self) -> SVec<f64, N> {
        self.lower + 0.5 * self.range()
    }

    fn lower(&self) -> SVec<f64, N> {
        self.lower
    }

    fn upper(&self) -> SVec<f64, N> {
        self.upper
    }

    fn set_lower(&mut self, l: SVec<f64, N>) {
        self.lower = l;
    }

    fn set_upper(&mut self, u: SVec<f64, N>) {
        self.upper = u;
    }
}

/// A n-dimensional "box" interval which might be empty.
///
/// A box region is a Cartesian product of intervals or, in other words, an axis-aligned bounding
/// box.
///
/// See [IsRegionBase] and [IsRegion] for details about the region concept.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BoxRegion<const N: usize> {
    non_empty_region: Option<NonEmptyBoxRegion<N>>,
}

impl BoxRegion<1> {
    /// Convert one-dimensional region to the [Interval] type.
    pub fn to_interval(self) -> Interval {
        Interval { region1: self }
    }
}

impl<const N: usize> IsRegionBase<N, SVec<f64, N>> for BoxRegion<N> {
    type Region = BoxRegion<N>;
    type NonEmptyRegion = NonEmptyBoxRegion<N>;

    fn unbounded() -> Self {
        Self {
            non_empty_region: Some(NonEmptyBoxRegion::unbounded()),
        }
    }

    fn from_bounds(bound_a: SVec<f64, N>, bound_b: SVec<f64, N>) -> Self {
        Self {
            non_empty_region: Some(NonEmptyBoxRegion::from_bounds(bound_a, bound_b)),
        }
    }

    fn is_degenerated(&self) -> bool {
        if let Some(non_empty) = self.non_empty_region {
            return non_empty.is_degenerated();
        }
        false
    }

    fn is_unbounded(&self) -> bool {
        if let Some(non_empty) = self.non_empty_region {
            return non_empty.is_unbounded();
        }
        false
    }

    fn extend(&mut self, point: SVec<f64, N>) {
        match &mut self.non_empty_region {
            Some(nz) => nz.extend(point),
            None => {
                self.non_empty_region = Some(NonEmptyBoxRegion::from_bounds(point, point));
            }
        }
    }

    fn clamp_point(&self, p: SVec<f64, N>) -> SVec<f64, N> {
        if let Some(non_empty) = self.non_empty_region {
            return non_empty.clamp_point(p);
        }
        p
    }

    fn intersect(self, other: Self) -> Self {
        match (self.non_empty_region, other.non_empty_region) {
            (Some(s), Some(o)) => s.intersect(o),
            _ => BoxRegion::empty(),
        }
    }

    fn contains(&self, p: SVec<f64, N>) -> bool {
        if let Some(non_empty) = self.non_empty_region {
            return non_empty.contains(p);
        }
        false
    }

    fn range(&self) -> SVec<f64, N> {
        if let Some(non_empty) = self.non_empty_region {
            return non_empty.range();
        }
        SVec::zeros()
    }

    fn to_region(self) -> Self::Region {
        self
    }

    fn to_non_empty_region(self) -> Option<Self::NonEmptyRegion> {
        if let Some(non_empty) = self.non_empty_region {
            return Some(non_empty);
        }
        None
    }

    fn from_point(point: SVec<f64, N>) -> Self {
        Self::from_bounds(point, point)
    }
}

impl<const N: usize> IsRegion<N, SVec<f64, N>> for BoxRegion<N> {
    fn empty() -> Self {
        BoxRegion {
            non_empty_region: None,
        }
    }

    fn try_lower(&self) -> Option<SVec<f64, N>> {
        if let Some(non_empty) = self.non_empty_region {
            return Some(non_empty.lower);
        }
        None
    }

    fn try_upper(&self) -> Option<SVec<f64, N>> {
        if let Some(non_empty) = self.non_empty_region {
            return Some(non_empty.upper);
        }
        None
    }

    fn is_empty(&self) -> bool {
        self.non_empty_region.is_none()
    }

    fn try_center(&self) -> Option<SVec<f64, N>> {
        if let Some(non_empty) = self.non_empty_region {
            return Some(non_empty.center());
        }
        None
    }
}
#[cfg(test)]
mod tests {
    use sophus_autodiff::linalg::SVec;

    use super::*;

    #[test]
    fn region_basics() {
        // 1) Test empty region
        let empty_f64 = BoxRegion::<2>::empty();
        assert!(empty_f64.is_empty(), "Empty region should be empty");
        assert!(
            !empty_f64.is_degenerated(),
            "Empty region is not degenerate"
        );
        assert!(!empty_f64.is_proper(), "Empty region is not proper");
        assert!(!empty_f64.is_unbounded(), "Empty region is not unbounded");

        // 2) Test unbounded region
        let unbounded = BoxRegion::<2>::unbounded();
        assert!(
            !unbounded.is_empty(),
            "Unbounded region should not be empty"
        );
        assert!(
            !unbounded.is_degenerated(),
            "Unbounded region is not degenerate"
        );
        assert!(
            unbounded.is_proper(),
            "Unbounded region is typically considered 'proper'"
        );
        assert!(unbounded.is_unbounded(), "Should be unbounded indeed");

        // 3) Test a single-point (degenerate) region
        let two_f64 = BoxRegion::<2>::from_point(SVec::<f64, 2>::repeat(2.0));
        assert!(!two_f64.is_empty(), "A single-point region is not empty");
        assert!(
            two_f64.is_degenerated(),
            "Single-point region is degenerate (lower == upper)"
        );
        assert!(!two_f64.is_proper(), "Degenerate region is not 'proper'");
        assert!(
            !two_f64.is_unbounded(),
            "Single-point region is certainly not unbounded"
        );
    }

    #[test]
    fn region_extend() {
        // Extending an empty region
        let mut region = BoxRegion::<2>::empty();
        assert!(region.is_empty());
        region.extend(SVec::<f64, 2>::repeat(3.0));

        // Now region should be a degenerate bounding box around (3,3)
        assert!(!region.is_empty());
        assert!(region.is_degenerated());
        assert_eq!(region.try_lower().unwrap(), VecF64::<2>::repeat(3.0));
        assert_eq!(region.try_upper().unwrap(), VecF64::<2>::repeat(3.0));

        // Extending further
        region.extend(SVec::<f64, 2>::from([4.0, 2.0]));
        assert!(!region.is_empty());
        assert!(!region.is_degenerated());

        // Now the bounding box should be from (3,2) to (4,3)
        assert_eq!(region.try_lower().unwrap(), SVec::from([3.0, 2.0]));
        assert_eq!(region.try_upper().unwrap(), SVec::from([4.0, 3.0]));
    }

    #[test]
    fn region_intersect() {
        // Intersect empty with anything => empty
        let empty = BoxRegion::<2>::empty();
        let region_a = BoxRegion::<2>::from_bounds(SVec::from([0.0, 0.0]), SVec::from([5.0, 5.0]));
        let inter1 = empty.intersect(region_a);
        assert!(inter1.is_empty(), "Intersection with empty should be empty");

        // Intersect two overlapping regions
        let region_b =
            BoxRegion::<2>::from_bounds(SVec::from([3.0, 3.0]), SVec::from([10.0, 10.0]));
        let inter2 = region_a.intersect(region_b);
        assert!(!inter2.is_empty(), "Should have a valid intersection");
        assert_eq!(inter2.try_lower().unwrap(), SVec::from([3.0, 3.0]));
        assert_eq!(inter2.try_upper().unwrap(), SVec::from([5.0, 5.0]));
        assert!(inter2.is_proper(), "Intersection is a non-empty box");

        // Intersect two disjoint regions => empty
        let region_c = BoxRegion::<2>::from_bounds(SVec::from([6.0, 6.0]), SVec::from([7.0, 7.0]));
        let inter3 = region_a.intersect(region_c);
        assert!(inter3.is_empty(), "Disjoint intersection should be empty");
    }

    #[test]
    fn region_clamp() {
        let region = BoxRegion::<2>::from_bounds(SVec::from([0.0, 0.0]), SVec::from([10.0, 10.0]));
        // 1) Point inside => should remain unchanged
        let inside = SVec::from([5.0, 5.0]);
        let clamped_inside = region.clamp_point(inside);
        assert_eq!(clamped_inside, inside);

        // 2) Point below => clamp to lower
        let below = SVec::from([-1.0, -2.0]);
        let clamped_below = region.clamp_point(below);
        assert_eq!(clamped_below, SVec::from([0.0, 0.0]));

        // 3) Point above => clamp to upper
        let above = SVec::from([20.0, 30.0]);
        let clamped_above = region.clamp_point(above);
        assert_eq!(clamped_above, SVec::from([10.0, 10.0]));

        // 4) If region is empty, clamp should just return the input
        let empty = BoxRegion::<2>::empty();
        let clamped_empty = empty.clamp_point(above);
        assert_eq!(clamped_empty, above, "Empty region clamps to input");
    }

    // Helper to define a 1D point type if needed;
    // in your codebase, you might just use `f64` directly
    // or rely on existing definitions.
    type OneD = SVec<f64, 1>;

    #[test]
    fn test_from_bounds_reordering() {
        // from_bounds should reorder if bound_a > bound_b in any dimension
        let reg = BoxRegion::<2>::from_bounds(SVec::from([10.0, 1.0]), SVec::from([0.0, 5.0]));
        let lower = reg.try_lower().unwrap();
        let upper = reg.try_upper().unwrap();

        assert_eq!(lower, SVec::from([0.0, 1.0]));
        assert_eq!(upper, SVec::from([10.0, 5.0]));
        assert!(!reg.is_empty());
        assert!(!reg.is_degenerated());
    }

    #[test]
    fn test_partial_overlap_creates_degenerate_region() {
        // Overlapping only on a boundary => degenerate intersection
        // For example: [0..5] ∩ [5..10] => [5..5], degenerate but not empty

        let reg_a = BoxRegion::<1>::from_bounds(OneD::from([0.0]), OneD::from([5.0]));
        let reg_b = BoxRegion::<1>::from_bounds(OneD::from([5.0]), OneD::from([10.0]));

        let inter = reg_a.intersect(reg_b);
        assert!(
            !inter.is_empty(),
            "Boundary overlap yields a degenerate region, not empty"
        );
        assert!(
            inter.is_degenerated(),
            "Intersection should be degenerate: [5..5]"
        );

        let lower = inter.try_lower().unwrap();
        let upper = inter.try_upper().unwrap();
        assert_eq!(lower[0], 5.0);
        assert_eq!(upper[0], 5.0);
    }

    #[test]
    fn test_nonemptyregion_setters() {
        // Testing set_lower / set_upper on a nonempty region
        let mut nonempty =
            NonEmptyBoxRegion::from_bounds(SVec::from([2.0, 3.0]), SVec::from([4.0, 6.0]));
        assert_eq!(nonempty.lower(), SVec::from([2.0, 3.0]));
        assert_eq!(nonempty.upper(), SVec::from([4.0, 6.0]));

        // Use trait methods to set new bounds
        nonempty.set_lower(SVec::from([1.0, 1.0]));
        nonempty.set_upper(SVec::from([5.0, 10.0]));

        // Confirm changes
        assert_eq!(nonempty.lower(), SVec::from([1.0, 1.0]));
        assert_eq!(nonempty.upper(), SVec::from([5.0, 10.0]));
        assert!(!nonempty.is_degenerated());
    }

    #[test]
    fn test_center_calculation() {
        // Non-degenerate center
        let nonempty =
            NonEmptyBoxRegion::from_bounds(SVec::from([2.0, 3.0]), SVec::from([6.0, 7.0]));
        let center = nonempty.center();
        assert_eq!(center, SVec::from([4.0, 5.0]));

        // Degenerate center (e.g., lower == upper)
        let deg = NonEmptyBoxRegion::from_bounds(SVec::from([5.0, 5.0]), SVec::from([5.0, 5.0]));
        let center_deg = deg.center();
        assert_eq!(center_deg, SVec::from([5.0, 5.0]));
    }

    #[test]
    fn test_conversions_between_region_and_non_emptyregion() {
        // Start with a NonEmptyRegion
        let ner = NonEmptyBoxRegion::from_bounds(SVec::from([1.0, 2.0]), SVec::from([3.0, 4.0]));
        // Convert to Region
        let reg = ner.to_region();
        assert!(!reg.is_empty());
        assert_eq!(reg.try_lower().unwrap(), SVec::from([1.0, 2.0]));
        assert_eq!(reg.try_upper().unwrap(), SVec::from([3.0, 4.0]));

        // Convert back
        let maybe_ner = reg.to_non_empty_region();
        assert!(maybe_ner.is_some());
        let unwrapped = maybe_ner.unwrap();
        assert_eq!(unwrapped.lower(), SVec::from([1.0, 2.0]));
        assert_eq!(unwrapped.upper(), SVec::from([3.0, 4.0]));

        // Converting an empty region yields None
        let empty_reg = BoxRegion::<2>::empty();
        let none_ner = empty_reg.to_non_empty_region();
        assert!(none_ner.is_none());
    }

    #[test]
    fn test_1d_interval_conversions() {
        // Start with a 1D NonEmptyRegion
        let one_d_region = NonEmptyBoxRegion::from_bounds(OneD::from([2.0]), OneD::from([5.0]));
        // Convert to "Interval" via `to_interval()` (assuming your code uses that)
        let interval = one_d_region.to_interval();
        assert_eq!(interval.region1.lower(), OneD::from([2.0]));
        assert_eq!(interval.region1.upper(), OneD::from([5.0]));

        // Convert a 1D Region -> Interval
        let reg_1d = BoxRegion::<1>::from_bounds(OneD::from([2.0]), OneD::from([2.0]));
        let interval2 = reg_1d.to_interval();
        assert!(interval2.region1.is_degenerated());
    }

    #[test]
    fn test_is_proper_logic() {
        // A region is "proper" if it's not empty and not degenerate
        // 1) empty
        let r_empty = BoxRegion::<2>::empty();
        assert!(!r_empty.is_proper());

        // 2) degenerate
        let r_deg = BoxRegion::<2>::from_bounds(SVec::from([1.0, 2.0]), SVec::from([1.0, 2.0]));
        assert!(!r_deg.is_empty());
        assert!(r_deg.is_degenerated());
        assert!(!r_deg.is_proper());

        // 3) unbounded is typically "proper" by your definition
        let r_unbounded = BoxRegion::<2>::unbounded();
        assert!(!r_unbounded.is_empty());
        assert!(!r_unbounded.is_degenerated());
        assert!(r_unbounded.is_proper());

        // 4) normal region
        let normal = BoxRegion::<2>::from_bounds(SVec::from([0.0, 0.0]), SVec::from([1.0, 2.0]));
        assert!(!normal.is_empty());
        assert!(!normal.is_degenerated());
        assert!(normal.is_proper());
    }

    #[test]
    fn test_extend_beyond_unbounded() {
        let mut unbounded = BoxRegion::<2>::unbounded();
        assert!(unbounded.is_unbounded());

        // Extend with a point inside the "min..max" range
        unbounded.extend(SVec::from([100.0, 100.0]));
        // It should remain unbounded if your logic doesn't try to surpass f64::MAX
        assert!(unbounded.is_unbounded());
    }

    #[test]
    fn test_intersection_with_self() {
        // Intersection with itself should yield the same region
        let reg_a = BoxRegion::<2>::from_bounds(SVec::from([1.0, 2.0]), SVec::from([3.0, 4.0]));
        let inter = reg_a.intersect(reg_a);
        assert_eq!(inter.try_lower(), Some(SVec::from([1.0, 2.0])));
        assert_eq!(inter.try_upper(), Some(SVec::from([3.0, 4.0])));
        assert_eq!(reg_a.is_degenerated(), inter.is_degenerated());
        assert_eq!(reg_a.is_empty(), inter.is_empty());
    }

    #[test]
    fn test_clamp_edges() {
        // region [0..5] x [10..20]
        let reg = BoxRegion::<2>::from_bounds(SVec::from([0.0, 10.0]), SVec::from([5.0, 20.0]));
        // clamp a point exactly on the lower bound
        let p_lower = SVec::from([0.0, 10.0]);
        let clamp_lower = reg.clamp_point(p_lower);
        assert_eq!(clamp_lower, p_lower);

        // clamp a point exactly on the upper bound
        let p_upper = SVec::from([5.0, 20.0]);
        let clamp_upper = reg.clamp_point(p_upper);
        assert_eq!(clamp_upper, p_upper);

        // clamp a point in the "corner" (x=0, y=20)
        let p_corner = SVec::from([0.0, 25.0]);
        let clamp_corner = reg.clamp_point(p_corner);
        assert_eq!(clamp_corner, SVec::from([0.0, 20.0]));
    }
}
