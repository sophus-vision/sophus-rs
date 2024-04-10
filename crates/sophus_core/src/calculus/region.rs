use nalgebra::SVector;

use crate::IsPoint;

/// Floating-point interval
#[derive(Debug, Copy, Clone)]
pub struct Interval {
    /// min and max of the interval
    pub min_max: Option<(f64, f64)>,
}

/// Integer interval
#[derive(Debug, Copy, Clone)]
pub struct IInterval {
    /// min and max of the interval
    pub min_max: Option<(i64, i64)>,
}

/// Region - n-dimensional interval
#[derive(Debug, Copy, Clone)]
pub struct Region<const D: usize> {
    /// min and max of the region
    pub min_max: Option<(SVector<f64, D>, SVector<f64, D>)>,
}

/// Integer Region - n-dimensional interval
#[derive(Debug, Copy, Clone)]
pub struct IRegion<const D: usize> {
    /// min and max of the region
    pub min_max: Option<(SVector<i64, D>, SVector<i64, D>)>,
}

impl<const D: usize> IRegion<D> {
    /// convert integer region to floating point region
    pub fn to_region(&self) -> Region<D> {
        if self.is_empty() {
            return Region::empty();
        }
        // example: [2, 5] -> [1.5, 5.5]
        Region::from_min_max(
            self.min().cast() - SVector::repeat(0.5),
            self.max().cast() + SVector::repeat(0.5),
        )
    }
}

/// Traits for regions
pub trait IsRegion<const D: usize, P: IsPoint<D>> {
    /// Region type
    type Region;

    /// create unbounded region
    fn unbounded() -> Self;

    /// create empty region
    fn empty() -> Self::Region;

    /// create region from min and max values
    fn from_min_max(min: P, max: P) -> Self::Region;

    /// create region from point
    fn from_point(point: P) -> Self::Region {
        Self::from_min_max(point, point)
    }

    /// is region empty
    fn is_empty(&self) -> bool;

    /// is region degenerated
    ///
    /// A region is degenerated if it is not empty and min == max
    fn is_degenerated(&self) -> bool;

    /// is region proper
    ///
    /// A region is proper if it is not empty and not degenerated
    fn is_proper(&self) -> bool {
        !self.is_empty() && !self.is_degenerated()
    }

    /// is region unbounded
    fn is_unbounded(&self) -> bool;

    /// extend region to include point
    fn extend(&mut self, point: &P);

    /// min of the region
    ///
    /// panics if the region is empty
    fn min(&self) -> P {
        self.try_min().unwrap()
    }

    /// max of the region
    ///
    /// panics if the region is empty
    fn max(&self) -> P {
        self.try_max().unwrap()
    }

    /// return min of the region or None if the region is empty
    fn try_min(&self) -> Option<P>;

    /// return max of the region or None if the region is empty
    fn try_max(&self) -> Option<P>;

    /// clamp point to the region
    fn clamp(&self, p: P) -> P;

    /// check if the region contains a point
    fn contains(&self, p: P) -> bool {
        if self.is_empty() {
            return false;
        }
        self.min().is_less_equal(self.min()) && p.is_less_equal(self.max())
    }

    /// range of the region
    fn range(&self) -> P;

    /// mid of the region
    fn mid(&self) -> P;
}

impl IsRegion<1, f64> for Interval {
    type Region = Self;

    fn unbounded() -> Self {
        Self {
            min_max: Option::Some((f64::NEG_INFINITY, f64::INFINITY)),
        }
    }

    fn empty() -> Self::Region {
        Self {
            min_max: Option::None,
        }
    }

    fn from_min_max(min: f64, max: f64) -> Self::Region {
        Self {
            min_max: Option::Some((min, max)),
        }
    }

    fn is_empty(&self) -> bool {
        self.min_max.is_none()
    }

    fn is_degenerated(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        self.min() == self.max()
    }

    fn is_unbounded(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        self.min() == f64::NEG_INFINITY && self.max() == f64::INFINITY
    }

    fn extend(&mut self, point: &f64) {
        if self.is_empty() {
            *self = Self::from_point(*point);
        }
        let (min, max) = (self.min().min(*point), self.max().max(*point));

        *self = Self::from_min_max(min, max)
    }

    fn try_min(&self) -> Option<f64> {
        Some(self.min_max?.0)
    }

    fn try_max(&self) -> Option<f64> {
        Some(self.min_max?.1)
    }

    fn clamp(&self, p: f64) -> f64 {
        p.clamp(self.min(), self.max())
    }

    fn range(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        self.max() - self.min()
    }

    fn mid(&self) -> f64 {
        self.min() + 0.5 * self.range()
    }
}

impl<const D: usize> IsRegion<D, SVector<f64, D>> for Region<D> {
    type Region = Self;

    fn unbounded() -> Self {
        let s: SVector<f64, D> = SVector::<f64, D>::smallest();
        let l: SVector<f64, D> = SVector::<f64, D>::largest();
        Self::from_min_max(s, l)
    }

    fn empty() -> Self {
        Self {
            min_max: Option::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.min_max.is_none()
    }

    fn is_degenerated(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        self.min() == self.max()
    }

    fn is_unbounded(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        self.min() == SVector::<f64, D>::smallest() && self.max() == SVector::<f64, D>::largest()
    }

    fn from_min_max(min: SVector<f64, D>, max: SVector<f64, D>) -> Self {
        Self {
            min_max: Option::Some((min, max)),
        }
    }

    fn try_min(&self) -> Option<SVector<f64, D>> {
        Some(self.min_max?.0)
    }
    fn try_max(&self) -> Option<SVector<f64, D>> {
        Some(self.min_max?.1)
    }

    fn clamp(&self, p: SVector<f64, D>) -> SVector<f64, D> {
        p.clamp(self.min(), self.max())
    }

    fn range(&self) -> SVector<f64, D> {
        let p: SVector<f64, D>;
        if self.is_empty() {
            p = SVector::<f64, D>::zeros();
            return p;
        }
        p = self.max() - self.min();
        p
    }

    fn mid(&self) -> SVector<f64, D> {
        self.min() + 0.5 * self.range()
    }

    fn extend(&mut self, point: &SVector<f64, D>) {
        if self.is_empty() {
            *self = Self::from_point(*point);
        }
        let (min, max) = self.min().inf_sup(point);

        *self = Self::from_min_max(min, max)
    }
}

impl IsRegion<1, i64> for IInterval {
    type Region = Self;

    fn unbounded() -> Self {
        Self {
            min_max: Option::Some((i64::MIN, i64::MAX)),
        }
    }

    fn empty() -> Self::Region {
        Self {
            min_max: Option::None,
        }
    }

    fn from_min_max(min: i64, max: i64) -> Self::Region {
        Self {
            min_max: Option::Some((min, max)),
        }
    }

    fn is_empty(&self) -> bool {
        self.min_max.is_none()
    }

    fn is_degenerated(&self) -> bool {
        false
    }

    fn is_unbounded(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        self.min() == i64::MIN && self.max() == i64::MAX
    }

    fn extend(&mut self, point: &i64) {
        if self.is_empty() {
            *self = Self::from_point(*point);
        }
        let (min, max) = (self.min().min(*point), self.max().max(*point));

        *self = Self::from_min_max(min, max)
    }

    fn try_min(&self) -> Option<i64> {
        Some(self.min_max?.0)
    }

    fn try_max(&self) -> Option<i64> {
        Some(self.min_max?.1)
    }

    fn clamp(&self, p: i64) -> i64 {
        p.clamp(self.min(), self.max())
    }

    fn range(&self) -> i64 {
        if self.is_empty() {
            return 0;
        }
        self.max() - self.min()
    }

    fn mid(&self) -> i64 {
        self.min() + self.range() / 2
    }
}

impl<const D: usize> IsRegion<D, SVector<i64, D>> for IRegion<D> {
    type Region = Self;

    fn unbounded() -> Self {
        let s: SVector<i64, D> = SVector::<i64, D>::smallest();
        let l: SVector<i64, D> = SVector::<i64, D>::largest();
        Self::from_min_max(s, l)
    }

    fn empty() -> Self {
        Self {
            min_max: Option::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.min_max.is_none()
    }

    fn is_degenerated(&self) -> bool {
        false
    }

    fn is_unbounded(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        self.min() == SVector::<i64, D>::smallest() && self.max() == SVector::<i64, D>::largest()
    }

    fn from_min_max(min: SVector<i64, D>, max: SVector<i64, D>) -> Self {
        Self {
            min_max: Option::Some((min, max)),
        }
    }

    fn try_min(&self) -> Option<SVector<i64, D>> {
        Some(self.min_max?.0)
    }
    fn try_max(&self) -> Option<SVector<i64, D>> {
        Some(self.min_max?.1)
    }

    fn clamp(&self, p: SVector<i64, D>) -> SVector<i64, D> {
        p.clamp(self.min(), self.max())
    }

    fn range(&self) -> SVector<i64, D> {
        let p: SVector<i64, D>;
        if self.is_empty() {
            p = SVector::<i64, D>::zeros();
            return p;
        }
        p = self.max() - self.min() + SVector::<i64, D>::repeat(1);
        p
    }

    fn mid(&self) -> SVector<i64, D> {
        self.min() + self.range() / 2
    }

    fn extend(&mut self, point: &SVector<i64, D>) {
        if self.is_empty() {
            *self = Self::from_point(*point);
        }
        let (min, max) = self.min().inf_sup(point);

        *self = Self::from_min_max(min, max)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn region() {
        let empty_f64 = Region::<2>::empty();
        assert!(empty_f64.is_empty());
        assert!(!empty_f64.is_degenerated());
        assert!(!empty_f64.is_proper());
        assert!(!empty_f64.is_unbounded());

        let unbounded = Region::<2>::unbounded();
        assert!(!unbounded.is_empty());
        assert!(!unbounded.is_degenerated());
        assert!(unbounded.is_proper());
        assert!(unbounded.is_unbounded());

        let one_i64 = IRegion::<2>::from_point(SVector::<i64, 2>::repeat(1));
        assert!(!one_i64.is_empty());
        assert!(!one_i64.is_degenerated());
        assert!(one_i64.is_proper());
        assert!(!one_i64.is_unbounded());

        let two_f64 = Region::<2>::from_point(SVector::<f64, 2>::repeat(2.0));
        assert!(!two_f64.is_empty());
        assert!(two_f64.is_degenerated());
        assert!(!two_f64.is_proper());
        assert!(!two_f64.is_unbounded());
    }

    #[test]
    fn interval() {
        let empty_f64 = Interval::empty();
        assert!(empty_f64.is_empty());
        assert!(!empty_f64.is_degenerated());
        assert!(!empty_f64.is_proper());
        assert!(!empty_f64.is_unbounded());

        let unbounded = Interval::unbounded();
        assert!(!unbounded.is_empty());
        assert!(!unbounded.is_degenerated());
        assert!(unbounded.is_proper());
        assert!(unbounded.is_unbounded());

        let one_i64 = IRegion::<1>::from_point(SVector::<i64, 1>::repeat(1));
        assert!(!one_i64.is_empty());
        assert!(!one_i64.is_degenerated());
        assert!(one_i64.is_proper());
        assert!(!one_i64.is_unbounded());

        let two_f64 = Interval::from_point(2.0);
        assert!(!two_f64.is_empty());
        assert!(two_f64.is_degenerated());
        assert!(!two_f64.is_proper());
        assert!(!two_f64.is_unbounded());
    }
}
