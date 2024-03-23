use nalgebra::SVector;
use num_traits::Bounded;

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

/// Traits for points
pub trait IsPoint<const D: usize>: Copy + Bounded + std::ops::Index<usize> {
    /// Point type
    type Point: Bounded;

    /// smallest point
    fn smallest() -> Self::Point {
        Bounded::min_value()
    }

    /// largest point
    fn largest() -> Self::Point {
        Bounded::max_value()
    }

    /// clamp point
    fn clamp(&self, min: Self, max: Self) -> Self::Point;

    /// check if point is less or equal to another point
    fn is_less_equal(&self, rhs: Self) -> bool;
}

impl<const D: usize> IsPoint<D> for SVector<f64, D> {
    type Point = Self;

    fn clamp(&self, min: Self, max: Self) -> Self::Point {
        let mut p: Self::Point = Self::Point::zeros();
        for i in 0..D {
            p[i] = self[i].clamp(min[i], max[i]);
        }
        p
    }

    fn is_less_equal(&self, _rhs: Self::Point) -> bool {
        todo!()
    }
}

impl<const D: usize> IsPoint<D> for SVector<i64, D> {
    type Point = Self;

    fn clamp(&self, min: Self::Point, max: Self::Point) -> Self::Point {
        let mut p: Self::Point = Self::Point::zeros();
        for i in 0..D {
            p[i] = self[i].clamp(min[i], max[i]);
        }
        p
    }

    fn is_less_equal(&self, _rhs: Self::Point) -> bool {
        todo!()
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
        let empty_f64 = Region::<1>::empty();
        assert!(empty_f64.is_empty());
        assert!(!empty_f64.is_degenerated());
        assert!(!empty_f64.is_proper());
        assert!(!empty_f64.is_unbounded());

        let unbounded = Region::<1>::unbounded();
        assert!(!unbounded.is_empty());
        assert!(!unbounded.is_degenerated());
        assert!(unbounded.is_proper());
        assert!(unbounded.is_unbounded());

        let one_i64 = IRegion::<1>::from_point(SVector::<i64, 1>::repeat(1));
        assert!(!one_i64.is_empty());
        assert!(!one_i64.is_degenerated());
        assert!(one_i64.is_proper());
        assert!(!one_i64.is_unbounded());

        let two_f64 = Region::<1>::from_point(SVector::<f64, 1>::repeat(2.0));
        assert!(!two_f64.is_empty());
        assert!(two_f64.is_degenerated());
        assert!(!two_f64.is_proper());
        assert!(!two_f64.is_unbounded());
    }
}
