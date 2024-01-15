use nalgebra::SVector;
use num_traits::Bounded;

#[derive(Debug, Copy, Clone)]
pub struct Region<const D: usize> {
    pub min_max: Option<(SVector<f64, D>, SVector<f64, D>)>,
}

#[derive(Debug, Copy, Clone)]
pub struct IRegion<const D: usize> {
    pub min_max: Option<(SVector<i64, D>, SVector<i64, D>)>,
}

impl<const D: usize> IRegion<D> {
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

pub trait PointTraits<const D: usize>: Copy + Bounded + std::ops::Index<usize> {
    type Point: Bounded;

    fn smallest() -> Self::Point {
        Bounded::min_value()
    }
    fn largest() -> Self::Point {
        Bounded::max_value()
    }

    fn clamp(&self, min: Self, max: Self) -> Self::Point;

    fn is_less_equal(&self, rhs: Self) -> bool;
}

impl<const D: usize> PointTraits<D> for SVector<f64, D> {
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

impl<const D: usize> PointTraits<D> for SVector<i64, D> {
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

pub trait RegionTraits<const D: usize, P: PointTraits<D>> {
    type Region;

    fn unbounded() -> Self;

    fn empty() -> Self::Region;

    fn from_min_max(min: P, max: P) -> Self::Region;

    fn is_empty(&self) -> bool;

    fn is_degenerated(&self) -> bool;

    fn is_proper(&self) -> bool {
        !self.is_empty() && !self.is_degenerated()
    }

    fn is_unbounded(&self) -> bool;

    fn from_point(point: P) -> Self::Region {
        Self::from_min_max(point, point)
    }

    fn extend(&mut self, point: &P);

    fn min(&self) -> P {
        self.try_min().unwrap()
    }
    fn max(&self) -> P {
        self.try_max().unwrap()
    }

    fn try_min(&self) -> Option<P>;
    fn try_max(&self) -> Option<P>;

    fn clamp(&self, p: P) -> P;

    fn contains(&self, p: P) -> bool {
        if self.is_empty() {
            return false;
        }
        self.min().is_less_equal(self.min()) && p.is_less_equal(self.max())
    }

    fn range(&self) -> P;

    fn mid(&self) -> P;
}

impl<const D: usize> RegionTraits<D, SVector<f64, D>> for Region<D> {
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

impl<const D: usize> RegionTraits<D, SVector<i64, D>> for IRegion<D> {
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
