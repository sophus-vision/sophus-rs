use super::{points::*, batch_types::*};
use dfdx::{tensor::*, tensor_ops::*};

#[derive(Debug, Clone)]
pub struct Region<const BATCH: usize, const D: usize> {
    min: Option<V<BATCH, D>>, // min.is_some() <=> max.is_some()
    max: Option<V<BATCH, D>>,
}

trait RegionTraits<const BATCH: usize, const D: usize> {
    type Region;
    type Point: PointTraits<BATCH, D>;

    fn unbounded() -> Self;

    fn empty() -> Self::Region;

    fn from_min_max(min: Self::Point, max: Self::Point) -> Self::Region;

    fn is_empty(&self) -> bool;

    fn is_degenerated(&self) -> bool;

    fn is_proper(&self) -> bool {
        !self.is_empty() && !self.is_degenerated()
    }

    fn is_unbounded(&self) -> bool;

    fn from_point(point: Self::Point) -> Self::Region {
        Self::from_min_max(point.clone(), point)
    }

    fn min(&self) -> &Self::Point;

    fn max(&self) -> &Self::Point;

    fn try_min(&self) -> Option<&Self::Point>;
    fn try_max(&self) -> Option<&Self::Point>;

    fn clamp(&self, p: Self::Point) -> Self::Point;

    fn contains(&self, p: Self::Point) -> bool {
        if self.is_empty() {
            return false;
        }
        p.is_less_equal(self.min()) && p.is_greater_equal(self.max())
    }

    fn range(&self) -> Self::Point;

    fn mid(&self) -> Self::Point;
}

impl<const BATCH: usize, const D: usize> RegionTraits<BATCH, D> for Region<BATCH, D> {
    type Region = Self;
    type Point = V<BATCH, D>;

    fn unbounded() -> Self {
        let s: V<BATCH, D> = V::<BATCH, D>::smallest();
        let l: V<BATCH, D> = V::<BATCH, D>::largest();
        Self::from_min_max(s, l)
    }

    fn empty() -> Self {
        Self {
            min: Option::default(),
            max: Option::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.min.is_none()
    }

    fn is_degenerated(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        self.min()
            .eq(self.max())
            .array()
            .iter()
            .flatten()
            .all(|x| *x)
    }

    fn is_unbounded(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        self.min().array().iter().flatten().all(|x| *x == f64::MIN)
            && self.max().array().iter().flatten().all(|x| *x == f64::MAX)
    }

    fn from_min_max(min: V<BATCH, D>, max: V<BATCH, D>) -> Self {
        Self {
            min: Option::Some(min),
            max: Option::Some(max),
        }
    }

    fn min(&self) -> &Self::Point {
        self.try_min().unwrap()
    }
    fn max(&self) -> &Self::Point {
        &self.try_max().unwrap()
    }

    fn try_min(&self) -> Option<&V<BATCH, D>> {
        Some(self.min.as_ref()?)
    }

    fn try_max(&self) -> Option<&V<BATCH, D>> {
        Some(self.max.as_ref()?)
    }

    fn clamp(&self, p: V<BATCH, D>) -> V<BATCH, D> {
        p.maximum(self.min().clone().minimum(self.max().clone()))
    }

    fn range(&self) -> V<BATCH, D> {
        let mut p: V<BATCH, D> = dfdx::tensor::Cpu::default().zeros();
        if self.is_empty() {
            return p;
        }
        p = self.max().clone() - self.min().clone();
        p
    }

    fn mid(&self) -> V<BATCH, D> {
        self.min().clone() + self.range() * 0.5
    }
}

#[derive(Debug, Clone)]
pub struct IRegion<const BATCH: usize, const D: usize> {
    min: Option<IV<BATCH, D>>, // min.is_some() <=> max.is_some()
    max: Option<IV<BATCH, D>>,
}

impl<const BATCH: usize, const D: usize> IRegion<BATCH, D> {
    pub fn to_region(&self) -> Region<BATCH, D> {
        if self.is_empty() {
            return Region::empty();
        }
        // example: [2, 5] -> [1.5, 5.5]
        Region::from_min_max(
            self.min().clone().to_dtype() - 0.5,
            self.max().clone().to_dtype() + 0.5,
        )
    }
}

impl<const BATCH: usize, const D: usize> RegionTraits<BATCH, D> for IRegion<BATCH, D> {
    type Region = Self;
    type Point = IV<BATCH, D>;

    fn unbounded() -> Self {
        let s: IV<BATCH, D> = IV::<BATCH, D>::smallest();
        let l: IV<BATCH, D> = IV::<BATCH, D>::largest();
        Self::from_min_max(s, l)
    }

    fn empty() -> Self {
        Self {
            min: Option::default(),
            max: Option::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.min.is_none()
    }

    fn is_degenerated(&self) -> bool {
        false
    }

    fn is_unbounded(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        self.min().array().iter().flatten().all(|x| *x == i64::MIN)
            && self.max().array().iter().flatten().all(|x| *x == i64::MAX)
    }

    fn from_min_max(min: IV<BATCH, D>, max: IV<BATCH, D>) -> Self {
        Self {
            min: Option::Some(min),
            max: Option::Some(max),
        }
    }

    fn try_min(&self) -> Option<&IV<BATCH, D>> {
        Some(self.min.as_ref()?)
    }
    fn try_max(&self) -> Option<&IV<BATCH, D>> {
        Some(self.max.as_ref()?)
    }

    fn clamp(&self, p: IV<BATCH, D>) -> IV<BATCH, D> {
        p.to_dtype::<f64>()
            .maximum(
                self.min()
                    .clone()
                    .to_dtype::<f64>()
                    .minimum(self.max().clone().to_dtype::<f64>()),
            )
            .to_dtype()
    }

    fn range(&self) -> IV<BATCH, D> {
        let mut p: IV<BATCH, D> = dfdx::tensor::Cpu::default().zeros();
        if self.is_empty() {
            return p;
        }
        p = (self.max().clone().to_dtype::<f64>() - self.min().clone().to_dtype::<f64>())
            .to_dtype();
        p
    }

    fn mid(&self) -> IV<BATCH, D> {
        let half_range = self.range().to_dtype() * 0.5;
        (self.min().clone().to_dtype() + half_range).to_dtype()
    }

    fn min(&self) -> &Self::Point {
        self.try_min().unwrap()
    }

    fn max(&self) -> &Self::Point {
        &self.try_max().unwrap()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn batched_region<const BATCH: usize>() {
        let dev = dfdx::tensor::Cpu::default();

        let empty_f64 = Region::<BATCH, 1>::empty();
        assert!(empty_f64.is_empty());
        assert!(!empty_f64.is_degenerated());
        assert!(!empty_f64.is_proper());
        assert!(!empty_f64.is_unbounded());

        let unbounded = Region::<BATCH, 1>::unbounded();
        assert!(!unbounded.is_empty());
        assert!(!unbounded.is_degenerated());
        assert!(unbounded.is_proper());
        assert!(unbounded.is_unbounded());

        let one_i64 = IRegion::<BATCH, 1>::from_point(dev.ones());
        assert!(!one_i64.is_empty());
        assert!(!one_i64.is_degenerated());
        assert!(one_i64.is_proper());
        assert!(!one_i64.is_unbounded());

        let two_f64 = Region::<BATCH, 1>::from_point(dev.ones() * 2.0);
        assert!(!two_f64.is_empty());
        assert!(two_f64.is_degenerated());
        assert!(!two_f64.is_proper());
        assert!(!two_f64.is_unbounded());
    }

    #[test]
    fn region() {
        batched_region::<1>();
    }
}
