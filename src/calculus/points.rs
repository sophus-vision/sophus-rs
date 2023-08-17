use dfdx::prelude::*;
use num_traits::Bounded;

use super::{batch_types::*, make::*};

pub trait PointTraits<const BATCH: usize, const D: usize>: Clone {
    fn smallest() -> Self;

    fn largest() -> Self;

    fn is_less_equal(&self, rhs: &Self) -> bool;

    fn is_greater_equal(&self, rhs: &Self) -> bool;
}

impl<const BATCH: usize, const D: usize> PointTraits<BATCH, D> for V<BATCH, D> {
    fn is_less_equal(&self, rhs: &Self) -> bool {
        self.le(rhs).array().iter().flatten().all(|x| *x)
    }

    fn is_greater_equal(&self, rhs: &Self) -> bool {
        self.ge(rhs).array().iter().flatten().all(|x| *x)
    }

    fn smallest() -> Self {
        let s: f64 = f64::min_value();
        let dev = dfdx::tensor::Cpu::default();
        dev.ones() * s
    }

    fn largest() -> Self {
        let l: f64 = f64::max_value();
        let dev = dfdx::tensor::Cpu::default();
        dev.ones() * l
    }
}

impl<const BATCH: usize, const D: usize> PointTraits<BATCH, D> for IV<BATCH, D> {
    fn is_less_equal(&self, rhs: &Self) -> bool {
        self.le(rhs).array().iter().flatten().all(|x| *x)
    }

    fn is_greater_equal(&self, rhs: &Self) -> bool {
        self.ge(rhs).array().iter().flatten().all(|x| *x)
    }

    fn smallest() -> Self {
        let s: i64 = i64::min_value();
        dfdx::tensor::Cpu::default().tensor([[s; D]; BATCH])
    }

    fn largest() -> Self {
        let l: i64 = i64::max_value();
        dfdx::tensor::Cpu::default().tensor([[l; D]; BATCH])
    }
}

pub fn tutil_point_examples<const BATCH: usize, const D: usize>() -> Vec<V<BATCH, D>> {
    let dev = dfdx::tensor::Cpu::default();
    vec![
        dev.zeros(),
        dev.ones() * 0.5,
        dev.ones().negate() * 0.5,
        dev.sample_uniform(),
        dev.sample_uniform(),
    ]
}

pub fn tutil_points_examples<const BATCH: usize, const D: usize, const NUM_POINTS: usize>(
) -> Vec<M<BATCH, D, NUM_POINTS>> {
    let dev = dfdx::tensor::Cpu::default();
    vec![
        dev.zeros(),
        dev.ones() * 0.5,
        dev.ones().negate() * 0.5,
        dev.sample_uniform(),
        dev.sample_uniform(),
    ]
}

pub fn normalized<const BATCH: usize, const DIM: usize>(v: V<BATCH, DIM>) -> V<BATCH, DIM> {
    let nrm: S<BATCH> = v.clone().square().sum::<_, Axis<1>>().sqrt();
    v / nrm.broadcast()
}

pub fn cross<const BATCH: usize, LhsTape: SophusTape + Merge<RhsTape>, RhsTape: SophusTape>(
    a: GenV<BATCH, 3, LhsTape>,
    b: GenV<BATCH, 3, RhsTape>,
) -> GenV<BATCH, 3, LhsTape> {
    let a_x: GenV<BATCH, 1, LhsTape> = a.clone().slice((.., 0..1)).realize();
    let a_y: GenV<BATCH, 1, LhsTape> = a.clone().slice((.., 1..2)).realize();
    let a_z: GenV<BATCH, 1, LhsTape> = a.slice((.., 2..3)).realize();

    let b_x: GenV<BATCH, 1, RhsTape> = b.clone().slice((.., 0..1)).realize();
    let b_y: GenV<BATCH, 1, RhsTape> = b.clone().slice((.., 1..2)).realize();
    let b_z: GenV<BATCH, 1, RhsTape> = b.slice((.., 2..3)).realize();

    make_blockvec3(
        a_y.clone() * b_z.clone() - a_z.clone() * b_y.clone(),
        a_z * b_x.clone() - a_x.clone() * b_z,
        a_x * b_y - a_y * b_x,
    )
}

pub fn unit_x<const BATCH: usize>() -> V<BATCH, 3> {
    let dev = dfdx::tensor::Cpu::default();
    let o: V<BATCH, 1> = dev.zeros();
    let l: V<BATCH, 1> = dev.ones();

    make_blockvec3(l, o.clone(), o)
}

pub fn unit_y<const BATCH: usize>() -> V<BATCH, 3> {
    let dev = dfdx::tensor::Cpu::default();
    let o: V<BATCH, 1> = dev.zeros();
    let l: V<BATCH, 1> = dev.ones();

    make_blockvec3(o.clone(), l, o)
}

pub fn unit_z<const BATCH: usize>() -> V<BATCH, 3> {
    let dev = dfdx::tensor::Cpu::default();
    let o: V<BATCH, 1> = dev.zeros();
    let l: V<BATCH, 1> = dev.ones();

    make_blockvec3(o.clone(), o, l)
}

pub fn Identity3<const BATCH: usize>() -> M<BATCH, 3,3> {

    make_3colvec_mat(unit_x(), unit_y(), unit_z())

}