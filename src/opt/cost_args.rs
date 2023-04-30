use std::fmt::Debug;

use crate::{lie::rotation3::Isometry3, manifold::traits::Manifold};

use super::tuple::SimpleTuple;

#[derive(Copy, Clone, Debug)]
pub struct CostTermArg<T, const ET: char> {
    pub arg: T,
}

pub trait CostArg: std::fmt::Debug {
    const ENTITY_TYPE: char;
    const DOF: usize;
}

impl<const ET: char> CostArg for CostTermArg<f64, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 1;
}

impl<const ET: char> CostArg for CostTermArg<nalgebra::Vector1<f64>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 1;
}

impl<const ET: char> CostArg for CostTermArg<nalgebra::Vector2<f64>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 2;
}

impl<const ET: char> CostArg for CostTermArg<nalgebra::Vector3<f64>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 3;
}

impl<const ET: char> CostArg for CostTermArg<nalgebra::Vector6<f64>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 6;
}

impl<const ET: char> CostArg for CostTermArg<Isometry3, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 6;
}

pub trait CostArgTuple: std::fmt::Debug {
    const N: usize;

    fn populate_dims(arr: &mut [usize], i: usize);

    fn get_dims<const M: usize>() -> [usize; M];
}

// Now we have to implement trait for an empty tuple,
// thus defining initial condition.
impl CostArgTuple for () {
    const N: usize = 0;

    fn populate_dims(arr: &mut [usize], i: usize) {
        assert!(arr.len() == i);
    }

    fn get_dims<const M: usize>() -> [usize; M] {
        [0; M]
    }
}

// Now we can implement trait for a non-empty tuple list,
// thus defining recursion and supporting tuple lists of arbitrary length.
impl<Head, Tail> CostArgTuple for (Head, Tail)
where
    Head: CostArg,
    Tail: CostArgTuple,
{
    const N: usize = 1 + Tail::N;

    fn populate_dims(arr: &mut [usize], i: usize) {
        arr[i] = if Head::ENTITY_TYPE == 'v' {
            Head::DOF
        } else {
            0
        };
        Tail::populate_dims(arr, i + 1);
    }

    fn get_dims<const M: usize>() -> [usize; M] {
        let mut arr = [0; M];
        Self::populate_dims(&mut arr, 0);
        arr
    }
}

pub struct CostFnArg<'a, const DOF: usize, T, const AT: char> {
    v: &'a Vec<T>,
}

impl<'a, const DOF: usize, T: Manifold<DOF>, const AT: char> CostFnArg<'a, DOF, T, AT> {}

impl<'a, const DOF: usize, T: Manifold<DOF>> CostFnArg<'a, DOF, T, 'v'> {
    pub fn var(v: &'a Vec<T>) -> CostFnArg<'a, DOF, T, 'v'> {
        CostFnArg::<'a, DOF, T, 'v'> { v }
    }
}

impl<'a, const DOF: usize, T: Manifold<DOF>> CostFnArg<'a, DOF, T, 'c'> {
    pub fn cond(v: &'a Vec<T>) -> CostFnArg<'a, DOF, T, 'c'> {
        CostFnArg::<'a, DOF, T, 'c'> { v }
    }
}

pub trait ManifoldV {
    type Arg: CostArg;
    const CAT: char;

    fn get_elem(&self, idx: usize) -> Self::Arg;
}

impl<'a, const AT: char> ManifoldV for CostFnArg<'a, 1, nalgebra::Vector1<f64>, AT> {
    type Arg = CostTermArg<nalgebra::Vector1<f64>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg { arg: self.v[idx] }
    }

    const CAT: char = AT;
}
impl<'a, const AT: char> ManifoldV for CostFnArg<'a, 2, nalgebra::Vector2<f64>, AT> {
    type Arg = CostTermArg<nalgebra::Vector2<f64>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg { arg: self.v[idx] }
    }
    const CAT: char = AT;
}

impl<'a, const AT: char> ManifoldV for CostFnArg<'a, 3, nalgebra::Vector3<f64>, AT> {
    type Arg = CostTermArg<nalgebra::Vector3<f64>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg { arg: self.v[idx] }
    }
    const CAT: char = AT;
}

impl<'a, const AT: char> ManifoldV for CostFnArg<'a, 6, nalgebra::Vector6<f64>, AT> {
    type Arg = CostTermArg<nalgebra::Vector6<f64>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg { arg: self.v[idx] }
    }
    const CAT: char = AT;
}

impl<'a, const AT: char> ManifoldV for CostFnArg<'a, 6, Isometry3, AT> {
    type Arg = CostTermArg<Isometry3, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg { arg: self.v[idx] }
    }
    const CAT: char = AT;
}

pub trait ManifoldVTuple {
    const N: usize;
    type ArgTuple: CostArgTuple;
    type EntityIndexTuple: SimpleTuple<usize>;

    type C: SimpleTuple<char>;

    //  fn get_dofs() -> Self::EntityIndexTuple;

    fn get_elem(&self, idx: &Self::EntityIndexTuple) -> Self::ArgTuple;

    fn get_var_at() -> Self::C;
}

// Now we have to implement trait for an empty tuple,
// thus defining initial condition.
impl ManifoldVTuple for () {
    //fn plus_one(&mut self) {}
    const N: usize = 0;

    type ArgTuple = ();
    type EntityIndexTuple = ();
    type C = ();

    //fn get_dofs() -> () {}

    fn get_elem(&self, _idx: &()) {}

    fn get_var_at() {}
}

// Now we can implement trait for a non-empty tuple list,
// thus defining recursion and supporting tuple lists of arbitrary length.
impl<Head, Tail> ManifoldVTuple for (Head, Tail)
where
    Head: ManifoldV,
    Tail: ManifoldVTuple,
{
    const N: usize = 1 + Tail::N;
    type ArgTuple = (Head::Arg, Tail::ArgTuple);
    type EntityIndexTuple = (usize, Tail::EntityIndexTuple);
    type C = (char, Tail::C);

    fn get_elem(&self, idx: &(usize, Tail::EntityIndexTuple)) -> Self::ArgTuple {
        (self.0.get_elem(idx.0), self.1.get_elem(&idx.1))
    }

    fn get_var_at() -> Self::C {
        (Head::CAT, Tail::get_var_at())
    }
}
