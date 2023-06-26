use std::fmt::Debug;

use crate::{lie::rotation3::Isometry3, manifold::traits::Manifold};
use sophus_macros::create_impl;

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
    type Idx;
    type Output: Debug;
    type CatArray: Debug;
    const CAT: Self::CatArray;

    fn get_elem(&self, idx: &Self::Idx) -> Self::Output;
}

impl<M0: ManifoldV> ManifoldVTuple for M0 {
    type Idx = [usize; 1];
    type Output = (M0::Arg, ());
    type CatArray = [char; 1];
    const CAT: Self::CatArray = [M0::CAT];

    fn get_elem(&self, idx: &Self::Idx) -> Self::Output {
        (self.get_elem(idx[0]), ())
    }
}

impl<M0: ManifoldV, M1: ManifoldV> ManifoldVTuple for (M0, M1) {
    type Idx = [usize; 2];
    type Output = (M0::Arg, (M1::Arg, ()));
    type CatArray = [char; 2];
    const CAT: Self::CatArray = [M0::CAT, M1::CAT];

    fn get_elem(&self, idx: &Self::Idx) -> Self::Output {
        (self.0.get_elem(idx[0]), (self.1.get_elem(idx[1]), ()))
    }
}

impl<M0: ManifoldV, M1: ManifoldV, M2: ManifoldV> ManifoldVTuple for (M0, M1, M2) {
    type Idx = [usize; 3];
    type Output = (M0::Arg, (M1::Arg, (M2::Arg, ())));
    type CatArray = [char; 3];
    const CAT: Self::CatArray = [M0::CAT, M1::CAT, M2::CAT];

    fn get_elem(&self, idx: &Self::Idx) -> Self::Output {
        (
            self.0.get_elem(idx[0]),
            (self.1.get_elem(idx[1]), (self.2.get_elem(idx[2]), ())),
        )
    }
}

const fn less_than<const N: usize>(
    c: &[char],
    lhs: [usize; N],
    rhs: [usize; N],
) -> std::cmp::Ordering {
    let mut permutation: [usize; N] = [0; N];

    let mut v_count = 0;
    let mut h = 0;
    loop {
        if h >= N {
            break;
        }
        if c[h] == 'm' {
            permutation[h] = v_count;
            v_count += 1;
        }
        h += 1;
    }

    let mut i = 0;
    loop {
        if i >= N {
            break;
        }
        if c[i] == 'v' {
            permutation[i] = v_count;
            v_count += 1;
        }
        i += 1;
    }
    let mut j = 0;
    loop {
        if j >= N {
            break;
        }
        if c[j] == 'c' {
            permutation[j] = v_count;
            v_count += 1;
        }
        j += 1;
    }

    let mut permuted_lhs: [usize; N] = [0; N];
    let mut permuted_rhs: [usize; N] = [0; N];

    let mut k = 0;
    loop {
        if k >= N {
            break;
        }
        permuted_lhs[permutation[k]] = lhs[k];
        permuted_rhs[permutation[k]] = rhs[k];
        k += 1;
    }

    let mut l = 0;
    loop {
        if l >= N {
            break;
        }
        if permuted_lhs[l] < permuted_rhs[l] {
            return std::cmp::Ordering::Less;
        } else if permuted_lhs[l] > permuted_rhs[l] {
            return std::cmp::Ordering::Greater;
        }
        l += 1;
    }
    return std::cmp::Ordering::Equal;
}

pub struct Less<C>
where
    C: AsRef<[char]>,
{
    pub(crate) c: C,
}

impl<C> Less<C>
where
    C: AsRef<[char]>,
{
    pub fn less_than<const N: usize>(
        &self,
        lhs: [usize; N],
        rhs: [usize; N],
    ) -> std::cmp::Ordering {
        less_than(self.c.as_ref(), lhs, rhs)
    }
}

mod test {

    use crate::opt::cost_args::Less;

    use super::less_than;

    #[test]
    fn test() {
        // Length 2
        const vv: [char; 2] = ['v', 'v'];
        const vc: [char; 2] = ['v', 'c'];
        const cv: [char; 2] = ['c', 'v'];
        const cc: [char; 2] = ['c', 'c'];

        assert_eq!(less_than(&vv, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(less_than(&vv, [1, 0], [0, 0]), std::cmp::Ordering::Greater);
        assert_eq!(less_than(&vc, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(less_than(&vc, [1, 0], [0, 0]), std::cmp::Ordering::Greater);
        assert_eq!(less_than(&cv, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(less_than(&cv, [1, 0], [0, 0]), std::cmp::Ordering::Greater);
        assert_eq!(less_than(&cc, [0, 0], [0, 0]), std::cmp::Ordering::Equal);

        const mv: [char; 2] = ['m', 'v'];
        const vm: [char; 2] = ['v', 'm'];
        const mm: [char; 2] = ['m', 'm'];

        assert_eq!(less_than(&mv, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(less_than(&vm, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(less_than(&mm, [0, 0], [0, 0]), std::cmp::Ordering::Equal);

        const vvv: [char; 3] = ['v', 'v', 'v'];

        assert_eq!(
            less_than(&vvv, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&vvv, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&vvv, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&vvv, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&vvv, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&vvv, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const cvv: [char; 3] = ['c', 'v', 'v'];

        assert_eq!(
            less_than(&cvv, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&cvv, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvv, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvv, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvv, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvv, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const cvc: [char; 3] = ['c', 'v', 'c'];
        assert_eq!(
            less_than(&cvc, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&cvc, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvc, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvc, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&cvc, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvc, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const ccv: [char; 3] = ['c', 'c', 'v'];

        assert_eq!(
            less_than(&ccv, [0, 0, 0], [0, 0, 0]),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            less_than(&ccv, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&ccv, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&ccv, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&ccv, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&ccv, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&ccv, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const cvm: [char; 3] = ['c', 'v', 'm'];
        assert_eq!(
            less_than(&cvm, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&cvm, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvm, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvm, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvm, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&cvm, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        // Length 4
        const vvvv: [char; 4] = ['v', 'v', 'v', 'v'];
        const cvvv: [char; 4] = ['c', 'v', 'v', 'v'];
        const ccvv: [char; 4] = ['c', 'c', 'v', 'v'];
        const cvcv: [char; 4] = ['c', 'v', 'c', 'v'];
        const vvcc: [char; 4] = ['v', 'v', 'c', 'c'];
        const ccvc: [char; 4] = ['c', 'c', 'v', 'c'];
        const vccv: [char; 4] = ['v', 'c', 'c', 'v'];

        assert_eq!(
            less_than(&vvvv, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&vvvv, [1, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvvv, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&cvvv, [1, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&ccvv, [0, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            less_than(&ccvv, [0, 1, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&cvcv, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&cvcv, [1, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&vvcc, [0, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            less_than(&vvcc, [0, 0, 0, 1], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&ccvc, [0, 0, 0, 0], [0, 1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&ccvc, [0, 1, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&vccv, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&vccv, [0, 0, 1, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );

        const mvvm: [char; 4] = ['m', 'v', 'v', 'm'];
        const mmvv: [char; 4] = ['m', 'm', 'v', 'v'];
        const vmmv: [char; 4] = ['v', 'm', 'm', 'v'];
        const mmmm: [char; 4] = ['m', 'm', 'm', 'm'];

        assert_eq!(
            less_than(&mvvm, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&mmvv, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&vmmv, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&mmmm, [0, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Equal
        );

        let c: [char; 3] = ['v', 'v', 'c'];
        let mut l: Vec<[usize; 3]> = vec![
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1],
        ];

        let less = Less { c };
        l.sort_by(|a, b| less.less_than(*a, *b));

        println!("{:?}", l);
    }
}
