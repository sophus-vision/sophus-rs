use std::fmt::Debug;
use std::mem::swap;

use crate::calculus::batch_types::*;
use crate::lie::isometry2::*;
use crate::manifold::traits::*;
use dfdx::prelude::*;
//use sophus_macros::create_impl;

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

impl<const ET: char> CostArg for CostTermArg<V<1, 1>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 1;
}

impl<const ET: char> CostArg for CostTermArg<V<1, 2>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 2;
}

impl<const ET: char> CostArg for CostTermArg<V<1, 3>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 3;
}

impl<const ET: char> CostArg for CostTermArg<Isometry2<1>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 3;
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

#[derive(Debug)]
pub struct CostFnArg<const DOF: usize, T: Manifold<DOF>, const AT: char> {
    pub v: Vec<T>,
}

impl<const DOF: usize, T: Manifold<DOF>, const AT: char> CostFnArg<DOF, T, AT> {}

impl<const DOF: usize, T: Manifold<DOF>> CostFnArg<DOF, T, 'v'> {
    pub fn var(v: Vec<T>) -> CostFnArg<DOF, T, 'v'> {
        CostFnArg::<DOF, T, 'v'> { v }
    }
}

impl<const DOF: usize, T: Manifold<DOF>> CostFnArg<DOF, T, 'c'> {
    pub fn cond(v: Vec<T>) -> CostFnArg<DOF, T, 'c'> {
        CostFnArg::<DOF, T, 'c'> { v }
    }
}

impl<const DOF: usize, T: Manifold<DOF>> CostFnArg<DOF, T, 'm'> {
    pub fn marg(v: Vec<T>) -> CostFnArg<DOF, T, 'm'> {
        CostFnArg::<DOF, T, 'm'> { v }
    }
}

pub trait ManifoldV {
    type Arg: CostArg;
    const CAT: char;
    const DOF: usize;

    fn get_elem(&self, idx: usize) -> Self::Arg;

    fn len(&self) -> usize;

    fn update(&mut self, delta: &[f64]);
}

impl<const AT: char> ManifoldV for CostFnArg<1, V<1, 1>, AT> {
    type Arg = CostTermArg<V<1, 1>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg {
            arg: self.v[idx].clone(),
        }
    }

    fn len(&self) -> usize {
        self.v.len()
    }

    const CAT: char = AT;
    const DOF: usize = 1;

    fn update(&mut self, delta: &[f64]) {
        assert!(self.v.len() * Self::DOF == delta.len());
        let mut i = 0;
        let dev = dfdx::tensor::Cpu::default();
        for e in self.v.iter_mut() {
            let u: Tensor<(Const<1>, _), _, _> = dev.tensor_from_vec(
                delta[i * Self::DOF..(i + 1) * Self::DOF].to_vec(),
                (Const, Self::DOF),
            );
            let mut c = e.clone().into_oplus(u.realize());
            swap(e, &mut c);
            i += 1;
        }
    }
}
impl<const AT: char> ManifoldV for CostFnArg<2, V<1, 2>, AT> {
    type Arg = CostTermArg<V<1, 2>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg {
            arg: self.v[idx].clone(),
        }
    }

    fn len(&self) -> usize {
        self.v.len()
    }

    const CAT: char = AT;
    const DOF: usize = 2;

    fn update(&mut self, delta: &[f64]) {
        assert!(self.v.len() * Self::DOF == delta.len());
        let mut i = 0;
        let dev = dfdx::tensor::Cpu::default();
        for e in self.v.iter_mut() {
            let u: Tensor<(Const<1>, _), _, _> = dev.tensor_from_vec(
                delta[i * Self::DOF..(i + 1) * Self::DOF].to_vec(),
                (Const, Self::DOF),
            );
            let mut c = e.clone().into_oplus(u.realize());
            swap(e, &mut c);
            i += 1;
        }
    }
}

impl<const AT: char> ManifoldV for CostFnArg<3, V<1, 3>, AT> {
    type Arg = CostTermArg<V<1, 3>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg {
            arg: self.v[idx].clone(),
        }
    }

    fn len(&self) -> usize {
        self.v.len()
    }

    const CAT: char = AT;
    const DOF: usize = 3;

    fn update(&mut self, delta: &[f64]) {
        assert!(self.v.len() * Self::DOF == delta.len());
        let mut i = 0;
        let dev = dfdx::tensor::Cpu::default();
        for e in self.v.iter_mut() {
            let u: Tensor<(Const<1>, _), _, _> = dev.tensor_from_vec(
                delta[i * Self::DOF..(i + 1) * Self::DOF].to_vec(),
                (Const, Self::DOF),
            );
            let mut c = e.clone().into_oplus(u.realize());
            swap(e, &mut c);
            i += 1;
        }
    }
}

impl<const AT: char> ManifoldV for CostFnArg<3, Isometry2<1>, AT> {
    type Arg = CostTermArg<Isometry2<1>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg {
            arg: self.v[idx].clone(),
        }
    }

    fn len(&self) -> usize {
        self.v.len()
    }

    const CAT: char = AT;
    const DOF: usize = 3;

   fn update(&mut self, delta: &[f64]) {
        assert!(self.v.len() * Self::DOF == delta.len());
        let mut i = 0;
        let dev = dfdx::tensor::Cpu::default();
        for e in self.v.iter_mut() {
            let u: Tensor<(Const<1>, _), _, _> = dev.tensor_from_vec(
                delta[i * Self::DOF..(i + 1) * Self::DOF].to_vec(),
                (Const, Self::DOF),
            );
            let mut c = e.clone().into_oplus(u.realize());
            swap(e, &mut c);
            i += 1;
        }
    }
}

pub trait ManifoldVTuple {
    type Idx;
    type GetElemTReturn: Debug;
    type CatArray: Debug;
    const CAT: Self::CatArray;
    type DofArray: Debug;
    const DOF_T: Self::DofArray;

    fn len_t(&self) -> Self::Idx;
    fn get_elem_t(&self, idx: &Self::Idx) -> Self::GetElemTReturn;

    fn update_t(&mut self, delta: &[f64]);
}

impl<M0: ManifoldV> ManifoldVTuple for M0 {
    type Idx = [usize; 1];
    type GetElemTReturn = (M0::Arg, ());
    type CatArray = [char; 1];
    const CAT: Self::CatArray = [M0::CAT];
    type DofArray = [usize; 1];
    const DOF_T: Self::DofArray = [M0::DOF];

    fn get_elem_t(&self, idx: &Self::Idx) -> Self::GetElemTReturn {
        (self.get_elem(idx[0]), ())
    }

    fn len_t(&self) -> Self::Idx {
        [self.len()]
    }

    fn update_t(&mut self, delta: &[f64]) {
        self.update(delta)
    }
}

impl<M0: ManifoldV, M1: ManifoldV> ManifoldVTuple for (M0, M1) {
    type Idx = [usize; 2];
    type GetElemTReturn = (M0::Arg, (M1::Arg, ()));
    type CatArray = [char; 2];
    const CAT: Self::CatArray = [M0::CAT, M1::CAT];
    type DofArray = [usize; 2];
    const DOF_T: Self::DofArray = [M0::DOF, M1::DOF];

    fn get_elem_t(&self, idx: &Self::Idx) -> Self::GetElemTReturn {
        (self.0.get_elem(idx[0]), (self.1.get_elem(idx[1]), ()))
    }

    fn len_t(&self) -> Self::Idx {
        [self.0.len(), self.1.len()]
    }

    fn update_t(&mut self, delta: &[f64]) {
        let offset_1 = self.0.len() * Self::DOF_T[0];
        self.0.update(&delta[..offset_1]);
        self.1.update(&delta[offset_1..]);
    }
}

impl<M0: ManifoldV, M1: ManifoldV, M2: ManifoldV> ManifoldVTuple for (M0, M1, M2) {
    type Idx = [usize; 3];
    type GetElemTReturn = (M0::Arg, (M1::Arg, (M2::Arg, ())));
    type CatArray = [char; 3];
    const CAT: Self::CatArray = [M0::CAT, M1::CAT, M2::CAT];
    type DofArray = [usize; 3];
    const DOF_T: Self::DofArray = [M0::DOF, M1::DOF, M2::DOF];

    fn get_elem_t(&self, idx: &Self::Idx) -> Self::GetElemTReturn {
        (
            self.0.get_elem(idx[0]),
            (self.1.get_elem(idx[1]), (self.2.get_elem(idx[2]), ())),
        )
    }

    fn len_t(&self) -> Self::Idx {
        [self.0.len(), self.1.len(), self.2.len()]
    }

    fn update_t(&mut self, delta: &[f64]) {
        let offset_1 = self.0.len() * Self::DOF_T[0];
        let offset_2 = self.1.len() * Self::DOF_T[1];

        self.0.update(&delta[..offset_1]);
        self.1.update(&delta[offset_1..offset_2]);
        self.2.update(&delta[offset_2..]);
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

pub struct CompareIdx<C>
where
    C: AsRef<[char]>,
{
    pub(crate) c: C,
}

impl<C> CompareIdx<C>
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

    pub fn all_var_eq(&self, lhs: &[usize], rhs: &[usize]) -> bool {
        let mut i = 0;
        loop {
            if i >= lhs.len() {
                break;
            }
            if self.c.as_ref()[i] == 'v' && lhs[i] != rhs[i] {
                return false;
            }
            i += 1;
        }
        return true;
    }
}

mod test {

    use crate::opt::cost_args::CompareIdx;

    use super::less_than;

    #[test]
    fn test() {
        // Length 2
        const VV: [char; 2] = ['v', 'v'];
        const VC: [char; 2] = ['v', 'c'];
        const CV: [char; 2] = ['c', 'v'];
        const CC: [char; 2] = ['c', 'c'];

        assert_eq!(less_than(&VV, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(less_than(&VV, [1, 0], [0, 0]), std::cmp::Ordering::Greater);
        assert_eq!(less_than(&VC, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(less_than(&VC, [1, 0], [0, 0]), std::cmp::Ordering::Greater);
        assert_eq!(less_than(&CV, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(less_than(&CV, [1, 0], [0, 0]), std::cmp::Ordering::Greater);
        assert_eq!(less_than(&CC, [0, 0], [0, 0]), std::cmp::Ordering::Equal);

        const MV: [char; 2] = ['m', 'v'];
        const VM: [char; 2] = ['v', 'm'];
        const MM: [char; 2] = ['m', 'm'];

        assert_eq!(less_than(&MV, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(less_than(&VM, [0, 0], [1, 0]), std::cmp::Ordering::Less);
        assert_eq!(less_than(&MM, [0, 0], [0, 0]), std::cmp::Ordering::Equal);

        const VVV: [char; 3] = ['v', 'v', 'v'];

        assert_eq!(
            less_than(&VVV, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&VVV, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&VVV, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&VVV, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&VVV, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&VVV, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const CVV: [char; 3] = ['c', 'v', 'v'];

        assert_eq!(
            less_than(&CVV, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CVV, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVV, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVV, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVV, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVV, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const CVC: [char; 3] = ['c', 'v', 'c'];
        assert_eq!(
            less_than(&CVC, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CVC, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVC, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVC, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CVC, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVC, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const CCV: [char; 3] = ['c', 'c', 'v'];

        assert_eq!(
            less_than(&CCV, [0, 0, 0], [0, 0, 0]),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            less_than(&CCV, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CCV, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CCV, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CCV, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CCV, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CCV, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        const CVM: [char; 3] = ['c', 'v', 'm'];
        assert_eq!(
            less_than(&CVM, [0, 0, 0], [1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CVM, [0, 2, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVM, [0, 1, 0], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVM, [0, 0, 1], [1, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVM, [0, 1, 0], [0, 0, 1]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CVM, [0, 0, 1], [0, 0, 2]),
            std::cmp::Ordering::Less
        );

        // Length 4
        const VVVV: [char; 4] = ['v', 'v', 'v', 'v'];
        const CVVV: [char; 4] = ['c', 'v', 'v', 'v'];
        const CCVV: [char; 4] = ['c', 'c', 'v', 'v'];
        const CVCV: [char; 4] = ['c', 'v', 'c', 'v'];
        const VVCC: [char; 4] = ['v', 'v', 'c', 'c'];
        const CCVC: [char; 4] = ['c', 'c', 'v', 'c'];
        const VCCV: [char; 4] = ['v', 'c', 'c', 'v'];

        assert_eq!(
            less_than(&VVVV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&VVVV, [1, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVVV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CVVV, [1, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CCVV, [0, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            less_than(&CCVV, [0, 1, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CVCV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CVCV, [1, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&VVCC, [0, 0, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            less_than(&VVCC, [0, 0, 0, 1], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&CCVC, [0, 0, 0, 0], [0, 1, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&CCVC, [0, 1, 0, 0], [0, 0, 0, 0]),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            less_than(&VCCV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&VCCV, [0, 0, 1, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );

        const MVVM: [char; 4] = ['m', 'v', 'v', 'm'];
        const MMVV: [char; 4] = ['m', 'm', 'v', 'v'];
        const VMMV: [char; 4] = ['v', 'm', 'm', 'v'];
        const MMMM: [char; 4] = ['m', 'm', 'm', 'm'];

        assert_eq!(
            less_than(&MVVM, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&MMVV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&VMMV, [0, 0, 0, 0], [1, 0, 0, 0]),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            less_than(&MMMM, [0, 0, 0, 0], [0, 0, 0, 0]),
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

        let less = CompareIdx { c };
        l.sort_by(|a, b| less.less_than(*a, *b));

        println!("{:?}", l);
    }
}
