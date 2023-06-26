use std::fmt::Debug;

use super::{
    block::{BlockMatrix, BlockVector},
    cost_args::{CostArgTuple, ManifoldV, ManifoldVTuple},
    tuple::SimpleTuple,
};

#[derive(Copy, Clone, Debug)]
pub struct CostTerm<const MAX_DIM: usize, const NUM_BLOCKS: usize> {
    pub hessian: BlockMatrix<MAX_DIM, NUM_BLOCKS>,
    pub gradient: BlockVector<MAX_DIM, NUM_BLOCKS>,
    pub cost: f64,
}

impl<const MAX_DIM: usize, const NUM_BLOCKS: usize> CostTerm<MAX_DIM, NUM_BLOCKS> {
    pub fn new(dims: [usize; NUM_BLOCKS]) -> Self {
        Self {
            hessian: BlockMatrix::new(&dims),
            gradient: BlockVector::new(&dims),
            cost: 0.0,
        }
    }
}

pub trait ResidualFn<const MAX_DIM: usize, const NUM_BLOCKS: usize>: Copy {
    type Args: CostArgTuple;
    type Constants;

    fn cost(&self, args: &Self::Args, constants: &Self::Constants)
        -> CostTerm<MAX_DIM, NUM_BLOCKS>;
}

pub trait CostTermSignature<const N: usize> {
    type Constants;
    //type EntityIndexTuple: SimpleTuple<usize>;

    const DOF_TUPLE: [i64; N];

    fn c_ref(&self) -> &Self::Constants;

    fn idx_ref(&self) -> &[usize; N];
}

pub fn apply<R, M, C, CC, AA: Debug, const NNN: usize, const NN: usize, const N: usize>(
    res_fn: R,
    cost_terms: &mut Vec<C>,
    families: &M,
) where
    R: ResidualFn<NNN, NN, Constants = CC, Args = AA>,
    M: ManifoldVTuple<Idx = [usize; N], Output = AA>,
    C: CostTermSignature<N, Constants = CC>,
    <M as ManifoldVTuple>::CatArray: AsRef<[char]>,
{
    println!("{:?}", M::CAT);
    use crate::opt::cost_args::Less;

    let less = Less { c: M::CAT };
    cost_terms.sort_by(|a, b| less.less_than(*a.idx_ref(), *b.idx_ref()));

    for t in cost_terms {
        let elem = families.get_elem(t.idx_ref());
        println!("{:?}", elem);

        res_fn.cost(&elem, t.c_ref());
    }
}
