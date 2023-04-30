use std::fmt::Debug;

use super::{
    block::{BlockMatrix, BlockVector},
    cost_args::{CostArgTuple, ManifoldVTuple},
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
    type EntityIndexTuple: SimpleTuple<usize>;

    const DOF_TUPLE: [i64; N];

    fn c_ref(&self) -> &Self::Constants;

    fn idx_ref(&self) -> &Self::EntityIndexTuple;
}

pub fn apply<R, I, M, C, CC, AA: Debug, const NNN: usize, const NN: usize, const N: usize>(
    res_fn: R,
    cost_terms: &Vec<C>,
    families: &M,
) where
    R: ResidualFn<NNN, NN, Constants = CC, Args = AA>,
    I: SimpleTuple<usize>,
    M: ManifoldVTuple<EntityIndexTuple = I, ArgTuple = AA>,
    C: CostTermSignature<N, EntityIndexTuple = I, Constants = CC>,
{
    println!("{:?}", M::get_var_at());
    //println!("{:?}", M::get_dofs());

    for t in cost_terms {
        println!("{:?}", families.get_elem(t.idx_ref()));

        res_fn.cost(&families.get_elem(t.idx_ref()), t.c_ref());
    }
}
