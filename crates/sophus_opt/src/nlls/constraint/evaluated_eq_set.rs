use core::fmt::Debug;

use dyn_clone::DynClone;

use super::evaluated_eq_constraint::EvaluatedEqConstraint;
use crate::{
    block::{
        block_vector::BlockVector,
        symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder,
    },
    variables::VarFamilies,
};

extern crate alloc;

/// Evaluated eq constraints
#[derive(Debug, Clone)]
pub struct EvaluatedEqSet<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const NUM_ARGS: usize>
{
    /// one name (of the corresponding variable family) for each argument (of the cost function
    pub family_names: [String; NUM_ARGS],
    /// evaluated constraints
    pub evaluated_constraints:
        alloc::vec::Vec<EvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>>,
}

impl<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const NUM_ARGS: usize>
    EvaluatedEqSet<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>
{
    /// create a new equality constraint set
    pub fn new(family_names: [String; NUM_ARGS]) -> Self {
        EvaluatedEqSet {
            family_names,
            evaluated_constraints: alloc::vec::Vec::new(),
        }
    }
}

/// Is evaluated equality constraint set
pub trait IsEvaluatedEqConstraintSet: Debug + DynClone {
    /// sum of L1 norms of residuals
    fn calc_sum_of_l1_norms(&self) -> f64;

    /// populate upper triangular KKT matrix
    fn populate_upper_triangular_kkt_mat(
        &self,
        variables: &VarFamilies,
        lambda: &BlockVector,
        constraint_idx: usize,
        block_triplet: &mut SymmetricBlockSparseMatrixBuilder,
        b: &mut BlockVector,
    );
}
