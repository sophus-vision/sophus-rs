use core::fmt::Debug;

use dyn_clone::DynClone;

use super::evaluated_ineq_constraint::EvaluatedIneqConstraint;
use crate::{
    block::{
        block_sparse_matrix_builder::BlockSparseMatrixBuilder,
        block_vector::BlockVector,
    },
    variables::{
        var_families::VarFamilies,
        VarKind,
    },
};

extern crate alloc;

/// Evaluated inequality constraints
#[derive(Debug, Clone)]
pub struct EvaluatedIneqSet<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const NUM_ARGS: usize,
> {
    /// one name (of the corresponding variable family) for each argument (of the cost function
    pub family_names: [String; NUM_ARGS],
    /// evaluated constraints
    pub evaluated_constraints:
        alloc::vec::Vec<EvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>>,
}

impl<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const NUM_ARGS: usize>
    EvaluatedIneqSet<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>
{
    /// create a new equality constraint set
    pub fn new(family_names: [String; NUM_ARGS]) -> Self {
        EvaluatedIneqSet {
            family_names,
            evaluated_constraints: alloc::vec::Vec::new(),
        }
    }

    /// number of variables of a given kind
    pub fn num_of_kind(&self, kind: VarKind, pool: &VarFamilies) -> usize {
        let mut c = 0;

        for name in self.family_names.iter() {
            c += if pool.collection.get(name).unwrap().get_var_kind() == kind {
                1
            } else {
                0
            };
        }
        c
    }
}

/// Is evaluated inequality constraint set
pub trait IsEvaluatedIneqConstraintSet: Debug + DynClone {
    /// sum of L1 norms of the <= residuals
    fn calc_le_sum_of_l1_norms(&self) -> f64;

    /// sum of L1 norms of the >= residuals
    fn calc_ge_sum_of_l1_norms(&self) -> f64;

    /// populate linear constraint system
    fn populate_linear_constraint_system(
        &self,
        variables: &VarFamilies,
        constraint_idx: usize,
        constraint_matrix: &mut BlockSparseMatrixBuilder,
        lower: &mut BlockVector,
        upper: &mut BlockVector,
    );
}
