use super::evaluated_term::EvaluatedCostTerm;
use crate::block::block_vector::BlockVector;
use crate::block::symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder;
use crate::variables::var_families::VarFamilies;
use crate::variables::VarKind;
use core::fmt::Debug;
use dyn_clone::DynClone;
extern crate alloc;

/// Generic evaluated cost.
#[derive(Debug, Clone)]
pub struct EvaluatedCost<const NUM: usize, const NUM_ARGS: usize> {
    /// one name (of the corresponding variable family) for each argument (of the cost function)
    pub family_names: [String; NUM_ARGS],
    /// evaluated terms of the overall cost
    pub terms: alloc::vec::Vec<EvaluatedCostTerm<NUM, NUM_ARGS>>,
}

impl<const NUM: usize, const NUM_ARGS: usize> EvaluatedCost<NUM, NUM_ARGS> {
    /// create a new evaluated cost
    pub fn new(family_names: [String; NUM_ARGS]) -> Self {
        EvaluatedCost {
            family_names,
            terms: alloc::vec::Vec::new(),
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

/// Evaluated cost.
pub trait IsEvaluatedCost: Debug + DynClone {
    /// squared error
    fn calc_square_error(&self) -> f64;

    /// populate upper triangular normal equation
    fn populate_upper_triangulatr_normal_equation(
        &self,
        variables: &VarFamilies,
        nu: f64,
        hessian_block_triplet: &mut SymmetricBlockSparseMatrixBuilder,
        neg_grad: &mut BlockVector,
    );
}
