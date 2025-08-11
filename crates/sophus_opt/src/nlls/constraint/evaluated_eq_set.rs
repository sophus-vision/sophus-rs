use core::fmt::Debug;

use dyn_clone::DynClone;
use sophus_block::{
    BlockVector,
    SymmetricBlockSparseMatrix,
};

use super::evaluated_eq_constraint::EvaluatedEqConstraint;
use crate::variables::VarFamilies;

extern crate alloc;

/// Evaluated equality constraint set:
/// `{c(V⁰₀, V⁰₁, ..., V⁰ₙ₋₁), ..., c(Vⁱ₀, Vⁱ₁, ..., Vⁱₙ₋₁), ...}`.
///
/// ## Generic parameters
///
///  * `INPUT_DIM`
///    - Total input dimension of the constraint residual function `c`. It is the sum of argument
///      dimensions: `|Vⁱ₀| + |Vⁱ₁| + ... + |Vⁱₙ₋₁|`.
///  * `N`
///    - Number of arguments of the residual function `c`.
#[derive(Debug, Clone)]
pub struct EvaluatedEqSet<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const N: usize> {
    /// Variable family name for each argument.
    pub family_names: [String; N],
    /// Collection of evaluated constraints.
    pub evaluated_constraints: alloc::vec::Vec<EvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, N>>,
}

impl<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const N: usize>
    EvaluatedEqSet<RESIDUAL_DIM, INPUT_DIM, N>
{
    /// Create a new evaluated constraint set.
    pub fn new(family_names: [String; N]) -> Self {
        EvaluatedEqSet {
            family_names,
            evaluated_constraints: alloc::vec::Vec::new(),
        }
    }
}

/// Evaluated equality constraint set trait.
pub trait IsEvaluatedEqConstraintSet: Debug + DynClone {
    /// Return sum of L1 norms of constraint residuals: `∑ᵢ |c(Vⁱ₀, Vⁱ₁, ..., Vⁱₙ₋₁)|`.
    fn calc_sum_of_l1_norms(&self) -> f64;

    /// Populate upper triangular KKT matrix.
    fn populate_upper_triangular_kkt_mat(
        &self,
        variables: &VarFamilies,
        lambda: &BlockVector,
        constraint_idx: usize,
        block_triplet: &mut SymmetricBlockSparseMatrix,
        b: &mut BlockVector,
    );
}
