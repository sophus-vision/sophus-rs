use core::fmt::Debug;

use dyn_clone::DynClone;
use sophus_solver::{
    BlockVector,
    SymmetricMatrixBuilderEnum,
};

use super::evaluated_term::EvaluatedCostTerm;
use crate::variables::VarFamilies;
extern crate alloc;

/// Evaluated non-linear least squares cost.
///
/// ## Generic parameters
///
///  * `INPUT_DIM`
///    - Total input dimension of the residual function `g`. It is the sum of argument dimensions:
///      `|Vⁱ₀| + |Vⁱ₁| + ... + |Vⁱₙ₋₁|`.
///  * `N`
///    - Number of arguments of the residual function `g`.
#[derive(Debug, Clone)]
pub struct EvaluatedCost<const INPUT_DIM: usize, const N: usize> {
    /// Variable family name for each argument.
    pub family_names: [String; N],
    /// Collection of evaluated terms.
    pub terms: alloc::vec::Vec<EvaluatedCostTerm<INPUT_DIM, N>>,
}

impl<const INPUT_DIM: usize, const N: usize> EvaluatedCost<INPUT_DIM, N> {
    /// Create a new evaluated cost.
    pub fn new(family_names: [String; N]) -> Self {
        EvaluatedCost {
            family_names,
            terms: alloc::vec::Vec::new(),
        }
    }
}

/// Evaluated non-linear least squares cost trait.
pub trait IsEvaluatedCost: Debug + DynClone {
    /// Return the sum of squared errors.
    fn calc_square_error(&self) -> f64;

    /// Populate upper triangular matrix of the normal equation.
    fn populate_upper_triangulatr_normal_equation(
        &self,
        variables: &VarFamilies,
        nu: f64,
        hessian_block_triplet: &mut SymmetricMatrixBuilderEnum,
        neg_grad: &mut BlockVector,
    );
}
