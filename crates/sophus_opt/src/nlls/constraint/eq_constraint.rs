use core::ops::Range;
use std::fmt::Debug;

use super::evaluated_eq_constraint::EvaluatedEqConstraint;
use crate::variables::{
    IsVarTuple,
    VarKind,
};

extern crate alloc;

/// Equality constraint residual function.
///
/// This trait is implemented by the user to define the concrete equality constraint.
///
/// ## Generic parameters
///
///  * `RESIDUAL_DIM`
///    - Dimension of the constraint residual vector `c`.
///  * `INPUT_DIM`
///    - Total input dimension of the constraint residual function `c`. It is the sum of argument
///      dimensions: |Vⁱ₀| + |Vⁱ₁| + ... + |Vⁱₙ₋₁|.
///  * `N`
///    - Number of arguments of the constraint residual function `c`.
///  * `GlobalConstants`
///    - Type of the global constants which are passed to the residual function. If no global
///      constants are needed, use `()`.
///  * `Args`
///    - Tuple of input argument types: `(Vⁱ₀, Vⁱ₁, ..., Vⁱₙ₋₁)`.
pub trait HasEqConstraintResidualFn<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<N>,
>: Send + Sync + 'static + Debug
{
    /// Returns an array of variable indices for each argument in this constraint.
    ///
    /// For example, if `idx_ref() = [2, 7, 3]` and `Args` is `(Foo, Bar, Bar)`, then:
    ///
    /// - Argument 0 is the 2nd variable of the `Foo` family.
    /// - Argument 1 is the 7th variable of the `Bar` family.
    /// - Argument 2 is the 3rd variable of the `Bar` family.
    fn idx_ref(&self) -> &[usize; N];

    /// Evaluate the equality constraint.
    fn eval(
        &self,
        global_constants: &GlobalConstants,
        idx: [usize; N],
        args: Args,
        derivatives: [VarKind; N],
    ) -> EvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, N>;
}

/// Equality constraints, to be passed to the non-linear least squares optimizer.
///
/// ## Generic parameters
///
///  * `RESIDUAL_DIM`
///    - Dimension of the constraint residual vector `c`.
///  * `INPUT_DIM`
///    - Total input dimension of the constraint residual function `c`. It is the sum of argument
///      dimensions: |Vⁱ₀| + |Vⁱ₁| + ... + |Vⁱₙ₋₁|.
///  * `N`
///    - Number of arguments of the constraint residual function `c`.
///  * `GlobalConstants`
///    - Type of the global constants which are passed to the residual function. If no global
///      constants are needed, use `()`.
///  * `Args`
///    - Tuple of input argument types: `(Vⁱ₀, Vⁱ₁, ..., Vⁱₙ₋₁)`.
///  * `Constraint`
///    - The constraint residual function `c`.
#[derive(Debug, Clone)]
pub struct EqConstraints<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<N>,
    Constraint: HasEqConstraintResidualFn<RESIDUAL_DIM, INPUT_DIM, N, GlobalConstants, Args>,
> {
    /// Variable family name for each argument.
    pub family_names: [String; N],
    /// Collection of unevaluated constraints.
    pub collection: alloc::vec::Vec<Constraint>,
    pub(crate) reduction_ranges: Option<alloc::vec::Vec<Range<usize>>>,
    phantom: core::marker::PhantomData<(GlobalConstants, Args, Constraint)>,
}

impl<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<N>,
    Constraint: HasEqConstraintResidualFn<RESIDUAL_DIM, INPUT_DIM, N, GlobalConstants, Args>,
> EqConstraints<RESIDUAL_DIM, INPUT_DIM, N, GlobalConstants, Args, Constraint>
{
    /// Create a new set of terms
    pub fn new(family_names: [impl ToString; N], constraints: alloc::vec::Vec<Constraint>) -> Self {
        EqConstraints {
            family_names: family_names.map(|name| name.to_string()),
            collection: constraints,
            reduction_ranges: None,
            phantom: core::marker::PhantomData,
        }
    }
}
