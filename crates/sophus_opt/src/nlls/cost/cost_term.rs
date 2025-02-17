use core::ops::Range;
use std::fmt::Debug;

use super::evaluated_term::EvaluatedCostTerm;
use crate::{
    robust_kernel::RobustKernel,
    variables::{
        IsVarTuple,
        VarKind,
    },
};

extern crate alloc;

/// Residual function of the non-linear least squares problem.
///
/// This trait is implemented by the user to define the concrete residual function.
///
/// ## Generic parameter
///
///  * `INPUT_DIM`
///    - Total input dimension. It is the sum of argument dimensions: |Vⁱ₀| + |Vⁱ₁| + ... + |Vⁱₙ₋₁|.
///  * `N`
///    - Number of arguments.
///  * `GlobalConstants`
///    - Type of the global constants which are passed to the residual function. If no global
///      constants are needed, use `()`.
///  * `Args`
///    - Tuple of input argument types: `(Vⁱ₀, Vⁱ₁, ..., Vⁱₙ₋₁)`.
pub trait HasResidualFn<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<N>,
>: Send + Sync + 'static + Debug
{
    /// Returns an array of variable indices for each argument in this residual function.
    ///
    /// For example, if `idx_ref() = [2, 7, 3]` and `Args` is `(Foo, Bar, Bar)`, then:
    ///
    /// - Argument 0 is the 2nd variable of the `Foo` family.
    /// - Argument 1 is the 7th variable of the `Bar` family.
    /// - Argument 2 is the 3rd variable of the `Bar` family.
    fn idx_ref(&self) -> &[usize; N];

    /// Evaluate the residual function.
    fn eval(
        &self,
        global_constants: &GlobalConstants,
        idx: [usize; N],
        args: Args,
        derivatives: [VarKind; N],
        robust_kernel: Option<RobustKernel>,
    ) -> EvaluatedCostTerm<INPUT_DIM, N>;
}

/// Cost terms, to be passed to the non-linear least squares optimizer.
///
/// ## Generic parameters
///
///  * `INPUT_DIM`
///    - Total input dimension of the common residual function `g`. It is the sum of argument
///      dimensions: |Vⁱ₀| + |Vⁱ₁| + ... + |Vⁱₙ₋₁|.
///  * `N`
///    - Number of arguments of the common residual function `g`.
///  * `GlobalConstants`
///    - Type of the global constants which are passed to the residual function. If no global
///      constants are needed, use `()`.
///  * `Args`
///    - Tuple of input argument types: `(Vⁱ₀, Vⁱ₁, ..., Vⁱₙ₋₁)`.
///  * `ResidualFn`
///    - The common residual function `g`.
#[derive(Debug, Clone)]
pub struct CostTerms<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<N>,
    ResidualFn: HasResidualFn<INPUT_DIM, N, GlobalConstants, Args>,
> {
    /// Variable family name for each argument.
    pub family_names: [String; N],
    /// Collection of unevaluated terms.
    pub collection: alloc::vec::Vec<ResidualFn>,
    pub(crate) reduction_ranges: Option<alloc::vec::Vec<Range<usize>>>,
    phantom: core::marker::PhantomData<(GlobalConstants, Args)>,
}

impl<
        const INPUT_DIM: usize,
        const N: usize,
        GlobalConstants: 'static + Send + Sync,
        Args: IsVarTuple<N>,
        ResidualFn: HasResidualFn<INPUT_DIM, N, GlobalConstants, Args>,
    > CostTerms<INPUT_DIM, N, GlobalConstants, Args, ResidualFn>
{
    /// Create a new set of cost terms.
    pub fn new(family_names: [impl ToString; N], terms: alloc::vec::Vec<ResidualFn>) -> Self {
        CostTerms {
            family_names: family_names.map(|name| name.to_string()),
            collection: terms,
            reduction_ranges: None,
            phantom: core::marker::PhantomData,
        }
    }
}
