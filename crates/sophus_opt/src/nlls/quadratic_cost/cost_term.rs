use super::evaluated_term::EvaluatedCostTerm;
use crate::robust_kernel::RobustKernel;
use crate::variables::var_tuple::IsVarTuple;
use crate::variables::VarKind;
use core::ops::Range;
use std::fmt::Debug;

extern crate alloc;

/// (Unevaluated) term of the cost function
pub trait IsCostTerm<
    const NUM: usize,
    const NUM_ARGS: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<NUM_ARGS>,
    Constants,
>: Send + Sync + 'static + Debug
{
    /// associated constants such as measurements, etc.
    type Constants;

    /// reference to the constants
    fn c_ref(&self) -> &Self::Constants;

    /// one index (into the variable family) for each argument
    fn idx_ref(&self) -> &[usize; NUM_ARGS];

    /// evaluate the residual function which shall be defined by the user
    fn eval(
        &self,
        global_constants: &GlobalConstants,
        idx: [usize; NUM_ARGS],
        args: Args,
        derivatives: [VarKind; NUM_ARGS],
        robust_kernel: Option<RobustKernel>,
        constants: &Constants,
    ) -> EvaluatedCostTerm<NUM, NUM_ARGS>;
}

/// (Unevaluated) cost
#[derive(Debug, Clone)]
pub struct CostTerms<
    const NUM: usize,
    const NUM_ARGS: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<NUM_ARGS>,
    Constants,
    Term: IsCostTerm<NUM, NUM_ARGS, GlobalConstants, Args, Constants>,
> {
    /// one variable family name for each argument
    pub family_names: [String; NUM_ARGS],
    /// collection of unevaluated terms
    pub collection: alloc::vec::Vec<Term>,
    pub(crate) reduction_ranges: Option<alloc::vec::Vec<Range<usize>>>,
    phantom: core::marker::PhantomData<(GlobalConstants, Args, Constants)>,
}

impl<
        const NUM: usize,
        const NUM_ARGS: usize,
        GlobalConstants: 'static + Send + Sync,
        Args: IsVarTuple<NUM_ARGS>,
        Constants,
        Term: IsCostTerm<NUM, NUM_ARGS, GlobalConstants, Args, Constants>,
    > CostTerms<NUM, NUM_ARGS, GlobalConstants, Args, Constants, Term>
{
    /// Create a new set of terms
    pub fn new(family_names: [impl ToString; NUM_ARGS], terms: alloc::vec::Vec<Term>) -> Self {
        CostTerms {
            family_names: family_names.map(|name| name.to_string()),
            collection: terms,
            reduction_ranges: None,
            phantom: core::marker::PhantomData,
        }
    }
}
