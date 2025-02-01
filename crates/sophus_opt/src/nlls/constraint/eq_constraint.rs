use core::ops::Range;
use std::fmt::Debug;

use super::evaluated_constraint::EvaluatedConstraint;
use crate::variables::{
    var_tuple::IsVarTuple,
    VarKind,
};

extern crate alloc;

/// (Unevaluated) equality constraint of the equality constraint function
pub trait IsEqConstraint<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
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

    /// Evaluate the equality constraint.
    fn eval(
        &self,
        global_constants: &GlobalConstants,
        idx: [usize; NUM_ARGS],
        args: Args,
        derivatives: [VarKind; NUM_ARGS],
        constants: &Constants,
    ) -> EvaluatedConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>;
}

/// (Unevaluated) equality constraints
#[derive(Debug, Clone)]
pub struct EqConstraints<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const NUM_ARGS: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<NUM_ARGS>,
    Constants,
    Constraint: IsEqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, Args, Constants>,
> {
    /// one variable family name for each argument
    pub family_names: [String; NUM_ARGS],
    /// collection of unevaluated terms
    pub collection: alloc::vec::Vec<Constraint>,
    pub(crate) reduction_ranges: Option<alloc::vec::Vec<Range<usize>>>,
    phantom: core::marker::PhantomData<(GlobalConstants, Args, Constants)>,
}

impl<
        const RESIDUAL_DIM: usize,
        const INPUT_DIM: usize,
        const NUM_ARGS: usize,
        GlobalConstants: 'static + Send + Sync,
        Args: IsVarTuple<NUM_ARGS>,
        Constants,
        Constraint: IsEqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, Args, Constants>,
    >
    EqConstraints<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, Args, Constants, Constraint>
{
    /// Create a new set of terms
    pub fn new(
        family_names: [impl ToString; NUM_ARGS],
        constraints: alloc::vec::Vec<Constraint>,
    ) -> Self {
        EqConstraints {
            family_names: family_names.map(|name| name.to_string()),
            collection: constraints,
            reduction_ranges: None,
            phantom: core::marker::PhantomData,
        }
    }
}
