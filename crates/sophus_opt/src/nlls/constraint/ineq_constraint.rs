use core::ops::Range;
use std::fmt::Debug;

use sophus_geo::region::box_region::NonEmptyBoxRegion;

use super::evaluated_ineq_constraint::EvaluatedIneqConstraint;
use crate::variables::{
    var_tuple::IsVarTuple,
    VarKind,
};

extern crate alloc;

/// (Unevaluated) inequality constraint of the equality constraint function
pub trait IsIneqConstraint<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const NUM_ARGS: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<NUM_ARGS>,
>: Send + Sync + 'static + Debug
{
    /// reference to the lower/upper bounds
    fn bounds(&self) -> NonEmptyBoxRegion<RESIDUAL_DIM>;

    /// one index (into the variable family) for each argument
    fn idx_ref(&self) -> &[usize; NUM_ARGS];

    /// Evaluate the <=> constraint.
    fn eval(
        &self,
        global_constants: &GlobalConstants,
        idx: [usize; NUM_ARGS],
        args: Args,
        derivatives: [VarKind; NUM_ARGS],
    ) -> EvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>;
}

/// (Unevaluated) inequality constraints
#[derive(Debug, Clone)]
pub struct IneqConstraints<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const NUM_ARGS: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<NUM_ARGS>,
    Constraint: IsIneqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, Args>,
> {
    /// one variable family name for each argument
    pub family_names: [String; NUM_ARGS],
    /// collection of unevaluated terms
    pub collection: alloc::vec::Vec<Constraint>,
    pub(crate) reduction_ranges: Option<alloc::vec::Vec<Range<usize>>>,
    phantom: core::marker::PhantomData<(GlobalConstants, Args)>,
}

impl<
        const RESIDUAL_DIM: usize,
        const INPUT_DIM: usize,
        const NUM_ARGS: usize,
        GlobalConstants: 'static + Send + Sync,
        Args: IsVarTuple<NUM_ARGS>,
        Constraint: IsIneqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, Args>,
    > IneqConstraints<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, Args, Constraint>
{
    /// Create a new set of terms
    pub fn new(
        family_names: [impl ToString; NUM_ARGS],
        constraints: alloc::vec::Vec<Constraint>,
    ) -> Self {
        IneqConstraints {
            family_names: family_names.map(|name| name.to_string()),
            collection: constraints,
            reduction_ranges: None,
            phantom: core::marker::PhantomData,
        }
    }
}
