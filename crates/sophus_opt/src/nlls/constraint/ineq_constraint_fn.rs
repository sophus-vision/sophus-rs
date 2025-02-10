use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::Range,
};

use snafu::Snafu;

use super::{
    evaluated_ineq_constraint::EvaluatedIneqConstraint,
    evaluated_ineq_set::{
        EvaluatedIneqSet,
        IsEvaluatedIneqConstraintSet,
    },
    ineq_constraint::{
        IneqConstraints,
        IsIneqConstraint,
    },
};
use crate::{
    nlls::{
        cost::compare_idx::{
            c_from_var_kind,
            CompareIdx,
        },
        linear_system::EvalMode,
    },
    variables::{
        var_families::{
            VarFamilies,
            VarFamilyError,
        },
        var_tuple::IsVarTuple,
        VarKind,
    },
};

extern crate alloc;

/// Equality constraint function
#[derive(Debug, Clone)]
pub struct IneqConstraintFn<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const NUM_ARGS: usize,
    GlobalConstants: 'static + Send + Sync,
    Constraint: IsIneqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, VarTuple>,
    VarTuple: IsVarTuple<NUM_ARGS> + 'static,
> {
    global_constants: GlobalConstants,
    constraints:
        IneqConstraints<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, VarTuple, Constraint>,
    phantom: PhantomData<VarTuple>,
}

impl<
        const RESIDUAL_DIM: usize,
        const INPUT_DIM: usize,
        const NUM_ARGS: usize,
        GlobalConstants: 'static + Send + Sync,
        Constraint: IsIneqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, VarTuple>,
        VarTuple: IsVarTuple<NUM_ARGS> + 'static,
    > IneqConstraintFn<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, Constraint, VarTuple>
{
    /// Create a new equality constraint function from the given eq-constrains and a
    /// eq-constraint functor.
    pub fn new_box(
        global_constants: GlobalConstants,
        terms: IneqConstraints<
            RESIDUAL_DIM,
            INPUT_DIM,
            NUM_ARGS,
            GlobalConstants,
            VarTuple,
            Constraint,
        >,
    ) -> alloc::boxed::Box<dyn IsIneqConstraintsFn> {
        alloc::boxed::Box::new(Self {
            global_constants,
            constraints: terms,
            phantom: PhantomData,
        })
    }
}

/// Is Equality constraint function.
pub trait IsIneqConstraintsFn {
    /// Evaluate the equality constraint.
    fn eval(
        &self,
        var_pool: &VarFamilies,
        eval_mode: EvalMode,
    ) -> Result<alloc::boxed::Box<dyn IsEvaluatedIneqConstraintSet>, IneqConstraintError>;

    /// Sort the constraints.
    fn sort(&mut self, variables: &VarFamilies);

    /// Number of arguments.
    fn num_args(&self) -> usize;

    /// residual dimension
    fn residual_dim(&self) -> usize;
}

/// Errors that can occur when working with variable families
#[derive(Snafu, Debug)]
pub enum IneqConstraintError {
    /// Error when working with variable families
    #[snafu(display("IneqConstraintError({})", source))]
    VariableFamilyError {
        /// The source of the error
        source: VarFamilyError,
    },
}

impl<
        const RESIDUAL_DIM: usize,
        const INPUT_DIM: usize,
        const NUM_ARGS: usize,
        GlobalConstants: 'static + Send + Sync,
        Constraint: IsIneqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, VarTuple>,
        VarTuple: IsVarTuple<NUM_ARGS> + 'static,
    > IsIneqConstraintsFn
    for IneqConstraintFn<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, Constraint, VarTuple>
{
    fn eval(
        &self,
        variables: &VarFamilies,
        eval_mode: EvalMode,
    ) -> Result<alloc::boxed::Box<dyn IsEvaluatedIneqConstraintSet>, IneqConstraintError> {
        let mut var_kind_array =
            VarTuple::var_kind_array(variables, self.constraints.family_names.clone());

        if eval_mode == EvalMode::DontCalculateDerivatives {
            var_kind_array = var_kind_array.map(|_x| VarKind::Conditioned)
        }

        let mut evaluated_eq_constraints =
            EvaluatedIneqSet::new(self.constraints.family_names.clone());

        let var_family_tuple =
            VarTuple::ref_var_family_tuple(variables, self.constraints.family_names.clone()).map_err(|e| IneqConstraintError::VariableFamilyError { source: e })?;

        let eval_res = |term: &Constraint| {
            term.eval(
                &self.global_constants,
                *term.idx_ref(),
                VarTuple::extract(&var_family_tuple, *term.idx_ref()),
                var_kind_array,
            )
        };

        let reduction_ranges = self.constraints.reduction_ranges.as_ref().unwrap();

        evaluated_eq_constraints
            .evaluated_constraints
            .reserve(reduction_ranges.len());
        for range in reduction_ranges.iter() {
            let mut evaluated_term_sum: Option<
                EvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>,
            > = None;

            for term in self.constraints.collection[range.start..range.end].iter() {
                match evaluated_term_sum {
                    Some(mut sum) => {
                        sum.reduce(eval_res(term));
                        evaluated_term_sum = Some(sum);
                    }
                    None => evaluated_term_sum = Some(eval_res(term)),
                }
            }

            evaluated_eq_constraints
                .evaluated_constraints
                .push(evaluated_term_sum.unwrap());
        }

        Ok(alloc::boxed::Box::new(evaluated_eq_constraints))
    }

    fn sort(&mut self, variables: &VarFamilies) {
        let var_kind_array =
            &VarTuple::var_kind_array(variables, self.constraints.family_names.clone());

        let c_array = c_from_var_kind(var_kind_array);

        let less = CompareIdx::new(&c_array);

        assert!(!self.constraints.collection.is_empty());

        self.constraints
            .collection
            .sort_by(|a, b| less.le_than(*a.idx_ref(), *b.idx_ref()));

        for t in 0..self.constraints.collection.len() - 1 {
            assert!(
                less.le_than(
                    *self.constraints.collection[t].idx_ref(),
                    *self.constraints.collection[t + 1].idx_ref()
                ) != core::cmp::Ordering::Greater
            );
        }

        let mut reduction_ranges: alloc::vec::Vec<Range<usize>> = alloc::vec![];
        let mut i = 0;
        while i < self.constraints.collection.len() {
            let outer_term = &self.constraints.collection[i];
            let outer_term_idx = i;
            while i < self.constraints.collection.len()
                && less.free_vars_equal(
                    outer_term.idx_ref(),
                    self.constraints.collection[i].idx_ref(),
                )
            {
                i += 1;
            }
            reduction_ranges.push(outer_term_idx..i);
        }

        self.constraints.reduction_ranges = Some(reduction_ranges);
    }

    fn num_args(&self) -> usize {
        NUM_ARGS
    }

    fn residual_dim(&self) -> usize {
        RESIDUAL_DIM
    }
}
