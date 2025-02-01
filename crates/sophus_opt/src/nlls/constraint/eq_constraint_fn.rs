use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::Range,
};

use super::{
    eq_constraint::{
        EqConstraints,
        IsEqConstraint,
    },
    evaluated_eq_set::IsEvaluatedEqConstraintSet,
};
use crate::{
    nlls::{
        constraint::{
            evaluated_constraint::EvaluatedConstraint,
            evaluated_eq_set::EvaluatedEqSet,
        },
        linear_system::EvalMode,
        quadratic_cost::compare_idx::{
            c_from_var_kind,
            CompareIdx,
        },
    },
    variables::{
        var_families::VarFamilies,
        var_tuple::IsVarTuple,
        VarKind,
    },
};

extern crate alloc;

/// Equality constraint function
#[derive(Debug, Clone)]
pub struct EqConstraintFn<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const NUM_ARGS: usize,
    GlobalConstants: 'static + Send + Sync,
    Constants: Debug,
    Constraint: IsEqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS, GlobalConstants, VarTuple, Constants>,
    VarTuple: IsVarTuple<NUM_ARGS> + 'static,
> {
    global_constants: GlobalConstants,
    constraint: EqConstraints<
        RESIDUAL_DIM,
        INPUT_DIM,
        NUM_ARGS,
        GlobalConstants,
        VarTuple,
        Constants,
        Constraint,
    >,
    phantom: PhantomData<VarTuple>,
}

impl<
        const RESIDUAL_DIM: usize,
        const INPUT_DIM: usize,
        const NUM_ARGS: usize,
        GlobalConstants: 'static + Send + Sync,
        Constants: 'static + Debug,
        Constraint: IsEqConstraint<
            RESIDUAL_DIM,
            INPUT_DIM,
            NUM_ARGS,
            GlobalConstants,
            VarTuple,
            Constants,
            Constants = Constants,
        >,
        VarTuple: IsVarTuple<NUM_ARGS> + 'static,
    >
    EqConstraintFn<
        RESIDUAL_DIM,
        INPUT_DIM,
        NUM_ARGS,
        GlobalConstants,
        Constants,
        Constraint,
        VarTuple,
    >
{
    /// Create a new equality constraint function from the given eq-constrains and a
    /// eq-constraint functor.
    pub fn new_box(
        global_constants: GlobalConstants,
        terms: EqConstraints<
            RESIDUAL_DIM,
            INPUT_DIM,
            NUM_ARGS,
            GlobalConstants,
            VarTuple,
            Constants,
            Constraint,
        >,
    ) -> alloc::boxed::Box<dyn IsEqConstraintsFn> {
        alloc::boxed::Box::new(Self {
            global_constants,
            constraint: terms,
            phantom: PhantomData,
        })
    }
}

/// Is Equality constraint function.
pub trait IsEqConstraintsFn {
    /// Evaluate the equality constraint.
    fn eval(
        &self,
        var_pool: &VarFamilies,
        eval_mode: EvalMode,
    ) -> alloc::boxed::Box<dyn IsEvaluatedEqConstraintSet>;

    /// Sort the constraints.
    fn sort(&mut self, variables: &VarFamilies);

    /// Number of arguments.
    fn num_args(&self) -> usize;

    /// residual dimension
    fn residual_dim(&self) -> usize;
}

impl<
        const RESIDUAL_DIM: usize,
        const INPUT_DIM: usize,
        const NUM_ARGS: usize,
        GlobalConstants: 'static + Send + Sync,
        Constants: Debug,
        Constraint: IsEqConstraint<
            RESIDUAL_DIM,
            INPUT_DIM,
            NUM_ARGS,
            GlobalConstants,
            VarTuple,
            Constants,
            Constants = Constants,
        >,
        VarTuple: IsVarTuple<NUM_ARGS> + 'static,
    > IsEqConstraintsFn
    for EqConstraintFn<
        RESIDUAL_DIM,
        INPUT_DIM,
        NUM_ARGS,
        GlobalConstants,
        Constants,
        Constraint,
        VarTuple,
    >
{
    fn eval(
        &self,
        var_pool: &VarFamilies,
        eval_mode: EvalMode,
    ) -> alloc::boxed::Box<dyn IsEvaluatedEqConstraintSet> {
        let mut var_kind_array =
            VarTuple::var_kind_array(var_pool, self.constraint.family_names.clone());

        if eval_mode == EvalMode::DontCalculateDerivatives {
            var_kind_array = var_kind_array.map(|_x| VarKind::Conditioned)
        }

        let mut evaluated_eq_constraints =
            EvaluatedEqSet::new(self.constraint.family_names.clone());

        let var_family_tuple =
            VarTuple::ref_var_family_tuple(var_pool, self.constraint.family_names.clone());

        let eval_res = |term: &Constraint| {
            term.eval(
                &self.global_constants,
                *term.idx_ref(),
                VarTuple::extract(&var_family_tuple, *term.idx_ref()),
                var_kind_array,
                term.c_ref(),
            )
        };

        let reduction_ranges = self.constraint.reduction_ranges.as_ref().unwrap();

        evaluated_eq_constraints
            .evaluated_constraints
            .reserve(reduction_ranges.len());
        for range in reduction_ranges.iter() {
            let mut evaluated_term_sum: Option<
                EvaluatedConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>,
            > = None;

            for term in self.constraint.collection[range.start..range.end].iter() {
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

        alloc::boxed::Box::new(evaluated_eq_constraints)
    }

    fn sort(&mut self, variables: &VarFamilies) {
        let var_kind_array =
            &VarTuple::var_kind_array(variables, self.constraint.family_names.clone());

        let c_array = c_from_var_kind(var_kind_array);

        let less = CompareIdx::new(&c_array);

        assert!(!self.constraint.collection.is_empty());

        self.constraint
            .collection
            .sort_by(|a, b| less.le_than(*a.idx_ref(), *b.idx_ref()));

        for t in 0..self.constraint.collection.len() - 1 {
            assert!(
                less.le_than(
                    *self.constraint.collection[t].idx_ref(),
                    *self.constraint.collection[t + 1].idx_ref()
                ) != core::cmp::Ordering::Greater
            );
        }

        let mut reduction_ranges: alloc::vec::Vec<Range<usize>> = alloc::vec![];
        let mut i = 0;
        while i < self.constraint.collection.len() {
            let outer_term = &self.constraint.collection[i];
            let outer_term_idx = i;
            while i < self.constraint.collection.len()
                && less.free_vars_equal(
                    outer_term.idx_ref(),
                    self.constraint.collection[i].idx_ref(),
                )
            {
                i += 1;
            }
            reduction_ranges.push(outer_term_idx..i);
        }

        self.constraint.reduction_ranges = Some(reduction_ranges);
    }

    fn num_args(&self) -> usize {
        NUM_ARGS
    }

    fn residual_dim(&self) -> usize {
        RESIDUAL_DIM
    }
}
