use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::Range,
};

use snafu::Snafu;

use super::{
    eq_constraint::{
        EqConstraints,
        HasEqConstraintResidualFn,
    },
    evaluated_eq_set::IsEvaluatedEqConstraintSet,
};
use crate::{
    nlls::{
        constraint::{
            evaluated_eq_constraint::EvaluatedEqConstraint,
            evaluated_eq_set::EvaluatedEqSet,
        },
        cost::compare_idx::{
            c_from_var_kind,
            CompareIdx,
        },
        linear_system::EvalMode,
    },
    variables::{
        IsVarTuple,
        VarFamilies,
        VarFamilyError,
        VarKind,
    },
};

extern crate alloc;

/// Equality constraint function.
pub trait IsEqConstraintsFn {
    /// Evaluate the equality constraint.
    fn eval(
        &self,
        var_pool: &VarFamilies,
        eval_mode: EvalMode,
    ) -> Result<alloc::boxed::Box<dyn IsEvaluatedEqConstraintSet>, EqConstraintError>;

    /// Sort the constraints (to ensure more efficient evaluation and reduction over
    /// conditioned variables).
    fn sort(&mut self, variables: &VarFamilies);

    /// Number of constraint arguments.
    fn num_args(&self) -> usize;

    /// Dimension of the constraint residual.
    fn residual_dim(&self) -> usize;
}

/// Generic equality constraint function.
///
/// It represents a set of equality constraints:
///
/// `{c(V⁰₀, V⁰₁, ..., V⁰ₙ₋₁), ..., c(Vⁱ₀, Vⁱ₁, ..., Vⁱₙ₋₁), ...}`
///
/// All terms are based on a common constraint residual function `c` and a set
///  of input arguments `Vⁱ₀, Vⁱ₁, ...,  Vⁱₙ₋₁`.
///
/// This struct is passed as a box of [IsEqConstraintsFn] to the optimizer.
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
///  * `Constraint`
///    - The constraint residual function `c`.
///  * `Args`
///    - Tuple of input argument types: `(Vⁱ₀, Vⁱ₁, ..., Vⁱₙ₋₁)`.
#[derive(Debug, Clone)]
pub struct EqConstraintFn<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Constraint: HasEqConstraintResidualFn<RESIDUAL_DIM, INPUT_DIM, N, GlobalConstants, Args>,
    Args: IsVarTuple<N> + 'static,
> {
    global_constants: GlobalConstants,
    constraints: EqConstraints<RESIDUAL_DIM, INPUT_DIM, N, GlobalConstants, Args, Constraint>,
    phantom: PhantomData<Args>,
}

impl<
        const RESIDUAL_DIM: usize,
        const INPUT_DIM: usize,
        const N: usize,
        GlobalConstants: 'static + Send + Sync,
        Constraint: HasEqConstraintResidualFn<RESIDUAL_DIM, INPUT_DIM, N, GlobalConstants, Args>,
        Args: IsVarTuple<N> + 'static,
    > EqConstraintFn<RESIDUAL_DIM, INPUT_DIM, N, GlobalConstants, Constraint, Args>
{
    /// Create a new equality constraint function from global constants and a set of constraints.
    pub fn new_boxed(
        global_constants: GlobalConstants,
        constraints: EqConstraints<RESIDUAL_DIM, INPUT_DIM, N, GlobalConstants, Args, Constraint>,
    ) -> alloc::boxed::Box<dyn IsEqConstraintsFn> {
        alloc::boxed::Box::new(Self {
            global_constants,
            constraints,
            phantom: PhantomData,
        })
    }
}

/// Errors that can occur when working with variables as constraint arguments.
#[derive(Snafu, Debug)]
pub enum EqConstraintError {
    /// Error when working with variable as constraint arguments.
    #[snafu(display("EqConstraintError( {} )", source))]
    VariableFamilyError {
        /// source
        source: VarFamilyError,
    },
}

impl<
        const RESIDUAL_DIM: usize,
        const INPUT_DIM: usize,
        const N: usize,
        GlobalConstants: 'static + Send + Sync,
        Constraint: HasEqConstraintResidualFn<RESIDUAL_DIM, INPUT_DIM, N, GlobalConstants, Args>,
        Args: IsVarTuple<N> + 'static,
    > IsEqConstraintsFn
    for EqConstraintFn<RESIDUAL_DIM, INPUT_DIM, N, GlobalConstants, Constraint, Args>
{
    fn eval(
        &self,
        variables: &VarFamilies,
        eval_mode: EvalMode,
    ) -> Result<alloc::boxed::Box<dyn IsEvaluatedEqConstraintSet>, EqConstraintError> {
        let mut var_kind_array =
            Args::var_kind_array(variables, self.constraints.family_names.clone());

        if eval_mode == EvalMode::DontCalculateDerivatives {
            var_kind_array = var_kind_array.map(|_x| VarKind::Conditioned)
        }

        let mut evaluated_eq_constraints =
            EvaluatedEqSet::new(self.constraints.family_names.clone());

        let var_family_tuple =
            Args::ref_var_family_tuple(variables, self.constraints.family_names.clone())
                .map_err(|e| EqConstraintError::VariableFamilyError { source: e })?;

        let eval_res = |term: &Constraint| {
            term.eval(
                &self.global_constants,
                *term.idx_ref(),
                Args::extract(&var_family_tuple, *term.idx_ref()),
                var_kind_array,
            )
        };

        let reduction_ranges = self.constraints.reduction_ranges.as_ref().unwrap();

        evaluated_eq_constraints
            .evaluated_constraints
            .reserve(reduction_ranges.len());
        for range in reduction_ranges.iter() {
            let mut evaluated_term_sum: Option<EvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, N>> =
                None;

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
            &Args::var_kind_array(variables, self.constraints.family_names.clone());

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
        N
    }

    fn residual_dim(&self) -> usize {
        RESIDUAL_DIM
    }
}
