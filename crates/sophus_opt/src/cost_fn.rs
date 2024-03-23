use std::marker::PhantomData;

use crate::cost::Cost;
use crate::cost_args::c_from_var_kind;
use crate::variables::VarKind;

use crate::cost::IsCost;
use crate::robust_kernel::RobustKernel;
use crate::term::Term;
use crate::variables::IsVarTuple;
use crate::variables::VarPool;

/// Signature of a term of a cost function
pub trait IsTermSignature<const N: usize> {
    /// associated constants such as measurements, etc.
    type Constants;

    /// one DOF for each argument
    const DOF_TUPLE: [i64; N];

    /// reference to the constants
    fn c_ref(&self) -> &Self::Constants;

    /// one index (into the variable family) for each argument
    fn idx_ref(&self) -> &[usize; N];
}

/// Signature of a cost function    
#[derive(Debug, Clone)]
pub struct CostSignature<
    const NUM_ARGS: usize,
    Constants,
    TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants>,
> {
    /// one variable family name for each argument
    pub family_names: [String; NUM_ARGS],
    /// terms of the unevaluated cost function
    pub terms: Vec<TermSignature>,
}

/// Residual function
pub trait IsResidualFn<
    const NUM: usize,
    const NUM_ARGS: usize,
    Args: IsVarTuple<NUM_ARGS>,
    Constants,
>: Copy
{
    /// evaluate the residual function which shall be defined by the user
    fn eval(
        &self,
        args: Args,
        derivatives: [VarKind; NUM_ARGS],
        robust_kernel: Option<RobustKernel>,
        constants: &Constants,
    ) -> Term<NUM, NUM_ARGS>;
}

/// Quadratic cost function of the non-linear least squares problem
pub trait IsCostFn {
    /// evaluate the cost function
    fn eval(&self, var_pool: &VarPool, calc_derivatives: bool) -> Box<dyn IsCost>;

    /// sort the terms of the cost function (to ensure more efficient evaluation and reduction over
    /// conditioned variables)
    fn sort(&mut self, variables: &VarPool);

    /// get the robust kernel function
    fn robust_kernel(&self) -> Option<RobustKernel>;
}

/// Generic cost function of the non-linear least squares problem
#[derive(Debug, Clone)]
pub struct CostFn<
    const NUM: usize,
    const NUM_ARGS: usize,
    Constants,
    TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants>,
    ResidualFn,
    VarTuple: IsVarTuple<NUM_ARGS> + 'static,
> where
    ResidualFn: IsResidualFn<NUM, NUM_ARGS, VarTuple, Constants>,
{
    signature: CostSignature<NUM_ARGS, Constants, TermSignature>,
    residual_fn: ResidualFn,
    robust_kernel: Option<RobustKernel>,
    phantom: PhantomData<VarTuple>,
}

impl<
        const NUM: usize,
        const NUM_ARGS: usize,
        Constants: 'static,
        TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants> + 'static,
        ResidualFn,
        VarTuple: IsVarTuple<NUM_ARGS> + 'static,
    > CostFn<NUM, NUM_ARGS, Constants, TermSignature, ResidualFn, VarTuple>
where
    ResidualFn: IsResidualFn<NUM, NUM_ARGS, VarTuple, Constants> + 'static,
{
    /// create a new cost function from a signature and a residual function
    pub fn new(
        signature: CostSignature<NUM_ARGS, Constants, TermSignature>,
        residual_fn: ResidualFn,
    ) -> Box<dyn IsCostFn> {
        Box::new(Self {
            signature,
            residual_fn,
            robust_kernel: None,
            phantom: PhantomData,
        })
    }

    /// create a new robust cost function from a signature, a residual function and a robust kernel
    pub fn new_robust(
        signature: CostSignature<NUM_ARGS, Constants, TermSignature>,
        residual_fn: ResidualFn,
        robust_kernel: RobustKernel,
    ) -> Box<dyn IsCostFn> {
        Box::new(Self {
            signature,
            residual_fn,
            robust_kernel: Some(robust_kernel),
            phantom: PhantomData,
        })
    }
}

impl<
        const NUM: usize,
        const NUM_ARGS: usize,
        Constants,
        TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants>,
        ResidualFn,
        VarTuple: IsVarTuple<NUM_ARGS> + 'static,
    > IsCostFn for CostFn<NUM, NUM_ARGS, Constants, TermSignature, ResidualFn, VarTuple>
where
    ResidualFn: IsResidualFn<NUM, NUM_ARGS, VarTuple, Constants>,
{
    fn eval(&self, var_pool: &VarPool, calc_derivatives: bool) -> Box<dyn IsCost> {
        use crate::cost_args::CompareIdx;
        let mut var_kind_array =
            VarTuple::var_kind_array(var_pool, self.signature.family_names.clone());
        let c_array = c_from_var_kind(&var_kind_array);

        if !calc_derivatives {
            var_kind_array = var_kind_array.map(|_x| VarKind::Conditioned)
        }
        let less = CompareIdx { c: c_array };

        let mut evaluated_terms = Cost::new(
            self.signature.family_names.clone(),
            TermSignature::DOF_TUPLE,
        );

        let mut i = 0;

        let var_family_tuple =
            VarTuple::ref_var_family_tuple(var_pool, self.signature.family_names.clone());

        let eval_res = |term_signature: &TermSignature| {
            self.residual_fn.eval(
                VarTuple::extract(&var_family_tuple, *term_signature.idx_ref()),
                var_kind_array,
                self.robust_kernel,
                term_signature.c_ref(),
            )
        };

        evaluated_terms.terms.reserve(self.signature.terms.len());

        while i < self.signature.terms.len() {
            let term_signature = &self.signature.terms[i];

            let outer_idx = term_signature.idx_ref();

            let mut evaluated_term = eval_res(term_signature);
            evaluated_term.idx.push(*term_signature.idx_ref());

            i += 1;

            // perform reduction over conditioned variables
            while i < self.signature.terms.len() {
                let inner_term_signature = &self.signature.terms[i];

                if !less.are_all_non_cond_vars_equal(outer_idx, inner_term_signature.idx_ref()) {
                    // end condition for reduction over conditioned variables
                    break;
                }

                i += 1;

                let inner_evaluated_term = eval_res(inner_term_signature);

                evaluated_term.hessian.mat += inner_evaluated_term.hessian.mat;
                evaluated_term.gradient.vec += inner_evaluated_term.gradient.vec;
                evaluated_term.cost += inner_evaluated_term.cost;
            }

            evaluated_terms.terms.push(evaluated_term);
        }

        Box::new(evaluated_terms)
    }

    fn sort(&mut self, variables: &VarPool) {
        let var_kind_array =
            &VarTuple::var_kind_array(variables, self.signature.family_names.clone());
        use crate::cost_args::CompareIdx;

        let c_array = c_from_var_kind(var_kind_array);

        let less = CompareIdx { c: c_array };

        assert!(!self.signature.terms.is_empty());

        self.signature
            .terms
            .sort_by(|a, b| less.le_than(*a.idx_ref(), *b.idx_ref()));

        for t in 0..self.signature.terms.len() - 1 {
            assert!(
                less.le_than(
                    *self.signature.terms[t].idx_ref(),
                    *self.signature.terms[t + 1].idx_ref()
                ) == std::cmp::Ordering::Less
            );
        }
    }

    fn robust_kernel(&self) -> Option<RobustKernel> {
        self.robust_kernel
    }
}
