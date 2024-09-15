use crate::cost::Cost;
use crate::cost::IsCost;
use crate::cost_args::c_from_var_kind;
use crate::robust_kernel::RobustKernel;
use crate::term::Term;
use crate::variables::IsVarTuple;
use crate::variables::VarKind;
use crate::variables::VarPool;
use std::marker::PhantomData;
use std::ops::Range;
use std::time::Instant;

/// Signature of a term of a cost function
pub trait IsTermSignature<const N: usize>: Send + Sync + 'static {
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
    pub(crate) reduction_ranges: Option<Vec<Range<usize>>>,
}

impl<
        const NUM_ARGS: usize,
        Constants,
        TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants>,
    > CostSignature<NUM_ARGS, Constants, TermSignature>
{
    /// Create a new cost signature
    pub fn new(family_names: [String; NUM_ARGS], terms: Vec<TermSignature>) -> Self {
        CostSignature {
            family_names,
            terms,
            reduction_ranges: None,
        }
    }
}

/// Residual function
pub trait IsResidualFn<
    const NUM: usize,
    const NUM_ARGS: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<NUM_ARGS>,
    Constants,
>: Copy + Send + Sync + 'static
{
    /// evaluate the residual function which shall be defined by the user
    fn eval(
        &self,
        global_constants: &GlobalConstants,
        idx: [usize; NUM_ARGS],
        args: Args,
        derivatives: [VarKind; NUM_ARGS],
        robust_kernel: Option<RobustKernel>,
        constants: &Constants,
    ) -> Term<NUM, NUM_ARGS>;
}

/// Quadratic cost function of the non-linear least squares problem
pub trait IsCostFn {
    /// evaluate the cost function
    fn eval(
        &self,
        var_pool: &VarPool,
        calc_derivatives: bool,
        parallelize: bool,
    ) -> Box<dyn IsCost>;

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
    GlobalConstants: 'static + Send + Sync,
    Constants,
    TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants>,
    ResidualFn,
    VarTuple: IsVarTuple<NUM_ARGS> + 'static,
> where
    ResidualFn: IsResidualFn<NUM, NUM_ARGS, GlobalConstants, VarTuple, Constants>,
{
    global_constants: GlobalConstants,
    signature: CostSignature<NUM_ARGS, Constants, TermSignature>,
    residual_fn: ResidualFn,
    robust_kernel: Option<RobustKernel>,
    phantom: PhantomData<VarTuple>,
}

impl<
        const NUM: usize,
        const NUM_ARGS: usize,
        GlobalConstants: 'static + Send + Sync,
        Constants: 'static,
        TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants> + 'static,
        ResidualFn,
        VarTuple: IsVarTuple<NUM_ARGS> + 'static,
    > CostFn<NUM, NUM_ARGS, GlobalConstants, Constants, TermSignature, ResidualFn, VarTuple>
where
    ResidualFn: IsResidualFn<NUM, NUM_ARGS, GlobalConstants, VarTuple, Constants> + 'static,
{
    /// create a new cost function from a signature and a residual function
    pub fn new_box(
        global_constants: GlobalConstants,
        signature: CostSignature<NUM_ARGS, Constants, TermSignature>,
        residual_fn: ResidualFn,
    ) -> Box<dyn IsCostFn> {
        Box::new(Self {
            global_constants,
            signature,
            residual_fn,
            robust_kernel: None,
            phantom: PhantomData,
        })
    }

    /// create a new robust cost function from a signature, a residual function and a robust kernel
    pub fn new_robust(
        global_constants: GlobalConstants,
        signature: CostSignature<NUM_ARGS, Constants, TermSignature>,
        residual_fn: ResidualFn,
        robust_kernel: RobustKernel,
    ) -> Box<dyn IsCostFn> {
        Box::new(Self {
            global_constants,
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
        GlobalConstants: 'static + Send + Sync,
        Constants,
        TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants>,
        ResidualFn,
        VarTuple: IsVarTuple<NUM_ARGS> + 'static,
    > IsCostFn
    for CostFn<NUM, NUM_ARGS, GlobalConstants, Constants, TermSignature, ResidualFn, VarTuple>
where
    ResidualFn: IsResidualFn<NUM, NUM_ARGS, GlobalConstants, VarTuple, Constants>,
{
    fn eval(
        &self,
        var_pool: &VarPool,
        calc_derivatives: bool,
        parallelize: bool,
    ) -> Box<dyn IsCost> {
        let mut var_kind_array =
            VarTuple::var_kind_array(var_pool, self.signature.family_names.clone());

        if !calc_derivatives {
            var_kind_array = var_kind_array.map(|_x| VarKind::Conditioned)
        }

        let mut evaluated_terms = Cost::new(
            self.signature.family_names.clone(),
            TermSignature::DOF_TUPLE,
        );

        let var_family_tuple =
            VarTuple::ref_var_family_tuple(var_pool, self.signature.family_names.clone());

        let eval_res = |term_signature: &TermSignature| {
            self.residual_fn.eval(
                &self.global_constants,
                *term_signature.idx_ref(),
                VarTuple::extract(&var_family_tuple, *term_signature.idx_ref()),
                var_kind_array,
                self.robust_kernel,
                term_signature.c_ref(),
            )
        };

        let reduction_ranges = self.signature.reduction_ranges.as_ref().unwrap();

        #[derive(Debug)]
        enum ParallelizationStrategy {
            None,
            OuterLoop,
            InnerLoop,
        }
        const OUTER_LOOP_THRESHOLD: usize = 100;
        const INNER_LOOP_THRESHOLD: f64 = 100.0;
        const REDUCTION_RATIO_THRESHOLD: f64 = 1.0;

        let average_inner_loop_size =
            self.signature.terms.len() as f64 / reduction_ranges.len() as f64;
        let reduction_ratio = average_inner_loop_size / reduction_ranges.len() as f64;

        let parallelization_strategy = match parallelize {
            true => {
                if reduction_ranges.len() >= OUTER_LOOP_THRESHOLD
                    && reduction_ratio < REDUCTION_RATIO_THRESHOLD
                {
                    // There are many outer terms, and significantly less inner terms on average
                    ParallelizationStrategy::OuterLoop
                } else if average_inner_loop_size >= INNER_LOOP_THRESHOLD {
                    // There are many inner terms on average.
                    ParallelizationStrategy::InnerLoop
                } else {
                    ParallelizationStrategy::None
                }
            }
            false => ParallelizationStrategy::None,
        };

        match parallelization_strategy {
            ParallelizationStrategy::None => {
                // This functional style code is slightly less efficient, than the nested while
                // loop below.
                //
                // evaluated_terms.terms = reduction_ranges
                //     .iter() // sequential outer loop
                //     .map(|range| {
                //         let evaluated_term_sum = self.signature.terms[range.start..range.end]
                //             .iter() // sequential inner loop
                //             .fold(None, |acc: Option<Term<NUM, NUM_ARGS>>, term| {
                //                 let evaluated_term = eval_res(term);
                //                 match acc {
                //                     Some(mut sum) => {
                //                         sum.reduce(evaluated_term);
                //                         Some(sum)
                //                     }
                //                     None => Some(evaluated_term),
                //                 }
                //             });

                //         evaluated_term_sum.unwrap()
                //     })
                //     .collect();

                evaluated_terms.terms.reserve(reduction_ranges.len());
                for range in reduction_ranges.iter() {
                    let mut evaluated_term_sum: Option<Term<NUM, NUM_ARGS>> = None;

                    for term in self.signature.terms[range.start..range.end].iter() {
                        match evaluated_term_sum {
                            Some(mut sum) => {
                                sum.reduce(eval_res(term));
                                evaluated_term_sum = Some(sum);
                            }
                            None => evaluated_term_sum = Some(eval_res(term)),
                        }
                    }

                    evaluated_terms.terms.push(evaluated_term_sum.unwrap());
                }
            }
            ParallelizationStrategy::OuterLoop => {
                use rayon::prelude::*;

                evaluated_terms.terms = reduction_ranges
                    .par_iter() // parallelize over the outer terms
                    .map(|range| {
                        let evaluated_term_sum = self.signature.terms[range.start..range.end]
                            .iter() // sequential inner loop
                            .fold(None, |acc: Option<Term<NUM, NUM_ARGS>>, term| {
                                let evaluated_term = eval_res(term);
                                match acc {
                                    Some(mut sum) => {
                                        sum.reduce(evaluated_term);
                                        Some(sum)
                                    }
                                    None => Some(evaluated_term),
                                }
                            });

                        evaluated_term_sum.unwrap()
                    })
                    .collect();
            }
            ParallelizationStrategy::InnerLoop => {
                use rayon::prelude::*;

                evaluated_terms.terms = reduction_ranges
                    .iter() // sequential outer loop
                    .map(|range| {
                        // We know on average there are many inner terms, however, there might be
                        // outliers.
                        //
                        // todo: Consider adding an if statement here and only parallelize the
                        //       inner loop if the range length is greater than some threshold.
                        let evaluated_term_sum = self.signature.terms[range.start..range.end]
                            .par_iter() // parallelize over the inner terms
                            .fold(
                                || None,
                                |acc: Option<Term<NUM, NUM_ARGS>>, term| {
                                    let evaluated_term = eval_res(term);
                                    match acc {
                                        Some(mut sum) => {
                                            sum.reduce(evaluated_term);
                                            Some(sum)
                                        }
                                        None => Some(evaluated_term),
                                    }
                                },
                            )
                            .reduce(
                                || None,
                                |acc, evaluated_term| match (acc, evaluated_term) {
                                    (Some(mut sum), Some(evaluated_term)) => {
                                        sum.reduce(evaluated_term);
                                        Some(sum)
                                    }
                                    (None, Some(evaluated_term)) => Some(evaluated_term),
                                    _ => None,
                                },
                            );

                        evaluated_term_sum.unwrap()
                    })
                    .collect();
            }
        }

        Box::new(evaluated_terms)
    }

    fn sort(&mut self, variables: &VarPool) {
        let now = Instant::now();

        let var_kind_array =
            &VarTuple::var_kind_array(variables, self.signature.family_names.clone());
        use crate::cost_args::CompareIdx;

        let c_array = c_from_var_kind(var_kind_array);

        let less = CompareIdx::new(&c_array);

        assert!(!self.signature.terms.is_empty());

        self.signature
            .terms
            .sort_by(|a, b| less.le_than(*a.idx_ref(), *b.idx_ref()));

        println!("sorting took: {:.2?}", now.elapsed());

        let now = Instant::now();

        for t in 0..self.signature.terms.len() - 1 {
            assert!(
                less.le_than(
                    *self.signature.terms[t].idx_ref(),
                    *self.signature.terms[t + 1].idx_ref()
                ) != std::cmp::Ordering::Greater
            );
        }

        println!("sorting val took: {:.2?}", now.elapsed());

        let now = Instant::now();

        let mut reduction_ranges: Vec<Range<usize>> = vec![];
        let mut i = 0;
        while i < self.signature.terms.len() {
            let outer_term_signature = &self.signature.terms[i];
            let outer_term_idx = i;
            while i < self.signature.terms.len()
                && less.free_vars_equal(
                    outer_term_signature.idx_ref(),
                    self.signature.terms[i].idx_ref(),
                )
            {
                i += 1;
            }
            reduction_ranges.push(outer_term_idx..i);
        }

        self.signature.reduction_ranges = Some(reduction_ranges);

        println!("reduction_ranges took: {:.2?}", now.elapsed());
    }

    fn robust_kernel(&self) -> Option<RobustKernel> {
        self.robust_kernel
    }
}
