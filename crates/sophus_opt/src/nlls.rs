use crate::cost::IsCost;
use crate::cost_fn::IsCostFn;
use crate::solvers::solve;
use crate::variables::VarPool;
use log::debug;
use log::info;
use std::fmt::Debug;

/// Optimization parameters
#[derive(Copy, Clone, Debug)]
pub struct OptParams {
    /// number of iterations
    pub num_iter: usize,
    /// initial value of the Levenberg-Marquardt regularization parameter
    pub initial_lm_nu: f64,
}

impl Default for OptParams {
    fn default() -> Self {
        Self {
            num_iter: 20,
            initial_lm_nu: 10.0,
        }
    }
}

/// Non-linear least squares optimization
pub fn optimize(
    mut variables: VarPool,
    mut cost_fns: Vec<Box<dyn IsCostFn>>,
    params: OptParams,
) -> VarPool {
    let mut init_costs: Vec<Box<dyn IsCost>> = Vec::new();
    for cost_fn in cost_fns.iter_mut() {
        // sort to achieve more efficient evaluation and reduction
        cost_fn.sort(&variables);
        init_costs.push(cost_fn.eval(&variables, false));
    }

    let mut nu = params.initial_lm_nu;

    let mut mse = 0.0;
    for init_cost in init_costs.iter() {
        mse += init_cost.calc_square_error();
    }
    let initial_mse = mse;

    use std::time::Instant;
    let now = Instant::now();

    for _i in 0..params.num_iter {
        debug!("nu: {:?}", nu);

        let mut evaluated_costs: Vec<Box<dyn IsCost>> = Vec::new();
        for cost_fn in cost_fns.iter_mut() {
            evaluated_costs.push(cost_fn.eval(&variables, true));
        }

        let updated_families = solve(&variables, evaluated_costs, nu);

        let mut new_costs: Vec<Box<dyn IsCost>> = Vec::new();
        for cost_fn in cost_fns.iter_mut() {
            new_costs.push(cost_fn.eval(&updated_families, true));
        }
        let mut new_mse = 0.0;
        for init_cost in new_costs.iter() {
            new_mse += init_cost.calc_square_error();
        }

        if new_mse < mse {
            nu *= 0.0333;
            variables = updated_families;
            mse = new_mse;
        } else {
            nu *= 2.0;
        }
    }
    info!("e^2: {:?} -> {:?}", initial_mse, mse);
    info!("{} iters took: {:.2?}", params.num_iter, now.elapsed());

    variables
}
