use crate::quadratic_cost::cost_fn::IsCostFn;
use crate::quadratic_cost::evaluated_cost::IsEvaluatedCost;
use crate::solvers::normal_equation::solve;
use crate::solvers::normal_equation::SolveError;
use crate::solvers::normal_equation::TripletType;
use crate::solvers::sparse_ldlt::SparseLdltParams;
use crate::variables::VarPool;
use core::fmt::Debug;
use log::debug;
use log::info;

extern crate alloc;

/// Linear solver type
#[derive(Copy, Clone, Debug)]
pub enum LinearSolverType {
    /// Sparse LDLT solver (using faer crate)
    FaerSparseLdlt(SparseLdltParams),
    /// Sparse partial pivoting LU solver (using faer crate)
    FaerSparsePartialPivLu,
    /// Sparse QR solver (using faer crate)
    FaerSparseQR,
    /// Dense cholesky (using nalgebra crate)
    NalgebraDenseCholesky,
    /// Dense full-piovt. LU solver (using nalgebra crate)
    NalgebraDenseFullPiVLu,
}

impl Default for LinearSolverType {
    fn default() -> Self {
        LinearSolverType::FaerSparseLdlt(Default::default())
    }
}

impl LinearSolverType {
    /// Get the triplet type
    pub fn triplet_type(&self) -> TripletType {
        match self {
            LinearSolverType::FaerSparseLdlt(_) => TripletType::Upper,
            LinearSolverType::FaerSparsePartialPivLu => TripletType::Full,
            LinearSolverType::FaerSparseQR => TripletType::Full,
            LinearSolverType::NalgebraDenseCholesky => TripletType::Full,
            LinearSolverType::NalgebraDenseFullPiVLu => TripletType::Full,
        }
    }
}

/// Optimization parameters
#[derive(Copy, Clone, Debug)]
pub struct OptParams {
    /// number of iterations
    pub num_iter: usize,
    /// initial value of the Levenberg-Marquardt regularization parameter
    pub initial_lm_nu: f64,
    /// whether to use parallelization
    pub parallelize: bool,
    /// linear solver type
    pub linear_solver: LinearSolverType,
}

impl Default for OptParams {
    fn default() -> Self {
        Self {
            num_iter: 20,
            initial_lm_nu: 10.0,
            parallelize: true,
            linear_solver: Default::default(),
        }
    }
}

/// Non-linear least squares optimization
pub fn optimize(
    mut variables: VarPool,
    mut cost_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
    params: OptParams,
) -> Result<VarPool, SolveError> {
    let mut init_costs: alloc::vec::Vec<alloc::boxed::Box<dyn IsEvaluatedCost>> =
        alloc::vec::Vec::new();

    for cost_fn in cost_fns.iter_mut() {
        // sort to achieve more efficient evaluation and reduction
        cost_fn.sort(&variables);
        init_costs.push(cost_fn.eval(&variables, false, params.parallelize));
    }

    let mut nu = params.initial_lm_nu;

    let mut mse = 0.0;
    for init_cost in init_costs.iter() {
        mse += init_cost.calc_square_error();
    }
    let initial_mse = mse;

    for i in 0..params.num_iter {
        let mut evaluated_costs: alloc::vec::Vec<alloc::boxed::Box<dyn IsEvaluatedCost>> =
            alloc::vec::Vec::new();
        for cost_fn in cost_fns.iter_mut() {
            evaluated_costs.push(cost_fn.eval(&variables, true, params.parallelize));
        }

        let updated_families = solve(&variables, evaluated_costs, nu, params.linear_solver)?;

        let mut new_costs: alloc::vec::Vec<alloc::boxed::Box<dyn IsEvaluatedCost>> =
            alloc::vec::Vec::new();
        for cost_fn in cost_fns.iter_mut() {
            new_costs.push(cost_fn.eval(&updated_families, true, params.parallelize));
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

        debug!(
            "i: {:?}, nu: {:?}, mse {:?}, (new_mse {:?})",
            i, nu, mse, new_mse
        );
    }
    info!("e^2: {:?} -> {:?}", initial_mse, mse);

    Ok(variables)
}
