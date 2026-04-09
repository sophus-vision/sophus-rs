use crate::{
    LinearSolverEnum,
    test_examples::LinearSystem,
};

pub(crate) mod medium;
pub(crate) mod multi_partition;
pub(crate) mod scalar_pattern;
pub(crate) mod single_partition;
pub(crate) mod small;

pub use medium::*;
pub use multi_partition::*;
pub use scalar_pattern::*;
pub use single_partition::*;
pub use small::*;

/// Returns vector of all small positive semi-definite linear systems.
pub fn create_all_small_linear_system(solver: &LinearSolverEnum) -> Vec<LinearSystem> {
    vec![
        create_small_linear_system(solver),
        create_single_partition_linear_system(solver),
        create_multi_partition_linear_system(solver),
        create_scalar_pattern_linear_system(solver),
    ]
}

mod tests {

    #[test]
    fn small_linear_systems_tests() {
        use approx::assert_abs_diff_eq;
        use nalgebra::DVector;

        use crate::{
            IsFactor,
            LinearSolverEnum,
            ldlt::DenseLdlt,
            test_examples::positive_semidefinite::create_all_small_linear_system,
        };

        let reference_solver = LinearSolverEnum::DenseLdlt(DenseLdlt::default());
        let solvers = LinearSolverEnum::all_solvers();

        for solver in &solvers {
            let problems = create_all_small_linear_system(solver);
            let reference_problems = create_all_small_linear_system(&reference_solver);

            for (problem, reference_problem) in problems.iter().zip(reference_problems) {
                let mat_a = &problem.mat_a;
                let dense_a = &reference_problem.mat_a.as_dense().unwrap();

                let b: DVector<f64> = problem.b.clone();
                let factor = solver.factorize(mat_a).unwrap();

                let x: DVector<f64> = factor.solve(&b).unwrap();

                assert_abs_diff_eq!((dense_a.view() * x), b, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn medium_linear_systems_tests() {
        use approx::assert_abs_diff_eq;
        use nalgebra::DVector;

        use crate::{
            IsFactor,
            LinearSolverEnum,
            ldlt::DenseLdlt,
            test_examples::positive_semidefinite::medium::create_medium_linear_problem,
        };

        let reference_solver = LinearSolverEnum::DenseLdlt(DenseLdlt::default());
        let solvers = LinearSolverEnum::all_solvers();

        for solver in &solvers {
            let problem = create_medium_linear_problem(solver);
            let reference_problem = create_medium_linear_problem(&reference_solver);

            let mat_a = &problem.mat_a;
            let dense_a = &reference_problem.mat_a.as_dense().unwrap();
            let b: DVector<f64> = problem.b.clone();
            let factor = solver.factorize(mat_a).unwrap();

            let x: DVector<f64> = factor.solve(&b).unwrap();

            assert_abs_diff_eq!((dense_a.view() * x), b, epsilon = 1e-6);
        }
    }
}
