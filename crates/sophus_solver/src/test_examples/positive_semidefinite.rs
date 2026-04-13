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
            IsInvertible,
            LinearSolverEnum,
            ldlt::DenseLdlt,
            matrix::PartitionBlockIndex,
            svd::pseudo_inverse,
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
                let mat_a_inverse_reference = pseudo_inverse(dense_a.view());

                let partitions = dense_a.partitions();

                let b: DVector<f64> = problem.b.clone();
                let factor = solver.factorize(mat_a).unwrap();

                let x: DVector<f64> = factor.solve(&b).unwrap();

                assert_abs_diff_eq!((dense_a.view() * x), b, epsilon = 1e-6);

                if let Some(mut min_norm_factor) = factor.into_invertible() {
                    let mat_a_inverse = min_norm_factor.pseudo_inverse();
                    assert_abs_diff_eq!(mat_a_inverse, mat_a_inverse_reference, epsilon = 1e-6);

                    for row_partition_idx in 0..partitions.len() {
                        let row_partition = &partitions.specs()[row_partition_idx];
                        for col_partition_idx in 0..partitions.len() {
                            let col_partition = &partitions.specs()[col_partition_idx];

                            for row_block in 0..row_partition.block_count {
                                let row_idx = PartitionBlockIndex {
                                    partition: row_partition_idx,
                                    block: row_block,
                                };

                                for col_block in 0..col_partition.block_count {
                                    let col_idx = PartitionBlockIndex {
                                        partition: col_partition_idx,
                                        block: col_block,
                                    };

                                    let block =
                                        min_norm_factor.pseudo_inverse_block(row_idx, col_idx);

                                    let r = partitions.scalar_offsets_by_partition()
                                        [row_partition_idx]
                                        + row_idx.block * row_partition.block_dim;
                                    let c = partitions.scalar_offsets_by_partition()
                                        [col_partition_idx]
                                        + col_idx.block * col_partition.block_dim;

                                    let reference_block =
                                        mat_a_inverse.view((r, c), block.shape()).into_owned();

                                    assert_abs_diff_eq!(block, reference_block, epsilon = 1e-6);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn medium_linear_systems_tests() {
        use approx::assert_abs_diff_eq;
        use nalgebra::DVector;

        use crate::{
            IsFactor,
            IsInvertible,
            LinearSolverEnum,
            ldlt::DenseLdlt,
            svd::pseudo_inverse,
            test_examples::positive_semidefinite::medium::create_medium_linear_problem,
        };

        let reference_solver = LinearSolverEnum::DenseLdlt(DenseLdlt::default());
        let solvers = LinearSolverEnum::all_solvers();

        for solver in &solvers {
            let problem = create_medium_linear_problem(solver);
            let reference_problem = create_medium_linear_problem(&reference_solver);

            let mat_a = &problem.mat_a;
            let dense_a = &reference_problem.mat_a.as_dense().unwrap();
            let mat_a_inverse_reference = pseudo_inverse(dense_a.view());
            let b: DVector<f64> = problem.b.clone();
            let factor = solver.factorize(mat_a).unwrap();

            let x: DVector<f64> = factor.solve(&b).unwrap();

            assert_abs_diff_eq!((dense_a.view() * x), b, epsilon = 1e-6);

            if let Some(mut min_norm_factor) = factor.into_invertible() {
                let mat_a_inverse = min_norm_factor.pseudo_inverse();
                assert_abs_diff_eq!(mat_a_inverse, mat_a_inverse_reference, epsilon = 1e-6);
            }
        }
    }
}
