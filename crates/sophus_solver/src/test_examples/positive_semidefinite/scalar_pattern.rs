use approx::assert_abs_diff_eq;
use nalgebra::DMatrix;

use crate::{
    LinearSolverEnum,
    matrix::{
        PartitionBlockIndex,
        PartitionSet,
        PartitionSpec,
    },
    prelude::*,
    test_examples::LinearSystem,
};

/// A small, scalar-sparse `8 x 8` linear system.
pub fn create_scalar_pattern_linear_system(solver: &LinearSolverEnum) -> LinearSystem {
    use nalgebra as na;
    let partitions = vec![PartitionSpec {
        eliminate_last: false,
        block_dim: 1,
        block_count: 8,
    }];
    let partitions = PartitionSet::new(partitions);
    let n = partitions.scalar_dim();
    let block_count = partitions.specs()[0].block_count;

    let mut mat_a_builder = solver.zero(partitions);

    let mut mat_l = na::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        mat_l[(i, i)] = 1.0;
        if i > 0 {
            mat_l[(i, i - 1)] = 0.2;
        }
        if i > 1 {
            mat_l[(i, i - 2)] = 0.05;
        }
    }
    let mat_a = &mat_l * mat_l.transpose();
    for i in 0..block_count {
        let idx_i = PartitionBlockIndex {
            partition: 0,
            block: i,
        };
        for j in 0..=i {
            let idx_j = PartitionBlockIndex {
                partition: 0,
                block: j,
            };
            mat_a_builder.add_lower_block(idx_i, idx_j, &mat_a.view((i, j), (1, 1)));
        }
    }

    let mat_a = mat_a_builder.build();
    let mat_a_dense = DMatrix::<f64>::from_row_slice(
        8,
        8,
        &[
            1.000, 0.2000, 0.0500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, //
            0.200, 1.0400, 0.2100, 0.0500, 0.0000, 0.0000, 0.0000, 0.0000, //
            0.050, 0.2100, 1.0425, 0.2100, 0.0500, 0.0000, 0.0000, 0.0000, //
            0.000, 0.0500, 0.2100, 1.0425, 0.2100, 0.0500, 0.0000, 0.0000, //
            0.000, 0.0000, 0.0500, 0.2100, 1.0425, 0.2100, 0.0500, 0.0000, //
            0.000, 0.0000, 0.0000, 0.0500, 0.2100, 1.0425, 0.2100, 0.0500, //
            0.000, 0.0000, 0.0000, 0.0000, 0.0500, 0.2100, 1.0425, 0.2100, //
            0.000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0500, 0.2100, 1.0425,
        ], //
    );
    assert_abs_diff_eq!(mat_a.to_dense(), mat_a_dense);

    let b = na::DVector::<f64>::from_row_slice(&[1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0]);
    LinearSystem { mat_a, b }
}
