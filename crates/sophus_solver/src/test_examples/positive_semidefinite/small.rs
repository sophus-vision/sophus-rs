use nalgebra::DMatrix;
use sophus_autodiff::linalg::MatF64;

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

/// A small, single partition, `6 x 6` linear system.
pub fn create_small_linear_system(solver: &LinearSolverEnum) -> LinearSystem {
    let partitions = vec![PartitionSpec {
        block_count: 2,
        block_dim: 3,
    }];
    let mut mat_a_builder = solver.zero(PartitionSet::new(partitions));
    let block_a00 = MatF64::<3, 3>::from_array2([
        [3.3, 1.0, 0.0], //
        [1.0, 3.2, 1.0],
        [0.0, 1.0, 3.1],
    ]);
    let block_a01 = MatF64::<3, 3>::from_array2([
        [0.0, 0.0, 0.0], //
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]);
    let block_a11 = MatF64::<3, 3>::from_array2([
        [2.3, 0.0, 0.0], //
        [0.0, 2.2, 0.0],
        [0.0, 0.0, 2.1],
    ]);

    let p0_b0 = PartitionBlockIndex {
        partition: 0,
        block: 0,
    };
    let b0_b1 = PartitionBlockIndex {
        partition: 0,
        block: 1,
    };

    mat_a_builder.add_lower_block(p0_b0, p0_b0, &block_a00.as_view());
    mat_a_builder.add_lower_block(b0_b1, p0_b0, &block_a01.as_view());
    mat_a_builder.add_lower_block(b0_b1, b0_b1, &block_a11.as_view());

    let b = nalgebra::DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mat_a = mat_a_builder.build();

    assert_eq!(mat_a.get_block(p0_b0, p0_b0), block_a00);
    assert_eq!(mat_a.get_block(b0_b1, p0_b0), block_a01);
    assert_eq!(mat_a.get_block(b0_b1, b0_b1), block_a11);

    let mat_a_dense = DMatrix::<f64>::from_row_slice(
        6,
        6,
        &[
            3.3, 1.0, 0.0, 0.0, 0.0, 1.0, //
            1.0, 3.2, 1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 3.1, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 2.3, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 2.2, 0.0, //
            1.0, 0.0, 0.0, 0.0, 0.0, 2.1,
        ], //
    );

    assert_eq!(mat_a.to_dense(), mat_a_dense);

    LinearSystem { mat_a, b }
}
