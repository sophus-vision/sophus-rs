use crate::{
    LinearSolverEnum,
    matrix::{
        PartitionBlockIndex,
        PartitionSet,
        PartitionSpec,
    },
    test_examples::LinearSystem,
};

/// A small, single partition, `8 x 8` linear system.
pub fn create_single_partition_linear_system(solver: &LinearSolverEnum) -> LinearSystem {
    use nalgebra as na;
    let partitions = PartitionSet::new(vec![PartitionSpec {
        eliminate_last: false,
        block_count: 4,
        block_dim: 2,
    }]);
    let scalar_dim = partitions.scalar_dim();
    let block_count = partitions.specs()[0].block_count;
    let block_dim = partitions.specs()[0].block_dim;

    let mut mat_a_builder = solver.zero(partitions);
    let mut mat_l = na::DMatrix::<f64>::zeros(scalar_dim, scalar_dim);

    let scalar_idx = |block_idx: usize| block_idx * block_dim;

    for col_j in 0..block_count {
        for d in 0..block_dim {
            mat_l[(scalar_idx(col_j) + d, scalar_idx(col_j) + d)] = 1.0;
        }
        if col_j > 0 {
            let mut block = mat_l.view_mut(
                (scalar_idx(col_j), scalar_idx(col_j - 1)),
                (block_dim, block_dim),
            );
            for c in 0..block_dim {
                for r in 0..block_dim {
                    block[(r, c)] = 0.05 * (1.0 + (r + c) as f64);
                }
            }
        }
    }

    let mat_a = &mat_l * mat_l.transpose();
    for row_i in 0..block_count {
        let row_idx = PartitionBlockIndex {
            partition: 0,
            block: row_i,
        };
        for col_j in 0..=row_i {
            let col_idx = PartitionBlockIndex {
                partition: 0,
                block: col_j,
            };
            mat_a_builder.add_lower_block(
                row_idx,
                col_idx,
                &mat_a.view(
                    (scalar_idx(row_i), scalar_idx(col_j)),
                    (block_dim, block_dim),
                ),
            );
        }
    }

    let b = na::DVector::<f64>::from_iterator(
        scalar_dim,
        (0..scalar_dim).map(|i| 1.0 + 0.1 * i as f64),
    );
    LinearSystem {
        mat_a: mat_a_builder.build(),
        b,
    }
}
