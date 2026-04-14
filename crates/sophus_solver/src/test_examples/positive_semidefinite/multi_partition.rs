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

/// A small, three partition, `11 x 11` linear system.
pub fn create_multi_partition_linear_system(solver: &LinearSolverEnum) -> LinearSystem {
    use nalgebra as na;
    let partitions = vec![
        PartitionSpec {
            eliminate_last: false,
            block_dim: 3,
            block_count: 2,
        },
        PartitionSpec {
            eliminate_last: false,
            block_dim: 1,
            block_count: 3,
        },
        PartitionSpec {
            eliminate_last: false,
            block_dim: 2,
            block_count: 1,
        },
    ];
    let mut mat_a_builder = solver.zero(PartitionSet::new(partitions.clone()));

    let mut base_by_partition = vec![0];
    let mut scalar_count = 0usize;
    for partition in &partitions {
        scalar_count += partition.block_dim * partition.block_count;
        base_by_partition.push(scalar_count);
    }

    let block_offsets_by_partition: Vec<usize> = {
        let mut offsets = vec![0];
        for partition in &partitions {
            offsets.push(offsets.last().unwrap() + partition.block_count);
        }
        offsets
    };
    let total_block_count = *block_offsets_by_partition.last().unwrap();

    let block_dim_by_global_block = |global_block_idx: usize| {
        let mut partition_idx = 0;
        while partition_idx + 1 < block_offsets_by_partition.len()
            && global_block_idx >= block_offsets_by_partition[partition_idx + 1]
        {
            partition_idx += 1;
        }
        partitions[partition_idx].block_dim
    };
    let scalar_offset_by_global_block = |global_block_idx: usize| {
        let mut partition_idx = 0;
        while partition_idx + 1 < block_offsets_by_partition.len()
            && global_block_idx >= block_offsets_by_partition[partition_idx + 1]
        {
            partition_idx += 1;
        }
        let block_idx = global_block_idx - block_offsets_by_partition[partition_idx];
        base_by_partition[partition_idx] + block_idx * partitions[partition_idx].block_dim
    };

    let mut mat_l = na::DMatrix::<f64>::zeros(scalar_count, scalar_count);
    for i in 0..total_block_count {
        let block_height = block_dim_by_global_block(i);
        let offset_i = scalar_offset_by_global_block(i);
        for d in 0..block_height {
            mat_l[(offset_i + d, offset_i + d)] = 1.0;
        }
        for j in 0..i {
            let block_width = block_dim_by_global_block(j);
            let offset_j = scalar_offset_by_global_block(j);
            let mut block = mat_l.view_mut((offset_i, offset_j), (block_height, block_width));
            for c in 0..block_width {
                for r in 0..block_height {
                    block[(r, c)] = 0.04 * ((r + 1 + c + 1) as f64);
                }
            }
        }
    }
    let mat_a = &mat_l * mat_l.transpose();
    for row_partition_idx in 0..partitions.len() {
        let block_height = partitions[row_partition_idx].block_dim;
        let block_row_count = partitions[row_partition_idx].block_count;
        let row_base = base_by_partition[row_partition_idx];
        for col_partition_idx in 0..=row_partition_idx {
            let block_width = partitions[col_partition_idx].block_dim;
            let block_col_count = partitions[col_partition_idx].block_count;
            let col_base = base_by_partition[col_partition_idx];

            for row_i in 0..block_row_count {
                let row_idx = PartitionBlockIndex {
                    partition: row_partition_idx,
                    block: row_i,
                };
                let r = row_base + row_i * block_height;
                let col_j_max = if row_partition_idx == col_partition_idx {
                    row_i
                } else {
                    block_col_count - 1
                };
                for col_j in 0..=col_j_max {
                    let col_idx = PartitionBlockIndex {
                        partition: col_partition_idx,
                        block: col_j,
                    };
                    let c = col_base + col_j * block_width;
                    mat_a_builder.add_lower_block(
                        row_idx,
                        col_idx,
                        &mat_a.view((r, c), (block_height, block_width)),
                    );
                }
            }
        }
    }

    let mat_a = mat_a_builder.build();

    let mat_a_dense_00 = DMatrix::<f64>::from_row_slice(
        6,
        6,
        &[
            1.0000, 0.0000, 0.0000, 0.0800, 0.1200, 0.1600, //
            0.0000, 1.0000, 0.0000, 0.1200, 0.1600, 0.2000, //
            0.0000, 0.0000, 1.0000, 0.1600, 0.2000, 0.2400, //
            0.0800, 0.1200, 0.1600, 1.0464, 0.0608, 0.0752, //
            0.1200, 0.1600, 0.2000, 0.0608, 1.0800, 0.0992, //
            0.1600, 0.2000, 0.2400, 0.0752, 0.0992, 1.1232,
        ], //
    );

    let mat_a_dense_01 = DMatrix::<f64>::from_row_slice(
        6,
        5,
        &[
            0.1600, 0.0800, 0.0800, 0.0800, 0.0800, //
            0.2000, 0.1200, 0.1200, 0.1200, 0.1200, //
            0.2400, 0.1600, 0.1600, 0.1600, 0.1600, //
            0.0752, 0.1264, 0.1264, 0.1264, 0.1264, //
            0.0992, 0.1808, 0.1808, 0.1808, 0.1808, //
            1.1232, 0.2352, 0.2352, 0.2352, 0.2352,
        ],
    );

    let mat_a_dense_11 = DMatrix::<f64>::from_row_slice(
        5,
        5,
        &[
            1.0928, 0.1728, 0.1728, 0.1728, 0.2416, //
            0.1728, 1.0992, 0.1792, 0.1792, 0.2512, //
            0.1728, 0.1792, 1.1056, 0.1856, 0.2608, //
            0.1728, 0.1792, 0.1856, 1.1120, 0.1504, //
            0.2416, 0.2512, 0.2608, 0.1504, 1.2032,
        ], //
    );

    assert_abs_diff_eq!(
        mat_a.to_dense().view((0, 0), (6, 6)).to_owned(),
        mat_a_dense_00.as_view(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        mat_a.to_dense().view((0, 5), (6, 5)).to_owned(),
        mat_a_dense_01.as_view(),
        epsilon = 1e-6,
    );
    assert_abs_diff_eq!(
        mat_a.to_dense().view((5, 0), (5, 6)).to_owned(),
        mat_a_dense_01.transpose().as_view(),
        epsilon = 1e-6,
    );
    assert_abs_diff_eq!(
        mat_a.to_dense().view((6, 6), (5, 5)).to_owned(),
        mat_a_dense_11.as_view(),
        epsilon = 1e-6,
    );

    let b = na::DVector::<f64>::from_iterator(
        scalar_count,
        (0..scalar_count).map(|k| (k as f64).sin() + 2.0),
    );

    LinearSystem { mat_a, b }
}
