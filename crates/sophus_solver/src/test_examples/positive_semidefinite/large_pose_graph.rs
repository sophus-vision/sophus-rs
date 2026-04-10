//! Large-scale pose graph linear system for benchmarking.
//!
//! Generates a sphere-like topology: N poses on a grid wrapped into a sphere,
//! each connected to ~6 neighbors. All blocks are 6×6 (SE3 tangent space).

use crate::{
    LinearSolverEnum,
    matrix::{
        PartitionBlockIndex,
        PartitionSet,
        PartitionSpec,
        SymmetricMatrixBuilderEnum,
    },
    test_examples::LinearSystem,
};

/// Create a large pose graph linear system.
///
/// - `num_poses`: number of 6×6 blocks (SE3 poses)
///
/// The topology is a grid wrapped into a cylinder/sphere with
/// cross-links to simulate loop closures.
pub fn create_large_pose_graph(solver: &LinearSolverEnum, num_poses: usize) -> LinearSystem {
    let block_dim = 6; // SE3

    let partitions = vec![PartitionSpec {
        block_count: num_poses,
        block_dim,
        eliminate_last: false,
    }];
    let mut mat_a_builder = solver.zero(PartitionSet::new(partitions));

    // Information matrix for a single edge (SPD 6×6).
    // Realistic: translation components have higher info than rotation.
    let info_edge = nalgebra::DMatrix::from_fn(block_dim, block_dim, |r, c| {
        if r == c {
            if r < 3 { 100.0 } else { 10.0 } // translation vs rotation
        } else if (r as isize - c as isize).abs() == 1 {
            0.5 // small coupling
        } else {
            0.0
        }
    });
    let neg_info = -&info_edge;

    // Grid dimensions for the sphere topology.
    let cols = (num_poses as f64).sqrt().ceil() as usize;
    let rows = (num_poses + cols - 1) / cols;
    let idx = |r: usize, c: usize| -> usize {
        let i = r * cols + c;
        if i < num_poses { i } else { num_poses - 1 }
    };

    // Base diagonal: prior on first pose.
    let prior_info =
        nalgebra::DMatrix::from_fn(
            block_dim,
            block_dim,
            |r, c| {
                if r == c { 1000.0 } else { 0.0 }
            },
        );
    let idx_0 = PartitionBlockIndex {
        partition: 0,
        block: 0,
    };
    mat_a_builder.add_lower_block(idx_0, idx_0, &prior_info.as_view());

    let add_edge = |builder: &mut SymmetricMatrixBuilderEnum, a: usize, b: usize| {
        if a == b || a >= num_poses || b >= num_poses {
            return;
        }
        let (i, j) = if a > b { (a, b) } else { (b, a) };
        let idx_i = PartitionBlockIndex {
            partition: 0,
            block: i,
        };
        let idx_j = PartitionBlockIndex {
            partition: 0,
            block: j,
        };

        // H[i,i] += info, H[j,j] += info, H[i,j] -= info
        builder.add_lower_block(idx_i, idx_i, &info_edge.as_view());
        builder.add_lower_block(idx_j, idx_j, &info_edge.as_view());
        builder.add_lower_block(idx_i, idx_j, &neg_info.as_view());
    };

    // Chain edges: connect consecutive poses.
    for i in 0..(num_poses - 1) {
        add_edge(&mut mat_a_builder, i, i + 1);
    }

    // Grid edges: 4-connectivity on the grid.
    for r in 0..rows {
        for c in 0..cols {
            let i = idx(r, c);
            if i >= num_poses {
                continue;
            }
            // Right neighbor.
            if c + 1 < cols {
                let j = idx(r, c + 1);
                if j < num_poses && j != i + 1 {
                    // Skip if already covered by chain.
                    add_edge(&mut mat_a_builder, i, j);
                }
            }
            // Down neighbor.
            if r + 1 < rows {
                let j = idx(r + 1, c);
                if j < num_poses {
                    add_edge(&mut mat_a_builder, i, j);
                }
            }
        }
    }

    // Loop closures: connect first/last row (sphere wrap).
    for c in 0..cols.min(num_poses) {
        let top = idx(0, c);
        let bot_row = rows - 1;
        let bot = idx(bot_row, c);
        if top != bot && bot < num_poses {
            add_edge(&mut mat_a_builder, top, bot);
        }
    }

    // Additional diagonal damping to ensure well-conditioned.
    let damping =
        nalgebra::DMatrix::from_fn(block_dim, block_dim, |r, c| if r == c { 1.0 } else { 0.0 });
    for i in 0..num_poses {
        let idx_i = PartitionBlockIndex {
            partition: 0,
            block: i,
        };
        mat_a_builder.add_lower_block(idx_i, idx_i, &damping.as_view());
    }

    let total_dim = num_poses * block_dim;
    let b = nalgebra::DVector::from_element(total_dim, 1.0);
    LinearSystem {
        mat_a: mat_a_builder.build(),
        b,
    }
}
