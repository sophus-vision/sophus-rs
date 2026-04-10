use sophus_autodiff::linalg::MatF64;

use crate::{
    LinearSolverEnum,
    matrix::{
        PartitionBlockIndex,
        PartitionSet,
        PartitionSpec,
        SymmetricMatrixBuilderEnum,
    },
    prelude::*,
    test_examples::LinearSystem,
};

/// A medium size `480 x 480`, two partition, linear system.
pub fn create_medium_linear_problem(solver: &LinearSolverEnum) -> LinearSystem {
    let partitions = vec![
        PartitionSpec {
            eliminate_last: false,
            block_count: 50,
            block_dim: 4,
        },
        PartitionSpec {
            eliminate_last: false,
            block_count: 60,
            block_dim: 3,
        },
    ];
    let mut mat_a_builder = solver.zero(PartitionSet::new(partitions));
    const R0: usize = 5;
    const C0: usize = 10;
    const NB0: usize = R0 * C0; // 250
    type M4 = MatF64<4, 4>;

    let i4 = M4::from_array2([
        [1.0, 0.01, 0.02, 0.01],
        [0.01, 1.1, 0.02, 0.0],
        [0.02, 0.02, 1.2, 0.0],
        [0.01, 0.0, 0.0, 1.3],
    ]);
    let four_i4 = M4::from_array2([
        [4.0, 0.001, 0.1, 0.002],
        [0.001, 4.2, 0.0009, 0.0],
        [0.1, 0.0009, 4.3, 0.0],
        [0.002, 0.0, 0.0, 4.0],
    ]);

    let idx0 = |r: usize, c: usize| -> usize { r * C0 + c };

    let add_edge4 =
        |builder: &mut SymmetricMatrixBuilderEnum, a: usize, b: usize, c: &M4, minus_c: &M4| {
            let (i, j) = if a > b { (a, b) } else { (b, a) };
            let idx_0i = PartitionBlockIndex {
                partition: 0,
                block: i,
            };
            let idx_0j = PartitionBlockIndex {
                partition: 0,
                block: j,
            };

            // Region [0,0] = (row partition 0, col partition 0)
            builder.add_lower_block(idx_0i, idx_0j, &minus_c.as_view());
            builder.add_lower_block(idx_0i, idx_0i, &c.as_view());
            builder.add_lower_block(idx_0j, idx_0j, &c.as_view());
        };

    // Base diagonal (partition 0)
    for i in 0..NB0 {
        let idx_0i = PartitionBlockIndex {
            partition: 0,
            block: i,
        };
        mat_a_builder.add_lower_block(idx_0i, idx_0i, &four_i4.as_view());
    }

    // 4-neighborhood (partition 0)
    let minus_i4: M4 = -i4;
    for r in 0..R0 {
        for c in 0..C0 {
            let i = idx0(r, c);
            if r + 1 < R0 {
                add_edge4(&mut mat_a_builder, i, idx0(r + 1, c), &i4, &minus_i4);
            }
            if c + 1 < C0 {
                add_edge4(&mut mat_a_builder, i, idx0(r, c + 1), &i4, &minus_i4);
            }
        }
    }

    // ---------------- Partition 1 (size 3 blocks) ----------------
    const NB1: usize = 60; // choose any length you like
    type M3 = MatF64<3, 3>;

    let i3 = M3::from_array2([[1.0, 0.02, 0.0], [0.02, 1.1, 0.01], [0.0, 0.01, 1.2]]);
    let three_i3 = M3::from_array2([[3.0, 0.001, 0.0], [0.001, 3.1, 0.0007], [0.0, 0.0007, 3.2]]);
    let minus_i3: M3 = -i3;

    let add_edge3 =
        |builder: &mut SymmetricMatrixBuilderEnum, a: usize, b: usize, c: &M3, minus_c: &M3| {
            let (i, j) = if a > b { (a, b) } else { (b, a) };

            let idx_1i = PartitionBlockIndex {
                partition: 1,
                block: i,
            };
            let idx_1j = PartitionBlockIndex {
                partition: 1,
                block: j,
            };

            // Region [1,1] = (row partition 1, col partition 1)
            builder.add_lower_block(idx_1i, idx_1j, &minus_c.as_view());
            builder.add_lower_block(idx_1i, idx_1i, &c.as_view());
            builder.add_lower_block(idx_1j, idx_1j, &c.as_view());
        };

    // Base diagonal (partition 1)
    for i in 0..NB1 {
        let idx_1i = PartitionBlockIndex {
            partition: 1,
            block: i,
        };
        mat_a_builder.add_lower_block(idx_1i, idx_1i, &three_i3.as_view());
    }

    // Simple chain coupling within partition 1
    for i in 0..(NB1 - 1) {
        add_edge3(&mut mat_a_builder, i, i + 1, &i3, &minus_i3);
    }

    // Region [1,0]: row-partition 1 (size 3), col-partition 0 (size 4)
    let c10 = MatF64::<3, 4>::from_array2([
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.08, 0.01, 0.0],
        [0.0, 0.01, 0.07, 0.0],
    ]);
    let minus_c10 = -c10;

    let idx_00 = PartitionBlockIndex {
        partition: 0,
        block: 0,
    };
    let idx_10 = PartitionBlockIndex {
        partition: 1,
        block: 0,
    };
    mat_a_builder.add_lower_block(idx_10, idx_00, &minus_c10.as_view());
    mat_a_builder.add_lower_block(idx_10, idx_10, &(/* 3x3 */i3).as_view()); // add to the two diagonals
    mat_a_builder.add_lower_block(idx_00, idx_00, &(/* 4x4 */i4).as_view());

    // ---------------- RHS vector ----------------
    // total scalar size = 250*4 + 60*3

    let b = nalgebra::DVector::from_element(NB0 * 4 + NB1 * 3, 1.0);
    LinearSystem {
        mat_a: mat_a_builder.build(),
        b,
    }
}

/// A medium-size indefinite (KKT) linear system: 490×490.
///
/// Extends the PD medium problem (480 variables) with 5 constraint multiplier
/// rows (1×1 blocks), each coupled to one variable block. The constraint
/// partition has `-εI` on the diagonal, making the system indefinite.
pub fn create_medium_indefinite_problem(solver: &LinearSolverEnum) -> LinearSystem {
    let num_constraints = 5;
    let eps = 1e-14;

    let partitions = vec![
        PartitionSpec {
            eliminate_last: false,
            block_count: 50,
            block_dim: 4,
        },
        PartitionSpec {
            eliminate_last: false,
            block_count: 60,
            block_dim: 3,
        },
        PartitionSpec {
            block_count: num_constraints,
            block_dim: 1,
            eliminate_last: true,
        },
    ];
    let mut mat_a_builder = solver.zero(PartitionSet::new(partitions));

    // ---- Reuse the PD structure for partitions 0 and 1 ----
    // (same as create_medium_linear_problem)
    const R0: usize = 5;
    const C0: usize = 10;
    const NB0: usize = R0 * C0;
    type M4 = MatF64<4, 4>;

    let i4 = M4::from_array2([
        [1.0, 0.01, 0.02, 0.01],
        [0.01, 1.1, 0.02, 0.0],
        [0.02, 0.02, 1.2, 0.0],
        [0.01, 0.0, 0.0, 1.3],
    ]);
    let four_i4 = M4::from_array2([
        [4.0, 0.001, 0.1, 0.002],
        [0.001, 4.2, 0.0009, 0.0],
        [0.1, 0.0009, 4.3, 0.0],
        [0.002, 0.0, 0.0, 4.0],
    ]);

    let idx0 = |r: usize, c: usize| -> usize { r * C0 + c };

    let add_edge4 =
        |builder: &mut SymmetricMatrixBuilderEnum, a: usize, b: usize, c: &M4, minus_c: &M4| {
            let (i, j) = if a > b { (a, b) } else { (b, a) };
            let idx_0i = PartitionBlockIndex {
                partition: 0,
                block: i,
            };
            let idx_0j = PartitionBlockIndex {
                partition: 0,
                block: j,
            };
            builder.add_lower_block(idx_0i, idx_0j, &minus_c.as_view());
            builder.add_lower_block(idx_0i, idx_0i, &c.as_view());
            builder.add_lower_block(idx_0j, idx_0j, &c.as_view());
        };

    for i in 0..NB0 {
        let idx_0i = PartitionBlockIndex {
            partition: 0,
            block: i,
        };
        mat_a_builder.add_lower_block(idx_0i, idx_0i, &four_i4.as_view());
    }

    let minus_i4: M4 = -i4;
    for r in 0..R0 {
        for c in 0..C0 {
            let i = idx0(r, c);
            if r + 1 < R0 {
                add_edge4(&mut mat_a_builder, i, idx0(r + 1, c), &i4, &minus_i4);
            }
            if c + 1 < C0 {
                add_edge4(&mut mat_a_builder, i, idx0(r, c + 1), &i4, &minus_i4);
            }
        }
    }

    const NB1: usize = 60;
    type M3 = MatF64<3, 3>;

    let i3 = M3::from_array2([[1.0, 0.02, 0.0], [0.02, 1.1, 0.01], [0.0, 0.01, 1.2]]);
    let three_i3 = M3::from_array2([[3.0, 0.001, 0.0], [0.001, 3.1, 0.0007], [0.0, 0.0007, 3.2]]);
    let minus_i3: M3 = -i3;

    let add_edge3 =
        |builder: &mut SymmetricMatrixBuilderEnum, a: usize, b: usize, c: &M3, minus_c: &M3| {
            let (i, j) = if a > b { (a, b) } else { (b, a) };
            let idx_1i = PartitionBlockIndex {
                partition: 1,
                block: i,
            };
            let idx_1j = PartitionBlockIndex {
                partition: 1,
                block: j,
            };
            builder.add_lower_block(idx_1i, idx_1j, &minus_c.as_view());
            builder.add_lower_block(idx_1i, idx_1i, &c.as_view());
            builder.add_lower_block(idx_1j, idx_1j, &c.as_view());
        };

    for i in 0..NB1 {
        let idx_1i = PartitionBlockIndex {
            partition: 1,
            block: i,
        };
        mat_a_builder.add_lower_block(idx_1i, idx_1i, &three_i3.as_view());
    }

    for i in 0..(NB1 - 1) {
        add_edge3(&mut mat_a_builder, i, i + 1, &i3, &minus_i3);
    }

    let c10 = MatF64::<3, 4>::from_array2([
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.08, 0.01, 0.0],
        [0.0, 0.01, 0.07, 0.0],
    ]);
    let minus_c10 = -c10;
    let idx_00 = PartitionBlockIndex {
        partition: 0,
        block: 0,
    };
    let idx_10 = PartitionBlockIndex {
        partition: 1,
        block: 0,
    };
    mat_a_builder.add_lower_block(idx_10, idx_00, &minus_c10.as_view());
    mat_a_builder.add_lower_block(idx_10, idx_10, &i3.as_view());
    mat_a_builder.add_lower_block(idx_00, idx_00, &i4.as_view());

    // ---- Constraint partition (partition 2): -εI diagonal + dense coupling ----
    // Dense Jacobian rows with large coefficients create near-zero pivots after
    // elimination, stressing solvers without BK pivoting.
    for k in 0..num_constraints {
        let idx_c = PartitionBlockIndex {
            partition: 2,
            block: k,
        };
        let neg_eps = nalgebra::DMatrix::from_row_slice(1, 1, &[-eps]);
        mat_a_builder.add_lower_block(idx_c, idx_c, &neg_eps.as_view());

        // Couple to variable block k with dense Jacobian row.
        let g = nalgebra::DMatrix::from_row_slice(1, 4, &[1.0, 0.5, 0.3, 0.1]);
        let idx_var = PartitionBlockIndex {
            partition: 0,
            block: k,
        };
        mat_a_builder.add_lower_block(idx_c, idx_var, &g.as_view());

        // Also couple to a second variable block (k+5) for more fill-in.
        let g2 = nalgebra::DMatrix::from_row_slice(1, 4, &[0.4, 0.7, 0.2, 0.5]);
        let idx_var2 = PartitionBlockIndex {
            partition: 0,
            block: k + 5,
        };
        mat_a_builder.add_lower_block(idx_c, idx_var2, &g2.as_view());
    }

    let total_dim = NB0 * 4 + NB1 * 3 + num_constraints;
    let b = nalgebra::DVector::from_element(total_dim, 1.0);
    LinearSystem {
        mat_a: mat_a_builder.build(),
        b,
    }
}
