use nalgebra::DVector;
use sophus_autodiff::linalg::MatF64;
use sophus_solver::{
    LinearSolverEnum,
    bench_utils::PuffinPrinter,
    matrix::{
        CompressedMatrixEnum,
        PartitionSpec,
        SymmetricMatrixBuilderEnum,
    },
    positive_semidefinite::DenseLdlt,
    prelude::*,
};

struct ExampleProblem {
    partitions: Vec<PartitionSpec>,
}
impl ExampleProblem {
    fn new() -> Self {
        // Partition 0: 250 blocks of size 4 (unchanged)
        // Partition 1: 60 blocks of size 3 (new)
        ExampleProblem {
            partitions: vec![
                PartitionSpec {
                    block_count: 250,
                    block_dimension: 4,
                },
                PartitionSpec {
                    block_count: 60,
                    block_dimension: 3,
                },
            ],
        }
    }

    fn partitions(&self) -> &[PartitionSpec] {
        &self.partitions
    }

    fn build(mut builder: SymmetricMatrixBuilderEnum) -> (CompressedMatrixEnum, DVector<f64>) {
        let b = {
            puffin::profile_scope!("build");

            // ---------------- Partition 0 (size 4 blocks) ----------------
            const R0: usize = 25;
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

            let add_edge4 = |builder: &mut SymmetricMatrixBuilderEnum,
                             a: usize,
                             b: usize,
                             c: &M4,
                             minus_c: &M4| {
                let (i, j) = if a > b { (a, b) } else { (b, a) };
                // Region [0,0] = (row partition 0, col partition 0)
                builder.add_lower_block(&[0, 0], [i, j], &minus_c.as_view());
                builder.add_lower_block(&[0, 0], [i, i], &c.as_view());
                builder.add_lower_block(&[0, 0], [j, j], &c.as_view());
            };

            // Base diagonal (partition 0)
            for i in 0..NB0 {
                builder.add_lower_block(&[0, 0], [i, i], &four_i4.as_view());
            }

            // 4-neighborhood (partition 0)
            let minus_i4: M4 = -i4;
            for r in 0..R0 {
                for c in 0..C0 {
                    let i = idx0(r, c);
                    if r + 1 < R0 {
                        add_edge4(&mut builder, i, idx0(r + 1, c), &i4, &minus_i4);
                    }
                    if c + 1 < C0 {
                        add_edge4(&mut builder, i, idx0(r, c + 1), &i4, &minus_i4);
                    }
                }
            }

            // ---------------- Partition 1 (size 3 blocks) ----------------
            const NB1: usize = 60; // choose any length you like
            type M3 = MatF64<3, 3>;

            let i3 = M3::from_array2([[1.0, 0.02, 0.0], [0.02, 1.1, 0.01], [0.0, 0.01, 1.2]]);
            let three_i3 =
                M3::from_array2([[3.0, 0.001, 0.0], [0.001, 3.1, 0.0007], [0.0, 0.0007, 3.2]]);
            let minus_i3: M3 = -i3;

            let add_edge3 = |builder: &mut SymmetricMatrixBuilderEnum,
                             a: usize,
                             b: usize,
                             c: &M3,
                             minus_c: &M3| {
                let (i, j) = if a > b { (a, b) } else { (b, a) };
                // Region [1,1] = (row partition 1, col partition 1)
                builder.add_lower_block(&[1, 1], [i, j], &minus_c.as_view());
                builder.add_lower_block(&[1, 1], [i, i], &c.as_view());
                builder.add_lower_block(&[1, 1], [j, j], &c.as_view());
            };

            // Base diagonal (partition 1)
            for i in 0..NB1 {
                builder.add_lower_block(&[1, 1], [i, i], &three_i3.as_view());
            }

            // Simple chain coupling within partition 1
            for i in 0..(NB1 - 1) {
                add_edge3(&mut builder, i, i + 1, &i3, &minus_i3);
            }

            // Region [1,0]: row-partition 1 (size 3), col-partition 0 (size 4)
            let c10 = MatF64::<3, 4>::from_array2([
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.08, 0.01, 0.0],
                [0.0, 0.01, 0.07, 0.0],
            ]);
            let minus_c10 = -c10;
            builder.add_lower_block(&[1, 0], [0, 0], &minus_c10.as_view());
            builder.add_lower_block(&[1, 1], [0, 0], &(/* 3x3 */i3).as_view()); // add to the two diagonals
            builder.add_lower_block(&[0, 0], [0, 0], &(/* 4x4 */i4).as_view());

            // ---------------- RHS vector ----------------
            // total scalar size = 250*4 + 60*3
            nalgebra::DVector::from_element(NB0 * 4 + NB1 * 3, 1.0)
        };

        puffin::profile_scope!("compress");
        (builder.build().compress(), b)
    }
}

fn main() {
    let problem = ExampleProblem::new();

    let dense_builder =
        LinearSolverEnum::DenseLdlt(DenseLdlt::default()).matrix_builder(problem.partitions());
    let (mat_a, b) = ExampleProblem::build(dense_builder);
    let dense_mat_a = mat_a.into_dense().unwrap();

    puffin::set_scopes_on(true);
    let printer = PuffinPrinter::new();

    for solver in LinearSolverEnum::all_solvers() {
        let builder = solver.matrix_builder(problem.partitions());

        puffin::GlobalProfiler::lock().new_frame();

        let x = {
            puffin::profile_scope!("total");
            let (mat_a, b) = ExampleProblem::build(builder);
            solver.solve(false, &mat_a, &b).unwrap()
        };
        puffin::GlobalProfiler::lock().new_frame();

        approx::assert_abs_diff_eq!(dense_mat_a.clone() * x, b.clone(), epsilon = 1e-6);

        printer.print_latest(&solver.name()).unwrap();
    }
}
