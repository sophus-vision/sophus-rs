use nalgebra::DVector;
use sophus_autodiff::linalg::MatF64;
use sophus_solver::{
    CompressedMatrixEnum,
    DenseLdlt,
    LinearSolverEnum,
    PartitionSpec,
    SymmetricMatrixBuilderEnum,
    prelude::*,
};
use tracing::{
    Dispatch,
    dispatcher,
    info_span,
    trace,
};
use tracing_timing::{
    Builder,
    Histogram,
    TimingSubscriber,
    group,
};

struct ExampleProblem {
    partitions: Vec<PartitionSpec>,
}

impl ExampleProblem {
    fn new() -> Self {
        ExampleProblem {
            partitions: vec![PartitionSpec {
                block_count: 250,
                block_dimension: 4,
            }],
        }
    }

    fn partitions(&self) -> &[PartitionSpec] {
        &self.partitions
    }

    fn build(mut builder: SymmetricMatrixBuilderEnum) -> (CompressedMatrixEnum, DVector<f64>) {
        const R: usize = 25;
        const C: usize = 10;
        const NB: usize = R * C; // 250
        type M = MatF64<4, 4>;
        let i4 = M::from_array2([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let four_i = M::from_array2([
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]);

        let idx = |r: usize, c: usize| -> usize { r * C + c };

        let add_edge =
            |builder: &mut SymmetricMatrixBuilderEnum, a: usize, b: usize, c: &M, minus_c: &M| {
                let (i, j) = if a > b { (a, b) } else { (b, a) };
                builder.add_lower_block(&[0, 0], [i, j], &minus_c.as_view());
                builder.add_lower_block(&[0, 0], [i, i], &c.as_view());

                builder.add_lower_block(&[0, 0], [j, j], &c.as_view());
            };

        // Base diagonal
        for i in 0..NB {
            builder.add_lower_block(&[0, 0], [i, i], &four_i.as_view());
        }

        // 4-neighborhood with uniform coupling
        let minus_c: M = -i4;

        for r in 0..R {
            for c in 0..C {
                let i = idx(r, c);
                if r + 1 < R {
                    add_edge(&mut builder, i, idx(r + 1, c), &i4, &minus_c);
                }
                if c + 1 < C {
                    add_edge(&mut builder, i, idx(r, c + 1), &i4, &minus_c);
                }
            }
        }

        let b = nalgebra::DVector::from_element(NB * 4, 1.0);
        (builder.build().compress(), b)
    }
}

fn main() {
    let problem = ExampleProblem::new();

    let dense_builder =
        LinearSolverEnum::DenseLdlt(DenseLdlt {}).matrix_builder(problem.partitions());
    let dense_mat_a = ExampleProblem::build(dense_builder).0.into_dense().unwrap();

    for solver in LinearSolverEnum::all_solvers() {
        let builder = solver.matrix_builder(problem.partitions());
        let (mat_a, b) = ExampleProblem::build(builder);
        let mut x = b.clone();

        type TS = TimingSubscriber<group::ByField, group::ByField>;

        let subscriber: TS = Builder::default()
            .spans(group::ByField::from("algo"))
            .events(group::ByField::from("path"))
            .build(|| Histogram::new_with_max(60_000_000_000, 3).unwrap());

        let dispatch = Dispatch::new(subscriber);
        println!("{}", solver.name());

        dispatcher::with_default(&dispatch, || {
            let span = info_span!("solver_bench", algo = ?solver.name());
            span.in_scope(|| {
                trace!(path = "solve/before");
                solver.solve_in_place(false, &mat_a, &mut x).unwrap();
                trace!(path = "solve");
            })
        });

        approx::assert_abs_diff_eq!(dense_mat_a.clone() * x, b.clone(), epsilon = 1e-6);

        let timing = dispatch.downcast_ref::<TS>().unwrap();
        timing.force_synchronize();

        timing.with_histograms(|hs| {
            let algo_key = format!("{:?}", solver.name());

            let mut algo_ms = 0.0;

            if let Some(events) = hs.get(&algo_key) {
                for (path, h) in events {
                    let p50_ms = h.value_at_quantile(0.50) as f64 / 1e6;

                    // Total time across all calls in this run.
                    let mut total_ns: u128 = 0;
                    for rec in h.iter_recorded() {
                        total_ns += (rec.count_since_last_iteration() as u128)
                            * (rec.value_iterated_to() as u128);
                    }
                    let call_count = h.len();
                    let total_ms = total_ns as f64 / 1e6;
                    algo_ms += total_ms;

                    println!(
                        "{path:<50} #call={call_count:>5}  p50={p50_ms:>7.2} ms  \
                        total={total_ms:>8.2} ms"
                    );
                }
            }

            println!("algo_ms: {algo_ms} ms");
        });
        println!();
    }
}
