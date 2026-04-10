//! Large-scale pose graph benchmark.
//!
//! Run: `cargo run --release --features std --bin pose_graph_bench`

use std::time::Instant;

use sophus_solver::{
    IsFactor,
    LinearSolverEnum,
    ldlt::{
        BlockSparseLdlt,
        FaerSparseLdlt,
        SparseLdlt,
    },
    test_examples::positive_semidefinite::large_pose_graph::create_large_pose_graph,
};

fn bench_solver(solver: LinearSolverEnum, n: usize) -> (f64, nalgebra::DVector<f64>) {
    let system = create_large_pose_graph(&solver, n);

    // Warm-up.
    let _ = solver.factorize(&system.mat_a).unwrap().solve(&system.b);

    let t = Instant::now();
    let factor = solver.factorize(&system.mat_a).unwrap();
    let x = factor.solve(&system.b).unwrap();
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    (ms, x)
}

fn main() {
    let sizes = [100, 500, 1000, 2500, 5000];

    let solvers: Vec<(LinearSolverEnum, &'static str)> = vec![
        (
            LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
            "block-LDLt",
        ),
        (
            LinearSolverEnum::SparseLdlt(SparseLdlt::default()),
            "sparse-LDLt",
        ),
        (
            LinearSolverEnum::FaerSparseLdlt(FaerSparseLdlt::default()),
            "faer-LDLt",
        ),
    ];

    let w = 90;
    println!();
    println!("{}", "=".repeat(w));
    println!("  Pose Graph Benchmark (6x6 SE3 blocks, grid+loop topology)");
    println!("{}", "=".repeat(w));
    print!("  {:>6}  {:>8}", "poses", "dim");
    for (_, name) in &solvers {
        print!("  {:>12}", name);
    }
    println!("  {:>8}", "OK");
    println!("{}", "-".repeat(w));

    for &n in &sizes {
        let dim = n * 6;
        let mut results: Vec<(&str, f64)> = Vec::new();
        let mut solutions: Vec<nalgebra::DVector<f64>> = Vec::new();

        for (solver, name) in &solvers {
            let (ms, x) = bench_solver(*solver, n);
            results.push((name, ms));
            solutions.push(x);
        }

        let ok = solutions
            .windows(2)
            .all(|pair| (&pair[0] - &pair[1]).norm() < 1e-4);

        print!("  {:>6}  {:>8}", n, dim);
        for (_, ms) in &results {
            print!("  {:>10.1} ms", ms);
        }
        println!("  {:>8}", if ok { "PASS" } else { "FAIL" });
    }
    println!("{}", "=".repeat(w));
}
