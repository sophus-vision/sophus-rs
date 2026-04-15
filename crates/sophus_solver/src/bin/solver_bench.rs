use std::time::Instant;

use sophus_solver::{
    IsFactor,
    LinearSolverEnum,
    ldlt::DenseLdlt,
    test_examples::positive_semidefinite::{
        create_medium_indefinite_problem,
        create_medium_linear_problem,
    },
};

struct Row {
    solver: String,
    ms: f64,
    residual: f64,
}

fn bench_solver(
    solver: LinearSolverEnum,
    dense_a: &nalgebra::DMatrix<f64>,
    b: &nalgebra::DVector<f64>,
    mat_a: &sophus_solver::matrix::SymmetricMatrixEnum,
) -> Row {
    // warm-up
    let _ = solver.factorize(mat_a).unwrap().solve(b);

    let t = Instant::now();
    let factor = solver.factorize(mat_a).unwrap();
    let x = factor.solve(b).unwrap();
    let ms = t.elapsed().as_secs_f64() * 1000.0;

    let residual = (dense_a * &x - b).norm();
    Row {
        solver: solver.name(),
        ms,
        residual,
    }
}

fn print_table(title: &str, dim: usize, rows: &[Row]) {
    let w = 70;
    println!("\n{}", "=".repeat(w));
    println!("  {}  ({}x{})", title, dim, dim);
    println!("{}", "=".repeat(w));
    println!(
        "  {:<30}  {:>10}  {:>12}  {:>5}",
        "Solver", "ms", "residual", "OK"
    );
    println!("{}", "-".repeat(w));
    for r in rows {
        let ok = r.residual < 1e-6;
        println!(
            "  {:<30}  {:>10.3}  {:>12.2e}  {:>5}",
            r.solver,
            r.ms,
            r.residual,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    println!("{}", "=".repeat(w));
}

fn main() {
    let ref_solver = LinearSolverEnum::DenseLdlt(DenseLdlt::default());

    // ── PD benchmark (480×480) ──────────────────────────────────────────
    let pd_ref = create_medium_linear_problem(&ref_solver);
    let dense_pd = pd_ref.mat_a.as_dense().unwrap().clone();
    let pd_dim = dense_pd.scalar_dimension();

    let mut pd_rows = Vec::new();
    for solver in LinearSolverEnum::all_solvers() {
        let system = create_medium_linear_problem(&solver);
        pd_rows.push(bench_solver(
            solver,
            &dense_pd.view().clone_owned(),
            &system.b,
            &system.mat_a,
        ));
    }
    print_table("Positive Definite", pd_dim, &pd_rows);

    // ── KKT benchmark (490×490) ─────────────────────────────────────────
    let kkt_ref = create_medium_indefinite_problem(&ref_solver);
    let dense_kkt = kkt_ref.mat_a.as_dense().unwrap().clone();
    let kkt_dim = dense_kkt.scalar_dimension();

    let mut kkt_rows = Vec::new();
    for solver in LinearSolverEnum::robust_indefinite_solvers() {
        let system = create_medium_indefinite_problem(&solver);
        kkt_rows.push(bench_solver(
            solver,
            &dense_kkt.view().clone_owned(),
            &system.b,
            &system.mat_a,
        ));
    }
    print_table("Indefinite (KKT)", kkt_dim, &kkt_rows);
}
