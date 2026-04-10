//! Bundle adjustment benchmark.
//!
//! Compares standard (all-free) vs Schur-complement optimization across solvers,
//! with and without conditioned cameras. Also benchmarks BA with scale constraints.
//!
//! Run: `just ba-bench`

use sophus_opt::example_problems::{
    ba_problem::BaProblem,
    ba_scale_constraint::BaScaleConstraintProblem,
};
use sophus_solver::LinearSolverEnum;

fn time_ms<T>(f: impl FnOnce() -> T) -> (f64, T) {
    let t = std::time::Instant::now();
    let v = f();
    (t.elapsed().as_secs_f64() * 1000.0, v)
}

// ── Standard BA ──────────────────────────────────────────────────────────────

struct Row {
    problem: &'static str,
    cam_kind: &'static str,
    solver: &'static str,
    standard_ms: f64,
    schur_ms: f64,
    standard_pose_rms: f64,
    standard_point_rms: f64,
    schur_pose_rms: f64,
    schur_point_rms: f64,
}

fn bench_solver(
    solver: LinearSolverEnum,
    solver_name: &'static str,
    cases: &[(&'static str, &BaProblem)],
    const_cams: bool,
    parallelize: bool,
    rows: &mut Vec<Row>,
) {
    let schur_solver = solver
        .to_schur()
        .expect("ba_solvers must have a Schur variant");
    let cam_kind = if const_cams { "conditioned" } else { "free" };
    for (prob_name, problem) in cases {
        // warm-up both paths
        if const_cams {
            problem.optimize_all_free_const_cams(solver, parallelize);
            problem.optimize_marg_points_const_cams(schur_solver, parallelize);
        } else {
            problem.optimize_all_free(solver, parallelize);
            problem.optimize_marg_points(schur_solver, parallelize);
        }

        let (standard_ms, standard_vars) = if const_cams {
            time_ms(|| problem.optimize_all_free_const_cams(solver, parallelize))
        } else {
            time_ms(|| problem.optimize_all_free(solver, parallelize))
        };
        let (schur_ms, schur_vars) = if const_cams {
            time_ms(|| problem.optimize_marg_points_const_cams(schur_solver, parallelize))
        } else {
            time_ms(|| problem.optimize_marg_points(schur_solver, parallelize))
        };

        let (standard_pose_rms, standard_point_rms) = problem.gt_errors(&standard_vars);
        let (schur_pose_rms, schur_point_rms) = problem.gt_errors(&schur_vars);

        rows.push(Row {
            problem: prob_name,
            cam_kind,
            solver: solver_name,
            standard_ms,
            schur_ms,
            standard_pose_rms,
            standard_point_rms,
            schur_pose_rms,
            schur_point_rms,
        });
    }
}

fn print_table(rows: &[Row], label: &str) {
    const POSE_TOL: f64 = 0.01;
    const POINT_TOL: f64 = 0.02;

    let w = 152;
    println!("\n{}", "=".repeat(w));
    println!("  Bundle Adjustment  (20 iters, {label})");
    println!("{}", "=".repeat(w));
    println!(
        "  {:<18}  {:<12}  {:<22}  {:>10}  {:>10}  {:>8}  {:>13}  {:>11}  {:>13}  {:>11}  {:>5}",
        "Problem",
        "Cameras",
        "Solver",
        "std ms",
        "schur ms",
        "speedup",
        "std pose RMS",
        "std pt RMS",
        "schur pose RMS",
        "schur pt RMS",
        "OK"
    );
    println!("{}", "-".repeat(w));
    for r in rows {
        let speedup = r.standard_ms / r.schur_ms;
        let ok = r.standard_pose_rms < POSE_TOL
            && r.standard_point_rms < POINT_TOL
            && r.schur_pose_rms < POSE_TOL
            && r.schur_point_rms < POINT_TOL;
        println!(
            "  {:<18}  {:<12}  {:<22}  {:>10.1}  {:>10.1}  {:>7.2}x  {:>12.4}m  {:>10.4}m  {:>12.4}m  {:>10.4}m  {:>5}",
            r.problem,
            r.cam_kind,
            r.solver,
            r.standard_ms,
            r.schur_ms,
            speedup,
            r.standard_pose_rms,
            r.standard_point_rms,
            r.schur_pose_rms,
            r.schur_point_rms,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    println!("{}", "=".repeat(w));
}

// ── BA + scale constraint ─────────────────────────────────────────────────────

struct ScaleRow {
    problem: &'static str,
    solver: &'static str,
    schur: bool,
    ms: f64,
    pose_rms: f64,
    point_rms: f64,
    constraint_ok: bool,
}

fn bench_scale(
    solver: LinearSolverEnum,
    solver_name: &'static str,
    schur: bool,
    cases: &[(&'static str, &BaScaleConstraintProblem)],
    parallelize: bool,
    rows: &mut Vec<ScaleRow>,
) {
    for (prob_name, problem) in cases {
        // warm-up
        if schur {
            problem.optimize_marg_points(solver, parallelize);
        } else {
            problem.optimize_all_free(solver, parallelize);
        }

        let (ms, vars) = if schur {
            time_ms(|| problem.optimize_marg_points(solver, parallelize))
        } else {
            time_ms(|| problem.optimize_all_free(solver, parallelize))
        };

        let (pose_rms, point_rms) = problem.gt_errors(&vars);
        let constraint_ok = problem.constraint_satisfied(&vars);

        rows.push(ScaleRow {
            problem: prob_name,
            solver: solver_name,
            schur,
            ms,
            pose_rms,
            point_rms,
            constraint_ok,
        });
    }
}

fn print_scale_table(rows: &[ScaleRow], label: &str) {
    const POSE_TOL: f64 = 0.06;
    const POINT_TOL: f64 = 0.06;

    let w = 108;
    println!("\n{}", "=".repeat(w));
    println!("  BA + Scale Constraint  (20 iters, {label})");
    println!("{}", "=".repeat(w));
    println!(
        "  {:<20}  {:<28}  {:<5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>5}",
        "Problem", "Solver", "Schur", "ms", "pose RMS", "point RMS", "constraint", "OK"
    );
    println!("{}", "-".repeat(w));
    for r in rows {
        let constraint_str = if r.constraint_ok { "satisfied" } else { "FAIL" };
        let ok = r.pose_rms < POSE_TOL && r.point_rms < POINT_TOL && r.constraint_ok;
        println!(
            "  {:<20}  {:<28}  {:<5}  {:>10.1}  {:>9.4}m  {:>9.4}m  {:>10}  {:>5}",
            r.problem,
            r.solver,
            if r.schur { "yes" } else { "no" },
            r.ms,
            r.pose_rms,
            r.point_rms,
            constraint_str,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    println!("{}", "=".repeat(w));
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    eprintln!("building problems...");
    let ba = BaProblem::new(10, 500);
    let scale = BaScaleConstraintProblem::new(10, 500);

    let ba_cases: &[(&'static str, &BaProblem)] = &[("10c/500pts", &ba)];
    let scale_cases: &[(&'static str, &BaScaleConstraintProblem)] = &[("10c/500pts", &scale)];

    let ba_solvers = LinearSolverEnum::ba_solvers();
    let ba_eq_solvers = LinearSolverEnum::ba_eq_solvers();

    for (label, parallelize) in [("sequential", false), ("parallel", true)] {
        // ── Standard BA ──────────────────────────────────────────────────────
        eprintln!("benchmarking BA ({label})...");
        let mut rows: Vec<Row> = Vec::new();
        for const_cams in [false, true] {
            for (solver, name) in &ba_solvers {
                bench_solver(*solver, name, ba_cases, const_cams, parallelize, &mut rows);
            }
        }

        // ── Scale-constraint BA ───────────────────────────────────────────────
        eprintln!("benchmarking BA+scale ({label})...");
        let mut scale_rows: Vec<ScaleRow> = Vec::new();
        for (solver, name, schur) in &ba_eq_solvers {
            bench_scale(
                *solver,
                name,
                *schur,
                scale_cases,
                parallelize,
                &mut scale_rows,
            );
        }

        print_table(&rows, label);
        print_scale_table(&scale_rows, label);
    }
}
