use sophus_bench::PuffinPrinter;
use sophus_solver::{
    IsFactor,
    LinearSolverEnum,
    ldlt::DenseLdlt,
    test_examples::positive_semidefinite::create_medium_linear_problem,
};

fn main() {
    let dense_mat_a =
        create_medium_linear_problem(&LinearSolverEnum::DenseLdlt(DenseLdlt::default()))
            .mat_a
            .into_dense()
            .unwrap();

    puffin::set_scopes_on(true);
    let printer = PuffinPrinter::new();

    for solver in LinearSolverEnum::all_solvers() {
        let linear_system = create_medium_linear_problem(&solver);

        puffin::GlobalProfiler::lock().new_frame();

        let x = {
            puffin::profile_scope!("total");
            let system = solver.factorize(&linear_system.mat_a).unwrap();
            system.solve(&linear_system.b).unwrap()
        };
        puffin::GlobalProfiler::lock().new_frame();

        approx::assert_abs_diff_eq!(
            dense_mat_a.view() * x,
            linear_system.b.clone(),
            epsilon = 1e-6
        );

        printer.print_latest(&solver.name()).unwrap();
    }
}
