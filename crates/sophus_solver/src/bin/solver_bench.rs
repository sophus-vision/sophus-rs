use sophus_solver::{
    IsLinearSolver,
    LinearSolverEnum,
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

fn main() {
    // for solver in LinearSolverEnum::sparse_solvers() {
    //     type TS = TimingSubscriber<group::ByField, group::ByField>;

    //     let subscriber: TS = Builder::default()
    //         .spans(group::ByField::from("algo")) // group spans by span field "algo"
    //         .events(group::ByField::from("path")) // group events by event field "path"
    //         .build(|| Histogram::new_with_max(10_000_000_000, 2).unwrap());

    //     let dispatch = Dispatch::new(subscriber);

    //     // run the workload under this subscriber
    //     dispatcher::with_default(&dispatch, || {
    //         println!("{}", solver.name());
    //         let span = info_span!("pose_circle_run", algo = ?solver.name());
    //         span.in_scope(|| {
    //             let pg = PoseCircleProblem::new(2500);

    //             let e0 = pg.calc_error(&pg.est_world_from_robot);
    //             assert!(e0 > 1.0, "{e0} > thr?");
    //             trace!(path = "pose_circle/calc_error_before");

    //             let refined = pg.optimize(solver);
    //             trace!(path = "pose_circle/optimize");

    //             let e1 = pg.calc_error(&refined);
    //             assert!(e1 < 0.05, "{e1} < thr?");
    //             trace!(path = "pose_circle/calc_error_after");
    //         });
    //     });

    //     // --- read back histograms (no &mut needed) ---
    //     let timing = dispatch.downcast_ref::<TS>().unwrap();
    //     timing.force_synchronize(); // refresh everything once, here.
    // :contentReference[oaicite:2]{index=2}

    //     timing.with_histograms(|hs| {
    //         let algo_key = format!("{:?}", solver.name()); // matches `?solver` on the span

    //         let mut algo_ms = 0.0;

    //         if let Some(events) = hs.get(&algo_key) {
    //             for (path, h) in events {
    //                 // 1) Per-call stats (median etc.)
    //                 let p50_ms = h.value_at_quantile(0.50) as f64 / 1e6;

    //                 // 2) Total time across all calls in this run
    //                 let mut total_ns: u128 = 0;
    //                 for rec in h.iter_recorded() {
    //                     // each bucket with value & count
    //                     total_ns += (rec.count_since_last_iteration() as u128)
    //                         * (rec.value_iterated_to() as u128);
    //                 }
    //                 let calls = h.len(); // total sample count
    //                 let total_ms = total_ns as f64 / 1e6;
    //                 algo_ms += total_ms;

    //                 println!(
    //                     "{path:<50} calls={calls:>5}  p50={p50_ms:>7.2} ms  total={total_ms:>8.2}
    // ms"                 );
    //             }
    //         }

    //         println!("algo_ms: {algo_ms} ms");
    //     });
    // }
}
