use crossbeam_channel::Sender;
use eframe::egui;
use log::warn;
use sophus_autodiff::linalg::{
    SVec,
    VecF64,
};
use sophus_image::ImageSize;
use sophus_lie::{
    IsAffineGroup,
    Isometry2F64,
    Isometry3,
    Isometry3F64,
    Rotation3,
};
use sophus_opt::{
    example_problems::inverse_depth_estimation::{
        InverseDepthProblem,
        to_cartesian_2d,
    },
    nlls::{
        OptParams,
        OptimizationSolution,
        Optimizer,
    },
};
use sophus_renderer::{
    camera::{
        ClippingPlanes,
        RenderCamera,
        RenderCameraProperties,
    },
    renderables::{
        Color,
        LineSegment3,
        axes3,
        make_point3,
        named_line3,
    },
};
use sophus_solver::{
    LinearSolverEnum,
    ldlt::{
        BlockSparseLdlt,
        FaerSparseLdlt,
    },
    lu::FaerSparseLu,
    qr::FaerSparseQr,
};
use sophus_viewer::packets::{
    ClearCondition,
    LineType,
    Packet,
    PlotViewPacket,
    ScalarCurveStyle,
    append_to_scene_packet,
    create_scene_packet,
    delete_scene_packet,
};

use super::opt_widget::{
    OptWidgetState,
    StepRequest,
};

const SCENE_LABEL: &str = "inv-depth - scene";
const PLOT_LABEL: &str = "inv-depth - cost";
const DAMPING_PLOT_LABEL: &str = "inv-depth - damping";
const MAX_ITERATIONS: usize = 200;

/// Solver for inverse depth BA with equality constraints.
#[derive(Copy, Clone, Debug, PartialEq)]
enum InvDepthSolver {
    FaerSparseLu,
    FaerSparseQr,
    SchurBlockSparseLdlt,
    SchurFaerSparseLdlt,
}

impl InvDepthSolver {
    fn label(self) -> &'static str {
        match self {
            InvDepthSolver::FaerSparseLu => "faer sparse LU",
            InvDepthSolver::FaerSparseQr => "faer sparse QR",
            InvDepthSolver::SchurBlockSparseLdlt => "Schur + block-sparse LDLᵀ",
            InvDepthSolver::SchurFaerSparseLdlt => "Schur + faer sparse LDLᵀ",
        }
    }

    fn uses_schur(self) -> bool {
        matches!(
            self,
            InvDepthSolver::SchurBlockSparseLdlt | InvDepthSolver::SchurFaerSparseLdlt
        )
    }

    fn to_linear_solver(self) -> LinearSolverEnum {
        match self {
            InvDepthSolver::FaerSparseLu => LinearSolverEnum::FaerSparseLu(FaerSparseLu {}),
            InvDepthSolver::FaerSparseQr => LinearSolverEnum::FaerSparseQr(FaerSparseQr {}),
            InvDepthSolver::SchurBlockSparseLdlt => {
                LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default())
            }
            InvDepthSolver::SchurFaerSparseLdlt => {
                LinearSolverEnum::SchurFaerSparseLdlt(FaerSparseLdlt::default())
            }
        }
    }

    fn all() -> [InvDepthSolver; 4] {
        [
            InvDepthSolver::FaerSparseLu,
            InvDepthSolver::FaerSparseQr,
            InvDepthSolver::SchurBlockSparseLdlt,
            InvDepthSolver::SchurFaerSparseLdlt,
        ]
    }
}

/// Embed a 2D pose (SE(2) in the xz-plane) into a 3D isometry for rendering.
///
/// SE(2) coordinates: `(lateral, depth)` → 3D: `(x, 0, z)`.
/// SE(2) rotation θ → 3D rotation around y-axis.
fn embed_pose_2d_to_3d(pose: &Isometry2F64) -> Isometry3F64 {
    let t = pose.translation();
    let angle = pose.rotation().log()[0];
    Isometry3::from_rotation_and_translation(
        Rotation3::exp(VecF64::<3>::new(0.0, angle, 0.0)),
        VecF64::<3>::new(t[0], 0.0, t[1]),
    )
}

/// Interactive inverse depth bundle adjustment demo.
///
/// Visualizes 2D point and pose estimation using inverse depth parameterization
/// with SE(2) poses and 1D camera. Covariance ellipses show pose translation
/// uncertainty in the xz-plane.
pub struct InverseDepthWidget {
    message_send: Sender<Vec<Packet>>,
    problem: InverseDepthProblem,
    solver: InvDepthSolver,
    opt: OptWidgetState,
    show_covariance: bool,
    cov_sigma: f32,
    /// Previous values for detecting UI changes requiring redraw.
    prev_show_covariance: bool,
    prev_cov_sigma: f32,
    /// Cached optimization solution for covariance queries.
    cached_solution: Option<OptimizationSolution>,
}

impl Drop for InverseDepthWidget {
    fn drop(&mut self) {
        match self.message_send.send(vec![
            delete_scene_packet(SCENE_LABEL),
            Packet::Plot(vec![PlotViewPacket::Delete(PLOT_LABEL.to_owned())]),
            Packet::Plot(vec![PlotViewPacket::Delete(DAMPING_PLOT_LABEL.to_owned())]),
        ]) {
            Ok(_) => {}
            Err(_) => {
                warn!("Failed to send delete packets.");
            }
        }
    }
}

impl InverseDepthWidget {
    /// Create the widget.
    pub fn new(message_send: Sender<Vec<Packet>>) -> Self {
        let problem = InverseDepthProblem::new();
        let solver = InvDepthSolver::SchurBlockSparseLdlt;
        let optimizer = Self::build_optimizer(&problem, solver);
        let opt = OptWidgetState::new(optimizer, MAX_ITERATIONS, PLOT_LABEL, DAMPING_PLOT_LABEL);

        let mut widget = InverseDepthWidget {
            message_send,
            problem,
            solver,
            opt,
            show_covariance: true,
            cov_sigma: 3.0,
            prev_show_covariance: true,
            prev_cov_sigma: 3.0,
            cached_solution: None,
        };
        widget.send_scene_setup();
        widget.send_current_estimates();
        widget
    }

    fn build_optimizer(problem: &InverseDepthProblem, solver: InvDepthSolver) -> Optimizer {
        let vars = if solver.uses_schur() {
            problem.build_variables_schur()
        } else {
            problem.build_variables()
        };
        Optimizer::new_with_eq(
            vars,
            vec![problem.build_cost()],
            vec![problem.build_eq_constraint()],
            OptParams {
                num_iterations: 10000,
                initial_lm_damping: 1.0,
                parallelize: false,
                solver: solver.to_linear_solver(),
                skip_final_hessian: false,
                ..Default::default()
            },
        )
        .expect("failed to build inverse depth optimizer")
    }

    /// Draw the left-panel controls.
    pub fn update_left_panel(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Inverse Depth BA (2D)");
        ui.label("SE(2) poses + inv-depth points.");

        ui.checkbox(&mut self.show_covariance, "show covariance");
        ui.add(egui::Slider::new(&mut self.cov_sigma, 1.0..=5.0).text("σ"));

        // Solver selection (disabled while running)
        let is_running = self.opt.step_request == StepRequest::Run;
        ui.add_enabled_ui(!is_running, |ui| {
            let mut changed = false;
            egui::ComboBox::from_label("solver")
                .selected_text(self.solver.label())
                .show_ui(ui, |ui| {
                    for choice in InvDepthSolver::all() {
                        if ui
                            .selectable_label(self.solver == choice, choice.label())
                            .clicked()
                        {
                            self.solver = choice;
                            changed = true;
                        }
                    }
                });
            if changed {
                self.reset();
            }
        });

        if self.opt.draw_buttons(ui) {
            self.reset();
        }
        self.opt.draw_status(ui, "");
    }

    /// Called each frame.
    pub fn update(&mut self) {
        let mut stepped = false;
        if let Some(info) = self.opt.update() {
            let cost = info.merit;
            stepped = true;
            // Invalidate cached solution when optimizer takes a step
            self.cached_solution = None;

            if let Some(opt) = &self.opt.optimizer {
                let damping = opt.lm_damping();
                self.send_current_estimates();
                self.send_cost_plot(cost);
                self.send_damping_plot(damping);
            }
        }

        // Redraw when UI controls change (slider dragged, checkbox toggled)
        if !stepped
            && (self.show_covariance != self.prev_show_covariance
                || self.cov_sigma != self.prev_cov_sigma)
        {
            self.send_current_estimates();
        }
        self.prev_show_covariance = self.show_covariance;
        self.prev_cov_sigma = self.cov_sigma;
    }

    fn reset(&mut self) {
        self.problem = InverseDepthProblem::new();
        let optimizer = Self::build_optimizer(&self.problem, self.solver);
        self.opt.reset();
        self.opt.optimizer = Some(optimizer);
        self.cached_solution = None;
        let _ = self.message_send.send(vec![
            Packet::Plot(vec![PlotViewPacket::Delete(PLOT_LABEL.to_owned())]),
            Packet::Plot(vec![PlotViewPacket::Delete(DAMPING_PLOT_LABEL.to_owned())]),
        ]);
        self.send_scene_setup();
    }

    fn send_scene_setup(&self) {
        let image_size = ImageSize {
            width: 640,
            height: 480,
        };
        let viewer_camera = RenderCamera {
            properties: RenderCameraProperties::new(
                sophus_sensor::DynCameraF64::new_pinhole(
                    VecF64::<4>::new(500.0, 500.0, 320.0, 240.0),
                    image_size,
                ),
                ClippingPlanes::default(),
            ),
            // Bird's-eye view looking down at the xz-plane.
            // Camera z-axis → world -y (looking down), camera y-axis → world +z (up in image).
            scene_from_camera: Isometry3::from_rotation_and_translation(
                Rotation3::exp(VecF64::<3>::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0)),
                VecF64::<3>::new(0.0, 50.0, 15.0),
            ),
        };
        let mut packets = vec![create_scene_packet(SCENE_LABEL, viewer_camera, true)];

        // Camera frustums (ground truth poses, embedded into 3D)
        let gt_poses_3d: Vec<Isometry3F64> = self
            .problem
            .true_world_from_cameras
            .iter()
            .map(embed_pose_2d_to_3d)
            .collect();
        let cam_lines = axes3(&gt_poses_3d).scale(0.5).line_width(1.0).build();
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![named_line3("cameras_gt", cam_lines)],
        ));

        // Ground truth points (green) — 2D (x,z) embedded into 3D (x, 0, z)
        let gt_pts: Vec<[f32; 3]> = self
            .problem
            .ground_truth_cartesian()
            .iter()
            .map(|p| [p[0] as f32, 0.0, p[1] as f32])
            .collect();
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![make_point3("pts_gt", &gt_pts, &Color::green(), 2.0)],
        ));

        // Initial estimate points (orange)
        let init_pts: Vec<[f32; 3]> = self
            .problem
            .init_points_inv
            .iter()
            .map(|p| {
                let c = to_cartesian_2d(p);
                [c[0] as f32, 0.0, c[1] as f32]
            })
            .collect();
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![make_point3("pts_init", &init_pts, &Color::orange(), 2.0)],
        ));

        let _ = self.message_send.send(packets);

        // Initialize plots
        let clear_cond = ClearCondition {
            max_x_range: MAX_ITERATIONS as f64 + 1.0,
        };
        let _ = self.message_send.send(vec![
            Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (PLOT_LABEL, "cost"),
                std::collections::VecDeque::new(),
                ScalarCurveStyle {
                    color: Color::blue(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )]),
            Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (DAMPING_PLOT_LABEL, "lm damping"),
                std::collections::VecDeque::new(),
                ScalarCurveStyle {
                    color: Color::green(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )]),
        ]);
    }

    fn send_current_estimates(&mut self) {
        let (points, poses) = {
            let opt = match &self.opt.optimizer {
                Some(o) => o,
                None => return,
            };
            let vars = opt.variables();
            (
                vars.get_members::<VecF64<2>>(InverseDepthProblem::POINTS),
                vars.get_members::<Isometry2F64>(InverseDepthProblem::POSES),
            )
        };

        // Estimated points in Cartesian (red) — 2D embedded into 3D
        let est_pts: Vec<[f32; 3]> = points
            .iter()
            .map(|p| {
                let c = to_cartesian_2d(p);
                [c[0] as f32, 0.0, c[1] as f32]
            })
            .collect();

        let mut renderables = vec![make_point3("pts_est", &est_pts, &Color::red(), 2.0)];

        // Estimated camera poses (yellow axes, embedded into 3D)
        let poses_3d: Vec<Isometry3F64> = poses.iter().map(embed_pose_2d_to_3d).collect();
        let cam_lines = axes3(&poses_3d).scale(0.3).line_width(1.0).build();
        renderables.push(named_line3("cameras_est", cam_lines));

        // Covariance visualization
        if self.show_covariance
            && self.opt.step_request == StepRequest::None
            && let Some(lines) = self.build_covariance_lines(&points, &poses)
        {
            renderables.push(named_line3("covariance", lines));
        }

        let _ = self
            .message_send
            .send(vec![append_to_scene_packet(SCENE_LABEL, renderables)]);
    }

    /// Build covariance ellipse line segments for points and poses in the xz-plane.
    ///
    /// For each point:
    /// 1. Extract the 2×2 covariance in (a, ψ) space
    /// 2. Sample the ellipse in (a, ψ) and map to Cartesian (x, z) (nonlinearly curved)
    ///
    /// For each free pose:
    /// 1. Extract the 3×3 covariance block in se2 tangent space
    /// 2. Take the translation sub-block (rows/cols 1..3)
    /// 3. Draw an ellipse at the camera position
    fn build_covariance_lines(
        &mut self,
        points: &[VecF64<2>],
        poses: &[Isometry2F64],
    ) -> Option<Vec<LineSegment3>> {
        // Cache the optimization solution so we don't re-run on every frame.
        let first_compute = self.cached_solution.is_none();
        if first_compute {
            self.cached_solution = Some(
                self.problem
                    .optimize_with_solver(self.solver.to_linear_solver(), self.solver.uses_schur()),
            );
        }
        let solution = self.cached_solution.as_mut().unwrap();

        let mut all_lines = Vec::new();
        let sigma = self.cov_sigma as f64;

        // Colors per point for distinguishing ellipsoids
        let point_colors = [
            Color::new(0.8, 0.2, 0.8, 1.0), // magenta
            Color::new(0.2, 0.8, 0.8, 1.0), // cyan
            Color::new(0.8, 0.8, 0.2, 1.0), // yellow
            Color::new(0.5, 0.5, 1.0, 1.0), // light blue
            Color::new(1.0, 0.5, 0.5, 1.0), // light red
            Color::new(0.5, 1.0, 0.5, 1.0), // light green
            Color::new(1.0, 0.7, 0.3, 1.0), // amber
            Color::new(0.7, 0.3, 1.0, 1.0), // purple
        ];

        // --- Point covariance (inverse-depth ellipses mapped to Cartesian) ---
        // Sample the covariance ellipse in (a, ψ) space, map to Cartesian (x, z).
        let n_pt_samples = 64;
        for (pt_idx, a_psi) in points.iter().enumerate() {
            let cov = match solution.covariance_block(
                InverseDepthProblem::POINTS,
                pt_idx,
                InverseDepthProblem::POINTS,
                pt_idx,
            ) {
                Some(c) => c,
                None => continue,
            };

            if cov.nrows() != 2 || cov.ncols() != 2 {
                continue;
            }

            let cov_mat = nalgebra::Matrix2::from_iterator(cov.iter().copied());
            let eig = nalgebra::SymmetricEigen::new(cov_mat);

            if eig.eigenvalues.iter().any(|&ev| ev < -1e-10) {
                continue;
            }

            let sqrt_evals = nalgebra::Vector2::new(
                eig.eigenvalues[0].max(0.0).sqrt(),
                eig.eigenvalues[1].max(0.0).sqrt(),
            );

            let color = &point_colors[pt_idx % point_colors.len()];

            // Sample ellipse in (a, ψ) space, map each sample to Cartesian
            let mut ring_points: Vec<[f32; 3]> = Vec::with_capacity(n_pt_samples + 1);
            for i in 0..=n_pt_samples {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n_pt_samples as f64;

                let e = nalgebra::Vector2::new(
                    sigma * sqrt_evals[0] * theta.cos(),
                    sigma * sqrt_evals[1] * theta.sin(),
                );
                let delta = eig.eigenvectors * e;
                let sample = VecF64::<2>::new(a_psi[0] + delta[0], a_psi[1] + delta[1]);

                // Skip samples with non-positive ψ (behind camera)
                if sample[1] > 0.0 {
                    let cart = to_cartesian_2d(&sample);
                    ring_points.push([cart[0] as f32, 0.0, cart[1] as f32]);
                }
            }

            for w in ring_points.windows(2) {
                all_lines.push(LineSegment3 {
                    p0: SVec::<f32, 3>::new(w[0][0], w[0][1], w[0][2]),
                    p1: SVec::<f32, 3>::new(w[1][0], w[1][1], w[1][2]),
                    color: *color,
                    line_width: 1.5,
                });
            }
        }

        // --- Pose covariance (translation ellipses) ---
        let pose_color = Color::new(1.0, 1.0, 0.0, 1.0); // bright yellow
        for (pose_idx, pose) in poses.iter().enumerate() {
            let cov = match solution.covariance_block(
                InverseDepthProblem::POSES,
                pose_idx,
                InverseDepthProblem::POSES,
                pose_idx,
            ) {
                Some(c) => c,
                None => continue, // conditioned (fixed) poses return None
            };

            if cov.nrows() != 3 || cov.ncols() != 3 {
                continue;
            }

            // Translation sub-block: rows/cols 1..3 of the 3×3 se2 covariance.
            // SE(2) tangent: [θ, v_lateral, v_depth] → translation = (v_lateral, v_depth).
            let trans_cov = cov.view((1, 1), (2, 2)).into_owned();
            let trans_cov_mat = nalgebra::Matrix2::from_iterator(trans_cov.iter().copied());
            let eig = nalgebra::SymmetricEigen::new(trans_cov_mat);

            // Skip only if eigenvalues are significantly negative (numerical issue).
            // Near-zero eigenvalues (e.g. from scale constraint) are clamped to zero.
            if eig.eigenvalues.iter().any(|&ev| ev < -1e-10) {
                continue;
            }

            let sqrt_evals = nalgebra::Vector2::new(
                eig.eigenvalues[0].max(0.0).sqrt(),
                eig.eigenvalues[1].max(0.0).sqrt(),
            );

            let n_samples = 48;
            let cam_pos = pose.translation(); // 2D: (lateral, depth)

            let mut ring_points: Vec<[f32; 3]> = Vec::with_capacity(n_samples + 1);
            for i in 0..=n_samples {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n_samples as f64;

                let e = nalgebra::Vector2::new(
                    sigma * sqrt_evals[0] * theta.cos(),
                    sigma * sqrt_evals[1] * theta.sin(),
                );
                let delta = eig.eigenvectors * e;
                // SE(2) translation delta → 3D: (x=lateral, y=0, z=depth)
                ring_points.push([
                    (cam_pos[0] + delta[0]) as f32,
                    0.0,
                    (cam_pos[1] + delta[1]) as f32,
                ]);
            }

            for w in ring_points.windows(2) {
                all_lines.push(LineSegment3 {
                    p0: SVec::<f32, 3>::new(w[0][0], w[0][1], w[0][2]),
                    p1: SVec::<f32, 3>::new(w[1][0], w[1][1], w[1][2]),
                    color: pose_color,
                    line_width: 3.0,
                });
            }
        }

        if all_lines.is_empty() {
            None
        } else {
            Some(all_lines)
        }
    }

    fn send_cost_plot(&self, cost: f64) {
        let clear_cond = ClearCondition {
            max_x_range: MAX_ITERATIONS as f64 + 1.0,
        };
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (PLOT_LABEL, "cost"),
                vec![(self.opt.total_steps as f64, cost)].into(),
                ScalarCurveStyle {
                    color: Color::blue(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )])]);
    }

    fn send_damping_plot(&self, damping: f64) {
        let clear_cond = ClearCondition {
            max_x_range: MAX_ITERATIONS as f64 + 1.0,
        };
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (DAMPING_PLOT_LABEL, "lm damping"),
                vec![(self.opt.total_steps as f64, damping)].into(),
                ScalarCurveStyle {
                    color: Color::green(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )])]);
    }
}
