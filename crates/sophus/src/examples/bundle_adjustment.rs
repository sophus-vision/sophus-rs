use crossbeam_channel::Sender;
use eframe::egui;
use log::warn;
use sophus_autodiff::linalg::VecF64;
use sophus_image::{
    ImageSize,
    MutImage4U8,
    prelude::IsMutImageView,
};
use sophus_lie::{
    IsAffineGroup,
    Isometry3,
    Isometry3F64,
    Rotation3,
};
use sophus_opt::{
    example_problems::{
        ba_problem::{
            BaProblem,
            TRUE_FOCAL,
        },
        ba_scale_constraint::BaScaleConstraintProblem,
    },
    nlls::{
        OptParams,
        OptProblem,
        Optimizer,
        optimize_nlls,
    },
    variables::VarKind,
};
use sophus_renderer::{
    camera::{
        ClippingPlanes,
        RenderCamera,
        RenderCameraProperties,
    },
    renderables::{
        Color,
        ImageFrame,
        axes3,
        make_point3,
        named_line3,
    },
};
use sophus_sensor::EnhancedUnifiedCameraF64;
use sophus_solver::matrix::PartitionBlockIndex;
use sophus_viewer::packets::{
    ClearCondition,
    ImageViewPacket,
    LineType,
    Packet,
    PlotViewPacket,
    ScalarCurveStyle,
    append_to_scene_packet,
    create_scene_packet,
    delete_scene_packet,
};

use super::opt_widget::{
    ALL_SOLVER_OPTIONS,
    EQ_SOLVER_OPTIONS,
    OptWidgetState,
    SolverSelector,
    StepRequest,
};

const SCENE_LABEL: &str = "bundle-adj - scene";
const SPARSITY_LABEL: &str = "bundle-adj - sparsity";
const PLOT_LABEL: &str = "bundle-adj - cost";
const FOCAL_PLOT_LABEL: &str = "bundle-adj - focal";
const CONSTRAINT_PLOT_LABEL: &str = "bundle-adj - constraint";
const DAMPING_PLOT_LABEL: &str = "bundle-adj - damping";
const MAX_ITERATIONS: usize = 500;

// ── Widget ───────────────────────────────────────────────────────────────

/// Interactive bundle adjustment demo (unified).
///
/// Toggle "scale constraint" to switch between:
/// - Standard BA: all cams + points optimized, focal length recovered.
/// - BA + scale constraint: pose 0 fixed, |t(pose 1)| = r constrained.
pub struct BundleAdjustmentWidget {
    message_send: Sender<Vec<Packet>>,
    /// Number of world points.
    pub num_points: usize,
    /// Number of cameras.
    num_cams: usize,

    // Mode
    /// Whether to use the scale constraint.
    use_scale_constraint: bool,

    // Problems
    ba_problem: BaProblem,
    scale_problem: BaScaleConstraintProblem,

    // Solver selection (shared across modes)
    solver: SolverSelector,
    use_schur: bool,

    // Shared optimizer state
    opt: OptWidgetState,
    latest_rms_px: Option<f64>,
    latest_focal: Option<f64>,
    latest_constraint_err: Option<f64>,
}

impl Drop for BundleAdjustmentWidget {
    fn drop(&mut self) {
        match self.message_send.send(vec![
            delete_scene_packet(SCENE_LABEL),
            Packet::Image(ImageViewPacket {
                view_label: SPARSITY_LABEL.to_owned(),
                frame: None,
                pixel_renderables: vec![],
                scene_renderables: vec![],
                delete: true,
            }),
            Packet::Plot(vec![PlotViewPacket::Delete(PLOT_LABEL.to_owned())]),
            Packet::Plot(vec![PlotViewPacket::Delete(FOCAL_PLOT_LABEL.to_owned())]),
            Packet::Plot(vec![PlotViewPacket::Delete(
                CONSTRAINT_PLOT_LABEL.to_owned(),
            )]),
            Packet::Plot(vec![PlotViewPacket::Delete(DAMPING_PLOT_LABEL.to_owned())]),
        ]) {
            Ok(_) => {}
            Err(_) => {
                warn!("Failed to send delete packets.");
            }
        }
    }
}

impl BundleAdjustmentWidget {
    /// Create the widget.
    pub fn new(message_send: Sender<Vec<Packet>>, use_scale_constraint: bool) -> Self {
        let num_points = 500;
        let num_cams = if use_scale_constraint { 10 } else { 50 };
        let ba_problem = BaProblem::new(num_cams, num_points);
        let scale_problem = BaScaleConstraintProblem::new(10, num_points);
        let solver = SolverSelector::BlockSparseLdlt;
        let use_schur = false;

        let optimizer = if use_scale_constraint {
            Self::build_scale_optimizer(&scale_problem, solver, use_schur)
        } else {
            Self::build_ba_optimizer(&ba_problem, solver, use_schur)
        };
        let opt = OptWidgetState::new(optimizer, MAX_ITERATIONS, PLOT_LABEL, DAMPING_PLOT_LABEL);

        let widget = BundleAdjustmentWidget {
            message_send,
            num_points,
            num_cams,
            use_scale_constraint,
            ba_problem,
            scale_problem,
            solver,
            use_schur,
            opt,
            latest_rms_px: None,
            latest_focal: None,
            latest_constraint_err: None,
        };
        widget.send_scene_setup();
        widget
    }

    fn build_ba_optimizer(
        problem: &BaProblem,
        solver: SolverSelector,
        use_schur: bool,
    ) -> Optimizer {
        let vars = problem.build_initial_variables(use_schur);
        Optimizer::new(
            vars,
            OptProblem::costs_only(vec![problem.build_cost()]),
            OptParams {
                num_iterations: 10000,
                initial_lm_damping: 1.0,
                parallelize: true,
                solver: solver.to_linear_solver(use_schur),
                skip_final_hessian: false,
                ..Default::default()
            },
        )
        .expect("failed to build BA optimizer")
    }

    fn build_scale_optimizer(
        problem: &BaScaleConstraintProblem,
        solver: SolverSelector,
        use_schur: bool,
    ) -> Optimizer {
        let points_kind = if use_schur {
            VarKind::Marginalized
        } else {
            VarKind::Free
        };
        let vars = problem.build_variables(points_kind);
        Optimizer::new(
            vars,
            OptProblem {
                costs: vec![problem.ba.build_cost()],
                eq_constraints: vec![problem.build_eq_constraint_fn()],
                ineq_constraints: vec![],
            },
            OptParams {
                num_iterations: 10000,
                initial_lm_damping: 1.0,
                parallelize: true,
                solver: solver.to_linear_solver(use_schur),
                skip_final_hessian: false,
                ..Default::default()
            },
        )
        .expect("failed to build scale-constraint optimizer")
    }

    fn active_ba(&self) -> &BaProblem {
        if self.use_scale_constraint {
            &self.scale_problem.ba
        } else {
            &self.ba_problem
        }
    }

    /// Draw the left-panel controls.
    pub fn update_left_panel(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Bundle Adjustment");
        if self.use_scale_constraint {
            ui.label("Cost: reprojection. Eq: |t1|=r. Pose 0 fixed.");
        } else {
            ui.label("Cost: reprojection. Poses 0,1 fixed (gauge).");
        }

        // Points slider
        let mut num_points = self.num_points;
        if ui
            .add(egui::Slider::new(&mut num_points, 5..=5000).text("# points"))
            .changed()
        {
            self.num_points = num_points;
            self.regenerate();
        }

        // Solver selection
        let is_running = self.opt.step_request == StepRequest::Run;
        ui.add_enabled_ui(!is_running, |ui| {
            let solver_options = if self.use_scale_constraint {
                EQ_SOLVER_OPTIONS
            } else {
                ALL_SOLVER_OPTIONS
            };

            let prev_solver = self.solver;
            egui::ComboBox::from_label("Solver")
                .selected_text(self.solver.label())
                .show_ui(ui, |ui| {
                    for &s in solver_options {
                        ui.selectable_value(&mut self.solver, s, s.label());
                    }
                });
            if self.solver != prev_solver {
                self.reset();
            }

            // Schur checkbox (only enabled when solver supports it).
            let can_schur = self.solver.has_schur();
            let mut use_schur = self.use_schur && can_schur;
            ui.add_enabled(
                can_schur,
                egui::Checkbox::new(&mut use_schur, "Schur complement"),
            );
            if use_schur != self.use_schur {
                self.use_schur = use_schur;
                self.reset();
            }
        });

        // Buttons
        if self.opt.draw_buttons(ui) {
            self.reset();
        }

        // Status
        self.opt.draw_status(ui, "");

        if let Some(rms) = self.latest_rms_px {
            ui.label(format!("reproj err: {rms:.3} px"));
        }
        if self.use_scale_constraint {
            if let Some(c) = self.latest_constraint_err {
                ui.label(format!(
                    "scale err: {c:+.6} m  (target={:.4})",
                    self.scale_problem.target_radius
                ));
            }
        } else {
            if let Some(f) = self.latest_focal {
                let err = f - TRUE_FOCAL;
                ui.label(format!("focal: {f:.1}  err: {err:+.1} px"));
            }
        }
    }

    /// Called each frame.
    pub fn update(&mut self) {
        if let Some(info) = self.opt.update() {
            let n_obs = self.active_ba().num_observations();
            let target_radius = self.scale_problem.target_radius;
            let use_scale = self.use_scale_constraint;
            let cost = info.merit;

            let rms_px = if n_obs > 0 {
                (cost / n_obs as f64).sqrt()
            } else {
                0.0
            };
            self.latest_rms_px = Some(rms_px);

            if let Some(opt) = &self.opt.optimizer {
                if use_scale {
                    let vars = opt.variables();
                    let poses = vars.get_members::<Isometry3F64>(BaProblem::POSES);
                    let t_norm = poses[1].translation().norm();
                    self.latest_constraint_err = Some(t_norm - target_radius);
                } else {
                    let cam = opt
                        .variables()
                        .get_members::<EnhancedUnifiedCameraF64>("cams");
                    let p = cam[0].params();
                    self.latest_focal = Some((p[0] + p[1]) * 0.5);
                }

                let damping = opt.lm_damping();

                self.send_current_estimates();
                self.send_reproj_plot(rms_px);
                self.send_damping_plot(damping);

                if self.use_scale_constraint {
                    if let Some(c) = self.latest_constraint_err {
                        self.send_constraint_plot(c);
                    }
                } else if let Some(f) = self.latest_focal {
                    self.send_focal_plot(f);
                }
            }
        }
    }

    // ── private ──────────────────────────────────────────────────────────

    fn regenerate(&mut self) {
        if self.use_scale_constraint {
            self.scale_problem = BaScaleConstraintProblem::new(10, self.num_points);
        } else {
            self.ba_problem = BaProblem::new(self.num_cams, self.num_points);
        }
        self.reset();
    }

    fn reset(&mut self) {
        let optimizer = if self.use_scale_constraint {
            Self::build_scale_optimizer(&self.scale_problem, self.solver, self.use_schur)
        } else {
            Self::build_ba_optimizer(&self.ba_problem, self.solver, self.use_schur)
        };
        self.opt.reset();
        self.opt.optimizer = Some(optimizer);
        self.latest_rms_px = None;
        self.latest_focal = None;
        self.latest_constraint_err = None;
        let _ = self.message_send.send(vec![
            Packet::Plot(vec![PlotViewPacket::Delete(PLOT_LABEL.to_owned())]),
            Packet::Plot(vec![PlotViewPacket::Delete(FOCAL_PLOT_LABEL.to_owned())]),
            Packet::Plot(vec![PlotViewPacket::Delete(
                CONSTRAINT_PLOT_LABEL.to_owned(),
            )]),
            Packet::Plot(vec![PlotViewPacket::Delete(DAMPING_PLOT_LABEL.to_owned())]),
        ]);
        self.send_scene_setup();
    }

    fn send_scene_setup(&self) {
        let ba = self.active_ba();
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
            scene_from_camera: Isometry3::from_rotation_and_translation(
                Rotation3::exp(VecF64::<3>::new(-0.4228, -0.0738, 0.0634)),
                VecF64::<3>::new(3.151, -13.113, -12.191),
            ),
        };
        let mut packets = vec![create_scene_packet(SCENE_LABEL, viewer_camera, false)];

        let true_cam_lines = axes3(ba.true_world_from_cameras.as_slice())
            .scale(0.35)
            .line_width(2.0)
            .build();
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![named_line3("cams_true", true_cam_lines)],
        ));

        let true_pts: Vec<[f32; 3]> = ba
            .true_points_in_world
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![make_point3("pts_true", &true_pts, &Color::green(), 6.0)],
        ));

        let init_cam_lines = axes3(ba.world_from_cameras.as_slice())
            .scale(0.35)
            .line_width(2.0)
            .build();
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![named_line3("cams_current", init_cam_lines)],
        ));

        let init_pts: Vec<[f32; 3]> = ba
            .points_in_world
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![make_point3("pts_current", &init_pts, &Color::orange(), 6.0)],
        ));

        // 1. Scene
        let _ = self.message_send.send(packets);

        // 2. Sparsity pattern image
        self.send_sparsity_image(ba);

        // 3. Plots — initialize all upfront so windows are visible immediately.
        let clear_cond = ClearCondition {
            max_x_range: MAX_ITERATIONS as f64 + 1.0,
        };
        let focal_or_constraint_label = if self.use_scale_constraint {
            (CONSTRAINT_PLOT_LABEL, "scale err (m)")
        } else {
            (FOCAL_PLOT_LABEL, "focal err (px)")
        };
        let _ = self.message_send.send(vec![
            Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (PLOT_LABEL, "reproj err (px)"),
                std::collections::VecDeque::new(),
                ScalarCurveStyle {
                    color: Color::blue(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )]),
            Packet::Plot(vec![PlotViewPacket::append_to_curve(
                focal_or_constraint_label,
                std::collections::VecDeque::new(),
                ScalarCurveStyle {
                    color: Color::orange(),
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

    /// Generate and send the block-sparse Hessian sparsity pattern as an image.
    ///
    /// When `use_schur` is true, shows only the free-variable (reduced) system.
    fn send_sparsity_image(&self, ba: &BaProblem) {
        let variables = ba.build_initial_variables(self.use_schur);
        let num_free_parts = variables.num_of_kind(VarKind::Free);

        let result = optimize_nlls(
            variables,
            vec![ba.build_cost()],
            vec![],
            OptParams {
                num_iterations: 0,
                skip_final_hessian: false,
                ..Default::default()
            },
        );
        let Ok(result) = result else { return };

        let sparsity = result.hessian_sparsity();
        let specs = sparsity.partition_specs().to_vec();

        // Partition colors.
        const COLORS: [[u8; 3]; 6] = [
            [50, 100, 220], // blue
            [220, 60, 60],  // red
            [50, 180, 80],  // green
            [200, 140, 30], // orange
            [160, 50, 200], // purple
            [30, 180, 180], // cyan
        ];

        // When Schur is active, only show the free partitions (reduced system).
        // The Schur complement S_ff has fill-in: free blocks (i,j) couple if
        // they share any marginalized block. Compute this from the sparsity graph.
        let show_specs: Vec<_> = if self.use_schur {
            specs.iter().take(num_free_parts).cloned().collect()
        } else {
            specs.clone()
        };

        // Block → partition mapping (only for shown partitions).
        let mut block_map: Vec<(usize, usize)> = Vec::new();
        for (p, s) in show_specs.iter().enumerate() {
            for b in 0..s.block_count {
                block_map.push((p, b));
            }
        }
        let total = block_map.len();
        if total == 0 {
            return;
        }

        // For Schur: compute the fill-in pattern of S_ff from the coupling graph.
        let schur_fill = if self.use_schur {
            Some(sparsity.schur_fill_in(num_free_parts))
        } else {
            None
        };

        let img_size = 400;
        let scale = (img_size / total).max(1);
        let w = total * scale;

        let mut img = MutImage4U8::from_image_size_and_val(
            ImageSize::new(w, w),
            sophus_autodiff::linalg::SVec::<u8, 4>::new(245, 245, 245, 255),
        );

        for gi in 0..total {
            let (pi, bi) = block_map[gi];
            let idx_i = PartitionBlockIndex {
                partition: pi,
                block: bi,
            };
            for gj in 0..total {
                let (pj, bj) = block_map[gj];
                let idx_j = PartitionBlockIndex {
                    partition: pj,
                    block: bj,
                };

                let is_nonzero = if let Some(ref fill) = schur_fill {
                    fill.contains(&(gi, gj))
                } else {
                    sparsity.has_block(idx_i, idx_j)
                };

                if !is_nonzero {
                    continue;
                }

                let ci = COLORS[pi % COLORS.len()];
                let cj = COLORS[pj % COLORS.len()];
                let [r, g, b] = if pi == pj {
                    ci
                } else {
                    [
                        ((ci[0] as u16 + cj[0] as u16) / 2) as u8,
                        ((ci[1] as u16 + cj[1] as u16) / 2) as u8,
                        ((ci[2] as u16 + cj[2] as u16) / 2) as u8,
                    ]
                };

                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = gj * scale + dx;
                        let py = gi * scale + dy;
                        if px < w && py < w {
                            *img.mut_pixel(px, py) =
                                sophus_autodiff::linalg::SVec::<u8, 4>::new(r, g, b, 255);
                        }
                    }
                }
            }
        }

        // Partition boundary lines.
        let line = sophus_autodiff::linalg::SVec::<u8, 4>::new(40, 40, 40, 255);
        let mut off = 0;
        for s in &show_specs {
            off += s.block_count;
            let lp = off * scale;
            if lp < w {
                for i in 0..w {
                    *img.mut_pixel(lp.min(w - 1), i) = line;
                    *img.mut_pixel(i, lp.min(w - 1)) = line;
                }
            }
        }

        let shared = img.to_shared();
        let _ = self.message_send.send(vec![Packet::Image(ImageViewPacket {
            view_label: SPARSITY_LABEL.to_owned(),
            frame: Some(ImageFrame::from_image(&shared)),
            pixel_renderables: vec![],
            scene_renderables: vec![],
            delete: false,
        })]);
    }

    fn send_current_estimates(&self) {
        let opt = match &self.opt.optimizer {
            Some(o) => o,
            None => return,
        };
        let vars = opt.variables();
        let poses = vars.get_members::<Isometry3F64>("poses");
        let points = vars.get_members::<VecF64<3>>("points");

        let cam_lines = axes3(&poses).scale(0.35).line_width(3.5).build();
        let pts: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();

        let _ = self.message_send.send(vec![append_to_scene_packet(
            SCENE_LABEL,
            vec![
                named_line3("cams_current", cam_lines),
                make_point3("pts_current", &pts, &Color::orange(), 6.0),
            ],
        )]);
    }

    fn send_reproj_plot(&self, rms_px: f64) {
        let clear_cond = ClearCondition {
            max_x_range: MAX_ITERATIONS as f64 + 1.0,
        };
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (PLOT_LABEL, "reproj err (px)"),
                vec![(self.opt.total_steps as f64, rms_px)].into(),
                ScalarCurveStyle {
                    color: Color::blue(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )])]);
    }

    fn send_focal_plot(&self, focal: f64) {
        let clear_cond = ClearCondition {
            max_x_range: MAX_ITERATIONS as f64 + 1.0,
        };
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (FOCAL_PLOT_LABEL, "focal err (px)"),
                vec![(self.opt.total_steps as f64, (focal - TRUE_FOCAL).abs())].into(),
                ScalarCurveStyle {
                    color: Color::orange(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )])]);
    }

    fn send_constraint_plot(&self, constraint_err: f64) {
        let clear_cond = ClearCondition {
            max_x_range: MAX_ITERATIONS as f64 + 1.0,
        };
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (CONSTRAINT_PLOT_LABEL, "scale err (m)"),
                vec![(self.opt.total_steps as f64, constraint_err.abs())].into(),
                ScalarCurveStyle {
                    color: Color::orange(),
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
