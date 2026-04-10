//! Interactive corridor navigation demo widget.
//!
//! Demonstrates constrained optimization for SE(3) pose trajectories that must
//! pass through a corridor defined by half-plane constraints. The cost is path
//! length + prior (pulling toward y=0), so the optimal trajectory hugs the north
//! wall (y=-2) through the corridor section.
//!
//! Two methods: IPM (Ch.19) and SQP (Ch.18).

use eframe::egui;
use sophus_autodiff::linalg::VecF64;
use sophus_image::ImageSize;
use sophus_lie::{
    IsAffineGroup,
    Isometry3,
    Isometry3F64,
    Rotation3,
};
use sophus_opt::{
    example_problems::corridor_navigation::CorridorNavigationProblem,
    nlls::{
        IneqMethod,
        OptParams,
        OptProblem,
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
        named_line3,
    },
};
use sophus_sensor::DynCameraF64;
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
use crossbeam_channel::Sender;

use super::opt_widget::OptWidgetState;

const SCENE_LABEL: &str = "corridor-navigation - scene";
const COST_PLOT_LABEL: &str = "corridor - cost";
const BARRIER_PLOT_LABEL: &str = "corridor - barrier";
const HMIN_PLOT_LABEL: &str = "corridor - h_min";
const MAX_STEPS: usize = 20000;

fn svec3(x: f32, y: f32, z: f32) -> sophus_autodiff::linalg::SVec<f32, 3> {
    sophus_autodiff::linalg::SVec::<f32, 3>::new(x, y, z)
}

use super::opt_widget::{
    IneqMethodSelector as Method,
    SolverSelector,
};

/// Interactive corridor navigation demo widget.
pub struct CorridorNavigationWidget {
    message_send: Sender<Vec<Packet>>,
    problem: CorridorNavigationProblem,
    /// Shared optimization widget state.
    opt: OptWidgetState,
    /// Selected optimization method.
    pub method: Method,
    /// Selected solver.
    solver: SolverSelector,
    /// Multiplicative decay factor.
    pub barrier_decay: f64,
    /// Inner steps per outer iteration.
    pub inner_iters: usize,
    /// Start from infeasible (y=0) instead of corridor center (y=-4).
    pub infeasible_init: bool,
    latest_min_h: Option<f64>,
}

impl Drop for CorridorNavigationWidget {
    fn drop(&mut self) {
        self.opt.delete_plots(&self.message_send);
        let _ = self.message_send.send(vec![
            delete_scene_packet(SCENE_LABEL),
            Packet::Plot(vec![PlotViewPacket::Delete(HMIN_PLOT_LABEL.to_owned())]),
        ]);
    }
}

impl CorridorNavigationWidget {
    /// Create the widget.
    pub fn new(message_send: Sender<Vec<Packet>>) -> Self {
        let barrier_decay = 0.5_f64;
        let inner_iters = 50_usize;
        let method = Method::Ipm;
        let solver = SolverSelector::DenseLdlt;

        let infeasible_init = false;
        let problem = CorridorNavigationProblem::new();
        let optimizer = Self::build_optimizer(
            &problem,
            method,
            solver,
            barrier_decay,
            inner_iters,
            infeasible_init,
        );

        let opt = OptWidgetState::new(optimizer, MAX_STEPS, COST_PLOT_LABEL, BARRIER_PLOT_LABEL);

        let widget = CorridorNavigationWidget {
            message_send,
            problem,
            opt,
            method,
            solver,
            barrier_decay,
            inner_iters,
            infeasible_init,
            latest_min_h: None,
        };
        widget.send_initial_scene();
        widget
    }

    fn build_optimizer(
        problem: &CorridorNavigationProblem,
        method: Method,
        solver: SolverSelector,
        barrier_decay: f64,
        inner_iters: usize,
        infeasible_init: bool,
    ) -> Optimizer {
        // Override init poses if starting from infeasible (y=0 for all poses).
        let variables = if infeasible_init {
            use sophus_opt::variables::{
                VarBuilder,
                VarFamily,
            };
            let n = problem.init_poses.len();
            let members: Vec<Isometry3F64> = (0..n)
                .map(|i| Isometry3F64::from_translation(VecF64::<3>::new(i as f64, 0.0, 0.0)))
                .collect();
            let mut constant_ids = std::collections::BTreeMap::new();
            constant_ids.insert(0, ());
            constant_ids.insert(n - 1, ());
            VarBuilder::new()
                .add_family(
                    sophus_opt::example_problems::corridor_navigation::POSES,
                    VarFamily::new_with_const_ids(
                        sophus_opt::variables::VarKind::Free,
                        members,
                        constant_ids,
                    ),
                )
                .build()
        } else {
            problem.build_variables()
        };

        let costs = problem.build_cost();
        let barrier = problem.build_barrier_cost();
        let params = OptParams {
            num_iterations: MAX_STEPS,
            initial_lm_damping: 1.0,
            parallelize: true,
            solver: solver.to_solver(),
            skip_final_hessian: false,
            ineq_method: match method {
                Method::Ipm => IneqMethod::Ipm {
                    tau: 0.99,
                    inner_iters,
                    lambda_decay: barrier_decay,
                },
                Method::Sqp => IneqMethod::Sqp {
                    tau: 0.99,
                    inner_iters,
                    mu_decay: barrier_decay,
                },
            },
            ..Default::default()
        };
        Optimizer::new(
            variables,
            OptProblem {
                costs,
                eq_constraints: vec![],
                ineq_constraints: vec![barrier],
            },
            params,
        )
        .expect("Optimizer::new failed")
    }

    /// Draw the left-panel controls.
    pub fn update_left_panel(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Corridor Navigation");
        ui.label("Cost: path length + prior (y=0). Ineq: y in [-6,-2].");

        let prev_method = self.method;
        egui::ComboBox::from_label("Method")
            .selected_text(self.method.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.method, Method::Ipm, Method::Ipm.label());
                ui.selectable_value(&mut self.method, Method::Sqp, Method::Sqp.label());
            });
        if self.method != prev_method {
            self.reset();
        }

        let prev_solver = self.solver;
        egui::ComboBox::from_label("Solver")
            .selected_text(self.solver.label())
            .show_ui(ui, |ui| {
                for &s in super::opt_widget::ALL_SOLVER_OPTIONS {
                    ui.selectable_value(&mut self.solver, s, s.label());
                }
            });
        if self.solver != prev_solver {
            self.reset();
        }

        let desc = match self.method {
            Method::Ipm => "IPM: λⱼhⱼ=μ, J re-eval each step",
            Method::Sqp => "SQP: J frozen per outer iter",
        };
        ui.label(desc);

        let decay_label = match self.method {
            Method::Sqp => "mu decay",
            Method::Ipm => "lambda decay",
        };
        ui.add(egui::Slider::new(&mut self.barrier_decay, 0.01..=0.99).text(decay_label));
        ui.add(egui::Slider::new(&mut self.inner_iters, 1..=500).text("inner iters"));

        let prev_infeasible = self.infeasible_init;
        ui.checkbox(&mut self.infeasible_init, "infeasible init (y=0)");
        if self.infeasible_init != prev_infeasible {
            self.reset();
        }

        if self.opt.draw_buttons(ui) {
            self.reset();
        }

        let barrier_label = self.method.barrier_label();
        self.opt.draw_status(ui, barrier_label);

        if let Some(h) = self.latest_min_h {
            let feasible = h > 0.0;
            ui.label(format!(
                "min h = {:.4} ({})",
                h,
                if feasible { "feasible" } else { "infeasible" }
            ));
        }
    }

    /// Called each frame.
    pub fn update(&mut self) {
        if let Some(_info) = self.opt.update() {
            self.update_display();
        }
    }

    fn update_display(&mut self) {
        let Some(opt) = &self.opt.optimizer else {
            return;
        };

        let poses = opt
            .variables()
            .get_members::<Isometry3F64>(sophus_opt::example_problems::corridor_navigation::POSES);

        let min_h: f64 = {
            let poses_ref = &poses;
            self.problem
                .constrained_pose_indices
                .iter()
                .flat_map(|&i| {
                    self.problem.walls.iter().map(move |wall| {
                        let t = poses_ref[i].translation();
                        wall.normal.dot(&t) + wall.offset
                    })
                })
                .fold(f64::INFINITY, f64::min)
        };

        self.latest_min_h = Some(min_h);

        self.send_trajectory_update(&poses);
        self.opt.send_plots(&self.message_send);
        self.send_hmin_plot(min_h);
    }

    fn reset(&mut self) {
        let optimizer = Self::build_optimizer(
            &self.problem,
            self.method,
            self.solver,
            self.barrier_decay,
            self.inner_iters,
            self.infeasible_init,
        );
        self.opt.reset();
        self.opt.latest_barrier = optimizer.barrier_param().unwrap_or(1.0);
        self.opt.optimizer = Some(optimizer);
        self.latest_min_h = None;

        self.opt.delete_plots(&self.message_send);
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::Delete(
                HMIN_PLOT_LABEL.to_owned(),
            )])]);
        self.send_initial_scene();
    }

    fn current_init_poses(&self) -> Vec<Isometry3F64> {
        if self.infeasible_init {
            let n = self.problem.init_poses.len();
            (0..n)
                .map(|i| Isometry3F64::from_translation(VecF64::<3>::new(i as f64, 0.0, 0.0)))
                .collect()
        } else {
            self.problem.init_poses.clone()
        }
    }

    fn send_initial_scene(&self) {
        let image_size = ImageSize {
            width: 640,
            height: 480,
        };
        let viewer_camera = RenderCamera {
            properties: RenderCameraProperties::new(
                DynCameraF64::new_pinhole(VecF64::<4>::new(800.0, 800.0, 320.0, 240.0), image_size),
                ClippingPlanes::default(),
            ),
            scene_from_camera: Isometry3::from_rotation_and_translation(
                Rotation3::exp(VecF64::<3>::new(0.0, 0.0, 0.0)),
                VecF64::<3>::new(14.5, -1.5, -50.0),
            ),
        };

        let mut packets = vec![create_scene_packet(SCENE_LABEL, viewer_camera, false)];

        // Corridor region: walls from x=8 to x=22
        let cx0 = 8.0_f32;
        let cx1 = 22.0_f32;
        let wall_lines = vec![
            LineSegment3 {
                p0: svec3(cx0, -2.0, 0.0),
                p1: svec3(cx1, -2.0, 0.0),
                color: Color {
                    r: 0.8,
                    g: 0.2,
                    b: 0.2,
                    a: 1.0,
                },
                line_width: 3.0,
            },
            LineSegment3 {
                p0: svec3(cx0, -6.0, 0.0),
                p1: svec3(cx1, -6.0, 0.0),
                color: Color {
                    r: 0.3,
                    g: 0.4,
                    b: 0.8,
                    a: 1.0,
                },
                line_width: 2.0,
            },
            LineSegment3 {
                p0: svec3(cx0, -2.0, 0.0),
                p1: svec3(cx0, -6.0, 0.0),
                color: Color {
                    r: 0.5,
                    g: 0.5,
                    b: 0.7,
                    a: 1.0,
                },
                line_width: 1.5,
            },
            LineSegment3 {
                p0: svec3(cx1, -2.0, 0.0),
                p1: svec3(cx1, -6.0, 0.0),
                color: Color {
                    r: 0.5,
                    g: 0.5,
                    b: 0.7,
                    a: 1.0,
                },
                line_width: 1.5,
            },
        ];
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![named_line3("walls", wall_lines)],
        ));

        let prior_segments = self.build_segments(
            &self.problem.prior_poses,
            Color {
                r: 0.5,
                g: 0.5,
                b: 0.5,
                a: 1.0,
            },
            2.0,
        );
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![named_line3("prior_trajectory", prior_segments)],
        ));

        let init_poses = self.current_init_poses();

        let init_segments = self.build_segments(
            &init_poses,
            Color {
                r: 0.4,
                g: 0.75,
                b: 0.4,
                a: 1.0,
            },
            1.5,
        );
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![named_line3("init_trajectory", init_segments)],
        ));

        let traj_segments = self.build_opt_segments(&init_poses);
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![named_line3("trajectory", traj_segments)],
        ));

        let pose_lines = axes3(&init_poses).scale(0.3).line_width(1.5).build();
        packets.push(append_to_scene_packet(
            SCENE_LABEL,
            vec![named_line3("poses", pose_lines)],
        ));

        let _ = self.message_send.send(packets);

        self.opt.init_plots(&self.message_send);

        // h_min plot (problem-specific)
        let clear_cond = ClearCondition {
            max_x_range: MAX_STEPS as f64 + 1.0,
        };
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (HMIN_PLOT_LABEL, "min h"),
                std::collections::VecDeque::new(),
                ScalarCurveStyle {
                    color: Color::orange(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )])]);
    }

    fn send_trajectory_update(&self, poses: &[Isometry3F64]) {
        let traj_segments = self.build_opt_segments(poses);
        let pose_lines = axes3(poses).scale(0.3).line_width(1.5).build();
        let _ = self.message_send.send(vec![append_to_scene_packet(
            SCENE_LABEL,
            vec![
                named_line3("trajectory", traj_segments),
                named_line3("poses", pose_lines),
            ],
        )]);
    }

    fn build_segments(
        &self,
        poses: &[Isometry3F64],
        color: Color,
        line_width: f32,
    ) -> Vec<LineSegment3> {
        poses
            .windows(2)
            .map(|w| {
                let t0 = w[0].translation();
                let t1 = w[1].translation();
                LineSegment3 {
                    p0: svec3(t0[0] as f32, t0[1] as f32, t0[2] as f32),
                    p1: svec3(t1[0] as f32, t1[1] as f32, t1[2] as f32),
                    color,
                    line_width,
                }
            })
            .collect()
    }

    fn build_opt_segments(&self, poses: &[Isometry3F64]) -> Vec<LineSegment3> {
        self.build_segments(
            poses,
            Color {
                r: 1.0,
                g: 0.5,
                b: 0.0,
                a: 1.0,
            },
            3.0,
        )
    }

    fn send_hmin_plot(&self, min_h: f64) {
        let clear_cond = ClearCondition {
            max_x_range: MAX_STEPS as f64 + 1.0,
        };
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (HMIN_PLOT_LABEL, "min h"),
                vec![(self.opt.total_steps as f64, min_h)].into(),
                ScalarCurveStyle {
                    color: Color::orange(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )])]);
    }
}
