//! Interactive spline trajectory optimization demo.
//!
//! 16 SE(2) control points form a trajectory from start (0,0) to end (7,0).
//! A smoothness cost pulls the trajectory straight; an obstacle at (3.5,0)
//! forces the path to bend around it.
//!
//! Trajectory visualisation uses the SE(2) Lie group spline directly —
//! translations are extracted for the 2D line path, headings from the
//! rotation component, and body-frame velocities from the spline velocity().

use std::f64::consts::PI;

use crossbeam_channel::Sender;
use eframe::egui;
use sophus_image::{
    ArcImage4U8,
    ImageSize,
    MutImage4U8,
    color_map::BlueWhiteRedBlackColorMap,
    prelude::IsMutImageView,
};
use sophus_lie::{
    Isometry2F64,
    prelude::{
        HasParams,
        IsAffineGroup,
    },
};
use sophus_opt::{
    example_problems::spline_trajectory::{
        SplineTrajProblem,
        WAYPOINTS,
    },
    nlls::{
        IneqMethod,
        OptParams,
        Optimizer,
    },
};
use sophus_renderer::renderables::{
    Color,
    ImageFrame,
    LineSegment2,
    PixelRenderable,
    Point2,
    named_line2,
    named_point2,
};
use sophus_spline::lie_group_spline::LieGroupCubicBSpline;
use sophus_viewer::packets::{
    ClearCondition,
    CurveVecWithConfStyle,
    ImageViewPacket,
    LineType,
    Packet,
    PlotViewPacket,
    ScalarCurveStyle,
    VerticalLine,
};

use super::opt_widget::{
    IneqMethodSelector as Method,
    OptWidgetState,
    SolverSelector,
};

const IMAGE_LABEL: &str = "spline-traj";
const MAX_STEPS: usize = 5000;

// Viewport bounds: wide enough to see full trajectory
const X_MIN: f64 = -1.0;
const X_MAX: f64 = 9.0;
const Y_MIN: f64 = -3.0;
const Y_MAX: f64 = 5.0;
const IMG_W: usize = 400;
const IMG_H: usize = 400;

fn svec2(x: f32, y: f32) -> sophus_autodiff::linalg::SVec<f32, 2> {
    sophus_autodiff::linalg::SVec::<f32, 2>::new(x, y)
}

fn world_to_pixel(x: f64, y: f64) -> [f32; 2] {
    let u = ((x - X_MIN) / (X_MAX - X_MIN) * IMG_W as f64) as f32;
    let v = ((Y_MAX - y) / (Y_MAX - Y_MIN) * IMG_H as f64) as f32;
    [u, v]
}

fn circle_segments(cx: f64, cy: f64, r: f64, n: usize, color: Color) -> Vec<LineSegment2> {
    (0..n)
        .map(|i| {
            let t0 = 2.0 * PI * i as f64 / n as f64;
            let t1 = 2.0 * PI * (i + 1) as f64 / n as f64;
            let [u0, v0] = world_to_pixel(cx + r * t0.cos(), cy + r * t0.sin());
            let [u1, v1] = world_to_pixel(cx + r * t1.cos(), cy + r * t1.sin());
            LineSegment2 {
                p0: svec2(u0, v0),
                p1: svec2(u1, v1),
                color,
                line_width: 2.0,
            }
        })
        .collect()
}

fn generate_heatmap(problem: &SplineTrajProblem) -> MutImage4U8 {
    let mut img = MutImage4U8::from_image_size_and_val(
        ImageSize::new(IMG_W, IMG_H),
        sophus_autodiff::linalg::SVec::<u8, 4>::new(0, 0, 0, 255),
    );
    // Use a distance-to-straight-line cost as the heatmap background.
    let max_cost = 5.0_f64;
    for py in 0..IMG_H {
        for px in 0..IMG_W {
            let x = X_MIN + (px as f64 + 0.5) / IMG_W as f64 * (X_MAX - X_MIN);
            let y = Y_MAX - (py as f64 + 0.5) / IMG_H as f64 * (Y_MAX - Y_MIN);
            // Cost: squared y-distance from the straight line (y=0)
            let cost = 0.5 * y * y;

            // Check obstacle
            let ox = x - problem.obstacle_center[0];
            let oy = y - problem.obstacle_center[1];
            let in_obstacle = ox * ox + oy * oy < problem.obstacle_radius * problem.obstacle_radius;

            let t = (cost / max_cost).min(1.0) as f32;
            let rgb = if in_obstacle {
                let c = BlueWhiteRedBlackColorMap::f32_to_rgb(t);
                sophus_autodiff::linalg::SVec::<u8, 3>::new(c[0] / 3, c[1] / 3, c[2] / 3)
            } else {
                BlueWhiteRedBlackColorMap::f32_to_rgb(t)
            };
            *img.mut_pixel(px, py) =
                sophus_autodiff::linalg::SVec::<u8, 4>::new(rgb[0], rgb[1], rgb[2], 255);
        }
    }
    img
}

const POSITION_PLOT_LABEL: &str = "spline-traj - position";
const VELOCITY_PLOT_LABEL: &str = "spline-traj - velocity";
const ANGLE_PLOT_LABEL: &str = "spline-traj - angle";
const ANGULAR_VELOCITY_PLOT_LABEL: &str = "spline-traj - angular velocity";

/// SE(2) Lie group spline type alias.
type SE2Spline =
    LieGroupCubicBSpline<f64, 3, 4, 2, 3, 0, 0, sophus_lie::Isometry2Impl<f64, 1, 0, 0>>;

/// Interactive spline trajectory optimization demo widget.
pub struct SplineTrajWidget {
    message_send: Sender<Vec<Packet>>,
    problem: SplineTrajProblem,
    opt: OptWidgetState,
    method: Method,
    solver: SolverSelector,
    inner_iters: usize,
    barrier_decay: f64,
    heatmap: ArcImage4U8,
    /// SE(2) Lie group spline for position, heading, and velocity.
    se2_spline: Option<SE2Spline>,
    /// Current playback time (None = not playing).
    playback_t: Option<f64>,
    /// Whether playback is active.
    playing: bool,
    /// Playback speed multiplier.
    playback_speed: f64,
}

impl Drop for SplineTrajWidget {
    fn drop(&mut self) {
        self.opt.delete_plots(&self.message_send);
        let _ = self.message_send.send(vec![
            Packet::Image(ImageViewPacket {
                view_label: IMAGE_LABEL.to_owned(),
                frame: None,
                pixel_renderables: vec![],
                scene_renderables: vec![],
                delete: true,
            }),
            Packet::Plot(vec![PlotViewPacket::Delete(POSITION_PLOT_LABEL.to_owned())]),
            Packet::Plot(vec![PlotViewPacket::Delete(VELOCITY_PLOT_LABEL.to_owned())]),
            Packet::Plot(vec![PlotViewPacket::Delete(ANGLE_PLOT_LABEL.to_owned())]),
            Packet::Plot(vec![PlotViewPacket::Delete(
                ANGULAR_VELOCITY_PLOT_LABEL.to_owned(),
            )]),
        ]);
    }
}

impl SplineTrajWidget {
    /// Create the widget.
    pub fn new(message_send: Sender<Vec<Packet>>) -> Self {
        let problem = SplineTrajProblem::new();
        let heatmap = generate_heatmap(&problem).to_shared();
        let method = Method::Ipm;
        let solver = SolverSelector::DenseLdlt;
        let inner_iters = 30;
        let barrier_decay = 0.5;
        let optimizer = Self::build_optimizer(&problem, method, solver, inner_iters, barrier_decay);
        let opt = OptWidgetState::new(
            optimizer,
            MAX_STEPS,
            "spline-traj-opt-cost",
            "spline-traj-opt-barrier",
        );

        let widget = Self {
            message_send,
            problem,
            opt,
            method,
            solver,
            inner_iters,
            barrier_decay,
            heatmap,
            se2_spline: None,
            playback_t: None,
            playing: false,
            playback_speed: 0.5,
        };
        widget.send_initial_frame();
        widget
    }

    fn build_optimizer(
        problem: &SplineTrajProblem,
        method: Method,
        solver: SolverSelector,
        inner_iters: usize,
        barrier_decay: f64,
    ) -> Optimizer {
        let params = OptParams {
            num_iterations: MAX_STEPS,
            initial_lm_damping: 0.1,
            parallelize: false,
            solver: solver.to_solver(),
            ineq_method: match method {
                Method::Ipm => IneqMethod::Ipm {
                    tau: 0.9,
                    inner_iters,
                    lambda_decay: barrier_decay,
                },
                Method::Sqp => IneqMethod::Sqp {
                    tau: 0.9,
                    inner_iters,
                    mu_decay: barrier_decay,
                },
            },
            ..Default::default()
        };
        problem.build_optimizer(params)
    }

    /// Draw the left-panel controls.
    pub fn update_left_panel(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Spline Trajectory");
        ui.label("Cost: smoothness + endpoints. Ineq: obstacle + |vy|<=max.");

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

        ui.add(egui::Slider::new(&mut self.inner_iters, 1..=200).text("inner iters"));
        ui.add(egui::Slider::new(&mut self.barrier_decay, 0.01..=0.99).text("decay"));

        if self.opt.draw_buttons(ui) {
            self.reset();
        }

        self.opt.draw_status(ui, self.method.barrier_label());

        ui.separator();
        ui.label("Trajectory playback");
        ui.add(egui::Slider::new(&mut self.playback_speed, 0.1..=1.0).text("speed"));
        ui.horizontal(|ui| {
            if self.playing {
                if ui.button("Stop playback").clicked() {
                    self.playing = false;
                }
            } else if ui.button("Run trajectory").clicked() {
                // Clear old plots before replaying.
                let _ = self.message_send.send(vec![
                    Packet::Plot(vec![PlotViewPacket::Delete(POSITION_PLOT_LABEL.to_owned())]),
                    Packet::Plot(vec![PlotViewPacket::Delete(VELOCITY_PLOT_LABEL.to_owned())]),
                    Packet::Plot(vec![PlotViewPacket::Delete(ANGLE_PLOT_LABEL.to_owned())]),
                    Packet::Plot(vec![PlotViewPacket::Delete(
                        ANGULAR_VELOCITY_PLOT_LABEL.to_owned(),
                    )]),
                ]);
                self.build_se2_spline();
                self.playback_t = Some(0.0);
                self.playing = true;
            }
        });
        if let (Some(t), Some(spline)) = (self.playback_t, self.se2_spline.as_ref()) {
            ui.label(format!("t = {t:.2} / {:.2}", spline.t_max()));
        }
    }

    /// Called each frame.
    pub fn update(&mut self) {
        if let Some(_info) = self.opt.update() {
            self.update_display();
        }

        // Playback: advance time and render the SE(2) pose.
        if self.playing
            && let Some(spline) = self.se2_spline.as_ref()
        {
            {
                let t_max = spline.t_max();
                let t = self.playback_t.unwrap_or(0.0);
                let dt = 0.02 * self.playback_speed;
                let new_t = t + dt;

                if new_t <= t_max {
                    self.playback_t = Some(new_t);
                    self.send_playback_frame(new_t);
                    self.send_playback_plots(new_t);
                } else {
                    // Stop at end.
                    self.playback_t = Some(t_max);
                    self.playing = false;
                }
            }
        }
    }

    /// Build an SE(2) Lie group spline from the current optimized SE(2) waypoints.
    fn build_se2_spline(&mut self) {
        let Some(opt) = &self.opt.optimizer else {
            return;
        };
        let poses = opt.variables().get_members::<Isometry2F64>(WAYPOINTS);
        if poses.len() < 2 {
            return;
        }

        self.se2_spline = Some(LieGroupCubicBSpline {
            control_points: poses,
            delta_t: 1.0,
            t0: 0.0,
        });

        // Initialize playback plots.
        self.init_playback_plots();
    }

    fn init_playback_plots(&self) {
        let t_max = self.se2_spline.as_ref().map(|s| s.t_max()).unwrap_or(10.0);
        let clear_cond = ClearCondition {
            max_x_range: t_max + 1.0,
        };
        let _ = self.message_send.send(vec![
            Packet::Plot(vec![PlotViewPacket::append_to_curve_vec2_with_conf(
                (POSITION_PLOT_LABEL, "x_y"),
                std::collections::VecDeque::new(),
                CurveVecWithConfStyle {
                    colors: [Color::red(), Color::green()],
                },
                clear_cond,
                None,
            )]),
            Packet::Plot(vec![PlotViewPacket::append_to_curve_vec2_with_conf(
                (VELOCITY_PLOT_LABEL, "vx_vy"),
                std::collections::VecDeque::new(),
                CurveVecWithConfStyle {
                    colors: [Color::red(), Color::green()],
                },
                clear_cond,
                None,
            )]),
            Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (ANGLE_PLOT_LABEL, "theta"),
                std::collections::VecDeque::new(),
                ScalarCurveStyle {
                    color: Color::blue(),
                    line_type: LineType::default(),
                },
                clear_cond,
                None,
            )]),
            Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (ANGULAR_VELOCITY_PLOT_LABEL, "omega"),
                std::collections::VecDeque::new(),
                ScalarCurveStyle {
                    color: Color::red(),
                    line_type: LineType::default(),
                },
                clear_cond,
                None,
            )]),
        ]);
    }

    fn send_playback_frame(&self, t: f64) {
        let Some(spline) = self.se2_spline.as_ref() else {
            return;
        };

        let pose = spline.interpolate(t);
        let pos = pose.translation();
        let [pu, pv] = world_to_pixel(pos[0], pos[1]);

        // Heading from the SE(2) rotation: extract angle from unit complex.
        let rotation = pose.factor();
        let rot_params = rotation.params();
        let theta = rot_params[1].atan2(rot_params[0]); // atan2(sin, cos)
        let arrow_len = 20.0_f32;
        let dx = arrow_len * theta.cos() as f32;
        let dy = -arrow_len * theta.sin() as f32;

        // Only send the pose overlay — static renderables (obstacle, trajectory)
        // are already in the viewer from the optimization phase and persist by name.
        let pixel_renderables = vec![
            named_line2(
                "pose_arrow",
                vec![LineSegment2 {
                    p0: svec2(pu, pv),
                    p1: svec2(pu + dx, pv + dy),
                    color: Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    },
                    line_width: 3.0,
                }],
            ),
            named_point2(
                "pose_point",
                vec![Point2 {
                    p: svec2(pu, pv),
                    color: Color {
                        r: 1.0,
                        g: 1.0,
                        b: 0.0,
                        a: 1.0,
                    },
                    point_size: 6.0,
                }],
            ),
        ];

        let _ = self.message_send.send(vec![Packet::Image(ImageViewPacket {
            view_label: IMAGE_LABEL.to_owned(),
            frame: None,
            pixel_renderables,
            scene_renderables: vec![],
            delete: false,
        })]);
    }

    fn send_playback_plots(&self, t: f64) {
        let Some(spline) = self.se2_spline.as_ref() else {
            return;
        };

        let pose = spline.interpolate(t);
        let pos = pose.translation();
        // Heading from SE(2) rotation (unit complex params: [cos θ, sin θ]).
        let rotation = pose.factor();
        let rot_params = rotation.params();
        let theta = rot_params[1].atan2(rot_params[0]);
        // Body-frame velocity from SE(2) spline.
        let vel = spline.velocity(t);

        let t_max = spline.t_max();
        let clear_cond = ClearCondition {
            max_x_range: t_max + 1.0,
        };
        let v_line = VerticalLine {
            x: t,
            name: "now".to_owned(),
        };

        let _ = self.message_send.send(vec![
            Packet::Plot(vec![PlotViewPacket::append_to_curve_vec2_with_conf(
                (POSITION_PLOT_LABEL, "x_y"),
                vec![(t, ([pos[0], pos[1]], [0.0, 0.0]))].into(),
                CurveVecWithConfStyle {
                    colors: [Color::red(), Color::green()],
                },
                clear_cond,
                Some(v_line.clone()),
            )]),
            Packet::Plot(vec![PlotViewPacket::append_to_curve_vec2_with_conf(
                (VELOCITY_PLOT_LABEL, "vx_vy"),
                vec![(t, ([vel[1], vel[2]], [0.0, 0.0]))].into(),
                CurveVecWithConfStyle {
                    colors: [Color::red(), Color::green()],
                },
                clear_cond,
                Some(v_line.clone()),
            )]),
            Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (ANGLE_PLOT_LABEL, "theta"),
                vec![(t, theta)].into(),
                ScalarCurveStyle {
                    color: Color::blue(),
                    line_type: LineType::default(),
                },
                clear_cond,
                Some(v_line.clone()),
            )]),
            Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (ANGULAR_VELOCITY_PLOT_LABEL, "omega"),
                vec![(t, vel[0])].into(),
                ScalarCurveStyle {
                    color: Color::red(),
                    line_type: LineType::default(),
                },
                clear_cond,
                Some(v_line),
            )]),
        ]);
    }

    fn update_display(&self) {
        let Some(opt) = &self.opt.optimizer else {
            return;
        };
        let poses = opt.variables().get_members::<Isometry2F64>(WAYPOINTS);
        self.send_overlay_update(&poses);
        self.opt.send_plots(&self.message_send);
    }

    fn reset(&mut self) {
        let optimizer = Self::build_optimizer(
            &self.problem,
            self.method,
            self.solver,
            self.inner_iters,
            self.barrier_decay,
        );
        self.opt.reset();
        self.opt.latest_barrier = optimizer.barrier_param().unwrap_or(1.0);
        self.opt.optimizer = Some(optimizer);

        self.opt.delete_plots(&self.message_send);
        self.send_initial_frame();
    }

    fn send_initial_frame(&self) {
        let init_poses = self.problem.build_init_points();
        let mut pixel_renderables = self.build_static_renderables();

        // Initial trajectory (green).
        pixel_renderables.push(named_line2(
            "trajectory",
            self.build_trajectory_segments(&init_poses, Color::green(), 1.0),
        ));

        // Start point (green dot).
        let [su, sv] = world_to_pixel(self.problem.start[0], self.problem.start[1]);
        pixel_renderables.push(named_point2(
            "start",
            vec![Point2 {
                p: svec2(su, sv),
                color: Color::green(),
                point_size: 8.0,
            }],
        ));

        // End point (blue dot).
        let [eu, ev] = world_to_pixel(self.problem.end_target[0], self.problem.end_target[1]);
        pixel_renderables.push(named_point2(
            "end",
            vec![Point2 {
                p: svec2(eu, ev),
                color: Color::blue(),
                point_size: 8.0,
            }],
        ));

        let _ = self.message_send.send(vec![Packet::Image(ImageViewPacket {
            view_label: IMAGE_LABEL.to_owned(),
            frame: Some(ImageFrame::from_image(&self.heatmap)),
            pixel_renderables,
            scene_renderables: vec![],
            delete: false,
        })]);
    }

    fn send_overlay_update(&self, poses: &[Isometry2F64]) {
        let mut pixel_renderables = self.build_static_renderables();

        // Current optimizing trajectory (orange).
        pixel_renderables.push(named_line2(
            "trajectory",
            self.build_trajectory_segments(poses, Color::orange(), 2.0),
        ));

        // Start point (green dot).
        let [su, sv] = world_to_pixel(self.problem.start[0], self.problem.start[1]);
        pixel_renderables.push(named_point2(
            "start",
            vec![Point2 {
                p: svec2(su, sv),
                color: Color::green(),
                point_size: 8.0,
            }],
        ));

        // End point (blue dot).
        let [eu, ev] = world_to_pixel(self.problem.end_target[0], self.problem.end_target[1]);
        pixel_renderables.push(named_point2(
            "end",
            vec![Point2 {
                p: svec2(eu, ev),
                color: Color::blue(),
                point_size: 8.0,
            }],
        ));

        let _ = self.message_send.send(vec![Packet::Image(ImageViewPacket {
            view_label: IMAGE_LABEL.to_owned(),
            frame: None,
            pixel_renderables,
            scene_renderables: vec![],
            delete: false,
        })]);
    }

    fn build_static_renderables(&self) -> Vec<PixelRenderable> {
        vec![
            // Obstacle boundary (red).
            named_line2(
                "obstacle",
                circle_segments(
                    self.problem.obstacle_center[0],
                    self.problem.obstacle_center[1],
                    self.problem.obstacle_radius,
                    64,
                    Color {
                        r: 1.0,
                        g: 0.2,
                        b: 0.2,
                        a: 1.0,
                    },
                ),
            ),
        ]
    }

    /// Evaluate the SE(2) Lie group cubic B-spline through control points and return line
    /// segments. Positions are extracted from the translation component.
    fn build_trajectory_segments(
        &self,
        poses: &[Isometry2F64],
        color: Color,
        line_width: f32,
    ) -> Vec<LineSegment2> {
        if poses.len() < 2 {
            return vec![];
        }
        let spline = LieGroupCubicBSpline {
            control_points: poses.to_vec(),
            delta_t: 1.0,
            t0: 0.0,
        };
        let num_samples = poses.len() * 10;
        let t_max = spline.t_max();
        let dt = t_max / num_samples as f64;
        let mut segments = Vec::with_capacity(num_samples);
        let mut prev_pos = spline.interpolate(0.0).translation();
        for i in 1..=num_samples {
            let t = (i as f64 * dt).min(t_max);
            let curr_pos = spline.interpolate(t).translation();
            let [u0, v0] = world_to_pixel(prev_pos[0], prev_pos[1]);
            let [u1, v1] = world_to_pixel(curr_pos[0], curr_pos[1]);
            segments.push(LineSegment2 {
                p0: svec2(u0, v0),
                p1: svec2(u1, v1),
                color,
                line_width,
            });
            prev_pos = curr_pos;
        }
        segments
    }
}
