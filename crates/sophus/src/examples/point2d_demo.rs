//! Interactive 2D constrained point demo.
//!
//! A 2D point is pulled toward a target while ellipse obstacles block the
//! direct path. The cost landscape is shown as a heatmap with infeasible
//! regions darkened and constraint boundaries overlaid.

use std::f64::consts::PI;

use crossbeam_channel::Sender;
use eframe::egui;
use sophus_autodiff::linalg::VecF64;
use sophus_image::{
    ArcImage4U8,
    ImageSize,
    MutImage4U8,
    color_map::BlueWhiteRedBlackColorMap,
    prelude::IsMutImageView,
};
use sophus_opt::{
    nlls::{
        CostFn,
        CostTerms,
        IneqBarrierCostFn,
        IneqConstraints,
        IneqMethod,
        OptParams,
        OptProblem,
        Optimizer,
        costs::Quadratic2CostTerm,
        ineq_constraints::ScalarEllipseConstraint,
    },
    variables::{
        VarBuilder,
        VarFamily,
        VarKind,
    },
};
use sophus_renderer::renderables::{
    Color,
    ImageFrame,
    LineSegment2,
    Point2,
    named_line2,
    named_point2,
};
use sophus_viewer::packets::{
    ClearCondition,
    ImageViewPacket,
    LineType,
    Packet,
    PlotViewPacket,
    ScalarCurveStyle,
};

use super::opt_widget::{
    OptWidgetState,
    SolverSelector,
};

const IMAGE_LABEL: &str = "point2d";
const COST_PLOT_LABEL: &str = "point2d - cost";
const BARRIER_PLOT_LABEL: &str = "point2d - barrier";
const HMIN_PLOT_LABEL: &str = "point2d - h_min";
const MAX_STEPS: usize = 5000;
const POINT: &str = "point";

const X_MIN: f64 = -1.0;
const X_MAX: f64 = 7.0;
const Y_MIN: f64 = -1.0;
const Y_MAX: f64 = 7.0;
const IMG_W: usize = 400;
const IMG_H: usize = 400;

#[derive(Clone)]
struct Obstacle {
    center: VecF64<2>,
    a: f64,
    b: f64,
    angle: f64,
}

fn svec2(x: f32, y: f32) -> sophus_autodiff::linalg::SVec<f32, 2> {
    sophus_autodiff::linalg::SVec::<f32, 2>::new(x, y)
}

fn world_to_pixel(x: f64, y: f64) -> [f32; 2] {
    let u = ((x - X_MIN) / (X_MAX - X_MIN) * IMG_W as f64) as f32;
    let v = ((Y_MAX - y) / (Y_MAX - Y_MIN) * IMG_H as f64) as f32;
    [u, v]
}

fn is_infeasible(x: f64, y: f64, obstacles: &[Obstacle]) -> bool {
    obstacles.iter().any(|o| {
        let dx = x - o.center[0];
        let dy = y - o.center[1];
        let c = o.angle.cos();
        let s = o.angle.sin();
        let u = c * dx + s * dy;
        let v = -s * dx + c * dy;
        (u / o.a).powi(2) + (v / o.b).powi(2) < 1.0
    })
}

fn generate_heatmap(target: VecF64<2>, obstacles: &[Obstacle]) -> MutImage4U8 {
    let mut img = MutImage4U8::from_image_size_and_val(
        ImageSize::new(IMG_W, IMG_H),
        sophus_autodiff::linalg::SVec::<u8, 4>::new(0, 0, 0, 255),
    );
    let max_cost = 20.0_f64;
    for py in 0..IMG_H {
        for px in 0..IMG_W {
            let x = X_MIN + (px as f64 + 0.5) / IMG_W as f64 * (X_MAX - X_MIN);
            let y = Y_MAX - (py as f64 + 0.5) / IMG_H as f64 * (Y_MAX - Y_MIN);
            let dx = x - target[0];
            let dy = y - target[1];
            let cost = 0.5 * (dx * dx + dy * dy);
            let infeasible = is_infeasible(x, y, obstacles);
            let t = (cost / max_cost).min(1.0) as f32;
            let rgb = if infeasible {
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

fn ellipse_segments(obs: &Obstacle, n: usize, color: Color) -> Vec<LineSegment2> {
    let c = obs.angle.cos();
    let s = obs.angle.sin();
    (0..n)
        .map(|i| {
            let t0 = 2.0 * PI * i as f64 / n as f64;
            let t1 = 2.0 * PI * (i + 1) as f64 / n as f64;
            let lx0 = obs.a * t0.cos();
            let ly0 = obs.b * t0.sin();
            let lx1 = obs.a * t1.cos();
            let ly1 = obs.b * t1.sin();
            let [u0, v0] = world_to_pixel(
                obs.center[0] + c * lx0 - s * ly0,
                obs.center[1] + s * lx0 + c * ly0,
            );
            let [u1, v1] = world_to_pixel(
                obs.center[0] + c * lx1 - s * ly1,
                obs.center[1] + s * lx1 + c * ly1,
            );
            LineSegment2 {
                p0: svec2(u0, v0),
                p1: svec2(u1, v1),
                color,
                line_width: 2.0,
            }
        })
        .collect()
}

fn default_obstacles() -> Vec<Obstacle> {
    vec![
        Obstacle {
            center: VecF64::<2>::new(2.0, 2.0),
            a: 1.2,
            b: 0.5,
            angle: PI / 4.0,
        },
        Obstacle {
            center: VecF64::<2>::new(4.5, 4.0),
            a: 1.0,
            b: 0.7,
            angle: -PI / 6.0,
        },
    ]
}

use super::opt_widget::IneqMethodSelector as Method;

/// Interactive 2D constrained point demo widget.
pub struct Point2dWidget {
    message_send: Sender<Vec<Packet>>,
    target: VecF64<2>,
    init: VecF64<2>,
    obstacles: Vec<Obstacle>,
    /// Shared optimization widget state.
    opt: OptWidgetState,
    /// Selected method.
    pub method: Method,
    /// Selected solver.
    solver: SolverSelector,
    /// Inner steps per outer iteration.
    pub inner_iters: usize,
    /// Decay factor.
    pub barrier_decay: f64,
    latest_min_h: Option<f64>,
    /// Whether to start from infeasible position.
    pub infeasible_init: bool,
    /// Heatmap image (cached).
    heatmap: ArcImage4U8,
    /// Trail of previous point positions (pixel coords, in_phase1 flag).
    trail: Vec<([f32; 2], bool)>,
}

impl Drop for Point2dWidget {
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
            Packet::Plot(vec![PlotViewPacket::Delete(HMIN_PLOT_LABEL.to_owned())]),
        ]);
    }
}

impl Point2dWidget {
    /// Create the widget.
    pub fn new(message_send: Sender<Vec<Packet>>) -> Self {
        let target = VecF64::<2>::new(5.5, 5.5);
        let init = VecF64::<2>::new(1.0, 0.5);
        let obstacles = default_obstacles();
        let heatmap = generate_heatmap(target, &obstacles).to_shared();
        let method = Method::Ipm;
        let solver = SolverSelector::DenseLdlt;
        let inner_iters = 50;
        let barrier_decay = 0.5;
        let optimizer = Self::build_optimizer(
            target,
            init,
            &obstacles,
            method,
            solver,
            inner_iters,
            barrier_decay,
        );

        let opt = OptWidgetState::new(optimizer, MAX_STEPS, COST_PLOT_LABEL, BARRIER_PLOT_LABEL);

        let widget = Point2dWidget {
            message_send,
            target,
            init,
            obstacles,
            opt,
            method,
            solver,
            inner_iters,
            barrier_decay,
            latest_min_h: None,
            infeasible_init: false,
            heatmap,
            trail: vec![(world_to_pixel(init[0], init[1]), false)],
        };
        widget.send_initial_frame();
        widget
    }

    fn build_optimizer(
        target: VecF64<2>,
        init: VecF64<2>,
        obstacles: &[Obstacle],
        method: Method,
        solver: SolverSelector,
        inner_iters: usize,
        barrier_decay: f64,
    ) -> Optimizer {
        let variables = VarBuilder::new()
            .add_family(POINT, VarFamily::new(VarKind::Free, vec![init]))
            .build();
        let prior = CostFn::new_boxed(
            (),
            CostTerms::new(
                [POINT],
                vec![Quadratic2CostTerm {
                    z: target,
                    entity_indices: [0],
                }],
            ),
        );
        let constraints: Vec<ScalarEllipseConstraint> = obstacles
            .iter()
            .map(|o| ScalarEllipseConstraint::new(o.center, o.a, o.b, o.angle, 0))
            .collect();
        let barrier = IneqBarrierCostFn::new_boxed((), IneqConstraints::new([POINT], constraints));

        Optimizer::new(
            variables,
            OptProblem {
                costs: vec![prior],
                eq_constraints: vec![],
                ineq_constraints: vec![barrier],
            },
            OptParams {
                num_iterations: MAX_STEPS,
                initial_lm_damping: 1.0,
                parallelize: false,
                solver: solver.to_solver(),
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
            },
        )
        .expect("Optimizer::new failed")
    }

    /// Draw the left-panel controls.
    pub fn update_left_panel(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("2D Constrained Point");
        ui.label("Cost: ||p-target||^2. Ineq: outside ellipses.");

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

        ui.add(egui::Slider::new(&mut self.inner_iters, 1..=200).text("inner iters / outer"));
        match self.method {
            Method::Ipm => {
                ui.add(
                    egui::Slider::new(&mut self.barrier_decay, 0.01..=0.99).text("lambda decay"),
                );
            }
            Method::Sqp => {
                ui.add(egui::Slider::new(&mut self.barrier_decay, 0.01..=0.99).text("mu decay"));
            }
        }

        let prev = self.infeasible_init;
        ui.checkbox(&mut self.infeasible_init, "infeasible init");
        if self.infeasible_init != prev {
            self.reset();
        }

        // Obstacle parameter sliders.
        let mut obstacles_changed = false;
        for (i, obs) in self.obstacles.iter_mut().enumerate() {
            ui.separator();
            ui.label(format!("Ellipse {i}"));
            obstacles_changed |= ui
                .add(egui::Slider::new(&mut obs.center[0], X_MIN..=X_MAX).text("cx"))
                .changed();
            obstacles_changed |= ui
                .add(egui::Slider::new(&mut obs.center[1], Y_MIN..=Y_MAX).text("cy"))
                .changed();
            obstacles_changed |= ui
                .add(egui::Slider::new(&mut obs.a, 0.1..=3.0).text("a"))
                .changed();
            obstacles_changed |= ui
                .add(egui::Slider::new(&mut obs.b, 0.1..=3.0).text("b"))
                .changed();
            obstacles_changed |= ui
                .add(egui::Slider::new(&mut obs.angle, -PI..=PI).text("angle"))
                .changed();
        }
        if obstacles_changed {
            self.heatmap = generate_heatmap(self.target, &self.obstacles).to_shared();
            self.reset();
        }

        ui.separator();
        if self.opt.draw_buttons(ui) {
            self.reset();
        }

        self.opt.draw_status(ui, self.method.barrier_label());

        if let Some(h) = self.latest_min_h {
            ui.label(format!(
                "min h = {:.4} ({})",
                h,
                if h > 0.0 { "feasible" } else { "infeasible" }
            ));
        }

        if let Some(opt) = &self.opt.optimizer {
            let points = opt.variables().get_members::<VecF64<2>>(POINT);
            if !points.is_empty() {
                let p = points[0];
                ui.label(format!("p = ({:.3}, {:.3})", p[0], p[1]));
            }
        }
    }

    /// Called each frame.
    pub fn update(&mut self) {
        if let Some(info) = self.opt.update() {
            self.update_display(info.in_phase1);
        }
    }

    /// Update plots and point overlay from current optimizer state.
    fn update_display(&mut self, in_phase1: bool) {
        let Some(opt) = &self.opt.optimizer else {
            return;
        };

        let points = opt.variables().get_members::<VecF64<2>>(POINT);
        let p = points[0];

        let min_h: f64 = self
            .obstacles
            .iter()
            .map(|o| Self::h_for_obstacle(p, o))
            .fold(f64::INFINITY, f64::min);

        self.latest_min_h = Some(min_h);

        let pix = world_to_pixel(p[0], p[1]);
        self.trail.push((pix, in_phase1));

        self.send_overlay_update(p);
        self.opt.send_plots(&self.message_send);
        self.send_hmin_plot(min_h);
    }

    fn reset(&mut self) {
        let init = if self.infeasible_init {
            self.obstacles[0].center
        } else {
            self.init
        };
        let optimizer = Self::build_optimizer(
            self.target,
            init,
            &self.obstacles,
            self.method,
            self.solver,
            self.inner_iters,
            self.barrier_decay,
        );
        self.opt.reset();
        self.opt.latest_barrier = optimizer.barrier_param().unwrap_or(1.0);
        self.opt.optimizer = Some(optimizer);
        self.latest_min_h = None;
        self.trail = vec![(world_to_pixel(init[0], init[1]), self.infeasible_init)];

        self.opt.delete_plots(&self.message_send);
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::Delete(
                HMIN_PLOT_LABEL.to_owned(),
            )])]);
        self.send_initial_frame();
    }

    fn send_initial_frame(&self) {
        let init = if self.infeasible_init {
            self.obstacles[0].center
        } else {
            self.init
        };

        let mut pixel_renderables = vec![];

        let mut all_segments = vec![];
        for o in &self.obstacles {
            all_segments.extend(ellipse_segments(
                o,
                64,
                Color {
                    r: 1.0,
                    g: 1.0,
                    b: 0.0,
                    a: 1.0,
                },
            ));
        }
        pixel_renderables.push(named_line2("obstacles", all_segments));

        let [pu, pv] = world_to_pixel(init[0], init[1]);
        pixel_renderables.push(named_point2(
            "current",
            vec![Point2 {
                p: svec2(pu, pv),
                color: Color::orange(),
                point_size: 4.0,
            }],
        ));

        let _ = self.message_send.send(vec![Packet::Image(ImageViewPacket {
            view_label: IMAGE_LABEL.to_owned(),
            frame: Some(ImageFrame::from_image(&self.heatmap)),
            pixel_renderables,
            scene_renderables: vec![],
            delete: false,
        })]);

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

    /// Compute h value for a point w.r.t. an obstacle.
    fn h_for_obstacle(p: VecF64<2>, o: &Obstacle) -> f64 {
        let u = p - o.center;
        let c = o.angle.cos();
        let s = o.angle.sin();
        let lu = c * u[0] + s * u[1];
        let lv = -s * u[0] + c * u[1];
        (lu / o.a).powi(2) + (lv / o.b).powi(2) - 1.0
    }

    fn send_overlay_update(&self, p: VecF64<2>) {
        let [pu, pv] = world_to_pixel(p[0], p[1]);

        let mut pixel_renderables = vec![];

        // Ellipse boundaries (static yellow).
        let mut all_segments = vec![];
        for o in &self.obstacles {
            all_segments.extend(ellipse_segments(
                o,
                64,
                Color {
                    r: 1.0,
                    g: 1.0,
                    b: 0.0,
                    a: 1.0,
                },
            ));
        }
        pixel_renderables.push(named_line2("obstacles", all_segments));

        // Linearized constraints: tangent line to each ellipse at the
        // nearest boundary point to the current iterate.
        // Both IPM and SQP linearize constraints (IPM re-linearizes every step,
        // SQP freezes at outer iterates).
        {
            let mut lin_segments = vec![];
            for o in &self.obstacles {
                let u = p - o.center;
                let c = o.angle.cos();
                let s = o.angle.sin();
                // Rotate to axis-aligned frame.
                let lx = c * u[0] + s * u[1];
                let ly = -s * u[0] + c * u[1];
                // Approximate nearest point on ellipse: normalize to unit circle,
                // project, scale back.
                let nx = lx / o.a;
                let ny = ly / o.b;
                let nr = (nx * nx + ny * ny).sqrt().max(1e-10);
                let qx_local = o.a * nx / nr;
                let qy_local = o.b * ny / nr;
                // Rotate back to world frame.
                let qx = o.center[0] + c * qx_local - s * qy_local;
                let qy = o.center[1] + s * qx_local + c * qy_local;
                // Gradient at q (in world frame): ∇h = 2A(q - center).
                let qu = VecF64::<2>::new(qx - o.center[0], qy - o.center[1]);
                let ia2 = 1.0 / (o.a * o.a);
                let ib2 = 1.0 / (o.b * o.b);
                let a00 = c * c * ia2 + s * s * ib2;
                let a01 = c * s * (ia2 - ib2);
                let a11 = s * s * ia2 + c * c * ib2;
                let gx = 2.0 * (a00 * qu[0] + a01 * qu[1]);
                let gy = 2.0 * (a01 * qu[0] + a11 * qu[1]);
                let gnorm = (gx * gx + gy * gy).sqrt();
                if gnorm < 1e-10 {
                    continue;
                }
                // Tangent direction at q.
                let tx = -gy / gnorm;
                let ty = gx / gnorm;
                let extent = 4.0;
                let x0 = qx - extent * tx;
                let y0 = qy - extent * ty;
                let x1 = qx + extent * tx;
                let y1 = qy + extent * ty;
                let [pu0, pv0] = world_to_pixel(x0, y0);
                let [pu1, pv1] = world_to_pixel(x1, y1);
                lin_segments.push(LineSegment2 {
                    p0: svec2(pu0, pv0),
                    p1: svec2(pu1, pv1),
                    color: Color {
                        r: 1.0,
                        g: 0.3,
                        b: 0.3,
                        a: 0.6,
                    },
                    line_width: 1.0,
                });
            }
            if !lin_segments.is_empty() {
                pixel_renderables.push(named_line2("linearized", lin_segments));
            }
        }

        // Trail as connected line segments.
        // Dark blue = phase-1 (feasibility), green = phase-2 (optimization).
        if self.trail.len() >= 2 {
            let phase1_color = Color {
                r: 0.1,
                g: 0.1,
                b: 0.6,
                a: 0.9,
            };
            let phase2_color = Color {
                r: 0.0,
                g: 1.0,
                b: 0.5,
                a: 0.8,
            };
            let segments: Vec<LineSegment2> = self
                .trail
                .windows(2)
                .map(|w| {
                    let color = if w[1].1 { phase1_color } else { phase2_color };
                    LineSegment2 {
                        p0: svec2(w[0].0[0], w[0].0[1]),
                        p1: svec2(w[1].0[0], w[1].0[1]),
                        color,
                        line_width: 1.5,
                    }
                })
                .collect();
            pixel_renderables.push(named_line2("trail", segments));
        }

        // Target (minimum) as a small blue circle.
        let [tu, tv] = world_to_pixel(self.target[0], self.target[1]);
        pixel_renderables.push(named_point2(
            "target",
            vec![Point2 {
                p: svec2(tu, tv),
                color: Color::red(),
                point_size: 6.0,
            }],
        ));

        // Current point.
        pixel_renderables.push(named_point2(
            "current",
            vec![Point2 {
                p: svec2(pu, pv),
                color: Color::orange(),
                point_size: 4.0,
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

    fn send_hmin_plot(&self, h: f64) {
        let clear_cond = ClearCondition {
            max_x_range: MAX_STEPS as f64 + 1.0,
        };
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (HMIN_PLOT_LABEL, "min h"),
                vec![(self.opt.total_steps as f64, h)].into(),
                ScalarCurveStyle {
                    color: Color::orange(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )])]);
    }
}
