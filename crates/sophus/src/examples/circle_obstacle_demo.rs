//! Interactive circle + obstacle constrained point demo.
//!
//! A 2D point is pulled toward a target while constrained to lie ON a circle
//! (equality constraint) and OUTSIDE an obstacle disk (inequality constraint).

use std::f64::consts::PI;

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
    example_problems::circle_obstacle::CircleObstacleProblem,
    nlls::{
        IneqMethod,
        OptParams,
        OptProblem,
        Optimizer,
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
use crossbeam_channel::Sender;

use super::opt_widget::{
    IneqMethodSelector as Method,
    OptWidgetState,
    SolverSelector,
};

const IMAGE_LABEL: &str = "circle-obstacle";
const COST_PLOT_LABEL: &str = "circle-obstacle - cost";
const BARRIER_PLOT_LABEL: &str = "circle-obstacle - barrier";
const EQ_PLOT_LABEL: &str = "circle-obstacle - eq residual";
const MAX_STEPS: usize = 5000;

const X_MIN: f64 = 1.5;
const X_MAX: f64 = 5.0;
const Y_MIN: f64 = 1.5;
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

fn generate_heatmap(problem: &CircleObstacleProblem) -> MutImage4U8 {
    let mut img = MutImage4U8::from_image_size_and_val(
        ImageSize::new(IMG_W, IMG_H),
        sophus_autodiff::linalg::SVec::<u8, 4>::new(0, 0, 0, 255),
    );
    let max_cost = 20.0_f64;
    for py in 0..IMG_H {
        for px in 0..IMG_W {
            let x = X_MIN + (px as f64 + 0.5) / IMG_W as f64 * (X_MAX - X_MIN);
            let y = Y_MAX - (py as f64 + 0.5) / IMG_H as f64 * (Y_MAX - Y_MIN);
            let dx = x - problem.target[0];
            let dy = y - problem.target[1];
            let cost = 0.5 * (dx * dx + dy * dy);

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

/// Interactive circle + obstacle demo widget.
pub struct CircleObstacleWidget {
    message_send: Sender<Vec<Packet>>,
    problem: CircleObstacleProblem,
    opt: OptWidgetState,
    method: Method,
    solver: SolverSelector,
    inner_iters: usize,
    barrier_decay: f64,
    heatmap: ArcImage4U8,
    trail: Vec<([f32; 2], bool)>,
}

impl Drop for CircleObstacleWidget {
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
            Packet::Plot(vec![PlotViewPacket::Delete(EQ_PLOT_LABEL.to_owned())]),
        ]);
    }
}

impl CircleObstacleWidget {
    /// Create the widget.
    pub fn new(message_send: Sender<Vec<Packet>>) -> Self {
        let problem = CircleObstacleProblem::new();
        let heatmap = generate_heatmap(&problem).to_shared();
        let method = Method::Ipm;
        let solver = SolverSelector::DenseLdlt;
        let inner_iters = 100;
        let barrier_decay = 0.5;
        let optimizer = Self::build_optimizer(&problem, method, solver, inner_iters, barrier_decay);
        let opt = OptWidgetState::new(optimizer, MAX_STEPS, COST_PLOT_LABEL, BARRIER_PLOT_LABEL);
        let init_pix = world_to_pixel(problem.init[0], problem.init[1]);

        let widget = Self {
            message_send,
            problem,
            opt,
            method,
            solver,
            inner_iters,
            barrier_decay,
            heatmap,
            trail: vec![(init_pix, false)],
        };
        widget.send_initial_frame();
        widget
    }

    fn build_optimizer(
        problem: &CircleObstacleProblem,
        method: Method,
        solver: SolverSelector,
        inner_iters: usize,
        barrier_decay: f64,
    ) -> Optimizer {
        Optimizer::new(
            problem.build_variables(),
            OptProblem {
                costs: vec![problem.build_prior_cost()],
                eq_constraints: vec![problem.build_eq_constraint()],
                ineq_constraints: vec![problem.build_obstacle_barrier()],
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
        ui.label("Circle + Obstacle");
        ui.label("Cost: ||p-target||^2. Eq: on circle. Ineq: outside obstacle.");

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
                for &s in super::opt_widget::EQ_SOLVER_OPTIONS {
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

        // Show eq constraint residual.
        if let Some(opt) = &self.opt.optimizer {
            let p = opt
                .variables()
                .get_members::<VecF64<2>>(sophus_opt::example_problems::circle_obstacle::POINT)[0];
            let dist = (p - self.problem.circle_center).norm();
            let eq_res = (dist - self.problem.circle_radius).abs();
            ui.label(format!("eq |‖p−c‖−r| = {eq_res:.4}"));
        }
    }

    /// Called each frame.
    pub fn update(&mut self) {
        if let Some(info) = self.opt.update() {
            self.update_display(info.in_phase1);
        }
    }

    fn update_display(&mut self, in_phase1: bool) {
        let Some(opt) = &self.opt.optimizer else {
            return;
        };
        let p = opt
            .variables()
            .get_members::<VecF64<2>>(sophus_opt::example_problems::circle_obstacle::POINT)[0];

        let pix = world_to_pixel(p[0], p[1]);
        self.trail.push((pix, in_phase1));

        // Eq residual plot.
        let dist = (p - self.problem.circle_center).norm();
        let eq_res = (dist - self.problem.circle_radius).abs();
        self.send_eq_plot(eq_res);

        self.send_overlay_update(p);
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
        self.trail = vec![(
            world_to_pixel(self.problem.init[0], self.problem.init[1]),
            false,
        )];

        self.opt.delete_plots(&self.message_send);
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::Delete(
                EQ_PLOT_LABEL.to_owned(),
            )])]);
        self.send_initial_frame();
    }

    fn send_initial_frame(&self) {
        let mut pixel_renderables = vec![];

        // Constraint circle (yellow).
        pixel_renderables.push(named_line2(
            "circle",
            circle_segments(
                self.problem.circle_center[0],
                self.problem.circle_center[1],
                self.problem.circle_radius,
                64,
                Color {
                    r: 1.0,
                    g: 1.0,
                    b: 0.0,
                    a: 1.0,
                },
            ),
        ));

        // Obstacle disk boundary (red).
        pixel_renderables.push(named_line2(
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
        ));

        // Target.
        let [tu, tv] = world_to_pixel(self.problem.target[0], self.problem.target[1]);
        pixel_renderables.push(named_point2(
            "target",
            vec![Point2 {
                p: svec2(tu, tv),
                color: Color::blue(),
                point_size: 6.0,
            }],
        ));

        // Init point.
        let [pu, pv] = world_to_pixel(self.problem.init[0], self.problem.init[1]);
        pixel_renderables.push(named_point2(
            "current",
            vec![Point2 {
                p: svec2(pu, pv),
                color: Color::orange(),
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

        self.opt.init_plots(&self.message_send);

        // Eq residual plot.
        let clear_cond = ClearCondition {
            max_x_range: MAX_STEPS as f64 + 1.0,
        };
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (EQ_PLOT_LABEL, "eq residual"),
                std::collections::VecDeque::new(),
                ScalarCurveStyle {
                    color: Color::orange(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )])]);
    }

    fn send_overlay_update(&self, p: VecF64<2>) {
        let [pu, pv] = world_to_pixel(p[0], p[1]);
        let mut pixel_renderables = vec![];

        // Constraint circle + obstacle (static).
        pixel_renderables.push(named_line2(
            "circle",
            circle_segments(
                self.problem.circle_center[0],
                self.problem.circle_center[1],
                self.problem.circle_radius,
                64,
                Color {
                    r: 1.0,
                    g: 1.0,
                    b: 0.0,
                    a: 1.0,
                },
            ),
        ));
        pixel_renderables.push(named_line2(
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
        ));

        // Trail.
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

        // Target.
        let [tu, tv] = world_to_pixel(self.problem.target[0], self.problem.target[1]);
        pixel_renderables.push(named_point2(
            "target",
            vec![Point2 {
                p: svec2(tu, tv),
                color: Color::blue(),
                point_size: 6.0,
            }],
        ));

        // Current point.
        pixel_renderables.push(named_point2(
            "current",
            vec![Point2 {
                p: svec2(pu, pv),
                color: Color::orange(),
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

    fn send_eq_plot(&self, eq_res: f64) {
        let clear_cond = ClearCondition {
            max_x_range: MAX_STEPS as f64 + 1.0,
        };
        let _ = self
            .message_send
            .send(vec![Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (EQ_PLOT_LABEL, "eq residual"),
                vec![(self.opt.total_steps as f64, eq_res)].into(),
                ScalarCurveStyle {
                    color: Color::orange(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )])]);
    }
}
