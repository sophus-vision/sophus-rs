//! Shared optimization widget state and UI helpers.
//!
//! Extracted from the common patterns across point2d, corridor, and BA demos.
//! Each demo widget owns an `OptWidgetState` and delegates step/button/status
//! logic to it, providing problem-specific visualization via callbacks.

use eframe::egui;
use sophus_opt::nlls::{
    Optimizer,
    StepInfo,
    TerminationReason,
};
use sophus_renderer::renderables::Color;
use sophus_solver::LinearSolverEnum;
use sophus_viewer::packets::{
    ClearCondition,
    LineType,
    Packet,
    PlotViewPacket,
    ScalarCurveStyle,
};
use crossbeam_channel::Sender;

/// Inequality constraint method selector for demo widgets.
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum IneqMethodSelector {
    /// Primal-dual IPM.
    Ipm,
    /// SQP with frozen Jacobians.
    Sqp,
}

impl IneqMethodSelector {
    /// UI label for the method.
    pub fn label(self) -> &'static str {
        match self {
            Self::Ipm => "IPM",
            Self::Sqp => "SQP",
        }
    }

    /// Label for the barrier/dual parameter.
    pub fn barrier_label(self) -> &'static str {
        match self {
            Self::Ipm => "lambda",
            Self::Sqp => "mu",
        }
    }
}

/// Linear solver selector for demo widgets.
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum SolverSelector {
    /// Dense LDLᵀ (handles PD, PSD, and indefinite).
    DenseLdlt,
    /// Dense LU.
    DenseLu,
    /// Block-sparse LDLᵀ with AMD ordering.
    BlockSparseLdlt,
    /// Faer sparse LDLᵀ (PD only).
    FaerSparseLdlt,
    /// Faer sparse LBLᵀ (Bunch-Kaufman, indefinite).
    FaerSparseLblt,
    /// Faer sparse LU.
    FaerSparseLu,
}

/// All solver options (for ineq-only problems where all solvers work).
pub const ALL_SOLVER_OPTIONS: &[SolverSelector] = &[
    SolverSelector::DenseLdlt,
    SolverSelector::DenseLu,
    SolverSelector::BlockSparseLdlt,
    SolverSelector::FaerSparseLdlt,
    SolverSelector::FaerSparseLblt,
    SolverSelector::FaerSparseLu,
];

/// Solver options for problems with equality constraints (indefinite KKT).
/// Excludes FaerSparseLdlt which assumes PD.
pub const EQ_SOLVER_OPTIONS: &[SolverSelector] = &[
    SolverSelector::DenseLdlt,
    SolverSelector::DenseLu,
    SolverSelector::BlockSparseLdlt,
    SolverSelector::FaerSparseLblt,
    SolverSelector::FaerSparseLu,
];

impl SolverSelector {
    /// UI label.
    pub fn label(self) -> &'static str {
        match self {
            Self::DenseLdlt => "Dense LDLt",
            Self::DenseLu => "Dense LU",
            Self::BlockSparseLdlt => "Block-sparse LDLt",
            Self::FaerSparseLdlt => "Faer sparse LDLt",
            Self::FaerSparseLblt => "Faer sparse LBLt",
            Self::FaerSparseLu => "Faer sparse LU",
        }
    }

    /// Whether this solver has a Schur-complement variant.
    pub fn has_schur(self) -> bool {
        self.to_solver().to_schur().is_some()
    }

    /// Convert to `LinearSolverEnum`, using Schur variant if `use_schur` is true
    /// and the solver supports it.
    pub fn to_linear_solver(self, use_schur: bool) -> LinearSolverEnum {
        let solver = self.to_solver();
        if use_schur {
            solver.to_schur().unwrap_or(solver)
        } else {
            solver
        }
    }

    /// Convert to `LinearSolverEnum`.
    pub fn to_solver(self) -> LinearSolverEnum {
        match self {
            Self::DenseLdlt => {
                LinearSolverEnum::DenseLdlt(sophus_solver::ldlt::DenseLdlt::default())
            }
            Self::DenseLu => LinearSolverEnum::DenseLu(sophus_solver::lu::DenseLu {}),
            Self::BlockSparseLdlt => {
                LinearSolverEnum::BlockSparseLdlt(sophus_solver::ldlt::BlockSparseLdlt::default())
            }
            Self::FaerSparseLdlt => {
                LinearSolverEnum::FaerSparseLdlt(sophus_solver::ldlt::FaerSparseLdlt::default())
            }
            Self::FaerSparseLblt => {
                LinearSolverEnum::FaerSparseLblt(sophus_solver::ldlt::FaerSparseLblt::default())
            }
            Self::FaerSparseLu => {
                LinearSolverEnum::FaerSparseLu(sophus_solver::lu::FaerSparseLu {})
            }
        }
    }
}

/// What kind of step to take next.
#[derive(PartialEq, Clone, Copy)]
pub enum StepRequest {
    /// No step requested.
    None,
    /// Run continuously (one step per frame).
    Run,
    /// Single step.
    Step,
}

/// Shared optimizer widget state.
///
/// Handles the step loop, buttons, counters, cost/param display,
/// termination reason, and plot sending. Each demo widget composes
/// this with its problem-specific state.
pub struct OptWidgetState {
    /// The optimizer.
    pub optimizer: Option<Optimizer>,
    /// Current step request.
    pub step_request: StepRequest,
    /// Total steps taken.
    pub total_steps: usize,
    /// Current inner step within outer iteration.
    pub current_inner: usize,
    /// Total outer iterations completed.
    pub total_outer: usize,
    /// Latest smooth cost.
    pub latest_cost: Option<f64>,
    /// Latest barrier parameter.
    pub latest_barrier: f64,
    /// Last termination reason.
    pub latest_reason: Option<TerminationReason>,
    /// Maximum steps before auto-stop.
    pub max_steps: usize,
    /// Cost plot label.
    pub cost_plot_label: String,
    /// Param plot label.
    pub barrier_plot_label: String,
    /// Damping plot label.
    pub damping_plot_label: String,
    /// Latest LM damping value.
    pub latest_damping: f64,
    /// Whether the optimizer is in phase-1 (feasibility).
    pub in_phase1: bool,
}

impl OptWidgetState {
    /// Create a new widget state.
    pub fn new(
        optimizer: Optimizer,
        max_steps: usize,
        cost_plot_label: &str,
        barrier_plot_label: &str,
    ) -> Self {
        let damping_plot_label = format!("{cost_plot_label} - damping");
        Self {
            latest_barrier: optimizer.barrier_param().unwrap_or(1.0),
            latest_damping: optimizer.lm_damping(),
            in_phase1: false,
            optimizer: Some(optimizer),
            step_request: StepRequest::None,
            total_steps: 0,
            current_inner: 0,
            total_outer: 0,
            latest_cost: None,
            latest_reason: None,
            max_steps,
            cost_plot_label: cost_plot_label.to_owned(),
            barrier_plot_label: barrier_plot_label.to_owned(),
            damping_plot_label,
        }
    }

    /// Execute one optimizer step. Returns the StepInfo if a step was taken.
    pub fn do_step(&mut self) -> Option<StepInfo> {
        let opt = self.optimizer.as_mut()?;

        match opt.step() {
            Ok(info) => {
                self.total_steps += 1;
                self.current_inner = info.inner_step;

                if info.did_outer_step {
                    self.total_outer += 1;
                    self.latest_barrier = opt.barrier_param().unwrap_or(self.latest_barrier);
                }

                self.latest_cost = Some(info.smooth_cost);
                self.latest_damping = info.lm_damping;
                self.in_phase1 = info.in_phase1;

                if info.termination.is_some() {
                    self.latest_reason = info.termination.clone();
                    self.step_request = StepRequest::None;
                }

                Some(info)
            }
            Err(_) => {
                self.latest_reason = Some(TerminationReason::LinearSolverFailed);
                self.step_request = StepRequest::None;
                None
            }
        }
    }

    /// Called each frame. Returns `Some(StepInfo)` if a step was taken.
    pub fn update(&mut self) -> Option<StepInfo> {
        match self.step_request {
            StepRequest::None => None,
            StepRequest::Step => {
                let info = self.do_step();
                self.step_request = StepRequest::None;
                info
            }
            StepRequest::Run => {
                let info = self.do_step();
                if self.total_steps >= self.max_steps {
                    self.step_request = StepRequest::None;
                }
                info
            }
        }
    }

    /// Draw the standard buttons: Optimize/Stop Opt, Step, Reset.
    /// Returns true if Reset was clicked.
    pub fn draw_buttons(&mut self, ui: &mut egui::Ui) -> bool {
        let mut reset = false;
        ui.horizontal(|ui| {
            if self.step_request == StepRequest::Run {
                if ui.button("Stop opt").clicked() {
                    self.step_request = StepRequest::None;
                }
            } else {
                if ui.button("Optimize").clicked() {
                    self.step_request = StepRequest::Run;
                }
                if ui.button("Step").clicked() {
                    self.step_request = StepRequest::Step;
                }
                if ui.button("Reset").clicked() {
                    reset = true;
                }
            }
        });
        reset
    }

    /// Draw step counters and status (cost, param, termination reason).
    pub fn draw_status(&self, ui: &mut egui::Ui, barrier_label: &str) {
        if self.in_phase1 {
            ui.colored_label(
                egui::Color32::from_rgb(30, 30, 150),
                "Phase 1 (feasibility)",
            );
        }
        ui.label(format!(
            "inner: {} outer: {} total: {}",
            self.current_inner, self.total_outer, self.total_steps,
        ));

        if self
            .optimizer
            .as_ref()
            .and_then(|o| o.barrier_param())
            .is_some()
        {
            ui.label(format!("{} = {:.2e}", barrier_label, self.latest_barrier));
        }

        if let Some(cost) = self.latest_cost {
            ui.label(format!("cost = {:.4e}", cost));
        }

        if let Some(ref reason) = self.latest_reason {
            ui.colored_label(egui::Color32::YELLOW, format!("stopped: {reason}"));
        }
    }

    /// Reset the optimizer state. Call this then set self.optimizer to the new one.
    pub fn reset(&mut self) {
        self.step_request = StepRequest::None;
        self.total_steps = 0;
        self.current_inner = 0;
        self.total_outer = 0;
        self.latest_cost = None;
        self.latest_reason = None;
    }

    /// Send cost and param plot data for the current step.
    pub fn send_plots(&self, sender: &Sender<Vec<Packet>>) {
        if let Some(cost) = self.latest_cost {
            self.send_plot(sender, &self.cost_plot_label, "cost", cost, Color::blue());
        }
        self.send_plot(
            sender,
            &self.barrier_plot_label,
            "barrier",
            self.latest_barrier,
            Color::green(),
        );
        self.send_plot(
            sender,
            &self.damping_plot_label,
            "damping",
            self.latest_damping,
            Color::red(),
        );
    }

    /// Delete all plot windows.
    pub fn delete_plots(&self, sender: &Sender<Vec<Packet>>) {
        let _ = sender.send(vec![
            Packet::Plot(vec![PlotViewPacket::Delete(self.cost_plot_label.clone())]),
            Packet::Plot(vec![PlotViewPacket::Delete(
                self.barrier_plot_label.clone(),
            )]),
            Packet::Plot(vec![PlotViewPacket::Delete(
                self.damping_plot_label.clone(),
            )]),
        ]);
    }

    /// Send empty plot curves so windows are visible from the start.
    pub fn init_plots(&self, sender: &Sender<Vec<Packet>>) {
        let clear_cond = ClearCondition {
            max_x_range: self.max_steps as f64 + 1.0,
        };
        let _ = sender.send(vec![
            Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (&*self.cost_plot_label, "cost"),
                std::collections::VecDeque::new(),
                ScalarCurveStyle {
                    color: Color::blue(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )]),
            Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (&*self.barrier_plot_label, "barrier"),
                std::collections::VecDeque::new(),
                ScalarCurveStyle {
                    color: Color::green(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )]),
            Packet::Plot(vec![PlotViewPacket::append_to_curve(
                (&*self.damping_plot_label, "damping"),
                std::collections::VecDeque::new(),
                ScalarCurveStyle {
                    color: Color::red(),
                    line_type: LineType::LineStrip,
                },
                clear_cond,
                None,
            )]),
        ]);
    }

    fn send_plot(
        &self,
        sender: &Sender<Vec<Packet>>,
        plot_label: &str,
        curve_name: &str,
        value: f64,
        color: Color,
    ) {
        let clear_cond = ClearCondition {
            max_x_range: self.max_steps as f64 + 1.0,
        };
        let _ = sender.send(vec![Packet::Plot(vec![PlotViewPacket::append_to_curve(
            (plot_label, curve_name),
            vec![(self.total_steps as f64, value)].into(),
            ScalarCurveStyle {
                color,
                line_type: LineType::LineStrip,
            },
            clear_cond,
            None,
        )])]);
    }
}
