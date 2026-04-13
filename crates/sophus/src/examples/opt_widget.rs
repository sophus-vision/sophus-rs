//! Shared optimization widget state and UI helpers.
//!
//! Extracted from the common patterns across point2d, corridor, and BA demos.
//! Each demo widget owns an `OptWidgetState` and delegates step/button/status
//! logic to it, providing problem-specific visualization via callbacks.

use crossbeam_channel::Sender;
use eframe::egui;
use sophus_opt::nlls::{
    Optimizer,
    StepInfo,
    TerminationReason,
};
use sophus_renderer::renderables::Color;
use sophus_viewer::packets::{
    ClearCondition,
    LineType,
    Packet,
    PlotViewPacket,
    ScalarCurveStyle,
};

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
    pub latest_param: f64,
    /// Last termination reason.
    pub latest_reason: Option<TerminationReason>,
    /// Maximum steps before auto-stop.
    pub max_steps: usize,
    /// Cost plot label.
    pub cost_plot_label: String,
    /// Param plot label.
    pub param_plot_label: String,
}

impl OptWidgetState {
    /// Create a new widget state.
    pub fn new(
        optimizer: Optimizer,
        max_steps: usize,
        cost_plot_label: &str,
        param_plot_label: &str,
    ) -> Self {
        Self {
            latest_param: 1.0,
            optimizer: Some(optimizer),
            step_request: StepRequest::None,
            total_steps: 0,
            current_inner: 0,
            total_outer: 0,
            latest_cost: None,
            latest_reason: None,
            max_steps,
            cost_plot_label: cost_plot_label.to_owned(),
            param_plot_label: param_plot_label.to_owned(),
        }
    }

    /// Execute one optimizer step. Returns the StepInfo if a step was taken.
    pub fn do_step(&mut self) -> Option<StepInfo> {
        let opt = self.optimizer.as_mut()?;

        match opt.step() {
            Ok(info) => {
                self.total_steps += 1;

                self.latest_cost = Some(info.smooth_cost);

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

    /// Draw the standard buttons: Run/Stop, Step, Reset.
    /// Returns true if Reset was clicked.
    pub fn draw_buttons(&mut self, ui: &mut egui::Ui) -> bool {
        let mut reset = false;
        ui.horizontal(|ui| {
            if self.step_request == StepRequest::Run {
                if ui.button("Stop").clicked() {
                    self.step_request = StepRequest::None;
                }
            } else {
                if ui.button("Run").clicked() {
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
    pub fn draw_status(&self, ui: &mut egui::Ui, _param_label: &str) {
        ui.label(format!("total steps: {}", self.total_steps));

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
            &self.param_plot_label,
            "param",
            self.latest_param,
            Color::green(),
        );
    }

    /// Delete all plot windows.
    pub fn delete_plots(&self, sender: &Sender<Vec<Packet>>) {
        let _ = sender.send(vec![
            Packet::Plot(vec![PlotViewPacket::Delete(self.cost_plot_label.clone())]),
            Packet::Plot(vec![PlotViewPacket::Delete(self.param_plot_label.clone())]),
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
                (&*self.param_plot_label, "param"),
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
