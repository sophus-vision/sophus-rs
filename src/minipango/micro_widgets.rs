use notan::egui::{self, emath::Numeric};
use num_traits::Num;
use std::{fmt::Display, sync::Arc};

pub trait MicroWidget {
    // add widget to egui
    fn add_to_egui(
        &mut self,
        ui: &mut egui::Ui,
        label: String,
        sender: &mut std::sync::mpsc::Sender<(String, MicroWidgetType)>,
    );

    // update widget
    fn update(&self, label: &str, widgets: &mut Self);
}

/// String representation of enum.
#[derive(Clone)]
pub struct EnumStringRepr {
    /// String representation of the current value.
    pub value: String,
    /// All possible values.
    pub values: std::vec::Vec<String>,
}

trait GuiUpdate {}

impl MicroWidget for EnumStringRepr {
    fn add_to_egui(
        &mut self,
        ui: &mut egui::Ui,
        label: String,
        sender: &mut std::sync::mpsc::Sender<(String, MicroWidgetType)>,
    ) {
        let mut selected = self.value.clone();
        egui::ComboBox::from_label(label.clone())
            .selected_text(format!("{:?}", &selected))
            .show_ui(ui, |ui| {
                for str in &self.values {
                    ui.selectable_value(&mut selected, str.to_string(), format!("{:?}", &str));
                }
            });
        if *self.value != selected {
            self.value = selected.to_string();
            sender
                .send((label, MicroWidgetType::EnumStringRepr(self.clone())))
                .unwrap();
        }
    }

    fn update(&self, _label: &str, other: &mut Self) {
        other.value = self.value.clone();
    }
}

pub trait Number: Num + Display + Numeric {
    fn from_var(var: Var<Self>) -> VarType;
    fn from_ranged_var(var: RangedVar<Self>) -> RangedVarType;
}

pub enum VarType {
    F32(Var<f32>),
    F64(Var<f64>),
    I32(Var<i32>),
    I64(Var<i64>),
}

impl Number for f32 {
    fn from_var(var: Var<Self>) -> VarType {
        VarType::F32(var)
    }
    fn from_ranged_var(var: RangedVar<Self>) -> RangedVarType {
        RangedVarType::F32(var)
    }
}
impl Number for f64 {
    fn from_var(var: Var<Self>) -> VarType {
        VarType::F64(var)
    }
    fn from_ranged_var(var: RangedVar<Self>) -> RangedVarType {
        RangedVarType::F64(var)
    }
}
impl Number for i32 {
    fn from_var(var: Var<Self>) -> VarType {
        VarType::I32(var)
    }
    fn from_ranged_var(var: RangedVar<Self>) -> RangedVarType {
        RangedVarType::I32(var)
    }
}
impl Number for i64 {
    fn from_var(var: Var<Self>) -> VarType {
        VarType::I64(var)
    }
    fn from_ranged_var(var: RangedVar<Self>) -> RangedVarType {
        RangedVarType::I64(var)
    }
}

/// Variable bool (checkbox) or numeric (read-only text box).
///
pub struct Var<T: Number> {
    /// Current value.
    pub value: T,
}

impl<T: Number> MicroWidget for Var<T> {
    fn add_to_egui(
        &mut self,
        ui: &mut egui::Ui,
        label: String,
        _sender: &mut std::sync::mpsc::Sender<(String, MicroWidgetType)>,
    ) {
        let _changed = false;
        ui.label(format!("{}: {}", label, self.value));
    }

    fn update(&self, _label: &str, other: &mut Self) {
        other.value = self.value.clone();
    }
}

/// A range value, represented as slider.
///
#[derive(Clone)]

pub struct RangedVar<T: Number> {
    /// Current value.
    pub value: T,
    /// Min, max bounds.
    pub min_max: (T, T),
}

pub enum RangedVarType {
    F32(RangedVar<f32>),
    F64(RangedVar<f64>),
    I32(RangedVar<i32>),
    I64(RangedVar<i64>),
}

/// A button.
///
/// Interfaced by [super::manager::UiButton].
pub struct Button {
    /// Is true is recently pressed.
    pub pressed: bool,
}

impl MicroWidget for Button {
    fn add_to_egui(
        &mut self,
        ui: &mut egui::Ui,
        label: String,
        _sender: &mut std::sync::mpsc::Sender<(String, MicroWidgetType)>,
    ) {
        let _ = ui.button(label);
    }

    fn update(&self, _label: &str, _other: &mut Self) {}
}

pub enum MicroWidgetType {
    EnumStringRepr(EnumStringRepr),
    Var(VarType),
    RangedVar(RangedVarType),
    Button(Button),
}

impl MicroWidgetType {}

impl<T: Number> MicroWidget for RangedVar<T> {
    fn add_to_egui(
        &mut self,
        ui: &mut egui::Ui,
        label: String,
        sender: &mut std::sync::mpsc::Sender<(String, MicroWidgetType)>,
    ) {
        if ui
            .add(
                egui::Slider::new(&mut self.value, self.min_max.0..=self.min_max.1)
                    .text(label.clone()),
            )
            .changed()
        {
            Arc::new(sender)
                .send((
                    label,
                    MicroWidgetType::RangedVar(T::from_ranged_var(self.clone())),
                ))
                .unwrap();
        }
    }

    fn update(&self, _label: &str, other: &mut Self) {
        other.value = self.value.clone();
    }
}

impl MicroWidget for MicroWidgetType {
    fn add_to_egui(
        &mut self,
        ui: &mut egui::Ui,
        label: String,
        sender: &mut std::sync::mpsc::Sender<(String, MicroWidgetType)>,
    ) {
        match self {
            MicroWidgetType::EnumStringRepr(widget) => widget.add_to_egui(ui, label, sender),
            MicroWidgetType::Var(widget) => match widget {
                VarType::F32(widget) => widget.add_to_egui(ui, label, sender),
                VarType::F64(widget) => widget.add_to_egui(ui, label, sender),
                VarType::I32(widget) => widget.add_to_egui(ui, label, sender),
                VarType::I64(widget) => widget.add_to_egui(ui, label, sender),
            },
            MicroWidgetType::RangedVar(widget) => match widget {
                RangedVarType::F32(widget) => widget.add_to_egui(ui, label, sender),
                RangedVarType::F64(widget) => widget.add_to_egui(ui, label, sender),
                RangedVarType::I32(widget) => widget.add_to_egui(ui, label, sender),
                RangedVarType::I64(widget) => widget.add_to_egui(ui, label, sender),
            },
            MicroWidgetType::Button(widget) => widget.add_to_egui(ui, label, sender),
        }
    }

    fn update(&self, label: &str, other: &mut Self) {
        match self {
            MicroWidgetType::EnumStringRepr(widget) => match other {
                MicroWidgetType::EnumStringRepr(other) => widget.update(label, other),
                _ => (),
            },
            MicroWidgetType::Var(widget) => match widget {
                VarType::F32(widget) => match other {
                    MicroWidgetType::Var(VarType::F32(other)) => widget.update(label, other),
                    _ => (),
                },
                VarType::F64(widget) => match other {
                    MicroWidgetType::Var(VarType::F64(other)) => widget.update(label, other),
                    _ => (),
                },
                VarType::I32(widget) => match other {
                    MicroWidgetType::Var(VarType::I32(other)) => widget.update(label, other),
                    _ => (),
                },
                VarType::I64(widget) => match other {
                    MicroWidgetType::Var(VarType::I64(other)) => widget.update(label, other),
                    _ => (),
                },
            },
            MicroWidgetType::RangedVar(widget) => match widget {
                RangedVarType::F32(widget) => match other {
                    MicroWidgetType::RangedVar(RangedVarType::F32(other)) => {
                        widget.update(label, other)
                    }
                    _ => (),
                },
                RangedVarType::F64(widget) => match other {
                    MicroWidgetType::RangedVar(RangedVarType::F64(other)) => {
                        widget.update(label, other)
                    }
                    _ => (),
                },
                RangedVarType::I32(widget) => match other {
                    MicroWidgetType::RangedVar(RangedVarType::I32(other)) => {
                        widget.update(label, other)
                    }
                    _ => (),
                },
                RangedVarType::I64(widget) => match other {
                    MicroWidgetType::RangedVar(RangedVarType::I64(other)) => {
                        widget.update(label, other)
                    }
                    _ => (),
                },
            },
            MicroWidgetType::Button(widget) => match other {
                MicroWidgetType::Button(other) => widget.update(label, other),
                _ => (),
            },
        }
    }
}
