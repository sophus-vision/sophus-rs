use crate::packets::plot_view_packet::ClearCondition;
use crate::packets::plot_view_packet::CurveTrait;
use crate::packets::plot_view_packet::LineType;
use crate::prelude::*;
use sophus_renderer::renderables::color::Color;

/// Scalar curve style
#[derive(Copy, Clone, Debug)]
pub struct ScalarCurveStyle {
    /// color
    pub color: Color,
    /// line type
    pub line_type: LineType,
}

/// Scalar curve
#[derive(Clone, Debug)]
pub struct ScalarCurve {
    /// data
    pub data: VecDeque<(f64, f64)>,
    /// style
    pub style: ScalarCurveStyle,

    /// clear condition
    pub clear_cond: ClearCondition,

    /// v-line
    pub v_line: Option<f64>,
}

impl ScalarCurve {
    /// Create a new scalar curve
    pub fn new(
        data: VecDeque<(f64, f64)>,
        color: Color,
        line_type: LineType,
        clear_cond: ClearCondition,
        v_line: Option<f64>,
    ) -> Self {
        ScalarCurve {
            data,
            style: ScalarCurveStyle { color, line_type },
            clear_cond,
            v_line,
        }
    }
}

impl CurveTrait<f64, ScalarCurveStyle> for ScalarCurve {
    fn mut_tuples(&mut self) -> &mut VecDeque<(f64, f64)> {
        &mut self.data
    }

    fn update_vline(&mut self, v_line: Option<f64>) {
        self.v_line = v_line;
    }

    fn assign_style(&mut self, style: ScalarCurveStyle) {
        self.style = style;
    }
}

/// Named scalar curve
#[derive(Clone, Debug)]
pub struct NamedScalarCurve {
    /// plot name
    pub plot_name: String,
    /// graph name
    pub graph_name: String,
    /// scalar curve
    pub scalar_curve: ScalarCurve,
}
