use sophus_renderer::renderables::Color;

use crate::{
    packets::plot_view_packet::{
        ClearCondition,
        CurveTrait,
        LineType,
    },
    prelude::*,
};

/// Vector of curves
#[derive(Clone, Debug)]
pub struct CurveVec<const N: usize> {
    /// data
    pub data: VecDeque<(f64, [f64; N])>,
    /// style
    pub style: CurveVecStyle<N>,
    /// clear condition
    pub clear_cond: ClearCondition,
    /// vertical line
    pub v_line: Option<f64>,
}

/// style of CurveVec
#[derive(Copy, Clone, Debug)]
pub struct CurveVecStyle<const N: usize> {
    /// colors, one for each curve
    pub colors: [Color; N],
    /// line type
    pub line_type: LineType,
}

impl<const N: usize> CurveVec<N> {
    /// Create a new curve vector
    pub fn new(
        data: VecDeque<(f64, [f64; N])>,
        color: [Color; N],
        line_type: LineType,
        clear_cond: ClearCondition,
        v_line: Option<f64>,
    ) -> Self {
        CurveVec {
            data,
            style: CurveVecStyle {
                colors: color,
                line_type,
            },
            clear_cond,
            v_line,
        }
    }
}

impl<const N: usize> CurveTrait<[f64; N], CurveVecStyle<N>> for CurveVec<N> {
    fn mut_tuples(&mut self) -> &mut VecDeque<(f64, [f64; N])> {
        &mut self.data
    }

    fn update_vline(&mut self, v_line: Option<f64>) {
        self.v_line = v_line;
    }

    fn assign_style(&mut self, style: CurveVecStyle<N>) {
        self.style = style;
    }
}

/// Curve vector with curve name and plot name
#[derive(Clone, Debug)]
pub struct NamedCurveVec<const N: usize> {
    /// plot name
    pub plot_name: String,
    /// curve name
    pub curve_name: String,
    /// scalar curve
    pub scalar_curve: CurveVec<N>,
}
