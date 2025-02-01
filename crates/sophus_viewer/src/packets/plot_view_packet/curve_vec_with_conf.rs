use sophus_renderer::renderables::color::Color;

use crate::{
    packets::plot_view_packet::{
        ClearCondition,
        CurveTrait,
    },
    prelude::*,
};

/// Vector of curves with confidence intervals
#[derive(Clone, Debug)]
pub struct CurveVecWithConf<const N: usize> {
    /// data
    pub data: DataVecDeque<N>,
    /// style
    pub style: CurveVecWithConfStyle<N>,
    /// clear condition
    pub clear_cond: ClearCondition,
    /// vertical line
    pub v_line: Option<f64>,
}

/// style of CurveVecWithConf
#[derive(Copy, Clone, Debug)]
pub struct CurveVecWithConfStyle<const N: usize> {
    /// colors, one for each curve
    pub colors: [Color; N],
}

/// vec conf curve data
pub type DataVecDeque<const N: usize> = VecDeque<(f64, ([f64; N], [f64; N]))>;

impl<const N: usize> CurveVecWithConf<N> {
    /// Create a new vec curve with confidence intervals
    pub fn new(
        data: DataVecDeque<N>,
        color: [Color; N],
        clear_cond: ClearCondition,
        v_line: Option<f64>,
    ) -> Self {
        CurveVecWithConf {
            data,
            style: CurveVecWithConfStyle { colors: color },
            clear_cond,
            v_line,
        }
    }
}

impl<const N: usize> CurveTrait<([f64; N], [f64; N]), CurveVecWithConfStyle<N>>
    for CurveVecWithConf<N>
{
    fn mut_tuples(&mut self) -> &mut VecDeque<(f64, ([f64; N], [f64; N]))> {
        &mut self.data
    }

    fn update_vline(&mut self, v_line: Option<f64>) {
        self.v_line = v_line;
    }

    fn assign_style(&mut self, style: CurveVecWithConfStyle<N>) {
        self.style = style;
    }
}

/// CurveVecWithConf with plot name and curve name
#[derive(Clone, Debug)]
pub struct NamedVecConfCurve<const N: usize> {
    /// plot name
    pub plot_name: String,
    /// curve name
    pub curve_name: String,
    /// scalar curve
    pub scalar_curve: CurveVecWithConf<N>,
}
