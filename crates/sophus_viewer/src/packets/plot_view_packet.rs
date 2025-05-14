// ported from https://github.com/farm-ng/farm-ng-core/tree/main/rs/plotting/src/graphs

mod curve_vec_with_conf;
mod scalar_curve;
mod vec_curve;

pub use curve_vec_with_conf::*;
pub use scalar_curve::*;
pub use vec_curve::*;

use crate::prelude::*;

/// clear condition
#[derive(Copy, Clone, Debug)]
pub struct ClearCondition {
    /// max x range
    pub max_x_range: f64,
}

/// Vertical Line
#[derive(Clone, Debug)]
pub struct VerticalLine {
    /// position
    pub x: f64,
    /// label
    pub name: String,
}

/// Curve trait
pub trait CurveTrait<DataChunk, Style> {
    /// mut tuples
    fn mut_tuples(&mut self) -> &mut VecDeque<(f64, DataChunk)>;

    /// append
    fn append_to(
        &mut self,
        mut new_tuples: VecDeque<(f64, DataChunk)>,
        style: Style,
        clear_cond: ClearCondition,
        v_line: Option<VerticalLine>,
    ) {
        self.mut_tuples().append(&mut new_tuples);

        self.drain_filter(clear_cond);

        self.assign_style(style);

        self.update_vline(v_line);
    }

    /// update vline
    fn update_vline(&mut self, v_line: Option<VerticalLine>);

    /// assign
    fn assign_style(&mut self, meta: Style);

    /// drain filter
    fn drain_filter(&mut self, pred: ClearCondition) {
        let max_x = self
            .mut_tuples()
            .iter()
            .fold(f64::MIN, |max, p| max.max(p.0));

        self.mut_tuples()
            .retain(|pair| pair.0 + pred.max_x_range > max_x);
    }
}

/// Line type
#[derive(Copy, Clone, Debug, Default)]
pub enum LineType {
    #[default]
    /// Solid line
    LineStrip,
    /// Points
    Points,
}

/// Packet to populate a scene view
#[derive(Clone, Debug)]
pub enum PlotViewPacket {
    /// a float value
    Scalar(NamedScalarCurve),
    /// a 2d vector curve
    Vec2(NamedCurveVec<2>),
    /// a 2d vector curve with confidence intervals
    Vec2Conf(NamedVecConfCurve<2>),
    /// a 3d vector curve
    Vec3(NamedCurveVec<3>),
    /// a 3d vector curve with confidence intervals
    Vec3Conf(NamedVecConfCurve<3>),
}

impl PlotViewPacket {
    /// Get the name of the plot
    pub fn name(&self) -> String {
        match self {
            PlotViewPacket::Scalar(named_scalar_curve) => named_scalar_curve.plot_name.clone(),
            PlotViewPacket::Vec2(named_vec_curve) => named_vec_curve.plot_name.clone(),
            PlotViewPacket::Vec2Conf(named_vec_conf_curve) => {
                named_vec_conf_curve.plot_name.clone()
            }
            PlotViewPacket::Vec3(named_vec_curve) => named_vec_curve.plot_name.clone(),
            PlotViewPacket::Vec3Conf(named_vec_conf_curve) => {
                named_vec_conf_curve.plot_name.clone()
            }
        }
    }
}

impl PlotViewPacket {
    /// Append data to a curve
    pub fn append_to_curve<S: Into<String>>(
        (plot, graph): (S, S),
        data: VecDeque<(f64, f64)>,
        style: ScalarCurveStyle,
        clear_cond: ClearCondition,
        v_line: Option<VerticalLine>,
    ) -> PlotViewPacket {
        let curve = NamedScalarCurve {
            plot_name: plot.into(),
            graph_name: graph.into(),
            scalar_curve: ScalarCurve {
                data,
                style,
                clear_cond,
                v_line,
            },
        };
        PlotViewPacket::Scalar(curve)
    }

    /// Append data to a 2-vector of curves
    pub fn append_to_curve_vec2<S: Into<String>>(
        (plot, graph): (S, S),
        data: VecDeque<(f64, [f64; 2])>,
        style: CurveVecStyle<2>,
        clear_cond: ClearCondition,
        v_line: Option<VerticalLine>,
    ) -> PlotViewPacket {
        let curve = NamedCurveVec {
            plot_name: plot.into(),
            curve_name: graph.into(),
            scalar_curve: CurveVec {
                data,
                style,
                clear_cond,
                v_line,
            },
        };

        PlotViewPacket::Vec2(curve)
    }

    /// Append data to a 3-vector of curves
    pub fn append_to_curve_vec3<S: Into<String>>(
        (plot, graph): (S, S),
        data: VecDeque<(f64, [f64; 3])>,
        style: CurveVecStyle<3>,
        clear_cond: ClearCondition,
        v_line: Option<VerticalLine>,
    ) -> PlotViewPacket {
        let curve = NamedCurveVec {
            plot_name: plot.into(),
            curve_name: graph.into(),
            scalar_curve: CurveVec {
                data,
                style,
                clear_cond,
                v_line,
            },
        };

        PlotViewPacket::Vec3(curve)
    }

    /// Append data to a 2-vector of curves with confidence intervals
    pub fn append_to_curve_vec2_with_conf<S: Into<String>>(
        (plot, graph): (S, S),
        data: DataVecDeque<2>,
        style: CurveVecWithConfStyle<2>,
        clear_cond: ClearCondition,
        v_line: Option<VerticalLine>,
    ) -> PlotViewPacket {
        let curve = NamedVecConfCurve {
            plot_name: plot.into(),
            curve_name: graph.into(),
            scalar_curve: CurveVecWithConf {
                data,
                style,
                clear_cond,
                v_line,
            },
        };

        PlotViewPacket::Vec2Conf(curve)
    }

    /// Append data to a 3-vector of curves with confidence intervals
    pub fn append_to_curve_vec3_with_conf<S: Into<String>>(
        (plot, graph): (S, S),
        data: DataVecDeque<3>,
        style: CurveVecWithConfStyle<3>,
        clear_cond: ClearCondition,
        v_line: Option<VerticalLine>,
    ) -> PlotViewPacket {
        let curve = NamedVecConfCurve {
            plot_name: plot.into(),
            curve_name: graph.into(),
            scalar_curve: CurveVecWithConf {
                data,
                style,
                clear_cond,
                v_line,
            },
        };

        PlotViewPacket::Vec3Conf(curve)
    }
}
