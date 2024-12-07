// ported from https://github.com/farm-ng/farm-ng-core/blob/main/rs/plotting/src/plotter_gui/mod.rs

use crate::interactions::InteractionEnum;
use crate::packets::plot_view_packet::curve_vec_with_conf::CurveVecWithConf;
use crate::packets::plot_view_packet::scalar_curve::ScalarCurve;
use crate::packets::plot_view_packet::vec_curve::CurveVec;
use crate::packets::plot_view_packet::CurveTrait;
use crate::packets::plot_view_packet::PlotViewPacket;
use crate::preludes::*;
use crate::views::View;
use linked_hash_map::LinkedHashMap;
use sophus_renderer::aspect_ratio::HasAspectRatio;

pub(crate) struct PlotView {
    pub(crate) enabled: bool,
    pub(crate) interaction: InteractionEnum,
    pub curves: BTreeMap<String, CurveStruct>,
    pub(crate) aspect_ratio: f32,
}

/// a single curve is a collection of data points
#[derive(Clone, Debug)]
pub struct CurveStruct {
    /// curve
    pub curve: GraphType,
    /// show graph
    pub show_graph: bool,
}

/// Graph type
#[derive(Clone, Debug)]
pub enum GraphType {
    /// Scalar curve
    Scalar(ScalarCurve),
    /// 2d vector curve
    Vec2(CurveVec<2>),
    /// 3d vector curve
    Vec3(CurveVec<3>),
    /// 2d vector curve with confidence intervals
    Vec2Conf(CurveVecWithConf<2>),
    /// 3d vector curve with confidence intervals
    Vec3Conf(CurveVecWithConf<3>),
}

impl PlotView {
    fn create_if_new(views: &mut LinkedHashMap<String, View>, packet: &PlotViewPacket) {
        if !views.contains_key(&packet.name()) {
            views.insert(
                packet.name().clone(),
                View::Plot(PlotView {
                    enabled: true,
                    interaction: InteractionEnum::No,
                    curves: BTreeMap::new(),
                    aspect_ratio: 1.0,
                }),
            );
        }
    }

    pub fn update(views: &mut LinkedHashMap<String, View>, packet: PlotViewPacket) {
        Self::create_if_new(views, &packet);

        let view = views.get_mut(&packet.name()).unwrap();
        let plot = match view {
            View::Plot(view) => view,
            _ => panic!("View type mismatch"),
        };

        match packet {
            PlotViewPacket::Scalar(new_value) => {
                let curve_name = new_value.graph_name.clone();

                plot.curves
                    .entry(curve_name.clone())
                    .and_modify(|curve_struct| match &mut curve_struct.curve {
                        GraphType::Scalar(g) => {
                            g.append_to(
                                new_value.scalar_curve.data.clone(),
                                new_value.scalar_curve.style,
                                new_value.scalar_curve.clear_cond,
                                new_value.scalar_curve.v_line,
                            );
                        }
                        GraphType::Vec3(_) => {}
                        GraphType::Vec3Conf(_) => {}
                        GraphType::Vec2(_) => {}
                        GraphType::Vec2Conf(_) => {}
                    })
                    .or_insert(CurveStruct {
                        curve: GraphType::Scalar(new_value.scalar_curve.clone()),
                        show_graph: true,
                    });
            }
            PlotViewPacket::Vec2(new_value) => {
                let curve_name = new_value.curve_name.clone();

                plot.curves
                    .entry(curve_name.clone())
                    .and_modify(|curve_struct| match &mut curve_struct.curve {
                        GraphType::Scalar(_s) => {}
                        GraphType::Vec2(g) => {
                            g.append_to(
                                new_value.scalar_curve.data.clone(),
                                new_value.scalar_curve.style,
                                new_value.scalar_curve.clear_cond,
                                new_value.scalar_curve.v_line,
                            );
                        }
                        GraphType::Vec3Conf(_) => {}
                        GraphType::Vec3(_) => {}
                        GraphType::Vec2Conf(_) => {}
                    })
                    .or_insert(CurveStruct {
                        curve: GraphType::Vec2(new_value.scalar_curve.clone()),
                        show_graph: true,
                    });
            }
            PlotViewPacket::Vec3(new_value) => {
                let curve_name = new_value.curve_name.clone();

                plot.curves
                    .entry(curve_name.clone())
                    .and_modify(|curve_struct| match &mut curve_struct.curve {
                        GraphType::Scalar(_s) => {}
                        GraphType::Vec3(g) => {
                            g.append_to(
                                new_value.scalar_curve.data.clone(),
                                new_value.scalar_curve.style,
                                new_value.scalar_curve.clear_cond,
                                new_value.scalar_curve.v_line,
                            );
                        }
                        GraphType::Vec3Conf(_) => {}
                        GraphType::Vec2(_) => {}
                        GraphType::Vec2Conf(_) => {}
                    })
                    .or_insert(CurveStruct {
                        curve: GraphType::Vec3(new_value.scalar_curve.clone()),
                        show_graph: true,
                    });
            }
            PlotViewPacket::Vec2Conf(new_value) => {
                let curve_name = new_value.curve_name.clone();

                plot.curves
                    .entry(curve_name.clone())
                    .and_modify(|curve_struct| match &mut curve_struct.curve {
                        GraphType::Scalar(_s) => {}
                        GraphType::Vec3(_) => {}
                        GraphType::Vec2Conf(g) => {
                            g.append_to(
                                new_value.scalar_curve.data.clone(),
                                new_value.scalar_curve.style,
                                new_value.scalar_curve.clear_cond,
                                new_value.scalar_curve.v_line,
                            );
                        }
                        GraphType::Vec2(_) => {}
                        GraphType::Vec3Conf(_) => {}
                    })
                    .or_insert(CurveStruct {
                        curve: GraphType::Vec2Conf(new_value.scalar_curve.clone()),
                        show_graph: true,
                    });
            }
            PlotViewPacket::Vec3Conf(new_value) => {
                let curve_name = new_value.curve_name.clone();

                plot.curves
                    .entry(curve_name.clone())
                    .and_modify(|curve_struct| match &mut curve_struct.curve {
                        GraphType::Scalar(_s) => {}
                        GraphType::Vec3(_) => {}
                        GraphType::Vec3Conf(g) => {
                            g.append_to(
                                new_value.scalar_curve.data.clone(),
                                new_value.scalar_curve.style,
                                new_value.scalar_curve.clear_cond,
                                new_value.scalar_curve.v_line,
                            );
                        }
                        GraphType::Vec2(_) => {}
                        GraphType::Vec2Conf(_) => {}
                    })
                    .or_insert(CurveStruct {
                        curve: GraphType::Vec3Conf(new_value.scalar_curve.clone()),
                        show_graph: true,
                    });
            }
        }
    }
}

impl HasAspectRatio for PlotView {
    fn aspect_ratio(&self) -> f32 {
        self.aspect_ratio
    }
}
