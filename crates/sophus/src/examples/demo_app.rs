//! Demo application with optics simulation, bundle adjustment, and 3D viewer examples.

use crossbeam_channel::{
    Sender,
    bounded,
};
use eframe::egui;
use egui::Slider;
use sophus_renderer::RenderContext;
use sophus_viewer::{
    ViewerBase,
    ViewerBaseConfig,
    packets::Packet,
};

use crate::examples::{
    bundle_adjustment::BundleAdjustmentWidget,
    circle_obstacle_demo::CircleObstacleWidget,
    corridor_demo::CorridorNavigationWidget,
    inverse_depth::InverseDepthWidget,
    optics_sim::OpticsSimWidget,
    point2d_demo::Point2dWidget,
    spline_traj_demo::SplineTrajWidget,
    viewer_example::ViewerExampleWidget,
};

#[derive(PartialEq)]
enum Demo {
    BundleAdjustment,
    ConstrainedOpt,
    OpticsSim,
    Viewer,
}

#[derive(PartialEq, Clone, Copy)]
enum BundleAdjustmentSub {
    VarIntrinsics,
    ScaleEqConstraint,
    InverseDepthCovariance,
}

#[derive(PartialEq, Clone, Copy)]
enum ConstrainedOptSub {
    ToyInequality,
    ToyEqIneq,
    CorridorIneq,
    SplineTraj,
}

enum ViewerEnum {
    BundleAdjustment(Box<BundleAdjustmentWidget>),
    InverseDepth(Box<InverseDepthWidget>),
    Point2d(Box<Point2dWidget>),
    CircleObstacle(Box<CircleObstacleWidget>),
    Corridor(Box<CorridorNavigationWidget>),
    SplineTraj(Box<SplineTrajWidget>),
    Optics(Box<OpticsSimWidget>),
    Viewer(Box<ViewerExampleWidget>),
}

/// Interactive demo app with optics simulation, bundle adjustment, and 3D viewer.
pub struct DemoApp {
    base: ViewerBase,
    message_send: Sender<Vec<Packet>>,
    selected_example: Demo,
    ba_sub: BundleAdjustmentSub,
    constrained_sub: ConstrainedOptSub,
    /// `Option` so we can drop-then-create when switching sub-tabs that share
    /// the same `ViewerEnum` variant (avoids delete-after-create packet races).
    content: Option<ViewerEnum>,
}

impl DemoApp {
    /// Create the content widget for the current BA sub-tab.
    fn make_ba_content(ba_sub: BundleAdjustmentSub, send: &Sender<Vec<Packet>>) -> ViewerEnum {
        match ba_sub {
            BundleAdjustmentSub::VarIntrinsics => ViewerEnum::BundleAdjustment(Box::new(
                BundleAdjustmentWidget::new(send.clone(), false),
            )),
            BundleAdjustmentSub::ScaleEqConstraint => ViewerEnum::BundleAdjustment(Box::new(
                BundleAdjustmentWidget::new(send.clone(), true),
            )),
            BundleAdjustmentSub::InverseDepthCovariance => {
                ViewerEnum::InverseDepth(Box::new(InverseDepthWidget::new(send.clone())))
            }
        }
    }

    /// Create the content widget for the current constrained-opt sub-tab.
    fn make_constrained_content(sub: ConstrainedOptSub, send: &Sender<Vec<Packet>>) -> ViewerEnum {
        match sub {
            ConstrainedOptSub::ToyInequality => {
                ViewerEnum::Point2d(Box::new(Point2dWidget::new(send.clone())))
            }
            ConstrainedOptSub::ToyEqIneq => {
                ViewerEnum::CircleObstacle(Box::new(CircleObstacleWidget::new(send.clone())))
            }
            ConstrainedOptSub::CorridorIneq => {
                ViewerEnum::Corridor(Box::new(CorridorNavigationWidget::new(send.clone())))
            }
            ConstrainedOptSub::SplineTraj => {
                ViewerEnum::SplineTraj(Box::new(SplineTrajWidget::new(send.clone())))
            }
        }
    }

    /// Drop old content (sends deletes), then create new content (sends creates).
    fn switch_content(&mut self, make_new: impl FnOnce(&Sender<Vec<Packet>>) -> ViewerEnum) {
        self.content = None; // drop old → sends delete packets
        self.content = Some(make_new(&self.message_send)); // create new → sends create packets
    }
}

impl eframe::App for DemoApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.base.update_data(ctx, frame);

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("sophus-rs demo");

                let examples = [
                    (Demo::BundleAdjustment, "bundle adjustment"),
                    (Demo::ConstrainedOpt, "constrained opt"),
                    (Demo::OpticsSim, "optics sim"),
                    (Demo::Viewer, "viewer"),
                ];

                for (example, label) in examples {
                    ui.selectable_value(&mut self.selected_example, example, label);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::RIGHT), |ui| {
                    ui.hyperlink("https://github.com/sophus-vision/sophus-rs/");
                });
            });

            // Sub-tabs for bundle adjustment.
            let mut sub_tab_switched = false;
            if self.selected_example == Demo::BundleAdjustment {
                ui.horizontal_wrapped(|ui| {
                    let prev = self.ba_sub;
                    ui.selectable_value(
                        &mut self.ba_sub,
                        BundleAdjustmentSub::VarIntrinsics,
                        "var intrinsics",
                    );
                    ui.selectable_value(
                        &mut self.ba_sub,
                        BundleAdjustmentSub::ScaleEqConstraint,
                        "scale eq constraint",
                    );
                    ui.selectable_value(
                        &mut self.ba_sub,
                        BundleAdjustmentSub::InverseDepthCovariance,
                        "inverse depth covariance",
                    );
                    if self.ba_sub != prev {
                        sub_tab_switched = true;
                        let ba_sub = self.ba_sub;
                        self.switch_content(|s| Self::make_ba_content(ba_sub, s));
                    }
                });
            }

            // Sub-tabs for constrained optimization.
            if self.selected_example == Demo::ConstrainedOpt {
                ui.horizontal_wrapped(|ui| {
                    let prev = self.constrained_sub;
                    ui.selectable_value(
                        &mut self.constrained_sub,
                        ConstrainedOptSub::ToyInequality,
                        "toy inequality",
                    );
                    ui.selectable_value(
                        &mut self.constrained_sub,
                        ConstrainedOptSub::ToyEqIneq,
                        "toy eq + ineq",
                    );
                    ui.selectable_value(
                        &mut self.constrained_sub,
                        ConstrainedOptSub::CorridorIneq,
                        "corridor ineq",
                    );
                    ui.selectable_value(
                        &mut self.constrained_sub,
                        ConstrainedOptSub::SplineTraj,
                        "spline trajectory",
                    );
                    if self.constrained_sub != prev {
                        sub_tab_switched = true;
                        let csub = self.constrained_sub;
                        self.switch_content(|s| Self::make_constrained_content(csub, s));
                    }
                });
            }

            // Initial content creation when switching top-level tabs.
            if !sub_tab_switched {
                let needs_switch = match (&self.selected_example, &self.content) {
                    (Demo::BundleAdjustment, Some(ViewerEnum::BundleAdjustment(_)))
                        if matches!(
                            self.ba_sub,
                            BundleAdjustmentSub::VarIntrinsics
                                | BundleAdjustmentSub::ScaleEqConstraint
                        ) =>
                    {
                        false
                    }
                    (Demo::BundleAdjustment, Some(ViewerEnum::InverseDepth(_)))
                        if self.ba_sub == BundleAdjustmentSub::InverseDepthCovariance =>
                    {
                        false
                    }
                    (Demo::ConstrainedOpt, Some(ViewerEnum::Point2d(_)))
                        if self.constrained_sub == ConstrainedOptSub::ToyInequality =>
                    {
                        false
                    }
                    (Demo::ConstrainedOpt, Some(ViewerEnum::CircleObstacle(_)))
                        if self.constrained_sub == ConstrainedOptSub::ToyEqIneq =>
                    {
                        false
                    }
                    (Demo::ConstrainedOpt, Some(ViewerEnum::Corridor(_)))
                        if self.constrained_sub == ConstrainedOptSub::CorridorIneq =>
                    {
                        false
                    }
                    (Demo::ConstrainedOpt, Some(ViewerEnum::SplineTraj(_)))
                        if self.constrained_sub == ConstrainedOptSub::SplineTraj =>
                    {
                        false
                    }
                    (Demo::OpticsSim, Some(ViewerEnum::Optics(_))) => false,
                    (Demo::Viewer, Some(ViewerEnum::Viewer(_))) => false,
                    _ => true,
                };
                if needs_switch {
                    let ba_sub = self.ba_sub;
                    let csub = self.constrained_sub;
                    let sel = &self.selected_example;
                    match sel {
                        Demo::BundleAdjustment => {
                            self.switch_content(|s| Self::make_ba_content(ba_sub, s))
                        }
                        Demo::ConstrainedOpt => {
                            self.switch_content(|s| Self::make_constrained_content(csub, s))
                        }
                        Demo::OpticsSim => self.switch_content(|s| {
                            ViewerEnum::Optics(Box::new(OpticsSimWidget::new(s.clone())))
                        }),
                        Demo::Viewer => self.switch_content(|s| {
                            ViewerEnum::Viewer(Box::new(ViewerExampleWidget::new(s.clone())))
                        }),
                    }
                }
            }

            self.base.update_top_bar(ui, ctx);
        });

        egui::SidePanel::left("left").show(ctx, |ui| {
            self.base.update_left_panel(ui, ctx);

            if let Some(content) = &mut self.content {
                match content {
                    ViewerEnum::BundleAdjustment(w) => w.update_left_panel(ui),
                    ViewerEnum::InverseDepth(w) => w.update_left_panel(ui),
                    ViewerEnum::Point2d(w) => w.update_left_panel(ui),
                    ViewerEnum::CircleObstacle(w) => w.update_left_panel(ui),
                    ViewerEnum::Corridor(w) => w.update_left_panel(ui),
                    ViewerEnum::SplineTraj(w) => w.update_left_panel(ui),
                    ViewerEnum::Optics(w) => {
                        ui.add(
                            Slider::new(&mut w.elements.scene_points.p[0][0], -3.000..=0.000)
                                .text("p0.x"),
                        );
                        ui.add(
                            Slider::new(&mut w.elements.scene_points.p[0][1], -1.000..=1.000)
                                .orientation(egui::SliderOrientation::Vertical)
                                .text("p0.y"),
                        );
                        ui.separator();
                        ui.add(
                            Slider::new(&mut w.elements.scene_points.p[1][0], -3.000..=0.000)
                                .text("p1.x"),
                        );
                        ui.add(
                            Slider::new(&mut w.elements.scene_points.p[1][1], -1.000..=1.000)
                                .orientation(egui::SliderOrientation::Vertical)
                                .text("p1.y"),
                        );
                        ui.separator();
                        ui.add(
                            Slider::new(&mut w.elements.detector.x, 0.100..=2.000)
                                .text("detector.x"),
                        );
                        ui.separator();
                        ui.add(
                            Slider::new(&mut w.elements.aperture.radius, 0.001..=0.25)
                                .text("aperture radius"),
                        );
                    }
                    ViewerEnum::Viewer(_) => {}
                }
            }
        });

        if let Some(content) = &mut self.content {
            match content {
                ViewerEnum::BundleAdjustment(w) => w.update(),
                ViewerEnum::InverseDepth(w) => w.update(),
                ViewerEnum::Point2d(w) => w.update(),
                ViewerEnum::CircleObstacle(w) => w.update(),
                ViewerEnum::Corridor(w) => w.update(),
                ViewerEnum::SplineTraj(w) => w.update(),
                ViewerEnum::Optics(w) => w.send_update(),
                ViewerEnum::Viewer(w) => w.update(),
            }
        }

        egui::TopBottomPanel::bottom("bottom").show(ctx, |ui| {
            self.base.update_bottom_status_bar(ui, ctx);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.base.update_central_panel(ui, ctx);
        });

        self.base.process_events();

        ctx.request_repaint();
    }
}

impl DemoApp {
    /// Create a new demo app.
    pub fn new(render_state: RenderContext) -> Box<DemoApp> {
        let (message_send, message_recv) = bounded(50);

        Box::new(DemoApp {
            base: ViewerBase::new(render_state, ViewerBaseConfig { message_recv }),
            message_send: message_send.clone(),
            selected_example: Demo::OpticsSim,
            ba_sub: BundleAdjustmentSub::VarIntrinsics,
            constrained_sub: ConstrainedOptSub::ToyInequality,
            content: Some(ViewerEnum::Optics(Box::new(OpticsSimWidget::new(
                message_send,
            )))),
        })
    }
}
