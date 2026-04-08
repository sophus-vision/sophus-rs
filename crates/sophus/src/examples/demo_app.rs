//! Demo application with optics simulation and 3D viewer examples.

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
    optics_sim::OpticsSimWidget,
    viewer_example::ViewerExampleWidget,
};

#[derive(PartialEq)]
enum Demo {
    OpticsSim,
    Viewer,
}

enum ViewerEnum {
    Optics(OpticsSimWidget),
    Viewer(ViewerExampleWidget),
}

/// Interactive demo app with optics simulation and 3D viewer.
pub struct DemoApp {
    base: ViewerBase,
    message_send: Sender<Vec<Packet>>,
    selected_example: Demo,
    content: ViewerEnum,
}

impl eframe::App for DemoApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.base.update_data(ctx, frame);

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("sophus-rs demo");

                let examples = [(Demo::OpticsSim, "optics sim"), (Demo::Viewer, "viewer")];

                for (example, label) in examples {
                    ui.selectable_value(&mut self.selected_example, example, label);
                }

                match self.selected_example {
                    Demo::OpticsSim => match &self.content {
                        ViewerEnum::Optics(_) => {}
                        _ => {
                            self.content =
                                ViewerEnum::Optics(OpticsSimWidget::new(self.message_send.clone()));
                        }
                    },
                    Demo::Viewer => match &self.content {
                        ViewerEnum::Viewer(_) => {}
                        _ => {
                            self.content = ViewerEnum::Viewer(ViewerExampleWidget::new(
                                self.message_send.clone(),
                            ));
                        }
                    },
                };
                ui.with_layout(egui::Layout::right_to_left(egui::Align::RIGHT), |ui| {
                    ui.hyperlink("https://github.com/sophus-vision/sophus-rs/");
                });
            });
            self.base.update_top_bar(ui, ctx);
        });

        egui::SidePanel::left("left").show(ctx, |ui| {
            self.base.update_left_panel(ui, ctx);

            if let ViewerEnum::Optics(optics_viewer_content) = &mut self.content {
                ui.add(
                    Slider::new(
                        &mut optics_viewer_content.elements.scene_points.p[0][0],
                        -3.000..=0.000,
                    )
                    .text("p0.x"),
                );
                ui.add(
                    Slider::new(
                        &mut optics_viewer_content.elements.scene_points.p[0][1],
                        -1.000..=1.000,
                    )
                    .orientation(egui::SliderOrientation::Vertical)
                    .text("p0.y"),
                );
                ui.separator();
                ui.add(
                    Slider::new(
                        &mut optics_viewer_content.elements.scene_points.p[1][0],
                        -3.000..=0.000,
                    )
                    .text("p1.x"),
                );
                ui.add(
                    Slider::new(
                        &mut optics_viewer_content.elements.scene_points.p[1][1],
                        -1.000..=1.000,
                    )
                    .orientation(egui::SliderOrientation::Vertical)
                    .text("p1.y"),
                );
                ui.separator();
                ui.add(
                    Slider::new(
                        &mut optics_viewer_content.elements.detector.x,
                        0.100..=2.000,
                    )
                    .text("detector.x"),
                );
                ui.separator();
                ui.add(
                    Slider::new(
                        &mut optics_viewer_content.elements.aperture.radius,
                        0.001..=0.25,
                    )
                    .text("aperture radius"),
                );
            }
        });
        match &mut self.content {
            ViewerEnum::Optics(optics_viewer_content) => {
                optics_viewer_content.send_update();
            }
            ViewerEnum::Viewer(viewer) => {
                viewer.update();
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
            content: ViewerEnum::Optics(OpticsSimWidget::new(message_send)),
        })
    }
}
