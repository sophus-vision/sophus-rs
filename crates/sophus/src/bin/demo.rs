use eframe::egui;
use sophus::examples::{
    optics_sim::OpticsSimWidget,
    viewer_example::ViewerExampleWidget,
};
use sophus_renderer::RenderContext;
use sophus_viewer::{
    ViewerBase,
    ViewerBaseConfig,
    packets::Packet,
};
use thingbuf::mpsc::blocking::{
    Sender,
    channel,
};

use crate::egui::Slider;

#[derive(PartialEq)]
enum Demo {
    OpticsSim,
    Viewer,
}

enum ViewerEnum {
    Optics(OpticsSimWidget),
    Viewer(ViewerExampleWidget),
}

pub struct DemoApp {
    base: ViewerBase,
    message_send: Sender<Vec<Packet>>,
    selected_example: Demo,
    content: ViewerEnum,
}

impl eframe::App for DemoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.base.update_data();

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
    /// Create a new app of sophus examples.
    pub fn new(render_state: RenderContext) -> Box<DemoApp> {
        let (message_send, message_recv) = channel(50);

        Box::new(DemoApp {
            base: ViewerBase::new(render_state, ViewerBaseConfig { message_recv }),
            message_send: message_send.clone(),
            selected_example: Demo::OpticsSim,
            content: ViewerEnum::Optics(OpticsSimWidget::new(message_send)),
        })
    }
}

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 220.0])
            .with_icon(
                // NOTE: Adding an icon is optional
                eframe::icon_data::from_png_bytes(&include_bytes!("../../assets/icon-256.png")[..])
                    .expect("Failed to load icon"),
            ),
        ..Default::default()
    };
    eframe::run_native(
        "sophus-rs examples",
        native_options,
        Box::new(|cc| Ok(DemoApp::new(RenderContext::from_egui_cc(cc)))),
    )
}

// When compiling to web using trunk:
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Trace).ok();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");

        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("Failed to find the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("the_canvas_id was not a HtmlCanvasElement");

        if !eframe::wgpu::util::is_browser_webgpu_supported().await {
            if let Some(loading_text) = document.get_element_by_id("loading_text") {
                loading_text.set_inner_html(
                    "<p> This browser does not support WebGPU. </p> \
                    <a href=\"https://github.com/gpuweb/gpuweb/wiki/Implementation-Status\">\
                    See here for details</a>",
                );
            }
            panic!("This browser does not support WebGPU.");
        }

        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(DemoApp::new(RenderContext::from_egui_cc(cc)))),
            )
            .await;

        // Remove the loading text and spinner:
        if let Some(loading_text) = document.get_element_by_id("loading_text") {
            match start_result {
                Ok(_) => {
                    loading_text.remove();
                }
                Err(e) => {
                    loading_text.set_inner_html(
                        "<p> The app has crashed. See the developer console for details. </p>",
                    );
                    panic!("Failed to start eframe: {e:?}");
                }
            }
        }
    });
}
