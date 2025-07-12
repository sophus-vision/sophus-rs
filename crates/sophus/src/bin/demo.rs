use eframe::egui::{
    self,
    Image,
    Pos2,
    Resize,
    Sense,
};
use sophus::examples::{
    optics_sim::OpticsSimWidget,
    viewer_example::ViewerExampleWidget,
};
use sophus_image::ImageSize;
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

#[derive(Clone, Copy, Debug)]
struct BBox {
    aspect: f32,
}

impl BBox {
    fn size(self, height: f32) -> egui::Vec2 {
        egui::Vec2::new(self.aspect * height, height)
    }

    fn height(boxes: &Vec<BBox>, total_width: f32) -> f32 {
        let mut w_sum = 0.0;
        for b in boxes {
            w_sum += b.aspect;
        }
        let factor = total_width / w_sum;
        factor
    }

    fn get(boxes: &Vec<BBox>, h: f32, clip: egui::Rect) -> Option<Vec<egui::Rect>> {
        let mut x_offset = clip.min.x;

        let mut y_offset = clip.min.y;

        let mut res = vec![];

        for b in boxes {
            let size = b.size(h);

            let w = size.x;

            if x_offset + w > clip.max.x {
                x_offset = clip.min.x;
                y_offset += h;
            }

            if y_offset + h > clip.max.y || x_offset + w > clip.max.x {
                return None;
            }

            res.push(egui::Rect::from_min_size(
                egui::Pos2::new(x_offset, y_offset),
                egui::Vec2::new(w, h),
            ));

            x_offset += w;
        }

        Some(res)
    }
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
    h: f32,
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

                ui.add(egui::Slider::new(&mut self.h, 0.0..=1000.0).text("My value"));
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
            egui_extras::install_image_loaders(ctx);

            let boxes = vec![
                ImageSize::new(150, 50),
                ImageSize::new(15, 60),
                ImageSize::new(150, 50),
                ImageSize::new(100, 30),
                ImageSize::new(150, 60),
                ImageSize::new(150, 50),
                ImageSize::new(100, 30),
            ];

            let boxes: Vec<BBox> = boxes
                .iter()
                .map(|b| BBox {
                    aspect: b.width as f32 / b.height as f32,
                })
                .collect();

            // let max_h = boxes
            //     .iter()
            //     .max_by(|x, y| x.height.cmp(&y.height))
            //     .unwrap()
            //     .height as f32;
            let num = boxes.len();

            // let w: Vec<f32> = boxes
            //     .iter()
            //     .map(|x| x.width as f32 / x.height as f32 * max_h)
            //     .collect();

            // let sum: f32 = w.iter().sum();

            println!("{:?}", boxes);

            let rect = ui.clip_rect();
            let width = rect.width();

            //  let factor = width / sum;

            let total_width = rect.size().x;

            let mut mi = 0.9 * BBox::height(&boxes, total_width);
            let mut ma = rect.size().y;
            let mut h = mi + ma / 2.0;

            println!("mi:{}", mi);

            assert!(BBox::get(&boxes, mi, rect).is_some());
            assert!(BBox::get(&boxes, ma, rect).is_none());

            for i in 0..10 {
                h = mi + (ma - mi) * 0.5;

                let bb = BBox::get(&boxes, h, rect);

                println!("h = {}", h);

                if bb.is_some() {
                    mi = h;
                } else {
                    ma = h;
                }
            }
            let bb = BBox::get(&boxes, mi, rect).unwrap();

            for i in 0..num {
                println!("{}", ui.clip_rect());

                let bb = bb[i];

                let r = egui::Window::new(format!("w{i}"))
                    //.resizable(true)
                    .movable(false)
                    .title_bar(false)
                    .collapsible(false)
                    .fixed_pos(bb.min)
                    .fixed_size(egui::Vec2::new(bb.size().x - 14.0, bb.size().y - 14.0))
                    .show(ctx, |ui| {
                        ui.add(
                            egui::Image::new(egui::include_image!("../../assets/ferris.png"))
                                .corner_radius(5)
                                .shrink_to_fit()
                                .maintain_aspect_ratio(false),
                        )
                    });
            }

            // println!(
            //     "{}",
            //     r.as_ref()
            //         .unwrap()
            //         .inner
            //         .as_ref()
            //         .unwrap()
            //         .intrinsic_size
            //         .unwrap()
            // );
            // let rect = r.unwrap().response.
            // println!("{}", rect);

            // self.base.update_central_panel(ui, ctx);

            // ui.push_id("inner", |ui| {
            //     let r = Resize::default().default_height(100.0).show(ui, |ui| {
            //         let sz = ui.available_size();

            //         ui.add(
            //             egui::Image::new(egui::include_image!("../../assets/ferris.png"))
            //                 .corner_radius(5)
            //                 .shrink_to_fit()
            //                 .maintain_aspect_ratio(true),
            //         )
            //     });
            // });
            // let r = Resize::default().default_height(100.0).show(ui, |ui| {
            //     ui.add(
            //         egui::Image::new(egui::include_image!("../../assets/ferris.png"))
            //             .corner_radius(5),
            //     )
            // });
            // let r = Resize::default().default_height(100.0).show(ui, |ui| {
            //     ui.add(
            //         egui::Image::new(egui::include_image!("../../assets/ferris.png"))
            //             .corner_radius(5),
            //     )
            // });

            // if r.1.interact_pointer_pos.is_some() {
            //     println!("{}", r.1.intrinsic_size.unwrap());
            // }
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
            h: 100.0,
        })
    }
}

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([640.0, 480.0])
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
