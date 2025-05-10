use std::sync::Arc;

use eframe::egui::{
    self,
    Slider,
};
use sophus::{
    examples::optics_sim::{
        aperture_stop::ApertureStop,
        convex_lens::BiConvexLens2,
        detector::Detector,
        element::Element,
        light_path::LightPath,
        refine_chief_ray_angle,
        scene_point::ScenePoints,
    },
    prelude::*,
    viewer::packets::Packet,
};
use sophus_autodiff::linalg::VecF64;
use sophus_geo::Circle;
use sophus_image::{
    ImageSize,
    MutImageF32,
};
use sophus_lie::Isometry3F64;
use sophus_renderer::{
    camera::{
        RenderCamera,
        RenderCameraProperties,
    },
    renderables::{
        Color,
        ImageFrame,
    },
    RenderContext,
};
use sophus_viewer::{
    packets::{
        append_to_scene_packet,
        create_scene_packet,
        ImageViewPacket,
    },
    ViewerBase,
    ViewerBaseConfig,
};
use thingbuf::mpsc::blocking::{
    channel,
    Sender,
};

pub struct OpticsElements {
    lens: Arc<BiConvexLens2<f64, 0, 0>>,
    detector: Detector,
    aperture: ApertureStop,
    scene_points: ScenePoints,
}

impl Default for OpticsElements {
    fn default() -> Self {
        Self::new()
    }
}

impl OpticsElements {
    pub fn new() -> Self {
        OpticsElements {
            detector: Detector { x: 0.5 },
            lens: Arc::new(BiConvexLens2::new(
                Circle {
                    center: VecF64::<2>::new(0.560, 0.0),
                    radius: 0.600,
                },
                Circle {
                    center: VecF64::<2>::new(-0.560, 0.0),
                    radius: 0.600,
                },
                1.5,
            )),
            scene_points: ScenePoints {
                p: [VecF64::<2>::new(-2.5, 0.1), VecF64::<2>::new(-2.0, -0.1)],
            },
            aperture: ApertureStop {
                x: 0.2,
                radius: 0.02,
            },
        }
    }
}

pub struct OpticsViewer {
    base: ViewerBase,
    elements: OpticsElements,
    message_send: Sender<Vec<Packet>>,
}

impl eframe::App for OpticsViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.base.update_data();

        egui::TopBottomPanel::bottom("bottom").show(ctx, |ui| {
            self.base.update_bottom_status_bar(ui, ctx);
        });

        egui::SidePanel::left("left").show(ctx, |ui| {
            self.base.update_left_panel(ui, ctx);

            ui.add(
                Slider::new(&mut self.elements.scene_points.p[0][0], -3.000..=0.000).text("p0.x"),
            );
            ui.add(
                Slider::new(&mut self.elements.scene_points.p[0][1], -1.000..=1.000).text("p0.y"),
            );
            ui.separator();
            ui.add(
                Slider::new(&mut self.elements.scene_points.p[1][0], -3.000..=0.000).text("p1.x"),
            );
            ui.add(
                Slider::new(&mut self.elements.scene_points.p[1][1], -1.000..=1.000).text("p1.y"),
            );
            ui.separator();
            ui.add(Slider::new(&mut self.elements.detector.x, 0.100..=2.000).text("detector.x"));
            ui.separator();
            ui.add(Slider::new(&mut self.elements.aperture.x, 0.100..=1.000).text("aperture.x"));
            ui.add(
                Slider::new(&mut self.elements.aperture.radius, 0.001..=0.25)
                    .text("aperture radius"),
            );
        });
        self.send_update();

        egui::CentralPanel::default().show(ctx, |ui| {
            self.base.update_central_panel(ui, ctx);
        });

        self.base.process_events();

        ctx.request_repaint();
    }
}

impl OpticsViewer {
    /// Create a new simple viewer
    pub fn new(render_state: RenderContext) -> Box<OpticsViewer> {
        let (message_send, message_recv) = channel(50);

        let packets = vec![create_scene_packet(
            "scene",
            RenderCamera {
                scene_from_camera: Isometry3F64::from_translation(VecF64::<3>::new(
                    -0.5, 0.0, -3.0,
                )),
                properties: RenderCameraProperties::default_from(ImageSize {
                    width: 640,
                    height: 480,
                }),
            },
            true,
        )];
        message_send.send(packets).unwrap();

        Box::new(OpticsViewer {
            base: ViewerBase::new(render_state, ViewerBaseConfig { message_recv }),
            elements: OpticsElements::new(),
            message_send,
        })
    }

    fn send_update(&mut self) {
        let mut light_path = vec![];

        let mut image = MutImageF32::from_image_size(ImageSize {
            width: 30,
            height: 50,
        });

        for i in 0..2 {
            let top_angle = refine_chief_ray_angle(
                0.0,
                self.elements.scene_points.p[i],
                self.elements.lens.clone(),
                VecF64::<2>::new(self.elements.aperture.x, self.elements.aperture.radius),
            );
            let center_angle = refine_chief_ray_angle(
                0.0,
                self.elements.scene_points.p[i],
                self.elements.lens.clone(),
                VecF64::<2>::new(self.elements.aperture.x, 0.0),
            );
            let bottom_angle = refine_chief_ray_angle(
                0.0,
                self.elements.scene_points.p[i],
                self.elements.lens.clone(),
                VecF64::<2>::new(self.elements.aperture.x, -self.elements.aperture.radius),
            );

            for angle in [
                ("top", top_angle),
                ("center", center_angle),
                ("bottom", bottom_angle),
            ] {
                light_path.push(LightPath::from(
                    format!("path_{}_{}", i, angle.0),
                    self.elements.scene_points.p[i],
                    angle.1,
                    self.elements.detector.clone(),
                    self.elements.aperture.clone(),
                    angle.0 == "center",
                    if i == 0 { Color::red() } else { Color::blue() },
                ));
            }

            for j in 0..100 {
                let angle = bottom_angle + j as f64 * 0.01 * (top_angle - bottom_angle);

                let mut path = LightPath::from(
                    format!("path_{i}_{j}"),
                    self.elements.scene_points.p[i],
                    angle,
                    self.elements.detector.clone(),
                    self.elements.aperture.clone(),
                    false,
                    if i == 0 { Color::red() } else { Color::blue() },
                );

                path.propagate(&self.elements.lens);

                if let Some(point) = path.image_point {
                    let pixel = (point * 90.0 + 25.0).round() as usize;

                    if (0..image.image_size().height).contains(&pixel) {
                        *image.mut_pixel(0, pixel) += 0.01;
                    }
                }
            }
        }

        for path in light_path.iter_mut() {
            path.propagate(&self.elements.lens);
        }

        let mut renderables3d = vec![
            self.elements.detector.to_renderable3(),
            self.elements.scene_points.to_renderable3(),
            self.elements.lens.to_renderable3(),
            self.elements.aperture.to_renderable3(),
        ];

        for path in light_path {
            renderables3d.push(path.to_renderable3());
        }

        let packets = vec![
            append_to_scene_packet("scene", renderables3d),
            Packet::Image(ImageViewPacket {
                view_label: "image".to_owned(),
                frame: Some(ImageFrame::from_image(&image.to_shared().to_rgba())),
                scene_renderables: vec![],
                pixel_renderables: vec![],
            }),
        ];

        self.message_send.send(packets).unwrap();
    }
}

fn main() {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([640.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,

        ..Default::default()
    };
    eframe::run_native(
        "Egui actor",
        options,
        Box::new(|cc| Ok(OpticsViewer::new(RenderContext::from_egui_cc(cc)))),
    )
    .unwrap();
}
