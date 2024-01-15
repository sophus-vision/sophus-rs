pub mod actor;
pub mod offscreen;
pub mod pixel_renderer;
pub mod scene_renderer;

use eframe::egui::load::SizedTexture;
use eframe::egui::Image;
use eframe::egui::Sense;
use eframe::egui::{self};
use eframe::egui_wgpu::Renderer;
use eframe::epaint::mutex::RwLock;
use hollywood::compute::pipeline::CancelRequest;
use nalgebra::SVector;
use std::sync::Arc;

use self::actor::ViewerCamera;
use self::actor::ViewerState;
use self::offscreen::OffscreenTexture;
use self::pixel_renderer::PixelRenderer;
use self::scene_renderer::depth_renderer::DepthRenderer;
use self::scene_renderer::SceneRenderer;
use crate::lie::rotation3::Isometry3;
use crate::sensor::perspective_camera::KannalaBrandtCamera;
use crate::viewer::actor::ViewerMessage;
use crate::viewer::pixel_renderer::LineVertex2;
use crate::viewer::pixel_renderer::PointVertex2;
use crate::viewer::scene_renderer::scene_line::LineVertex3;
use crate::viewer::scene_renderer::scene_point::PointVertex3;
use crate::viewer::scene_renderer::scene_triangle::MeshVertex3;

#[derive(Clone, Debug, Default)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Clone, Debug)]
pub enum Renderable {
    Lines2(Lines2),
    Points2(Points2),
    Lines3(Lines3),
    Points3(Points3),
    Triangles3(Triangles3),
}

#[derive(Clone, Debug, Default)]
pub struct Lines2 {
    pub name: String,
    pub lines: Vec<Line2>,
}

#[derive(Clone, Debug, Default)]
pub struct Points2 {
    pub name: String,
    pub points: Vec<Point2>,
}

#[derive(Clone, Debug, Default)]
pub struct Lines3 {
    pub name: String,
    pub lines: Vec<Line3>,
}

#[derive(Clone, Debug, Default)]
pub struct Points3 {
    pub name: String,
    pub points: Vec<Point3>,
}

#[derive(Clone, Debug, Default)]
pub struct Triangles3 {
    pub name: String,
    pub mesh: Vec<Triangle3>,
}

#[derive(Clone, Debug, Default)]
pub struct Line2 {
    pub p0: SVector<f32, 2>,
    pub p1: SVector<f32, 2>,
    pub color: Color,
    pub line_width: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Line3 {
    pub p0: SVector<f32, 3>,
    pub p1: SVector<f32, 3>,
    pub color: Color,
    pub line_width: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Point2 {
    pub p: SVector<f32, 2>,
    pub color: Color,
    pub point_size: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Point3 {
    pub p: SVector<f32, 3>,
    pub color: Color,
    pub point_size: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Triangle3 {
    pub p0: SVector<f32, 3>,
    pub p1: SVector<f32, 3>,
    pub p2: SVector<f32, 3>,
    pub color: Color,
}

pub struct ViewerBuilder {
    pub camera: ViewerCamera,

    pub update_receiver: std::sync::mpsc::Receiver<ViewerMessage>,
    pub view_pose_sender: tokio::sync::mpsc::Sender<Isometry3<f64>>,
    pub cancel_request_sender: Option<tokio::sync::mpsc::Sender<CancelRequest>>,

    pub viewer_state: ViewerState,
}

impl ViewerBuilder {
    pub fn new(camera: ViewerCamera) -> Self {
        let (update_sender, update_receiver) = std::sync::mpsc::channel();
        let (view_pose_sender, view_pose_receiver) = tokio::sync::mpsc::channel(100);
        let view_pose_receiver_arc = Arc::new(view_pose_receiver);

        Self {
            camera,
            update_receiver,
            cancel_request_sender: None,
            view_pose_sender,
            viewer_state: ViewerState {
                sender: Some(update_sender),
                view_pose_receiver: Some(view_pose_receiver_arc),
            },
        }
    }
}

#[derive(Clone)]
pub struct ViewerRenderState {
    pub wgpu_state: Arc<RwLock<Renderer>>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: Arc<wgpu::Adapter>,
}

pub struct SimpleViewer {
    state: ViewerRenderState,
    offscreen: OffscreenTexture,
    cam: KannalaBrandtCamera<f64>,
    pixel: PixelRenderer,
    scene: SceneRenderer,

    receiver: std::sync::mpsc::Receiver<ViewerMessage>,
    cancel_request_sender: tokio::sync::mpsc::Sender<CancelRequest>,
}

impl SimpleViewer {
    pub fn new(builder: ViewerBuilder, render_state: &ViewerRenderState) -> Self {
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        Self {
            state: render_state.clone(),
            offscreen: OffscreenTexture::new(render_state, &builder.camera.intrinsics),
            cam: builder.camera.intrinsics,
            pixel: PixelRenderer::new(render_state, &builder, depth_stencil.clone()),
            scene: SceneRenderer::new(render_state, &builder, depth_stencil),
            receiver: builder.update_receiver,
            cancel_request_sender: builder.cancel_request_sender.unwrap(),
        }
    }

    fn add_renderables_to_tables(&mut self) {
        for msg in self.receiver.try_iter() {
            match msg {
                ViewerMessage::Packets(msg_vec) => {
                    for m in msg_vec {
                        match m {
                            Renderable::Lines2(lines) => {
                                self.pixel
                                    .line_renderer
                                    .lines_table
                                    .insert(lines.name, lines.lines);
                            }
                            Renderable::Points2(points) => {
                                self.pixel
                                    .point_renderer
                                    .points_table
                                    .insert(points.name, points.points);
                            }
                            Renderable::Lines3(lines3) => {
                                self.scene
                                    .line_renderer
                                    .line_table
                                    .insert(lines3.name, lines3.lines);
                            }
                            Renderable::Points3(points3) => {
                                self.scene
                                    .point_renderer
                                    .point_table
                                    .insert(points3.name, points3.points);
                            }
                            Renderable::Triangles3(triangles3) => {
                                self.scene
                                    .mesh_renderer
                                    .mesh_table
                                    .insert(triangles3.name, triangles3.mesh);
                            }
                        }
                    }
                }
                ViewerMessage::RequestViewPose(request) => {
                    request.reply(|_| self.scene.interaction.scene_from_camera);
                }
            }
        }
        for (_, points) in self.pixel.point_renderer.points_table.iter() {
            for point in points.iter() {
                let v = PointVertex2 {
                    _pos: [point.p[0], point.p[1]],
                    _color: [point.color.r, point.color.g, point.color.b, point.color.a],
                    _point_size: point.point_size,
                };
                for _i in 0..6 {
                    self.pixel.point_renderer.vertex_data.push(v);
                }
            }
        }
        for (_, lines) in self.pixel.line_renderer.lines_table.iter() {
            for line in lines.iter() {
                let p0 = line.p0;
                let p1 = line.p1;
                let d = (p0 - p1).normalize();
                let normal = [d[1], -d[0]];

                let v0 = LineVertex2 {
                    _pos: [p0[0], p0[1]],
                    _normal: normal,
                    _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                    _line_width: line.line_width,
                };
                let v1 = LineVertex2 {
                    _pos: [p1[0], p1[1]],
                    _normal: normal,
                    _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                    _line_width: line.line_width,
                };
                self.pixel.line_renderer.vertex_data.push(v0);
                self.pixel.line_renderer.vertex_data.push(v0);
                self.pixel.line_renderer.vertex_data.push(v1);
                self.pixel.line_renderer.vertex_data.push(v0);
                self.pixel.line_renderer.vertex_data.push(v1);
                self.pixel.line_renderer.vertex_data.push(v1);
            }
        }
        for (_, points) in self.scene.point_renderer.point_table.iter() {
            for point in points.iter() {
                let v = PointVertex3 {
                    _pos: [point.p[0], point.p[1], point.p[2]],
                    _color: [point.color.r, point.color.g, point.color.b, point.color.a],
                    _point_size: point.point_size,
                };
                for _i in 0..6 {
                    self.scene.point_renderer.vertex_data.push(v);
                }
            }
        }
        for (_, lines) in self.scene.line_renderer.line_table.iter() {
            for line in lines.iter() {
                let p0 = line.p0;
                let p1 = line.p1;

                let v0 = LineVertex3 {
                    _p0: [p0[0], p0[1], p0[2]],
                    _p1: [p1[0], p1[1], p1[2]],
                    _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                    _line_width: line.line_width,
                };
                let v1 = LineVertex3 {
                    _p0: [p0[0], p0[1], p0[2]],
                    _p1: [p1[0], p1[1], p1[2]],
                    _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                    _line_width: line.line_width,
                };
                self.scene.line_renderer.vertex_data.push(v0);
                self.scene.line_renderer.vertex_data.push(v0);
                self.scene.line_renderer.vertex_data.push(v1);
                self.scene.line_renderer.vertex_data.push(v0);
                self.scene.line_renderer.vertex_data.push(v1);
                self.scene.line_renderer.vertex_data.push(v1);
            }
        }
        for (_, mesh) in self.scene.mesh_renderer.mesh_table.iter() {
            for trig in mesh.iter() {
                let v0 = MeshVertex3 {
                    _pos: [trig.p0[0], trig.p0[1], trig.p0[2]],
                    _color: [trig.color.r, trig.color.g, trig.color.b, trig.color.a],
                };
                let v1 = MeshVertex3 {
                    _pos: [trig.p1[0], trig.p1[1], trig.p1[2]],
                    _color: [trig.color.r, trig.color.g, trig.color.b, trig.color.a],
                };
                let v2 = MeshVertex3 {
                    _pos: [trig.p2[0], trig.p2[1], trig.p2[2]],
                    _color: [trig.color.r, trig.color.g, trig.color.b, trig.color.a],
                };
                self.scene.mesh_renderer.vertices.push(v0);
                self.scene.mesh_renderer.vertices.push(v1);
                self.scene.mesh_renderer.vertices.push(v2);
            }
        }
    }
}

impl eframe::App for SimpleViewer {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.cancel_request_sender
            .try_send(CancelRequest::Cancel(()))
            .unwrap();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui_extras::install_image_loaders(ctx);

        self.scene.clear_vertex_data();
        self.pixel.clear_vertex_data();

        self.add_renderables_to_tables();

        self.pixel.prepare(&self.state);
        self.scene.prepare(&self.state, &self.cam);

        let mut command_encoder = self
            .state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.scene.paint(
            &mut command_encoder,
            &self.offscreen.rgba_texture_view,
            &self.scene.depth_renderer,
        );
        self.pixel.paint(
            &mut command_encoder,
            &self.offscreen.rgba_texture_view,
            &self.scene.depth_renderer,
        );

        self.state.queue.submit(Some(command_encoder.finish()));

        self.pixel
            .show_interaction_marker(&self.state, &self.scene.interaction.maybe_state);

        let mut command_encoder = self
            .state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.scene.depth_paint(
            &mut command_encoder,
            &self.offscreen.depth_texture_view,
            &self.scene.depth_renderer,
        );

        let depth_image = self.offscreen.download_images(
            &self.state,
            command_encoder,
            &self.cam.image_size(),
        );

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.scope(|ui| {
                let response = ui.add(
                    Image::new(SizedTexture {
                        size: egui::Vec2::new(
                            self.cam.image_size().width as f32,
                            self.cam.image_size().height as f32,
                        ),
                        id: self.offscreen.rgba_tex_id,
                    })
                    .fit_to_original_size(1.0)
                    .sense(Sense::click_and_drag()),
                );
                self.scene.process_event(&self.cam, &response, depth_image);
            });
        });

        ctx.request_repaint();
    }
}
