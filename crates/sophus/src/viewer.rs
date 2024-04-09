pub mod actor;
pub mod offscreen;
pub mod pixel_renderer;
pub mod renderable;
pub mod scene_renderer;

use self::actor::ViewerBuilder;
use self::offscreen::OffscreenTexture;
use self::pixel_renderer::PixelRenderer;
use self::renderable::Renderable;
use self::scene_renderer::depth_renderer::DepthRenderer;
use self::scene_renderer::textured_mesh::TexturedMeshVertex3;
use self::scene_renderer::SceneRenderer;
use crate::image::arc_image::ArcImage4U8;
use crate::image::image_view::ImageSize;
use crate::image::image_view::IsImageView;
use crate::viewer::pixel_renderer::LineVertex2;
use crate::viewer::pixel_renderer::PointVertex2;
use crate::viewer::scene_renderer::line::LineVertex3;
use crate::viewer::scene_renderer::mesh::MeshVertex3;
use crate::viewer::scene_renderer::point::PointVertex3;
use eframe::egui::load::SizedTexture;
use eframe::egui::Image;
use eframe::egui::Sense;
use eframe::egui::{self};
use eframe::egui_wgpu::Renderer;
use eframe::epaint::mutex::RwLock;
use hollywood::actors::egui::EguiAppFromBuilder;
use hollywood::actors::egui::Stream;
use hollywood::compute::pipeline::CancelRequest;
use hollywood::core::request::RequestMessage;
use sophus_core::tensor::tensor_view::IsTensorLike;
use sophus_lie::groups::isometry3::Isometry3;
use sophus_sensor::dyn_camera::DynCamera;
use std::sync::Arc;

#[derive(Clone)]
pub struct ViewerRenderState {
    pub wgpu_state: Arc<RwLock<Renderer>>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: Arc<wgpu::Adapter>,
}

pub(crate) struct BackgroundTexture {
    pub(crate) background_image_texture: wgpu::Texture,
    pub(crate) background_image_tex_id: egui::TextureId,
}

impl BackgroundTexture {
    fn new(image_size: ImageSize, render_state: ViewerRenderState) -> Self {
        let background_image_target_target =
            render_state
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: image_size.width as u32,
                        height: image_size.height as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
                });

        let background_image_texture_view =
            background_image_target_target.create_view(&wgpu::TextureViewDescriptor::default());
        let background_image_tex_id = render_state.wgpu_state.write().register_native_texture(
            render_state.device.as_ref(),
            &background_image_texture_view,
            wgpu::FilterMode::Linear,
        );

        Self {
            background_image_texture: background_image_target_target,
            background_image_tex_id,
        }
    }
}

pub struct SimpleViewer {
    state: ViewerRenderState,
    offscreen: OffscreenTexture,
    cam: DynCamera<f64, 1>,
    pixel: PixelRenderer,
    scene: SceneRenderer,
    background_image: Option<ArcImage4U8>,
    background_texture: Option<BackgroundTexture>,
    message_recv: std::sync::mpsc::Receiver<Stream<Vec<Renderable>>>,
    request_recv: std::sync::mpsc::Receiver<RequestMessage<(), Isometry3<f64, 1>>>,
    cancel_request_sender: tokio::sync::mpsc::Sender<CancelRequest>,
}

impl EguiAppFromBuilder<ViewerBuilder> for SimpleViewer {
    fn new(builder: ViewerBuilder, render_state: ViewerRenderState) -> Box<SimpleViewer> {
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        Box::new(SimpleViewer {
            state: render_state.clone(),
            offscreen: OffscreenTexture::new(&render_state, &builder.config.camera.intrinsics),
            cam: builder.config.camera.intrinsics.clone(),
            pixel: PixelRenderer::new(&render_state, &builder, depth_stencil.clone()),
            scene: SceneRenderer::new(&render_state, &builder, depth_stencil),
            message_recv: builder.message_recv,
            request_recv: builder.request_recv,
            cancel_request_sender: builder.cancel_request_sender.unwrap(),
            background_image: None,
            background_texture: None,
        })
    }

    type Out = SimpleViewer;

    type State = ViewerRenderState;
}

impl SimpleViewer {
    fn add_renderables_to_tables(&mut self) {
        for stream in self.message_recv.try_iter() {
            for m in stream.msg {
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
                    Renderable::Mesh3(mesh) => {
                        self.scene
                            .mesh_renderer
                            .mesh_table
                            .insert(mesh.name, mesh.mesh);
                    }
                    Renderable::TexturedMesh3(textured_mesh) => {
                        self.scene
                            .textured_mesh_renderer
                            .mesh_table
                            .insert(textured_mesh.name, textured_mesh.mesh);
                    }
                    Renderable::BackgroundImage(image) => {
                        self.background_image = Some(image);
                    }
                }
            }
        }
        for request in self.request_recv.try_iter() {
            request.reply(|_| self.scene.interaction.scene_from_camera);
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
        for (_, mesh) in self.scene.textured_mesh_renderer.mesh_table.iter() {
            for trig in mesh.iter() {
                let v0 = TexturedMeshVertex3 {
                    _pos: [trig.p0[0], trig.p0[1], trig.p0[2]],
                    _tex: [trig.tex0[0], trig.tex0[1]],
                };
                let v1 = TexturedMeshVertex3 {
                    _pos: [trig.p1[0], trig.p1[1], trig.p1[2]],
                    _tex: [trig.tex1[0], trig.tex1[1]],
                };
                let v2 = TexturedMeshVertex3 {
                    _pos: [trig.p2[0], trig.p2[1], trig.p2[2]],
                    _tex: [trig.tex2[0], trig.tex2[1]],
                };
                self.scene.textured_mesh_renderer.vertices.push(v0);
                self.scene.textured_mesh_renderer.vertices.push(v1);
                self.scene.textured_mesh_renderer.vertices.push(v2);
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

        if let Some(image) = self.background_image.clone() {
            self.background_texture = Some(BackgroundTexture::new(
                image.image_size(),
                self.state.clone(),
            ));
            let tex_ref = self.background_texture.as_ref().unwrap();

            self.state.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &tex_ref.background_image_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(image.tensor.scalar_view().as_slice().unwrap()),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * image.image_size().width as u32),
                    rows_per_image: None,
                },
                tex_ref.background_image_texture.size(),
            );

            self.background_image = None;
        }
        // self.state.queue.write_texture(
        //     wgpu::ImageCopyTexture {
        //         texture: &self.buffers.dist_texture,
        //         mip_level: 0,
        //         origin: wgpu::Origin3d::ZERO,
        //         aspect: wgpu::TextureAspect::All,
        //     },
        //     bytemuck::cast_slice(distort_lut.table.tensor.scalar_view().as_slice().unwrap()),
        //     wgpu::ImageDataLayout {
        //         offset: 0,
        //         bytes_per_row: Some(8 * distort_lut.table.image_size().width as u32),
        //         rows_per_image: None,
        //     },
        //     self.buffers.dist_texture.size(),
        // );

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

        let depth_image =
            self.offscreen
                .download_images(&self.state, command_encoder, &self.cam.image_size());

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.scope(|ui| {
                if let Some(image) = &self.background_texture {
                    ui.add(
                        Image::new(SizedTexture {
                            size: egui::Vec2::new(
                                image.background_image_texture.size().width as f32,
                                image.background_image_texture.size().height as f32,
                            ),
                            id: image.background_image_tex_id,
                        })
                        .fit_to_original_size(1.0),
                    );
                }

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
