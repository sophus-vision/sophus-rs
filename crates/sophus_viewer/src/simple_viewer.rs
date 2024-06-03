use std::collections::HashMap;

use eframe::egui;
use eframe::egui::load::SizedTexture;
use hollywood::actors::egui::EguiAppFromBuilder;
use hollywood::actors::egui::Stream;
use hollywood::compute::pipeline::CancelRequest;
use hollywood::RequestWithReplyChannel;
use linked_hash_map::LinkedHashMap;
use sophus_core::linalg::VecF32;
use sophus_image::arc_image::ArcImageF32;
use sophus_lie::Isometry3;
use sophus_sensor::dyn_camera::DynCamera;

use crate::actor::ViewerBuilder;
use crate::pixel_renderer::LineVertex2;
use crate::pixel_renderer::PointVertex2;
use crate::renderables::Packet;
use crate::renderables::Packets;
use crate::scene_renderer::line::LineVertex3;
use crate::scene_renderer::mesh::MeshVertex3;
use crate::scene_renderer::point::PointVertex3;
use crate::scene_renderer::textured_mesh::TexturedMeshVertex3;
use crate::views::aspect_ratio::get_adjusted_view_size;
use crate::views::aspect_ratio::get_max_size;
use crate::views::view2d::View2d;
use crate::views::view3d::View3d;
use crate::views::View;
use crate::ViewerRenderState;

/// The simple viewer top-level struct.
pub struct SimpleViewer {
    state: ViewerRenderState,
    views: LinkedHashMap<String, View>,
    message_recv: tokio::sync::mpsc::UnboundedReceiver<Stream<Packets>>,
    request_recv:
        tokio::sync::mpsc::UnboundedReceiver<RequestWithReplyChannel<String, Isometry3<f64, 1>>>,
    cancel_request_sender: tokio::sync::mpsc::UnboundedSender<CancelRequest>,
}

impl EguiAppFromBuilder<ViewerBuilder> for SimpleViewer {
    fn new(builder: ViewerBuilder, render_state: ViewerRenderState) -> Box<SimpleViewer> {
        Box::new(SimpleViewer {
            state: render_state.clone(),
            views: LinkedHashMap::new(),
            message_recv: builder.message_from_actor_recv,
            request_recv: builder.in_request_from_actor_recv,
            cancel_request_sender: builder.cancel_request_sender.unwrap(),
        })
    }

    type Out = SimpleViewer;

    type State = ViewerRenderState;
}

impl SimpleViewer {
    fn add_renderables_to_tables(&mut self) {
        loop {
            let maybe_stream = self.message_recv.try_recv();
            if maybe_stream.is_err() {
                break;
            }
            let stream = maybe_stream.unwrap();
            for packet in stream.msg.packets {
                match packet {
                    Packet::View3d(packet) => View3d::update(&mut self.views, packet, &self.state),
                    Packet::View2d(packet) => View2d::update(&mut self.views, packet, &self.state),
                }
            }
        }

        loop {
            let maybe_request = self.request_recv.try_recv();
            if maybe_request.is_err() {
                break;
            }
            let request = maybe_request.unwrap();
            let view_label = request.request.clone();
            if self.views.contains_key(&view_label) {
                let view = self.views.get(&view_label).unwrap();
                if let View::View3d(view) = view {
                    request.reply(view.scene.interaction.scene_from_camera());
                }
            }
        }

        for (_, view) in self.views.iter_mut() {
            match view {
                View::View3d(view) => {
                    for (_, points) in view.scene.point_renderer.point_table.iter() {
                        for point in points.iter() {
                            let v = PointVertex3 {
                                _pos: [point.p[0], point.p[1], point.p[2]],
                                _color: [
                                    point.color.r,
                                    point.color.g,
                                    point.color.b,
                                    point.color.a,
                                ],
                                _point_size: point.point_size,
                            };
                            for _i in 0..6 {
                                view.scene.point_renderer.vertex_data.push(v);
                            }
                        }
                    }
                    for (_, lines) in view.scene.line_renderer.line_table.iter() {
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
                            view.scene.line_renderer.vertex_data.push(v0);
                            view.scene.line_renderer.vertex_data.push(v0);
                            view.scene.line_renderer.vertex_data.push(v1);
                            view.scene.line_renderer.vertex_data.push(v0);
                            view.scene.line_renderer.vertex_data.push(v1);
                            view.scene.line_renderer.vertex_data.push(v1);
                        }
                    }
                    for (_, mesh) in view.scene.mesh_renderer.mesh_table.iter() {
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
                            view.scene.mesh_renderer.vertices.push(v0);
                            view.scene.mesh_renderer.vertices.push(v1);
                            view.scene.mesh_renderer.vertices.push(v2);
                        }
                    }
                    for (_, mesh) in view.scene.textured_mesh_renderer.mesh_table.iter() {
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
                            view.scene.textured_mesh_renderer.vertices.push(v0);
                            view.scene.textured_mesh_renderer.vertices.push(v1);
                            view.scene.textured_mesh_renderer.vertices.push(v2);
                        }
                    }
                }
                View::View2d(view) => {
                    for (_, points) in view.pixel.point_renderer.points_table.iter() {
                        for point in points.iter() {
                            let v = PointVertex2 {
                                _pos: [point.p[0], point.p[1]],
                                _color: [
                                    point.color.r,
                                    point.color.g,
                                    point.color.b,
                                    point.color.a,
                                ],
                                _point_size: point.point_size,
                            };
                            for _i in 0..6 {
                                view.pixel.point_renderer.vertex_data.push(v);
                            }
                        }
                    }
                    for (_, lines) in view.pixel.line_renderer.lines_table.iter() {
                        println!("lines: {}", lines.len());

                        for line in lines.iter() {
                            let p0 = line.p0;
                            let p1 = line.p1;
                            let d = (p0 - p1).normalize();
                            let normal = [d[1], -d[0]];

                            println!("p0: {:?}, p1: {:?}", p0, p1);
                            println!("d: {:?}, normal: {:?}", d, normal);
                            println!("color: {:?}", line.color);

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
                            view.pixel.line_renderer.vertex_data.push(v0);
                            view.pixel.line_renderer.vertex_data.push(v0);
                            view.pixel.line_renderer.vertex_data.push(v1);
                            view.pixel.line_renderer.vertex_data.push(v0);
                            view.pixel.line_renderer.vertex_data.push(v1);
                            view.pixel.line_renderer.vertex_data.push(v1);
                        }
                    }
                    for (_, points) in view.scene.point_renderer.point_table.iter() {
                        for point in points.iter() {
                            let v = PointVertex3 {
                                _pos: [point.p[0], point.p[1], point.p[2]],
                                _color: [
                                    point.color.r,
                                    point.color.g,
                                    point.color.b,
                                    point.color.a,
                                ],
                                _point_size: point.point_size,
                            };
                            for _i in 0..6 {
                                view.scene.point_renderer.vertex_data.push(v);
                            }
                        }
                    }
                    for (_, lines) in view.scene.line_renderer.line_table.iter() {
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
                            view.scene.line_renderer.vertex_data.push(v0);
                            view.scene.line_renderer.vertex_data.push(v0);
                            view.scene.line_renderer.vertex_data.push(v1);
                            view.scene.line_renderer.vertex_data.push(v0);
                            view.scene.line_renderer.vertex_data.push(v1);
                            view.scene.line_renderer.vertex_data.push(v1);
                        }
                    }
                    for (_, mesh) in view.scene.mesh_renderer.mesh_table.iter() {
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
                            view.scene.mesh_renderer.vertices.push(v0);
                            view.scene.mesh_renderer.vertices.push(v1);
                            view.scene.mesh_renderer.vertices.push(v2);
                        }
                    }
                    for (_, mesh) in view.scene.textured_mesh_renderer.mesh_table.iter() {
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
                            view.scene.textured_mesh_renderer.vertices.push(v0);
                            view.scene.textured_mesh_renderer.vertices.push(v1);
                            view.scene.textured_mesh_renderer.vertices.push(v2);
                        }
                    }
                }
            }
        }
    }
}

struct Data {
    rgba_tex_id: egui::TextureId,
    depth_image: ArcImageF32,
    intrinsics: DynCamera<f64, 1>,
}

impl eframe::App for SimpleViewer {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.cancel_request_sender.send(CancelRequest).unwrap();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui_extras::install_image_loaders(ctx);

        // Clear vertex data
        for (_, view) in self.views.iter_mut() {
            if let View::View3d(view) = view {
                view.scene.clear_vertex_data();
                //view.pixel.clear_vertex_data();
            }
            if let View::View2d(view) = view {
                view.scene.clear_vertex_data();
                view.pixel.clear_vertex_data();
            }
        }

        // Add renderables to tables
        self.add_renderables_to_tables();

        let mut view_data_map = LinkedHashMap::new();

        // Prepare views and update background textures
        for (view_label, view) in self.views.iter_mut() {
            match view {
                View::View3d(view) => {
                    view.scene.prepare(&self.state, &view.intrinsics, &None);

                    let mut command_encoder = self
                        .state
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                    view.scene.paint(
                        &mut command_encoder,
                        &view.offscreen.rgba_texture_view,
                        &view.scene.depth_renderer,
                    );

                    self.state.queue.submit(Some(command_encoder.finish()));

                    let mut command_encoder = self
                        .state
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    view.scene.depth_paint(
                        &mut command_encoder,
                        &view.offscreen.depth_texture_view,
                        &view.scene.depth_renderer,
                    );

                    let depth_image = view.offscreen.download_images(
                        &self.state,
                        command_encoder,
                        &view.intrinsics.image_size(),
                    );

                    view_data_map.insert(
                        view_label.clone(),
                        Data {
                            rgba_tex_id: view.offscreen.rgba_tex_id,
                            depth_image,
                            intrinsics: view.intrinsics.clone(),
                        },
                    );
                }
                View::View2d(view) => {
                    view.scene
                        .prepare(&self.state, &view.intrinsics, &view.background_image);
                    view.pixel.prepare(&self.state);

                    view.background_image = None;

                    let mut command_encoder = self
                        .state
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                    view.scene.paint(
                        &mut command_encoder,
                        &view.offscreen.rgba_texture_view,
                        &view.scene.depth_renderer,
                    );
                    view.pixel.paint(
                        &mut command_encoder,
                        &view.offscreen.rgba_texture_view,
                        &view.scene.depth_renderer,
                    );

                    self.state.queue.submit(Some(command_encoder.finish()));

                    view.pixel
                        .show_interaction_marker(&self.state, &view.scene.interaction);

                    let mut command_encoder = self
                        .state
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    view.scene.depth_paint(
                        &mut command_encoder,
                        &view.offscreen.depth_texture_view,
                        &view.scene.depth_renderer,
                    );

                    let depth_image = view.offscreen.download_images(
                        &self.state,
                        command_encoder,
                        &view.intrinsics.image_size(),
                    );

                    view_data_map.insert(
                        view_label.clone(),
                        Data {
                            rgba_tex_id: view.offscreen.rgba_tex_id,
                            depth_image,
                            intrinsics: view.intrinsics.clone(),
                        },
                    );
                }
            }
        }

        let mut responses = HashMap::new();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.scope(|ui0| {
                if self.views.is_empty() {
                    return;
                }
                let (max_width, max_height) = get_max_size(
                    &self.views,
                    0.95 * ui0.available_width(),
                    0.95 * ui0.available_height(),
                );

                ui0.horizontal_wrapped(|ui| {
                    for (view_label, view_data) in view_data_map.iter() {
                        let adjusted_size =
                            get_adjusted_view_size(&self.views[view_label], max_width, max_height);

                        let response = ui.add(
                            egui::Image::new(SizedTexture {
                                size: egui::Vec2::new(
                                    view_data.intrinsics.image_size().width as f32,
                                    view_data.intrinsics.image_size().height as f32,
                                ),
                                id: view_data.rgba_tex_id,
                            })
                            .fit_to_exact_size(egui::Vec2 {
                                x: adjusted_size.0,
                                y: adjusted_size.1,
                            })
                            .sense(egui::Sense::click_and_drag()),
                        );

                        responses.insert(
                            view_label.clone(),
                            (
                                response,
                                VecF32::<2>::new(
                                    view_data.intrinsics.image_size().width as f32
                                        / adjusted_size.0,
                                    view_data.intrinsics.image_size().height as f32
                                        / adjusted_size.1,
                                ),
                            ),
                        );
                    }
                });
            });
        });

        for (view_label, view) in self.views.iter_mut() {
            match view {
                View::View3d(view) => {
                    let response = responses.get(view_label).unwrap();
                    let depth_image = view_data_map.get(view_label).unwrap().depth_image.clone();

                    view.scene.process_event(
                        &view.intrinsics,
                        &response.0,
                        &response.1,
                        depth_image,
                    );
                }
                View::View2d(_) => {}
            }
        }

        ctx.request_repaint();
    }
}
