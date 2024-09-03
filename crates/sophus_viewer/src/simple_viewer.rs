use std::collections::HashMap;

use eframe::egui;

use linked_hash_map::LinkedHashMap;
use sophus_core::linalg::VecF64;
use sophus_image::arc_image::ArcImageF32;
use sophus_image::ImageSize;
use sophus_lie::traits::IsTranslationProductGroup;
use sophus_lie::Isometry3;
use sophus_sensor::DynCamera;

use crate::offscreen_renderer::renderer::ClippingPlanes;
use crate::renderables::Packet;
use crate::renderables::Packets;
use crate::views::aspect_ratio::get_adjusted_view_size;
use crate::views::aspect_ratio::get_max_size;
use crate::views::aspect_ratio::HasAspectRatio;
use crate::views::interactions::ViewportScale;
use crate::views::view2d::View2d;
use crate::views::view3d::View3d;
use crate::views::View;
use crate::ViewerRenderState;

/// Cancel request.
pub struct CancelRequest {}

/// Viewer camera configuration.
#[derive(Clone, Debug)]
pub struct ViewerCamera {
    /// Camera intrinsics
    pub intrinsics: DynCamera<f64, 1>,
    /// Clipping planes
    pub clipping_planes: ClippingPlanes,
    /// Scene from camera pose
    pub scene_from_camera: Isometry3<f64, 1>,
}

impl Default for ViewerCamera {
    fn default() -> Self {
        ViewerCamera::default_from(ImageSize::new(639, 479))
    }
}

impl ViewerCamera {
    /// Create default viewer camera from image size
    pub fn default_from(image_size: ImageSize) -> ViewerCamera {
        ViewerCamera {
            intrinsics: DynCamera::default_pinhole(image_size),
            clipping_planes: ClippingPlanes::default(),
            scene_from_camera: Isometry3::from_t(&VecF64::<3>::new(0.0, 0.0, -5.0)),
        }
    }
}

/// The simple viewer top-level struct.
pub struct SimpleViewer {
    state: ViewerRenderState,
    views: LinkedHashMap<String, View>,
    message_recv: tokio::sync::mpsc::UnboundedReceiver<Packets>,
    cancel_request_sender: tokio::sync::mpsc::UnboundedSender<CancelRequest>,
}

/// Simple viewer builder.
pub struct SimplerViewerBuilder {
    /// Message receiver.
    pub message_recv: tokio::sync::mpsc::UnboundedReceiver<Packets>,
    /// Cancel request sender.
    pub cancel_request_sender: tokio::sync::mpsc::UnboundedSender<CancelRequest>,
}

impl SimpleViewer {
    /// Create a new simple viewer.
    pub fn new(
        builder: SimplerViewerBuilder,
        render_state: ViewerRenderState,
    ) -> Box<SimpleViewer> {
        Box::new(SimpleViewer {
            state: render_state.clone(),
            views: LinkedHashMap::new(),
            message_recv: builder.message_recv,
            cancel_request_sender: builder.cancel_request_sender,
        })
    }

    fn add_renderables_to_tables(&mut self) {
        loop {
            let maybe_stream = self.message_recv.try_recv();
            if maybe_stream.is_err() {
                break;
            }
            let stream = maybe_stream.unwrap();
            for packet in stream.packets {
                match packet {
                    Packet::View3d(packet) => View3d::update(&mut self.views, packet, &self.state),
                    Packet::View2d(packet) => View2d::update(&mut self.views, packet, &self.state),
                }
            }
        }
    }
}

struct ResponseStruct {
    ui_response: egui::Response,
    depth_image: ArcImageF32,
    scales: ViewportScale,
    view_port_size: ImageSize,
}

impl eframe::App for SimpleViewer {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.cancel_request_sender.send(CancelRequest {}).unwrap();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui_extras::install_image_loaders(ctx);

        // Add renderables to tables
        self.add_renderables_to_tables();

        let mut responses = HashMap::new();

        egui::SidePanel::left("left").show(ctx, |ui| {
            for (view_label, view) in self.views.iter_mut() {
                ui.checkbox(view.enabled_mut(), view_label);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.scope(|ui0| {
                if self.views.is_empty() {
                    return;
                }
                let maybe_max_size = get_max_size(
                    &self.views,
                    0.99 * ui0.available_width(),
                    0.99 * ui0.available_height(),
                );
                if maybe_max_size.is_none() {
                    return;
                }
                let (max_width, max_height) = maybe_max_size.unwrap();

                ui0.horizontal_wrapped(|ui| {
                    for (view_label, view) in self.views.iter_mut() {
                        if !view.enabled() {
                            continue;
                        }

                        let view_aspect_ratio = view.aspect_ratio();
                        let adjusted_size =
                            get_adjusted_view_size(view_aspect_ratio, max_width, max_height);
                        match view {
                            View::View3d(view) => {
                                let render_result = view.renderer.render_with_interaction_marker(
                                    &adjusted_size.image_size(),
                                    view.interaction.zoom2d(),
                                    view.interaction.scene_from_camera(),
                                    &view.interaction,
                                );

                                let ui_response = ui.add(
                                    egui::Image::new(egui::load::SizedTexture {
                                        size: egui::Vec2::new(
                                            adjusted_size.width,
                                            adjusted_size.height,
                                        ),
                                        id: render_result.rgba_tex_id,
                                    })
                                    .fit_to_exact_size(egui::Vec2 {
                                        x: adjusted_size.width,
                                        y: adjusted_size.height,
                                    })
                                    .sense(egui::Sense::click_and_drag()),
                                );

                                responses.insert(
                                    view_label.clone(),
                                    ResponseStruct {
                                        ui_response,
                                        scales: ViewportScale::from_image_size_and_viewport_size(
                                            view.intrinsics().image_size(),
                                            adjusted_size,
                                        ),
                                        depth_image: render_result.depth,
                                        view_port_size: adjusted_size.image_size(),
                                    },
                                );
                            }
                            View::View2d(view) => {
                                let render_result = view.renderer.render_with_interaction_marker(
                                    &adjusted_size.image_size(),
                                    view.interaction.zoom2d(),
                                    view.interaction.scene_from_camera(),
                                    &view.interaction,
                                );

                                let ui_response = ui.add(
                                    egui::Image::new(egui::load::SizedTexture {
                                        size: egui::Vec2::new(
                                            adjusted_size.width,
                                            adjusted_size.height,
                                        ),
                                        id: render_result.rgba_tex_id,
                                    })
                                    .fit_to_exact_size(egui::Vec2 {
                                        x: adjusted_size.width,
                                        y: adjusted_size.height,
                                    })
                                    .sense(egui::Sense::click_and_drag()),
                                );

                                responses.insert(
                                    view_label.clone(),
                                    ResponseStruct {
                                        ui_response,
                                        scales: ViewportScale::from_image_size_and_viewport_size(
                                            view.intrinsics().image_size(),
                                            adjusted_size,
                                        ),
                                        depth_image: render_result.depth,
                                        view_port_size: adjusted_size.image_size(),
                                    },
                                );
                            }
                        }
                    }
                });
            });
        });

        for (view_label, view) in self.views.iter_mut() {
            match view {
                View::View3d(view) => {
                    if let Some(response) = responses.get(view_label) {
                        view.interaction.process_event(
                            &view.intrinsics(),
                            &response.ui_response,
                            &response.scales,
                            response.view_port_size,
                            &response.depth_image,
                        );
                    }
                }
                View::View2d(view) => {
                    if let Some(response) = responses.get(view_label) {
                        view.interaction.process_event(
                            &view.intrinsics(),
                            &response.ui_response,
                            &response.scales,
                            response.view_port_size,
                            &response.depth_image,
                        );
                    }
                }
            }
        }

        ctx.request_repaint();
    }
}
