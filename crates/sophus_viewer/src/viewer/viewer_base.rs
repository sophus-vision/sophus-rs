use std::collections::HashMap;

use eframe::egui;
use linked_hash_map::LinkedHashMap;
use sophus_image::arc_image::ArcImageF32;
use sophus_image::ImageSize;

use crate::renderables::Packet;
use crate::renderables::Packets;
use crate::viewer::aspect_ratio::get_adjusted_view_size;
use crate::viewer::aspect_ratio::get_max_size;
use crate::viewer::aspect_ratio::HasAspectRatio;
use crate::viewer::interactions::ViewportScale;
use crate::viewer::views::view2d::View2d;
use crate::viewer::views::view3d::View3d;
use crate::viewer::views::View;
use crate::RenderContext;

/// Viewer top-level struct.
pub struct ViewerBase {
    state: RenderContext,
    views: LinkedHashMap<String, View>,
    message_recv: std::sync::mpsc::Receiver<Packets>,
    show_depth: bool,
    backface_culling: bool,
    responses: HashMap<String, ResponseStruct>,
}

pub(crate) struct ResponseStruct {
    pub(crate) ui_response: egui::Response,
    pub(crate) z_image: ArcImageF32,
    pub(crate) scales: ViewportScale,
    pub(crate) view_port_size: ImageSize,
}

impl ViewerBase {
    /// Create a new viewer.
    pub fn new(
        render_state: RenderContext,
        message_recv: std::sync::mpsc::Receiver<Packets>,
    ) -> ViewerBase {
        ViewerBase {
            state: render_state.clone(),
            views: LinkedHashMap::new(),
            message_recv,
            show_depth: false,
            backface_culling: false,
            responses: HashMap::new(),
        }
    }

    /// Update the data.
    pub fn update_data(&mut self) {
        self.add_renderables_to_tables();
    }

    /// Process events.
    pub fn process_events(&mut self) {
        for (view_label, view) in self.views.iter_mut() {
            match view {
                View::View3d(view) => {
                    if let Some(response) = self.responses.get(view_label) {
                        view.interaction.process_event(
                            &view.intrinsics(),
                            &response.ui_response,
                            &response.scales,
                            response.view_port_size,
                            &response.z_image,
                        );
                    }
                }
                View::View2d(view) => {
                    if let Some(response) = self.responses.get(view_label) {
                        view.interaction.process_event(
                            &view.intrinsics(),
                            &response.ui_response,
                            &response.scales,
                            response.view_port_size,
                            &response.z_image,
                        );
                    }
                }
            }
        }
        self.responses.clear();
    }

    /// Update the left panel.
    pub fn update_left_panel(&mut self, ui: &mut egui::Ui) {
        for (view_label, view) in self.views.iter_mut() {
            ui.checkbox(view.enabled_mut(), view_label);
        }
        ui.separator();
        ui.checkbox(&mut self.show_depth, "show depth");
        ui.checkbox(&mut self.backface_culling, "backface culling");
        ui.separator();
    }

    /// Update the central panel.
    pub fn update_central_panel(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
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
                            let render_result = view
                                .renderer
                                .render_params(
                                    &adjusted_size.image_size(),
                                    &view.interaction.scene_from_camera(),
                                )
                                .zoom(view.interaction.zoom2d())
                                .interaction(&view.interaction)
                                .backface_culling(self.backface_culling)
                                .compute_depth_texture(self.show_depth)
                                .render();

                            let egui_texture = if self.show_depth {
                                render_result.depth_egui_tex_id
                            } else {
                                render_result.rgba_egui_tex_id
                                // render_result.rgba_egui_tex_id
                            };

                            let ui_response = ui.add(
                                egui::Image::new(egui::load::SizedTexture {
                                    size: egui::Vec2::new(
                                        adjusted_size.width,
                                        adjusted_size.height,
                                    ),
                                    id: egui_texture,
                                })
                                .fit_to_exact_size(egui::Vec2 {
                                    x: adjusted_size.width,
                                    y: adjusted_size.height,
                                })
                                .sense(egui::Sense::click_and_drag()),
                            );

                            self.responses.insert(
                                view_label.clone(),
                                ResponseStruct {
                                    ui_response,
                                    scales: ViewportScale::from_image_size_and_viewport_size(
                                        view.intrinsics().image_size(),
                                        adjusted_size,
                                    ),
                                    z_image: render_result.depth_image.ndc_z_image,
                                    view_port_size: adjusted_size.image_size(),
                                },
                            );
                        }
                        View::View2d(view) => {
                            let render_result = view
                                .renderer
                                .render_params(
                                    &adjusted_size.image_size(),
                                    &view.interaction.scene_from_camera(),
                                )
                                .zoom(view.interaction.zoom2d())
                                .interaction(&view.interaction)
                                .backface_culling(self.backface_culling)
                                .render();

                            let ui_response = ui.add(
                                egui::Image::new(egui::load::SizedTexture {
                                    size: egui::Vec2::new(
                                        adjusted_size.width,
                                        adjusted_size.height,
                                    ),
                                    id: render_result.rgba_egui_tex_id,
                                })
                                .fit_to_exact_size(egui::Vec2 {
                                    x: adjusted_size.width,
                                    y: adjusted_size.height,
                                })
                                .sense(egui::Sense::click_and_drag()),
                            );

                            self.responses.insert(
                                view_label.clone(),
                                ResponseStruct {
                                    ui_response,
                                    scales: ViewportScale::from_image_size_and_viewport_size(
                                        view.intrinsics().image_size(),
                                        adjusted_size,
                                    ),
                                    z_image: render_result.depth_image.ndc_z_image,
                                    view_port_size: adjusted_size.image_size(),
                                },
                            );
                        }
                    }
                }
            });
        });
    }

    pub(crate) fn add_renderables_to_tables(&mut self) {
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
