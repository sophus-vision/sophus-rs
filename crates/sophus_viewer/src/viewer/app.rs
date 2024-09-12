use std::collections::HashMap;

use eframe::egui;

use crate::viewer::aspect_ratio::get_adjusted_view_size;
use crate::viewer::aspect_ratio::get_max_size;
use crate::viewer::aspect_ratio::HasAspectRatio;
use crate::viewer::interactions::ViewportScale;
use crate::viewer::plugin::IsUiPlugin;
use crate::viewer::types::CancelRequest;
use crate::viewer::types::ResponseStruct;
use crate::viewer::views::View;
use crate::viewer::Viewer;

impl<Plugin: IsUiPlugin> eframe::App for Viewer<Plugin> {
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
            ui.separator();
            ui.checkbox(&mut self.show_depth, "show depth");
            ui.checkbox(&mut self.backface_culling, "backface culling");
            ui.separator();
            self.plugin.update_left_panel(ui, ctx);
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
                                    self.show_depth,
                                    self.backface_culling,
                                    false,
                                );

                                let egui_texture = if self.show_depth {
                                    render_result.depth_egui_tex_id
                                } else {
                                    render_result.rgba_egui_tex_id
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

                                self.plugin.process_view3d_response(
                                    view_label,
                                    &ui_response,
                                    &view.interaction.scene_from_camera(),
                                );

                                responses.insert(
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
                                let render_result = view.renderer.render_with_interaction_marker(
                                    &adjusted_size.image_size(),
                                    view.interaction.zoom2d(),
                                    view.interaction.scene_from_camera(),
                                    &view.interaction,
                                    false,
                                    self.backface_culling,
                                    false,
                                );

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

                                self.plugin
                                    .process_view2d_response(view_label, &ui_response);

                                responses.insert(
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
                            &response.z_image,
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
                            &response.z_image,
                        );
                    }
                }
            }
        }

        ctx.request_repaint();
    }
}
