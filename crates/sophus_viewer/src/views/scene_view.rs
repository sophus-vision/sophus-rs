use eframe::egui::{
    self,
    Response,
    Ui,
};
use linked_hash_map::LinkedHashMap;
use log::warn;
use sophus_renderer::{
    FinalRenderResult,
    HasAspectRatio,
    OffscreenRenderer,
    RenderContext,
    camera::RenderIntrinsics,
    textures::download_depth,
};

use crate::{
    ResponseStruct,
    ViewportScale,
    ViewportSize,
    interactions::{
        InteractionEnum,
        orbit_interaction::OrbitalInteraction,
    },
    packets::{
        SceneViewCreation,
        SceneViewPacket,
        SceneViewPacketContent,
    },
    prelude::*,
    views::View,
};

pub(crate) struct SceneView {
    pub(crate) renderer: OffscreenRenderer,
    pub(crate) interaction: InteractionEnum,
    pub(crate) enabled: bool,
    pub(crate) locked_to_birds_eye_orientation: bool,
    #[cfg(target_arch = "wasm32")]
    pub(crate) final_render_result_promise: Option<poll_promise::Promise<FinalRenderResult>>,
    #[cfg(target_arch = "wasm32")]
    pub(crate) final_render_result: Option<FinalRenderResult>,
}

fn show_image(
    show_depth: bool,
    final_render_result: &FinalRenderResult,
    adjusted_size: ViewportSize,
    ui: &mut Ui,
) -> Response {
    let egui_texture = if show_depth {
        final_render_result.depth_egui_tex_id
    } else {
        final_render_result.rgba_egui_tex_id
    };

    ui.add(
        egui::Image::new(egui::load::SizedTexture {
            size: egui::Vec2::new(adjusted_size.width, adjusted_size.height),
            id: egui_texture,
        })
        .shrink_to_fit()
        // .fit_to_exact_size(egui::Vec2 {
        //     x: adjusted_size.width,
        //     y: adjusted_size.height,
        // })
        .sense(egui::Sense::click_and_drag()),
    )
}

impl SceneView {
    fn create(
        views: &mut LinkedHashMap<String, View>,
        view_label: &str,
        creation: &SceneViewCreation,
        context: &RenderContext,
    ) {
        views.insert(
            view_label.to_string(),
            View::Scene(SceneView {
                renderer: OffscreenRenderer::new(context, &creation.initial_camera.properties),
                interaction: InteractionEnum::Orbital(OrbitalInteraction::new(
                    view_label,
                    creation.initial_camera.scene_from_camera,
                    creation.initial_camera.properties.clipping_planes,
                )),
                enabled: true,
                locked_to_birds_eye_orientation: creation.locked_to_birds_eye_orientation,
                #[cfg(target_arch = "wasm32")]
                final_render_result: None,
                #[cfg(target_arch = "wasm32")]
                final_render_result_promise: None,
            }),
        );
    }

    pub fn update(
        views: &mut LinkedHashMap<String, View>,
        packet: SceneViewPacket,
        context: &RenderContext,
    ) {
        match &packet.content {
            SceneViewPacketContent::Delete => {
                views.remove(&packet.view_label);
            }
            SceneViewPacketContent::Creation(creation) => {
                Self::create(views, &packet.view_label, creation, context);
            }
            SceneViewPacketContent::Renderables(r) => {
                if let Some(view) = views.get_mut(&packet.view_label) {
                    if let View::Scene(scene_view) = view {
                        scene_view.renderer.update_scene(r.clone());
                    } else {
                        warn!("Is not a scene-view: {}", packet.view_label);
                    }
                } else {
                    warn!("View not found: {}", packet.view_label);
                }
            }
            SceneViewPacketContent::WorldFromSceneUpdate(world_from_scene) => {
                if let Some(view) = views.get_mut(&packet.view_label) {
                    if let View::Scene(scene_view) = view {
                        scene_view.renderer.scene.world_from_scene = *world_from_scene
                    } else {
                        warn!("Is not a scene-view: {}", packet.view_label);
                    }
                } else {
                    warn!("View not found: {}", packet.view_label);
                }
            }
        }
    }

    pub fn intrinsics(&self) -> RenderIntrinsics {
        self.renderer.intrinsics()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render(
        &mut self,
        ui: &mut Ui,
        show_depth: bool,
        backface_culling: bool,
        context: RenderContext,
        adjusted_size: ViewportSize,
    ) -> Option<ResponseStruct> {
        let render_result = self
            .renderer
            .render_params(
                &adjusted_size.image_size(),
                &self.interaction.scene_from_camera(),
            )
            .zoom(self.interaction.zoom2d())
            .interaction(self.interaction.marker())
            .backface_culling(backface_culling)
            .compute_depth_texture(show_depth)
            .render();
        let clipping_planes = self.renderer.camera_properties.clipping_planes.cast();
        let view_port_size = adjusted_size.image_size();

        // This is a non-wasm target, so we can block on the async function to
        // download the depth texture on the GPU to the CPU.
        let render_result = pollster::block_on(download_depth(
            show_depth,
            clipping_planes,
            context,
            &view_port_size,
            &render_result,
        ));

        let ui_response = show_image(show_depth, &render_result, adjusted_size, ui);
        let intrinsic_size = ui_response.intrinsic_size.unwrap();

        Some(ResponseStruct {
            ui_response,
            scales: ViewportScale::from_image_size_and_viewport_size(
                self.intrinsics().image_size(),
                ViewportSize {
                    width: intrinsic_size.x,
                    height: intrinsic_size.y,
                },
            ),
            z_image: Some(render_result.depth_image.ndc_z_image.clone()),
            view_port_size: adjusted_size.image_size(),
        })
    }

    // For wasm targets, we cannot block on an async function, since this will block
    // the execution forever. Instead we need a multi-stage approach:
    //
    // 1. Render what we can synchronously and store it in "render_result".
    // 2. Use spawn_local to start the async download of the depth image.
    // 3. Save result of spawn_local in "self.final_render_result_promise". The promise won't be
    //    ready until the next time we call this function.
    //.4. If the promise is ready, we cache the data in "self.final_render_result".
    // 5. If the cache "self.final_render_result" has data, we display the result.
    //.   We might display the same result several times in the row. But typically we expect
    //.   to see an update every frame, just with one frame latency.
    #[cfg(target_arch = "wasm32")]
    pub fn render(
        &mut self,
        ui: &mut Ui,
        show_depth: bool,
        backface_culling: bool,
        context: RenderContext,
        adjusted_size: ViewportSize,
    ) -> Option<ResponseStruct> {
        if let Some(r) = self.final_render_result_promise.as_mut() {
            if let Some(final_render_result) = r.ready() {
                self.final_render_result = Some(final_render_result.clone());
                self.final_render_result_promise = None;
            }
        }

        if self.final_render_result_promise.is_none() {
            let render_result = self
                .renderer
                .render_params(
                    &adjusted_size.image_size(),
                    &self.interaction.scene_from_camera(),
                )
                .zoom(self.interaction.zoom2d())
                .interaction(self.interaction.marker())
                .backface_culling(backface_culling)
                .compute_depth_texture(show_depth)
                .render();

            // Note: The code is refactored so that async download_depth is a free function,
            //.and that it only takes in arguments which can be moved into the spawn_local async
            // block. In particular, we cannot use mutable variables below the block, due to
            // lifetime. The async block outlives the function body.
            let context = context.clone();
            let view_port_size = adjusted_size.image_size();
            let clipping_planes = self.renderer.camera_properties.clipping_planes.cast();

            self.final_render_result_promise =
                Some(poll_promise::Promise::spawn_local(async move {
                    download_depth(
                        show_depth,
                        clipping_planes,
                        context,
                        &view_port_size,
                        &render_result,
                    )
                    .await
                }));
        }
        if let Some(final_render_result) = self.final_render_result.as_ref() {
            let ui_response = show_image(show_depth, &final_render_result, adjusted_size, ui);

            return Some(ResponseStruct {
                ui_response,
                scales: ViewportScale::from_image_size_and_viewport_size(
                    self.intrinsics().image_size(),
                    adjusted_size,
                ),
                z_image: Some(final_render_result.depth_image.ndc_z_image.clone()),
                view_port_size: adjusted_size.image_size(),
            });
        }
        None
    }
}

impl HasAspectRatio for SceneView {
    fn aspect_ratio(&self) -> f32 {
        self.renderer.aspect_ratio()
    }
}
