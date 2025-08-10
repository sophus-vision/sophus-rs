use eframe::egui;
use linked_hash_map::LinkedHashMap;
use log::warn;
#[cfg(target_arch = "wasm32")]
use sophus_renderer::FinalRenderResult;
use sophus_renderer::{
    HasAspectRatio,
    OffscreenRenderer,
    RenderContext,
    camera::RenderIntrinsics,
    textures::download_depth,
};

use crate::{
    ResponseStruct,
    ViewportScale,
    interactions::{
        InteractionEnum,
        orbit_interaction::OrbitalInteraction,
    },
    layout::{
        WindowPlacement,
        show_image,
    },
    packets::{
        SceneViewCreation,
        SceneViewPacket,
        SceneViewPacketContent,
    },
    prelude::*,
    views::View,
};

/// Window parameters
pub struct WindowParams {
    pub show_depth: bool,
    pub backface_culling: bool,
    pub floating_windows: bool,
    pub show_title_bars: bool,
}

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
        ctx: &egui::Context,
        context: RenderContext,
        placement: &WindowPlacement,
        params: WindowParams,
    ) -> Option<ResponseStruct> {
        let view_port_size = placement.viewport_size();

        let render_result = self
            .renderer
            .render_params(&view_port_size, &self.interaction.scene_from_camera())
            .zoom(self.interaction.zoom2d())
            .interaction(self.interaction.marker())
            .backface_culling(params.backface_culling)
            .compute_depth_texture(params.show_depth)
            .render();
        let clipping_planes = self.renderer.camera_properties.clipping_planes.cast();

        // This is a non-wasm target, so we can block on the async function to
        // download the depth texture on the GPU to the CPU.
        let render_result = pollster::block_on(download_depth(
            params.show_depth,
            clipping_planes,
            context,
            &view_port_size,
            &render_result,
        ));

        let egui_texture = if params.show_depth {
            render_result.depth_egui_tex_id
        } else {
            render_result.rgba_egui_tex_id
        };

        let (ui_response, view_disabled) = show_image(
            ctx,
            egui_texture,
            placement,
            params.floating_windows,
            params.show_title_bars,
        );

        Some(ResponseStruct {
            ui_response,
            scales: ViewportScale::from_image_size_and_viewport_size(
                self.intrinsics().image_size(),
                placement,
            ),
            z_image: Some(render_result.depth_image.ndc_z_image.clone()),
            view_port_size,
            view_disabled,
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
        ctx: &egui::Context,
        context: RenderContext,
        placement: &WindowPlacement,
        params: WindowParams,
    ) -> Option<ResponseStruct> {
        let view_port_size = placement.viewport_size();

        if let Some(r) = self.final_render_result_promise.as_mut() {
            if let Some(final_render_result) = r.ready() {
                self.final_render_result = Some(final_render_result.clone());
                self.final_render_result_promise = None;
            }
        }

        if self.final_render_result_promise.is_none() {
            let render_result = self
                .renderer
                .render_params(&view_port_size, &self.interaction.scene_from_camera())
                .zoom(self.interaction.zoom2d())
                .interaction(self.interaction.marker())
                .backface_culling(params.backface_culling)
                .compute_depth_texture(params.show_depth)
                .render();

            // Note: The code is refactored so that async download_depth is a free function,
            //.and that it only takes in arguments which can be moved into the spawn_local async
            // block. In particular, we cannot use mutable variables below the block, due to
            // lifetime. The async block outlives the function body.
            let context = context.clone();
            let clipping_planes = self.renderer.camera_properties.clipping_planes.cast();

            self.final_render_result_promise =
                Some(poll_promise::Promise::spawn_local(async move {
                    download_depth(
                        params.show_depth,
                        clipping_planes,
                        context,
                        &view_port_size,
                        &render_result,
                    )
                    .await
                }));
        }
        if let Some(final_render_result) = self.final_render_result.as_ref() {
            let egui_texture = if params.show_depth {
                final_render_result.depth_egui_tex_id
            } else {
                final_render_result.rgba_egui_tex_id
            };

            let (ui_response, view_disabled) = show_image(
                ctx,
                egui_texture,
                placement,
                params.floating_windows,
                params.show_title_bars,
            );

            return Some(ResponseStruct {
                ui_response,
                scales: ViewportScale::from_image_size_and_viewport_size(
                    self.intrinsics().image_size(),
                    placement,
                ),
                z_image: Some(final_render_result.depth_image.ndc_z_image.clone()),
                view_port_size,
                view_disabled,
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
