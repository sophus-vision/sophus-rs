use crate::interactions::orbit_interaction::OrbitalInteraction;
use crate::interactions::InteractionEnum;
use crate::packets::scene_view_packet::SceneViewCreation;
use crate::packets::scene_view_packet::SceneViewPacket;
use crate::packets::scene_view_packet::SceneViewPacketContent;
use crate::preludes::*;
use crate::views::View;
use linked_hash_map::LinkedHashMap;
use log::warn;
use sophus_renderer::aspect_ratio::HasAspectRatio;
use sophus_renderer::camera::intrinsics::RenderIntrinsics;
use sophus_renderer::offscreen_renderer::OffscreenRenderer;
use sophus_renderer::RenderContext;

pub(crate) struct SceneView {
    pub(crate) renderer: OffscreenRenderer,
    pub(crate) interaction: InteractionEnum,
    pub(crate) enabled: bool,
    pub(crate) locked_to_birds_eye_orientation: bool,
}

impl SceneView {
    fn create(
        views: &mut LinkedHashMap<String, View>,
        view_label: &str,
        creation: &SceneViewCreation,
        state: &RenderContext,
    ) {
        views.insert(
            view_label.to_string(),
            View::Scene(SceneView {
                renderer: OffscreenRenderer::new(state, &creation.initial_camera.properties),
                interaction: InteractionEnum::Orbital(OrbitalInteraction::new(
                    view_label,
                    creation.initial_camera.scene_from_camera,
                    creation.initial_camera.properties.clipping_planes,
                )),
                enabled: true,
                locked_to_birds_eye_orientation: creation.locked_to_birds_eye_orientation,
            }),
        );
    }

    pub fn update(
        views: &mut LinkedHashMap<String, View>,
        packet: SceneViewPacket,
        state: &RenderContext,
    ) {
        match &packet.content {
            SceneViewPacketContent::Creation(creation) => {
                Self::create(views, &packet.view_label, creation, state);
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
}

impl HasAspectRatio for SceneView {
    fn aspect_ratio(&self) -> f32 {
        self.renderer.aspect_ratio()
    }
}
