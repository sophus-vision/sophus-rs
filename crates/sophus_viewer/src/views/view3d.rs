use linked_hash_map::LinkedHashMap;
use sophus_sensor::DynCamera;

use crate::offscreen_renderer::OffscreenRenderer;
use crate::renderables::renderable3d::View3dPacket;
use crate::views::aspect_ratio::HasAspectRatio;
use crate::views::interactions::orbit_interaction::OrbitalInteraction;
use crate::views::interactions::InteractionEnum;
use crate::views::View;
use crate::ViewerRenderState;

pub(crate) struct View3d {
    pub(crate) renderer: OffscreenRenderer,
    pub(crate) interaction: InteractionEnum,
    pub(crate) enabled: bool,
}

impl View3d {
    fn create_if_new(
        views: &mut LinkedHashMap<String, View>,
        packet: &View3dPacket,
        state: &ViewerRenderState,
    ) {
        if !views.contains_key(&packet.view_label) {
            views.insert(
                packet.view_label.clone(),
                View::View3d(View3d {
                    renderer: OffscreenRenderer::new(state, &packet.initial_camera.intrinsics),
                    interaction: InteractionEnum::Orbital(OrbitalInteraction::new(
                        packet.initial_camera.scene_from_camera,
                        packet.initial_camera.clipping_planes,
                    )),
                    enabled: true,
                }),
            );
        }
    }

    pub fn update(
        views: &mut LinkedHashMap<String, View>,
        packet: View3dPacket,
        state: &ViewerRenderState,
    ) {
        Self::create_if_new(views, &packet, state);

        let view = views.get_mut(&packet.view_label).unwrap();
        let view = match view {
            View::View3d(view) => view,
            _ => panic!("View type mismatch"),
        };

        view.renderer.update_3d_renderables(packet.renderables3d);
    }

    pub fn intrinsics(&self) -> DynCamera<f64, 1> {
        self.renderer.intrinsics()
    }
}

impl HasAspectRatio for View3d {
    fn aspect_ratio(&self) -> f32 {
        self.renderer.aspect_ratio()
    }
}
