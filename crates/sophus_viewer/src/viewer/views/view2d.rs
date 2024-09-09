use linked_hash_map::LinkedHashMap;
use sophus_sensor::DynCamera;

use crate::renderables::renderable2d::View2dPacket;
use crate::renderer::Renderer;
use crate::viewer::aspect_ratio::HasAspectRatio;
use crate::viewer::interactions::inplane_interaction::InplaneInteraction;
use crate::viewer::interactions::InteractionEnum;
use crate::viewer::views::View;
use crate::RenderContext;

pub(crate) struct View2d {
    pub(crate) renderer: Renderer,
    pub(crate) interaction: InteractionEnum,
    pub(crate) enabled: bool,
}

impl View2d {
    fn create_if_new(
        views: &mut LinkedHashMap<String, View>,
        packet: &View2dPacket,
        state: &RenderContext,
    ) -> bool {
        if views.contains_key(&packet.view_label) {
            return false;
        }
        if let Some(frame) = &packet.frame {
            views.insert(
                packet.view_label.clone(),
                View::View2d(View2d {
                    renderer: Renderer::new(state, frame.intrinsics()),
                    interaction: InteractionEnum::InPlane(InplaneInteraction::new()),
                    enabled: true,
                }),
            );
            return true;
        }

        false
    }

    pub fn update(
        views: &mut LinkedHashMap<String, View>,
        packet: View2dPacket,
        state: &RenderContext,
    ) {
        Self::create_if_new(views, &packet, state);
        let view = views.get_mut(&packet.view_label).unwrap();

        let view = match view {
            View::View2d(view) => view,
            _ => panic!("View type mismatch"),
        };

        if let Some(frame) = packet.frame {
            let new_intrinsics = frame.intrinsics();

            // We got a new frame, hence we need to clear all renderables and then add the
            // intrinsics and background image if present. The easiest and most error-proof way to
            // do this is to create a new SceneRenderer and PixelRenderer and replace the old ones.
            view.renderer = Renderer::new(state, new_intrinsics);

            view.renderer
                .reset_2d_frame(new_intrinsics, frame.maybe_image());
        }

        let view = views.get_mut(&packet.view_label).unwrap();
        let view = match view {
            View::View2d(view) => view,
            _ => panic!("View type mismatch"),
        };

        view.renderer.update_2d_renderables(packet.renderables2d);
    }

    pub fn intrinsics(&self) -> DynCamera<f64, 1> {
        self.renderer.intrinsics()
    }
}

impl HasAspectRatio for View2d {
    fn aspect_ratio(&self) -> f32 {
        self.renderer.aspect_ratio()
    }
}
