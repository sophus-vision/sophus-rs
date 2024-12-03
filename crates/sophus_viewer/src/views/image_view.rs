use crate::interactions::inplane_interaction::InplaneInteraction;
use crate::interactions::InteractionEnum;
use crate::packets::image_view_packet::ImageViewPacket;
use crate::preludes::*;
use crate::views::View;
use sophus_renderer::aspect_ratio::HasAspectRatio;
use sophus_renderer::camera::intrinsics::RenderIntrinsics;
use sophus_renderer::offscreen_renderer::OffscreenRenderer;
use sophus_renderer::RenderContext;
use linked_hash_map::LinkedHashMap;

pub(crate) struct ImageView {
    pub(crate) renderer: OffscreenRenderer,
    pub(crate) interaction: InteractionEnum,
    pub(crate) enabled: bool,
}

impl ImageView {
    fn create_if_new(
        views: &mut LinkedHashMap<String, View>,
        packet: &ImageViewPacket,
        state: &RenderContext,
    ) -> bool {
        if views.contains_key(&packet.view_label) {
            return false;
        }
        if let Some(frame) = &packet.frame {
            views.insert(
                packet.view_label.clone(),
                View::Image(ImageView {
                    renderer: OffscreenRenderer::new(state, frame.camera_properties()),
                    interaction: InteractionEnum::InPlane(InplaneInteraction::new(
                        &packet.view_label,
                    )),
                    enabled: true,
                }),
            );
            return true;
        }

        false
    }

    pub fn update(
        views: &mut LinkedHashMap<String, View>,
        packet: ImageViewPacket,
        state: &RenderContext,
    ) {
        Self::create_if_new(views, &packet, state);
        let view = views.get_mut(&packet.view_label).unwrap();

        let view = match view {
            View::Image(view) => view,
            _ => panic!("View type mismatch"),
        };

        if let Some(frame) = packet.frame {
            let new_camera_properties = frame.camera_properties().clone();

            // We got a new frame, hence we need to clear all renderables and then add the
            // intrinsics and background image if present. The easiest and most error-proof way to
            // do this is to create a new SceneRenderer and PixelRenderer and replace the old ones.
            view.renderer = OffscreenRenderer::new(state, &new_camera_properties);

            view.renderer
                .reset_2d_frame(&new_camera_properties.intrinsics, frame.maybe_image());
        }

        let view = views.get_mut(&packet.view_label).unwrap();
        let view = match view {
            View::Image(view) => view,
            _ => panic!("View type mismatch"),
        };

        view.renderer.update_pixels(packet.pixel_renderables);
        view.renderer.update_scene(packet.scene_renderables);
    }

    pub fn intrinsics(&self) -> RenderIntrinsics {
        self.renderer.intrinsics()
    }
}

impl HasAspectRatio for ImageView {
    fn aspect_ratio(&self) -> f32 {
        self.renderer.aspect_ratio()
    }
}
