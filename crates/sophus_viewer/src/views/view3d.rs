use linked_hash_map::LinkedHashMap;
use sophus_image::ImageSize;
use sophus_sensor::DynCamera;

use crate::interactions::orbit_interaction::OrbitalInteraction;
use crate::interactions::InteractionEnum;
use crate::offscreen::OffscreenTexture;
use crate::renderables::renderable3d::Renderable3d;
use crate::renderables::renderable3d::View3dPacket;
use crate::scene_renderer::SceneRenderer;
use crate::views::aspect_ratio::HasAspectRatio;
use crate::views::View;
use crate::ViewerRenderState;

pub(crate) struct View3d {
    pub(crate) scene: SceneRenderer,
    pub(crate) intrinsics: DynCamera<f64, 1>,
    pub(crate) offscreen: OffscreenTexture,
}

impl View3d {
    fn create_if_new(
        views: &mut LinkedHashMap<String, View>,
        packet: &View3dPacket,
        state: &ViewerRenderState,
    ) {
        if !views.contains_key(&packet.view_label) {
            let depth_stencil = Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });
            views.insert(
                packet.view_label.clone(),
                View::View3d(View3d {
                    offscreen: OffscreenTexture::new(
                        state,
                        &packet.initial_camera.intrinsics.image_size(),
                    ),
                    scene: SceneRenderer::new(
                        state,
                        &packet.initial_camera.intrinsics,
                        depth_stencil,
                        InteractionEnum::OrbitalInteraction(OrbitalInteraction {
                            maybe_pointer_state: None,
                            maybe_scroll_state: None,
                            maybe_scene_focus: None,
                            clipping_planes: packet.initial_camera.clipping_planes,
                            scene_from_camera: packet.initial_camera.scene_from_camera,
                        }),
                    ),
                    intrinsics: packet.initial_camera.intrinsics.clone(),
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
        for m in packet.renderables3d {
            match m {
                Renderable3d::Lines3(lines3) => {
                    view.scene
                        .line_renderer
                        .line_table
                        .insert(lines3.name, lines3.lines);
                }
                Renderable3d::Points3(points3) => {
                    view.scene
                        .point_renderer
                        .point_table
                        .insert(points3.name, points3.points);
                }
                Renderable3d::Mesh3(mesh) => {
                    view.scene
                        .mesh_renderer
                        .mesh_table
                        .insert(mesh.name, mesh.mesh);
                }
            }
        }
    }
}

impl HasAspectRatio for View3d {
    fn aspect_ratio(&self) -> f32 {
        self.intrinsics.image_size().aspect_ratio()
    }

    fn view_size(&self) -> ImageSize {
        self.intrinsics.image_size()
    }

    fn intrinsics(&self) -> &DynCamera<f64, 1> {
        &self.intrinsics
    }
}
