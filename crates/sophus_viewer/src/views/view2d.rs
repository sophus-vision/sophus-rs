use linked_hash_map::LinkedHashMap;
use sophus_core::linalg::VecF64;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::ImageSize;
use sophus_lie::traits::IsTranslationProductGroup;
use sophus_lie::Isometry3;
use sophus_sensor::DynCamera;

use crate::interactions::inplane_interaction::InplaneInteraction;
use crate::interactions::InteractionEnum;
use crate::interactions::WgpuClippingPlanes;
use crate::offscreen::OffscreenTextures;
use crate::pixel_renderer::PixelRenderer;
use crate::renderables::renderable2d::Renderable2d;
use crate::renderables::renderable2d::View2dPacket;
use crate::renderables::renderable3d::Renderable3d;
use crate::renderables::renderable3d::TexturedMesh3;
use crate::scene_renderer::SceneRenderer;
use crate::views::aspect_ratio::HasAspectRatio;
use crate::views::View;
use crate::ViewerRenderState;

pub(crate) struct View2d {
    pub(crate) scene: SceneRenderer,
    pub(crate) pixel: PixelRenderer,
    pub(crate) intrinsics: DynCamera<f64, 1>,
    pub(crate) offscreen: Option<OffscreenTextures>,
    pub(crate) background_image: Option<ArcImage4U8>,
}

impl View2d {
    fn create_if_new(
        views: &mut LinkedHashMap<String, View>,
        packet: &View2dPacket,
        state: &ViewerRenderState,
    ) -> bool {
        if views.contains_key(&packet.view_label) {
            return false;
        }
        if let Some(frame) = &packet.frame {
            let depth_stencil = Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });
            views.insert(
                packet.view_label.clone(),
                View::View2d(View2d {
                    scene: SceneRenderer::new(
                        state,
                        frame.intrinsics(),
                        depth_stencil.clone(),
                        InteractionEnum::InplaneInteraction(InplaneInteraction {
                            maybe_pointer_state: None,
                            maybe_scroll_state: None,
                            maybe_scene_focus: None,
                            _clipping_planes: WgpuClippingPlanes {
                                near: 0.1,
                                far: 1000.0,
                            },
                            scene_from_camera: Isometry3::from_t(&VecF64::<3>::new(0.0, 0.0, -5.0)),
                        }),
                    ),
                    pixel: PixelRenderer::new(
                        state,
                        &frame.intrinsics().image_size(),
                        depth_stencil.clone(),
                    ),
                    intrinsics: frame.intrinsics().clone(),
                    background_image: frame.maybe_image().cloned(),
                    offscreen: None,
                }),
            );
            return true;
        }

        false
    }

    pub fn update(
        views: &mut LinkedHashMap<String, View>,
        packet: View2dPacket,
        state: &ViewerRenderState,
    ) {
        Self::create_if_new(views, &packet, state);
        let view = views.get_mut(&packet.view_label).unwrap();

        let view = match view {
            View::View2d(view) => view,
            _ => panic!("View type mismatch"),
        };

        if let Some(frame) = packet.frame {
            let depth_stencil = Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });

            let new_intrinsics = frame.intrinsics();

            // We got a new frame, hence we need to clear all renderables and then add the
            // intrinsics and background image if present. The easiest and most error-proof way to
            // do this is to create a new SceneRenderer and PixelRenderer and replace the old ones.
            view.pixel = PixelRenderer::new(
                state,
                &frame.intrinsics().image_size(),
                depth_stencil.clone(),
            );

            view.scene = SceneRenderer::new(
                state,
                frame.intrinsics(),
                depth_stencil,
                InteractionEnum::InplaneInteraction(InplaneInteraction {
                    maybe_pointer_state: None,
                    maybe_scroll_state: None,
                    maybe_scene_focus: None,
                    _clipping_planes: WgpuClippingPlanes {
                        near: 0.1,
                        far: 1000.0,
                    },
                    scene_from_camera: Isometry3::from_t(&VecF64::<3>::new(0.0, 0.0, 0.0)),
                }),
            );

            view.offscreen = Some(OffscreenTextures::new(state, &new_intrinsics.image_size()));

            view.intrinsics = new_intrinsics.clone();
            if let Some(background_image) = frame.maybe_image() {
                view.background_image = Some(background_image.clone());

                let w = view.intrinsics.image_size().width;
                let h = view.intrinsics.image_size().height;

                let far = 500.0;

                let p0 = view
                    .intrinsics
                    .cam_unproj_with_z(&VecF64::<2>::new(-0.5, -0.5), far)
                    .cast();
                let p1 = view
                    .intrinsics
                    .cam_unproj_with_z(&VecF64::<2>::new(w as f64 - 0.5, -0.5), far)
                    .cast();
                let p2 = view
                    .intrinsics
                    .cam_unproj_with_z(&VecF64::<2>::new(-0.5, h as f64 - 0.5), far)
                    .cast();
                let p3 = view
                    .intrinsics
                    .cam_unproj_with_z(&VecF64::<2>::new(w as f64 - 0.5, h as f64 - 0.5), far)
                    .cast();

                let tex_mesh = TexturedMesh3::make(&[
                    [(p0, [0.0, 0.0]), (p1, [1.0, 0.0]), (p2, [0.0, 1.0])],
                    [(p1, [1.0, 0.0]), (p2, [0.0, 1.0]), (p3, [1.0, 1.0])],
                ]);
                view.scene
                    .textured_mesh_renderer
                    .mesh_table
                    .insert("background".to_owned(), tex_mesh.mesh);
            }
        }

        let view = views.get_mut(&packet.view_label).unwrap();
        let view = match view {
            View::View2d(view) => view,
            _ => panic!("View type mismatch"),
        };

        for m in packet.renderables2d {
            match m {
                Renderable2d::Lines2(lines) => {
                    view.pixel
                        .line_renderer
                        .lines_table
                        .insert(lines.name, lines.lines);
                }
                Renderable2d::Points2(points) => {
                    view.pixel
                        .point_renderer
                        .points_table
                        .insert(points.name, points.points);
                }
            }
        }
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

    pub fn update_offscreen_texture(
        &mut self,
        render_state: &ViewerRenderState,
        view_port_size: &ImageSize,
    ) {
        match &mut self.offscreen {
            Some(offscreen) => {
                let current_view_port_size = offscreen.view_port_size();
                if current_view_port_size != view_port_size {
                    *offscreen = OffscreenTextures::new(render_state, view_port_size);
                }
            }
            None => {
                self.offscreen = Some(OffscreenTextures::new(render_state, view_port_size));
            }
        }
    }
}

impl HasAspectRatio for View2d {
    fn aspect_ratio(&self) -> f32 {
        self.intrinsics.image_size().aspect_ratio()
    }
}
