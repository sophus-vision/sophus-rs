/// offscreen texture for rendering
pub mod textures;

/// The pixel renderer for 2D rendering.
pub mod pixel_renderer;

/// The scene renderer for 3D rendering.
pub mod scene_renderer;

/// Types used in the renderer API
pub mod types;

use sophus_core::linalg::VecF64;
use sophus_core::IsTensorLike;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::prelude::IsImageView;
use sophus_image::ImageSize;
use sophus_lie::Isometry3F64;
use sophus_sensor::DynCamera;

use crate::renderables::renderable2d::Renderable2d;
use crate::renderables::renderable3d::make_textured_mesh3;
use crate::renderables::renderable3d::Renderable3d;
use crate::renderer::pixel_renderer::pixel_line::Line2dEntity;
use crate::renderer::pixel_renderer::pixel_point::Point2dEntity;
use crate::renderer::pixel_renderer::PixelRenderer;
use crate::renderer::scene_renderer::line::Line3dEntity;
use crate::renderer::scene_renderer::mesh::Mesh3dEntity;
use crate::renderer::scene_renderer::point::Point3dEntity;
use crate::renderer::scene_renderer::textured_mesh::TexturedMeshEntity;
use crate::renderer::scene_renderer::SceneRenderer;
use crate::renderer::textures::OffscreenTextures;
use crate::renderer::types::ClippingPlanesF64;
use crate::renderer::types::DepthImage;
use crate::renderer::types::RenderResult;
use crate::renderer::types::TranslationAndScaling;
use crate::viewer::interactions::InteractionEnum;
use crate::RenderContext;

/// Offscreen renderer
pub struct OffscreenRenderer {
    intrinsics: DynCamera<f64, 1>,
    pub(crate) clipping_planes: ClippingPlanesF64,
    state: RenderContext,
    scene: SceneRenderer,
    pixel: PixelRenderer,
    textures: OffscreenTextures,
    maybe_background_image: Option<ArcImage4U8>,
}

impl OffscreenRenderer {
    /// background image plane
    pub const BACKGROUND_IMAGE_PLANE: f64 = 900.0;

    /// create new offscreen renderer
    pub fn new(
        state: &RenderContext,
        intrinsics: DynCamera<f64, 1>,
        clipping_planes: ClippingPlanesF64,
    ) -> Self {
        let depth_bias_state = wgpu::DepthBiasState {
            constant: 2,      // Adjust this value as needed
            slope_scale: 1.0, // Adjust this value as needed
            clamp: 0.0,       // Adjust this value as needed
        };
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: depth_bias_state,
        });
        Self {
            scene: SceneRenderer::new(
                state,
                intrinsics.clone(),
                clipping_planes.clone(),
                depth_stencil.clone(),
            ),
            pixel: PixelRenderer::new(state, &intrinsics.image_size(), depth_stencil),
            textures: OffscreenTextures::new(state, &intrinsics.image_size()),
            intrinsics: intrinsics.clone(),
            clipping_planes: clipping_planes.clone(),
            state: state.clone(),
            maybe_background_image: None,
        }
    }

    /// get intrinsics
    pub fn intrinsics(&self) -> DynCamera<f64, 1> {
        self.intrinsics.clone()
    }

    /// reset 2d frame
    pub fn reset_2d_frame(
        &mut self,
        intrinsics: &DynCamera<f64, 1>,
        maybe_background_image: Option<&ArcImage4U8>,
    ) {
        self.intrinsics = intrinsics.clone();
        if let Some(background_image) = maybe_background_image {
            let w = self.intrinsics.image_size().width;
            let h = self.intrinsics.image_size().height;

            let far = Self::BACKGROUND_IMAGE_PLANE;

            let p0 = self
                .intrinsics
                .cam_unproj_with_z(&VecF64::<2>::new(-0.5, -0.5), far)
                .cast();
            let p1 = self
                .intrinsics
                .cam_unproj_with_z(&VecF64::<2>::new(w as f64 - 0.5, -0.5), far)
                .cast();
            let p2 = self
                .intrinsics
                .cam_unproj_with_z(&VecF64::<2>::new(-0.5, h as f64 - 0.5), far)
                .cast();
            let p3 = self
                .intrinsics
                .cam_unproj_with_z(&VecF64::<2>::new(w as f64 - 0.5, h as f64 - 0.5), far)
                .cast();

            let name = "background_image";
            let tex_mesh = make_textured_mesh3(
                name,
                &[
                    [(p0, [0.0, 0.0]), (p1, [1.0, 0.0]), (p2, [0.0, 1.0])],
                    [(p1, [1.0, 0.0]), (p2, [0.0, 1.0]), (p3, [1.0, 1.0])],
                ],
            );
            self.scene.textured_mesh_renderer.mesh_table.insert(
                name.to_owned(),
                TexturedMeshEntity::new(&self.state, &tex_mesh, background_image.clone()),
            );
        }
    }

    /// update 2d renderables
    pub fn update_2d_renderables(&mut self, renderables: Vec<Renderable2d>) {
        for m in renderables {
            match m {
                Renderable2d::Line(lines) => {
                    self.pixel
                        .line_renderer
                        .lines_table
                        .insert(lines.name.clone(), Line2dEntity::new(&self.state, &lines));
                }
                Renderable2d::Point(points) => {
                    self.pixel.point_renderer.points_table.insert(
                        points.name.clone(),
                        Point2dEntity::new(&self.state, &points),
                    );
                }
            }
        }
    }

    /// update 3d renerables
    pub fn update_3d_renderables(&mut self, renderables: Vec<Renderable3d>) {
        for m in renderables {
            match m {
                Renderable3d::Line(lines3) => {
                    self.scene
                        .line_renderer
                        .line_table
                        .insert(lines3.name.clone(), Line3dEntity::new(&self.state, &lines3));
                }
                Renderable3d::Point(points3) => {
                    self.scene.point_renderer.point_table.insert(
                        points3.name.clone(),
                        Point3dEntity::new(&self.state, &points3),
                    );
                }
                Renderable3d::Mesh3(mesh) => {
                    self.scene
                        .mesh_renderer
                        .mesh_table
                        .insert(mesh.name.clone(), Mesh3dEntity::new(&self.state, &mesh));
                }
            }
        }
    }

    fn render_impl(
        &mut self,
        view_port_size: &ImageSize,
        zoom: TranslationAndScaling,
        scene_from_camera: Isometry3F64,
        maybe_interaction_enum: Option<&InteractionEnum>,
        compute_depth_texture: bool,
        backface_culling: bool,
        download_rgba: bool,
    ) -> RenderResult {
        if self.textures.view_port_size != *view_port_size {
            self.textures = OffscreenTextures::new(&self.state, view_port_size);
        }

        self.scene.prepare(&self.state, zoom, &self.intrinsics);

        self.maybe_background_image = None;
        self.pixel
            .prepare(&self.state, view_port_size, &self.intrinsics, zoom);

        let mut command_encoder = self
            .state
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.scene.paint(
            &self.state,
            &scene_from_camera,
            &mut command_encoder,
            &self.textures.rgba.rgba_texture_view,
            &self.textures.z_buffer,
            backface_culling,
        );

        if let Some(interaction_enum) = maybe_interaction_enum {
            self.pixel
                .show_interaction_marker(&self.state, interaction_enum);
        }

        self.pixel.paint(
            &mut command_encoder,
            &self.textures.rgba.rgba_texture_view,
            &self.textures.z_buffer,
        );

        self.state.wgpu_queue.submit(Some(command_encoder.finish()));

        let mut command_encoder = self
            .state
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.scene.depth_paint(
            &self.state,
            &scene_from_camera,
            &mut command_encoder,
            &self.textures.depth.depth_texture_view_f32,
            &self.textures.z_buffer,
            backface_culling,
        );

        let depth_image = DepthImage {
            ndc_z_image: self.textures.depth.download_image(
                &self.state,
                command_encoder,
                view_port_size,
            ),
            clipping_planes: self.clipping_planes.cast(),
        };
        let mut image_4u8 = None;
        if download_rgba {
            let command_encoder = self
                .state
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            image_4u8 = Some(self.textures.rgba.download_rgba_image(
                &self.state,
                command_encoder,
                view_port_size,
            ));
        }
        if compute_depth_texture {
            let image_rgba = depth_image.color_mapped();

            self.state.wgpu_queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.textures.depth.visual_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(image_rgba.tensor.scalar_view().as_slice().unwrap()),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * image_rgba.image_size().width as u32),
                    rows_per_image: Some(image_rgba.image_size().height as u32),
                },
                self.textures.depth.visual_texture.size(),
            );
        }

        RenderResult {
            rgba_image: image_4u8,
            rgba_egui_tex_id: self.textures.rgba.egui_tex_id,
            depth_image,
            depth_egui_tex_id: self.textures.depth.egui_tex_id,
        }
    }

    /// render with interaction marker
    pub fn render_with_interaction_marker(
        &mut self,
        view_port_size: &ImageSize,
        zoom: TranslationAndScaling,
        scene_from_camera: Isometry3F64,
        interaction_enum: &InteractionEnum,
        compute_depth_texture: bool,
        backface_culling: bool,
        download_rgba: bool,
    ) -> RenderResult {
        self.render_impl(
            view_port_size,
            zoom,
            scene_from_camera,
            Some(interaction_enum),
            compute_depth_texture,
            backface_culling,
            download_rgba,
        )
    }

    /// render
    pub fn render(
        &mut self,
        view_port_size: &ImageSize,
        zoom: TranslationAndScaling,
        scene_from_camera: Isometry3F64,
        compute_depth_texture: bool,
        backface_culling: bool,
        download_rgba: bool,
    ) -> RenderResult {
        self.render_impl(
            view_port_size,
            zoom,
            scene_from_camera,
            None,
            compute_depth_texture,
            backface_culling,
            download_rgba,
        )
    }
}
