/// Render camera
pub mod camera;
/// The pixel renderer for 2D rendering.
pub mod pixel_renderer;
/// The scene renderer for 3D rendering.
pub mod scene_renderer;
/// offscreen texture for rendering
pub mod textures;
/// Types used in the renderer API
pub mod types;

use sophus_core::linalg::VecF64;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::ImageSize;
use sophus_lie::Isometry3F64;

use crate::renderables::renderable2d::Renderable2d;
use crate::renderables::renderable3d::make_textured_mesh3;
use crate::renderables::renderable3d::Renderable3d;
use crate::renderer::camera::intrinsics::RenderIntrinsics;
use crate::renderer::camera::properties::RenderCameraProperties;
use crate::renderer::pixel_renderer::pixel_line::Line2dEntity;
use crate::renderer::pixel_renderer::pixel_point::Point2dEntity;
use crate::renderer::pixel_renderer::PixelRenderer;
use crate::renderer::scene_renderer::line::Line3dEntity;
use crate::renderer::scene_renderer::mesh::Mesh3dEntity;
use crate::renderer::scene_renderer::point::Point3dEntity;
use crate::renderer::scene_renderer::textured_mesh::TexturedMeshEntity;
use crate::renderer::scene_renderer::SceneRenderer;
use crate::renderer::textures::Textures;
use crate::renderer::types::RenderResult;
use crate::renderer::types::TranslationAndScaling;
use crate::viewer::interactions::InteractionEnum;
use crate::RenderContext;

/// Offscreen renderer
pub struct OffscreenRenderer {
    pub(crate) camera_properties: RenderCameraProperties,
    state: RenderContext,
    scene: SceneRenderer,
    pixel: PixelRenderer,
    textures: Textures,
    maybe_background_image: Option<ArcImage4U8>,
}

struct RenderParams<'a> {
    view_port_size: ImageSize,
    zoom: TranslationAndScaling,
    scene_from_camera: Isometry3F64,
    maybe_interaction_enum: Option<&'a InteractionEnum>,
    compute_depth_texture: bool,
    backface_culling: bool,
    download_rgba: bool,
}

/// Render builder
pub struct RenderBuilder<'a> {
    params: RenderParams<'a>,
    offscreen_renderer: &'a mut OffscreenRenderer,
}

impl<'a> RenderBuilder<'a> {
    /// new
    pub fn new(
        view_port_size: ImageSize,
        scene_from_camera: Isometry3F64,
        offscreen_renderer: &'a mut OffscreenRenderer,
    ) -> Self {
        Self {
            params: RenderParams {
                view_port_size,
                zoom: TranslationAndScaling::identity(),
                scene_from_camera,
                maybe_interaction_enum: None,
                compute_depth_texture: false,
                backface_culling: false,
                download_rgba: false,
            },
            offscreen_renderer,
        }
    }

    /// set zoom
    pub fn zoom(mut self, zoom: TranslationAndScaling) -> Self {
        self.params.zoom = zoom;
        self
    }

    /// set interaction
    pub fn interaction(mut self, interaction_enum: &'a InteractionEnum) -> Self {
        self.params.maybe_interaction_enum = Some(interaction_enum);
        self
    }

    /// set compute depth texture
    pub fn compute_depth_texture(mut self, compute_depth_texture: bool) -> Self {
        self.params.compute_depth_texture = compute_depth_texture;
        self
    }

    /// set backface culling
    pub fn backface_culling(mut self, backface_culling: bool) -> Self {
        self.params.backface_culling = backface_culling;
        self
    }

    /// set download rgba
    pub fn download_rgba(mut self, download_rgba: bool) -> Self {
        self.params.download_rgba = download_rgba;
        self
    }

    /// render
    pub fn render(self) -> RenderResult {
        self.offscreen_renderer.render_impl(&self.params)
    }
}

impl OffscreenRenderer {
    /// background image plane
    pub const BACKGROUND_IMAGE_PLANE: f64 = 900.0;

    /// create new offscreen renderer
    pub fn new(state: &RenderContext, camera_properties: &RenderCameraProperties) -> Self {
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
        let textures = Textures::new(state, &camera_properties.intrinsics.image_size());
        Self {
            scene: SceneRenderer::new(state, camera_properties, depth_stencil.clone()),
            pixel: PixelRenderer::new(
                state,
                &camera_properties.intrinsics.image_size(),
                depth_stencil,
            ),
            textures,
            camera_properties: camera_properties.clone(),
            state: state.clone(),
            maybe_background_image: None,
        }
    }

    /// get intrinsics
    pub fn intrinsics(&self) -> RenderIntrinsics {
        self.camera_properties.intrinsics.clone()
    }

    /// reset 2d frame
    pub fn reset_2d_frame(
        &mut self,
        intrinsics: &RenderIntrinsics,
        maybe_background_image: Option<&ArcImage4U8>,
    ) {
        self.camera_properties.intrinsics = intrinsics.clone();
        if let Some(background_image) = maybe_background_image {
            let w = self.camera_properties.intrinsics.image_size().width;
            let h = self.camera_properties.intrinsics.image_size().height;

            let far = Self::BACKGROUND_IMAGE_PLANE;

            let p0 = self
                .camera_properties
                .intrinsics
                .cam_unproj_with_z(&VecF64::<2>::new(-0.5, -0.5), far)
                .cast();
            let p1 = self
                .camera_properties
                .intrinsics
                .cam_unproj_with_z(&VecF64::<2>::new(w as f64 - 0.5, -0.5), far)
                .cast();
            let p2 = self
                .camera_properties
                .intrinsics
                .cam_unproj_with_z(&VecF64::<2>::new(-0.5, h as f64 - 0.5), far)
                .cast();
            let p3 = self
                .camera_properties
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

    /// render
    pub fn render_params(
        &mut self,
        view_port_size: &ImageSize,
        world_from_camera: &Isometry3F64,
    ) -> RenderBuilder {
        RenderBuilder::new(*view_port_size, *world_from_camera, self)
    }

    fn render_impl(&mut self, state: &RenderParams) -> RenderResult {
        if self.textures.view_port_size != state.view_port_size {
            self.textures = Textures::new(&self.state, &state.view_port_size);
        }

        self.scene
            .prepare(&self.state, state.zoom, &self.camera_properties);

        self.maybe_background_image = None;
        self.pixel.prepare(
            &self.state,
            &state.view_port_size,
            &self.camera_properties.intrinsics,
            state.zoom,
        );

        let mut command_encoder = self
            .state
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // pixel currently invalidates the z-buffer

        self.scene.paint(
            &self.state,
            &state.scene_from_camera,
            &mut command_encoder,
            &self.textures.rgba.rgba_texture_view,
            &self.textures.depth,
            state.backface_culling,
        );

        let depth_image = self.textures.depth.download_depth_image(
            &self.state,
            command_encoder,
            &state.view_port_size,
            &self.camera_properties.clipping_planes,
        );
        let mut command_encoder = self
            .state
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.pixel.paint(
            &mut command_encoder,
            &self.textures.rgba.rgba_texture_view,
            &self.textures.depth,
        );

        self.state.wgpu_queue.submit(Some(command_encoder.finish()));

        if let Some(interaction_enum) = state.maybe_interaction_enum {
            self.pixel
                .show_interaction_marker(&self.state, interaction_enum);
        }

        let mut image_4u8 = None;
        if state.download_rgba {
            let command_encoder = self
                .state
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            image_4u8 = Some(self.textures.rgba.download_rgba_image(
                &self.state,
                command_encoder,
                &state.view_port_size,
            ));
        }
        if state.compute_depth_texture {
            self.textures
                .depth
                .compute_visual_depth_texture(&self.state, &depth_image);
        }

        RenderResult {
            rgba_image: image_4u8,
            rgba_egui_tex_id: self.textures.rgba.egui_tex_id,
            depth_image,
            depth_egui_tex_id: self.textures.depth.visual_depth_texture.egui_tex_id,
        }
    }
}
