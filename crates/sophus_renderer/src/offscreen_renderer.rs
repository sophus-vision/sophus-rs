use crate::camera::intrinsics::RenderIntrinsics;
use crate::camera::properties::RenderCameraProperties;
use crate::pixel_renderer::pixel_line::Line2dEntity;
use crate::pixel_renderer::pixel_point::Point2dEntity;
use crate::pixel_renderer::PixelRenderer;
use crate::prelude::*;
use crate::renderables::pixel_renderable::PixelRenderable;
use crate::renderables::scene_renderable::SceneRenderable;
use crate::scene_renderer::distortion::DistortionRenderer;
use crate::scene_renderer::line::Line3dEntity;
use crate::scene_renderer::mesh::Mesh3dEntity;
use crate::scene_renderer::point::Point3dEntity;
use crate::scene_renderer::SceneRenderer;
use crate::textures::Textures;
use crate::types::RenderResult;
use crate::types::SceneFocusMarker;
use crate::types::TranslationAndScaling;
use crate::uniform_buffers::VertexShaderUniformBuffers;
use crate::RenderContext;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::image_view::IsImageView;
use sophus_image::ImageSize;
use sophus_lie::Isometry3F64;

/// Offscreen renderer
pub struct OffscreenRenderer {
    pub(crate) camera_properties: RenderCameraProperties,
    render_context: RenderContext,
    /// Scene renderer
    pub scene: SceneRenderer,
    distortion: DistortionRenderer,
    pixel: PixelRenderer,
    textures: Textures,
    maybe_background_image: Option<wgpu::Texture>,
    uniforms: Arc<VertexShaderUniformBuffers>,
}

struct RenderParams {
    view_port_size: ImageSize,
    zoom: TranslationAndScaling,
    scene_from_camera: Isometry3F64,
    maybe_marker: Option<SceneFocusMarker>,
    compute_depth_texture: bool,
    backface_culling: bool,
    download_rgba: bool,
}

/// Render builder
pub struct RenderBuilder<'a> {
    params: RenderParams,
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
                maybe_marker: None,
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
    pub fn interaction(mut self, marker: Option<SceneFocusMarker>) -> Self {
        self.params.maybe_marker = marker;
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
    pub fn new(render_context: &RenderContext, camera_properties: &RenderCameraProperties) -> Self {
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
        let textures = Textures::new(render_context, &camera_properties.intrinsics.image_size());

        let uniforms = Arc::new(VertexShaderUniformBuffers::new(
            render_context,
            camera_properties,
        ));

        Self {
            scene: SceneRenderer::new(render_context, depth_stencil.clone(), uniforms.clone()),
            distortion: DistortionRenderer::new(render_context, uniforms.clone()),
            pixel: PixelRenderer::new(render_context, uniforms.clone()),
            textures,
            camera_properties: camera_properties.clone(),
            render_context: render_context.clone(),
            maybe_background_image: None,
            uniforms,
        }
    }

    /// get intrinsics
    pub fn intrinsics(&self) -> RenderIntrinsics {
        self.camera_properties.intrinsics.clone()
    }

    /// get camera properties
    pub fn camera_properties(&self) -> RenderCameraProperties {
        self.camera_properties.clone()
    }

    /// reset 2d frame
    pub fn reset_2d_frame(
        &mut self,
        intrinsics: &RenderIntrinsics,
        maybe_background_image: Option<&ArcImage4U8>,
    ) {
        self.camera_properties.intrinsics = intrinsics.clone();
        if let Some(image) = maybe_background_image {
            let device = &self.render_context.wgpu_device;

            let texture_size = wgpu::Extent3d {
                width: image.image_size().width as u32,
                height: image.image_size().height as u32,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("dist_texture"),
                view_formats: &[],
            });

            self.render_context.wgpu_queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(image.tensor.scalar_view().as_slice().unwrap()),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * image.image_size().width as u32),
                    rows_per_image: Some(image.image_size().height as u32),
                },
                texture.size(),
            );

            self.maybe_background_image = Some(texture);
        } else {
            self.maybe_background_image = None;
        }
    }

    /// update 2d renderables
    pub fn update_pixels(&mut self, renderables: Vec<PixelRenderable>) {
        for m in renderables {
            match m {
                PixelRenderable::Line(lines) => {
                    self.pixel.line_renderer.lines_table.insert(
                        lines.name.clone(),
                        Line2dEntity::new(&self.render_context, &lines),
                    );
                }
                PixelRenderable::Point(points) => {
                    self.pixel.point_renderer.points_table.insert(
                        points.name.clone(),
                        Point2dEntity::new(&self.render_context, &points),
                    );
                }
            }
        }
    }

    /// update 3d renerables
    pub fn update_scene(&mut self, renderables: Vec<SceneRenderable>) {
        for m in renderables {
            match m {
                SceneRenderable::Line(lines3) => {
                    self.scene.line_renderer.line_table.insert(
                        lines3.name.clone(),
                        Line3dEntity::new(&self.render_context, &lines3),
                    );
                }
                SceneRenderable::Point(points3) => {
                    self.scene.point_renderer.point_table.insert(
                        points3.name.clone(),
                        Point3dEntity::new(&self.render_context, &points3),
                    );
                }
                SceneRenderable::Mesh3(mesh) => {
                    self.scene.mesh_renderer.mesh_table.insert(
                        mesh.name.clone(),
                        Mesh3dEntity::new(&self.render_context, &mesh),
                    );
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

    fn render_impl(&mut self, params: &RenderParams) -> RenderResult {
        if self.textures.view_port_size != params.view_port_size {
            self.textures = Textures::new(&self.render_context, &params.view_port_size);
        }

        self.uniforms.update(
            &self.render_context,
            params.zoom,
            &self.camera_properties,
            params.view_port_size,
        );

        let mut command_encoder = self
            .render_context
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.scene.paint(
            &self.render_context,
            &params.scene_from_camera,
            &mut command_encoder,
            &self.textures.rgbd,
            &self.textures.depth,
            params.backface_culling,
        );

        self.render_context
            .wgpu_queue
            .submit(Some(command_encoder.finish()));

        let command_encoder = self
            .render_context
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.distortion.run(
            &self.render_context,
            command_encoder,
            &self.textures.rgbd,
            &self.textures.depth,
            &self.maybe_background_image,
            &params.view_port_size,
        );

        let mut command_encoder = self
            .render_context
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.pixel
            .paint(&mut command_encoder, &self.textures.rgbd.final_texture_view);

        let depth_image = self.textures.depth.download_depth_image(
            &self.render_context,
            command_encoder,
            &params.view_port_size,
            &self.camera_properties.clipping_planes,
        );

        self.pixel
            .show_interaction_marker(&self.render_context, &params.maybe_marker);

        let mut rgba_image = None;

        if params.download_rgba {
            let command_encoder = self
                .render_context
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            rgba_image = Some(self.textures.rgbd.download(
                &self.render_context,
                command_encoder,
                &params.view_port_size,
            ));
        }

        if params.compute_depth_texture {
            self.textures
                .depth
                .compute_visual_depth_texture(&self.render_context, &depth_image);
        }

        RenderResult {
            rgba_image,
            rgba_egui_tex_id: self.textures.rgbd.egui_tex_id,
            depth_image,
            depth_egui_tex_id: self.textures.depth.visual_depth_texture.egui_tex_id,
        }
    }
}
