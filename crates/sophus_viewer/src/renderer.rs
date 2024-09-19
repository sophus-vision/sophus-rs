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
/// render uniforms
pub mod uniform_buffers;

use sophus_core::IsTensorLike;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::intensity_image::intensity_arc_image::IsIntensityArcImage;
use sophus_image::ImageSize;
use sophus_lie::Isometry3F64;

use crate::renderables::renderable2d::Renderable2d;
use crate::renderables::renderable3d::Renderable3d;
use crate::renderer::camera::intrinsics::RenderIntrinsics;
use crate::renderer::camera::properties::RenderCameraProperties;
use crate::renderer::pixel_renderer::pixel_line::Line2dEntity;
use crate::renderer::pixel_renderer::pixel_point::Point2dEntity;
use crate::renderer::pixel_renderer::PixelRenderer;
use crate::renderer::scene_renderer::distortion::DistortionRenderer;
use crate::renderer::scene_renderer::line::Line3dEntity;
use crate::renderer::scene_renderer::mesh::Mesh3dEntity;
use crate::renderer::scene_renderer::point::Point3dEntity;
use crate::renderer::scene_renderer::SceneRenderer;
use crate::renderer::textures::Textures;
use crate::renderer::types::RenderResult;
use crate::renderer::types::TranslationAndScaling;
use crate::viewer::interactions::InteractionEnum;
use crate::RenderContext;
use sophus_image::image_view::IsImageView;

/// Offscreen renderer
pub struct OffscreenRenderer {
    pub(crate) camera_properties: RenderCameraProperties,
    state: RenderContext,
    scene: SceneRenderer,
    distortion: DistortionRenderer,
    pixel: PixelRenderer,
    textures: Textures,
    maybe_background_image: Option<wgpu::Texture>,
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
            distortion: DistortionRenderer::new(
                state,
                depth_stencil.clone(),
                &textures.rgbd,
                camera_properties,
            ),
            pixel: PixelRenderer::new(state, &camera_properties.intrinsics.image_size()),
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
        if let Some(image) = maybe_background_image {
            let device = &self.state.wgpu_device;
            let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

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

            self.state.wgpu_queue.write_texture(
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

        self.scene.paint(
            &self.state,
            &state.scene_from_camera,
            &mut command_encoder,
            &self.textures.rgbd.rgba_texture_view,
            &self.textures.depth,
            state.backface_culling,
        );

        self.state.wgpu_queue.submit(Some(command_encoder.finish()));

        let command_encoder = self
            .state
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.distortion.run(
            &self.state,
            command_encoder,
            &self.textures.rgbd,
            &self.textures.depth,
            &self.maybe_background_image,
            &state.view_port_size,
        );

        let mut command_encoder = self
            .state
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.pixel.paint(
            &mut command_encoder,
            &self.textures.rgbd.rgba_texture_view_distorted,
        );

        let depth_image = self.textures.depth.download_depth_image(
            &self.state,
            command_encoder,
            &state.view_port_size,
            &self.camera_properties.clipping_planes,
        );

        if let Some(interaction_enum) = state.maybe_interaction_enum {
            self.pixel
                .show_interaction_marker(&self.state, interaction_enum);
        }

        let mut image_4u8 = None;
        let command_encoder = self
            .state
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        image_4u8 = Some(self.textures.rgbd.download_rgb_and_depth(
            &self.state,
            command_encoder,
            &state.view_port_size,
        ));

        if state.compute_depth_texture {
            self.textures
                .depth
                .compute_visual_depth_texture(&self.state, &depth_image);
        }

        RenderResult {
            rgba_image: match image_4u8 {
                Some(i) => Some(i.rgba_u16.convert_to()),
                None => None,
            },
            rgba_egui_tex_id: self.textures.rgbd.egui_tex_id,
            depth_image,
            depth_egui_tex_id: self.textures.depth.visual_depth_texture.egui_tex_id,
        }
    }
}
