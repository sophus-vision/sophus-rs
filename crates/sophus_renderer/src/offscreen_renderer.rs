use eframe::wgpu;
use sophus_image::{
    ArcImage4U8,
    ImageSize,
};
use sophus_lie::Isometry3F64;

use crate::{
    RenderContext,
    camera::{
        Intermediate,
        RenderCameraProperties,
        RenderIntrinsics,
    },
    pixel_renderer::{
        Ellipse2dEntity,
        Line2dEntity,
        PixelRenderer,
        Point2dEntity,
    },
    prelude::*,
    renderables::{
        PixelRenderable,
        SceneRenderable,
    },
    scene_renderer::{
        DistortionRenderer,
        LIGHT_IN_CAMERA,
        Line3dEntity,
        Mesh3dEntity,
        Point3dEntity,
        SceneRenderer,
        TexturedMeshEntity,
        TracedPrimitives,
    },
    textures::{
        FaceTextures,
        Textures,
    },
    types::{
        RenderResult,
        ScenePivotMarker,
        TranslationAndScaling,
    },
    uniform_buffers::{
        DrawStyle,
        VertexShaderUniformBuffers,
    },
};

/// Offscreen renderer
pub struct OffscreenRenderer {
    /// Camera properties
    pub camera_properties: RenderCameraProperties,
    /// Render context
    pub render_context: RenderContext,
    /// Scene renderer
    pub scene: SceneRenderer,
    distortion: DistortionRenderer,
    pixel: PixelRenderer,
    textures: Textures,
    maybe_background_image: Option<wgpu::Texture>,
    uniforms: Arc<VertexShaderUniformBuffers>,
    /// primitives which are traced against each pixel's ray, rather than rasterized
    traced: TracedPrimitives,
}

struct RenderParams {
    view_port_size: ImageSize,
    zoom: TranslationAndScaling,
    scene_from_camera: Isometry3F64,
    maybe_marker: Option<ScenePivotMarker>,
    backface_culling: bool,
    download_rgba: bool,
    debug_frustum_planes: bool,
    wireframe: bool,
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
                backface_culling: false,
                download_rgba: false,
                debug_frustum_planes: false,
                wireframe: false,
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
    pub fn interaction(mut self, marker: Option<ScenePivotMarker>) -> Self {
        self.params.maybe_marker = marker;
        self
    }

    /// set backface culling
    pub fn backface_culling(mut self, backface_culling: bool) -> Self {
        self.params.backface_culling = backface_culling;
        self
    }

    /// Tint the image by which frustum each pixel came from - a debug view of the intermediate
    /// the scene was rendered into. A view which fits on a single plane gets one flat tint.
    pub fn debug_frustum_planes(mut self, debug_frustum_planes: bool) -> Self {
        self.params.debug_frustum_planes = debug_frustum_planes;
        self
    }

    /// Draw everything as edges rather than as surfaces.
    ///
    /// A rasterized surface is drawn as the edges of its triangles, and a traced one as its
    /// silhouette - which is the outline of the shape itself, not of any tessellation of it.
    pub fn wireframe(mut self, wireframe: bool) -> Self {
        self.params.wireframe = wireframe;
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

        let traced = TracedPrimitives::new(render_context);
        let uniforms = Arc::new(VertexShaderUniformBuffers::new(
            render_context,
            camera_properties,
        ));

        Self {
            scene: SceneRenderer::new(render_context, depth_stencil.clone(), uniforms.clone()),
            distortion: DistortionRenderer::new(render_context, uniforms.clone(), &traced),
            traced,
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

    /// Removes all renderables, but keeps the pipelines, textures and bind groups.
    ///
    /// This is what a view wants when a new frame arrives: recreating the whole renderer instead
    /// rebuilds every pipeline and texture, which costs orders of magnitude more than rendering
    /// the frame does.
    pub fn clear_renderables(&mut self) {
        self.scene.mesh_renderer.mesh_table.clear();
        self.scene.textured_mesh_renderer.mesh_table.clear();
        self.traced.clear();
        self.pixel.line_renderer.lines_table.clear();
        self.pixel.point_renderer.points_table.clear();
        self.scene.line_renderer.line_table.clear();
        self.scene.point_renderer.point_table.clear();
        self.pixel.ellipse_renderer.ellipses_table.clear();
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
            // A streaming view hands us a new image of the same size every frame, so reuse the
            // texture rather than allocating a new one each time.
            let reusable = self
                .maybe_background_image
                .as_ref()
                .is_some_and(|texture| texture.size() == texture_size);
            let texture = match reusable {
                true => self.maybe_background_image.take().unwrap(),
                false => device.create_texture(&wgpu::TextureDescriptor {
                    size: texture_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    label: Some("background image"),
                    view_formats: &[],
                }),
            };

            self.render_context.wgpu_queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(image.tensor.scalar_view().as_slice().unwrap()),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * image.image_size().width as u32),
                    rows_per_image: Some(image.image_size().height as u32),
                },
                texture.size(),
            );

            if !reusable {
                self.distortion.invalidate_bind_group();
            }
            self.maybe_background_image = Some(texture);
        } else {
            if self.maybe_background_image.is_some() {
                self.distortion.invalidate_bind_group();
            }
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
                PixelRenderable::Ellipse(ellipses) => {
                    self.pixel.ellipse_renderer.ellipses_table.insert(
                        ellipses.name.clone(),
                        Ellipse2dEntity::new(&self.render_context, &ellipses),
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
                SceneRenderable::Ellipsoid(ellipsoids) => {
                    self.traced.insert(&ellipsoids);
                }
                SceneRenderable::Planar(planars) => {
                    self.traced.insert_planars(&planars);
                }
                SceneRenderable::Capsule(capsules) => {
                    self.traced.insert_capsules(&capsules);
                }
                SceneRenderable::Cone(cones) => {
                    self.traced.insert_cones(&cones);
                }
                SceneRenderable::Mesh3(mesh) => {
                    self.scene.mesh_renderer.mesh_table.insert(
                        mesh.name.clone(),
                        Mesh3dEntity::new(&self.render_context, &mesh),
                    );
                }
                SceneRenderable::TexturedMesh3(mesh) => {
                    let renderer = &mut self.scene.textured_mesh_renderer;
                    let entity = TexturedMeshEntity::new(
                        &self.render_context,
                        &renderer.texture_bind_group_layout,
                        &mesh,
                    );
                    renderer.mesh_table.insert(mesh.name.clone(), entity);
                }
            }
        }
    }

    /// render
    pub fn render_params(
        &'_ mut self,
        view_port_size: &ImageSize,
        world_from_camera: &Isometry3F64,
    ) -> RenderBuilder<'_> {
        RenderBuilder::new(*view_port_size, *world_from_camera, self)
    }

    fn render_impl(&mut self, params: &RenderParams) -> RenderResult {
        if self.textures.view_port_size != params.view_port_size {
            self.textures = Textures::new(&self.render_context, &params.view_port_size);
            self.distortion.invalidate_bind_group();
        }

        // The light rides the camera, a little up and to the side of the optical axis: a light
        // exactly on the axis makes the shading term depend only on the angle between the normal
        // and the view direction, so a sphere comes out radially symmetric - a flat disc with a
        // vignette rather than a ball, with no terminator anywhere.
        let light_in_world = (self.scene.world_from_scene * params.scene_from_camera)
            .rotation()
            .transform(LIGHT_IN_CAMERA);
        self.traced.update(
            &self.render_context,
            &(self.scene.world_from_scene * params.scene_from_camera),
            LIGHT_IN_CAMERA,
        );

        // One plane, or several frusta for a field of view no plane holds.
        let style = DrawStyle {
            debug_frustum_planes: params.debug_frustum_planes,
            wireframe: params.wireframe,
        };
        let intermediate = Intermediate::choose(&self.camera_properties.intrinsics, params.zoom);
        let face_size = FaceTextures::face_size_for(params.view_port_size);
        match &intermediate {
            Intermediate::Plane(plane) => self.uniforms.update(
                &self.render_context,
                params.zoom,
                &self.camera_properties,
                params.view_port_size,
                style,
                *plane,
            ),
            Intermediate::Frusta(frusta) => {
                self.textures.ensure_faces(&self.render_context, face_size);
                self.uniforms.update_for_faces(
                    &self.render_context,
                    params.zoom,
                    &self.camera_properties,
                    params.view_port_size,
                    frusta,
                    style,
                );
            }
        }

        let mut command_encoder = self
            .render_context
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // All passes go into a single command buffer: wgpu inserts the barriers between them, and
        // one submission per frame avoids the per-submit overhead of several.
        match &intermediate {
            Intermediate::Frusta(frusta) => {
                let faces = self.textures.faces.as_ref().expect("just ensured to exist");
                let (multisample_view, resolve_view, depth_view) = faces.face_attachments();
                // one pose slot per entity *per face*, so the cursor runs across all of them
                let mut entity_slot = 0;
                for frustum in frusta {
                    self.scene.paint_into(
                        &self.render_context,
                        &frustum.world_from_face(&params.scene_from_camera),
                        light_in_world,
                        &mut command_encoder,
                        multisample_view,
                        resolve_view,
                        depth_view,
                        &mut entity_slot,
                        params.backface_culling,
                    );
                    faces.copy_face_into_atlas(&mut command_encoder, frustum.index() as u32);
                    faces.resolve_face_depth(&mut command_encoder, frustum.index() as u32);
                }
                self.distortion.run_faces(
                    &self.traced,
                    &self.render_context,
                    &mut command_encoder,
                    &self.textures.rgbd,
                    &self.textures.depth,
                    faces,
                    &self.maybe_background_image,
                    &params.view_port_size,
                );
            }
            Intermediate::Plane(_) => {
                self.scene.paint(
                    &self.render_context,
                    &params.scene_from_camera,
                    light_in_world,
                    &mut command_encoder,
                    &self.textures.rgbd,
                    &self.textures.depth,
                    params.backface_culling,
                );
                self.distortion.run(
                    &self.traced,
                    &self.render_context,
                    &mut command_encoder,
                    &self.textures.rgbd,
                    &self.textures.depth,
                    &self.maybe_background_image,
                    &params.view_port_size,
                );
            }
        }

        // Note: the marker state has to be set before `paint`, which records the draw calls
        // based on it - otherwise the marker would appear and disappear one frame late.
        self.pixel
            .show_interaction_marker(&self.render_context, &params.maybe_marker);

        self.pixel
            .paint(&mut command_encoder, &self.textures.rgbd.final_texture_view);

        self.render_context
            .wgpu_queue
            .submit(core::iter::once(command_encoder.finish()));

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

        RenderResult {
            rgba_image,
            rgba_egui_tex_id: self.textures.rgbd.egui_tex_id,
            depth_egui_tex_id: self.textures.depth.visual_depth_texture.egui_tex_id,
            depth_texture: self
                .textures
                .depth
                .main_render_ndc_z_texture
                .final_texture
                .clone(),
            depth_staging_buffer: self.textures.depth.staging_buffer.clone(),
            visual_depth_texture: self
                .textures
                .depth
                .visual_depth_texture
                .visual_texture
                .clone(),
        }
    }
}
