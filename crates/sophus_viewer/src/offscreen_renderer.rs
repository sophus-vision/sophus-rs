/// offscreen texture for rendering
pub mod textures;

/// The pixel renderer for 2D rendering.
pub mod pixel_renderer;

/// The scene renderer for 3D rendering.
pub mod scene_renderer;

use eframe::egui;
use sophus_core::linalg::SVec;
use sophus_core::linalg::VecF64;
use sophus_core::IsTensorLike;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::arc_image::GenArcImage;
use sophus_image::mut_image::MutImage4U8;
use sophus_image::prelude::IsImageView;
use sophus_image::prelude::IsMutImageView;
use sophus_image::ImageSize;
use sophus_lie::Isometry3F64;
use sophus_sensor::DynCamera;

use crate::offscreen_renderer::pixel_renderer::pixel_line::Line2dEntity;
use crate::offscreen_renderer::pixel_renderer::pixel_point::Point2dEntity;
use crate::offscreen_renderer::pixel_renderer::PixelRenderer;
use crate::offscreen_renderer::scene_renderer::line::Line3dEntity;
use crate::offscreen_renderer::scene_renderer::mesh::Mesh3dEntity;
use crate::offscreen_renderer::scene_renderer::point::Point3dEntity;
use crate::offscreen_renderer::scene_renderer::textured_mesh::TexturedMeshEntity;
use crate::offscreen_renderer::scene_renderer::SceneRenderer;
use crate::offscreen_renderer::textures::OffscreenTextures;
use crate::renderables::renderable2d::Renderable2d;
use crate::renderables::renderable3d::make_textured_mesh3;
use crate::renderables::renderable3d::Renderable3d;
use crate::views::aspect_ratio::HasAspectRatio;
use crate::views::interactions::InteractionEnum;
use crate::ViewerRenderState;

/// Clipping planes for the Wgpu renderer
#[derive(Clone, Copy, Debug)]
pub struct ClippingPlanes {
    /// Near clipping plane
    pub near: f64,
    /// Far clipping plane
    pub far: f64,
}

impl ClippingPlanes {
    /// default near clipping plabe
    pub const DEFAULT_NEAR: f64 = 1.0;
    /// default far clipping plabe
    pub const DEFAULT_FAR: f64 = 1000.0;
}

impl Default for ClippingPlanes {
    fn default() -> Self {
        ClippingPlanes {
            near: ClippingPlanes::DEFAULT_NEAR,
            far: ClippingPlanes::DEFAULT_FAR,
        }
    }
}

impl ClippingPlanes {
    pub(crate) fn z_from_ndc(&self, ndc: f64) -> f64 {
        -(self.far * self.near) / (-self.far + ndc * self.far - ndc * self.near)
    }

    pub(crate) fn _ndc_from_z(&self, z: f64) -> f64 {
        (self.far * (z - self.near)) / (z * (self.far - self.near))
    }
}

/// Offscreen renderer
pub struct OffscreenRenderer {
    intrinsics: DynCamera<f64, 1>,
    state: ViewerRenderState,
    scene: SceneRenderer,
    pixel: PixelRenderer,
    textures: OffscreenTextures,
    maybe_background_image: Option<ArcImage4U8>,
}

/// Render result
pub struct OffscreenRenderResult {
    /// depth image - might have a greater width than the requested width
    pub depth: GenArcImage<2, 0, f32, f32, 1, 1>,
    /// rgba egui texture id
    pub rgba_egui_tex_id: egui::TextureId,

    /// depth egui texture id
    pub depth_egui_tex_id: egui::TextureId,
}

impl HasAspectRatio for OffscreenRenderer {
    fn aspect_ratio(&self) -> f32 {
        self.intrinsics.image_size().aspect_ratio()
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Zoom2d {
    pub(crate) translation_x: f32,
    pub(crate) translation_y: f32,
    pub(crate) scaling_x: f32,
    pub(crate) scaling_y: f32,
}

impl Default for Zoom2d {
    fn default() -> Self {
        Zoom2d {
            translation_x: 0.0,
            translation_y: 0.0,
            scaling_x: 1.0,
            scaling_y: 1.0,
        }
    }
}

/// Translation and scaling
///
/// todo: move to sophus_lie
#[derive(Clone, Copy, Debug)]
pub struct TranslationAndScaling {
    /// translation
    pub translation: VecF64<2>,
    /// scaling
    pub scaling: VecF64<2>,
}

impl TranslationAndScaling {
    /// identity
    pub fn identity() -> Self {
        TranslationAndScaling {
            translation: VecF64::<2>::zeros(),
            scaling: VecF64::<2>::new(1.0, 1.0),
        }
    }

    /// apply translation and scaling
    pub fn apply(&self, xy: VecF64<2>) -> VecF64<2> {
        VecF64::<2>::new(
            xy[0] * self.scaling[0] + self.translation[0],
            xy[1] * self.scaling[1] + self.translation[1],
        )
    }
}

impl OffscreenRenderer {
    /// background image plane
    pub const BACKGROUND_IMAGE_PLANE: f64 = 900.0;

    /// create new offscreen renderer
    pub fn new(state: &ViewerRenderState, intrinsics: &DynCamera<f64, 1>) -> Self {
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
            scene: SceneRenderer::new(state, intrinsics, depth_stencil.clone()),
            pixel: PixelRenderer::new(state, &intrinsics.image_size(), depth_stencil),
            textures: OffscreenTextures::new(state, &intrinsics.image_size()),
            intrinsics: intrinsics.clone(),
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

    fn depth_to_rgb(depth: f32) -> (u8, u8, u8) {
        let depth = depth.clamp(0.0, 1.0); // Ensure the depth value is between 0 and 1

        let r = if depth < 0.375 {
            0.0
        } else if depth < 0.625 {
            255.0 * (depth - 0.375) / 0.25
        } else {
            255.0 * (1.0 - (depth - 0.625) / 0.25)
        };

        let g = if depth < 0.25 {
            255.0 * depth / 0.25
        } else if depth < 0.75 {
            255.0
        } else {
            255.0 * (1.0 - (depth - 0.75) / 0.25)
        };

        let b = if depth < 0.125 {
            255.0 * (0.125 - depth) / 0.125
        } else if depth < 0.375 {
            255.0
        } else if depth < 0.625 {
            255.0 * (1.0 - (depth - 0.375) / 0.25)
        } else {
            0.0
        };

        (r as u8, g as u8, b as u8)
    }

    fn render_impl(
        &mut self,
        view_port_size: &ImageSize,
        zoom: TranslationAndScaling,
        scene_from_camera: Isometry3F64,
        maybe_interaction_enum: Option<&InteractionEnum>,
        compute_visual_depth: bool,
        backface_culling: bool,
    ) -> OffscreenRenderResult {
        if self.textures.view_port_size != *view_port_size {
            self.textures = OffscreenTextures::new(&self.state, view_port_size);
        }

        self.scene.prepare(&self.state, zoom, &self.intrinsics);

        self.maybe_background_image = None;
        self.pixel
            .prepare(&self.state, view_port_size, &self.intrinsics, zoom);

        let mut command_encoder = self
            .state
            .device
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

        self.state.queue.submit(Some(command_encoder.finish()));

        let mut command_encoder = self
            .state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.scene.depth_paint(
            &self.state,
            &scene_from_camera,
            &mut command_encoder,
            &self.textures.depth.depth_texture_view_f32,
            &self.textures.z_buffer,
            backface_culling,
        );

        let depth_image =
            self.textures
                .depth
                .download_image(&self.state, command_encoder, view_port_size);

        if compute_visual_depth {
            let mut image_rgba = MutImage4U8::from_image_size_and_val(
                depth_image.image_size(),
                SVec::<u8, 4>::new(0, 255, 0, 255),
            );

            for v in 0..depth_image.image_size().height {
                for u in 0..depth_image.image_size().width {
                    let z = depth_image.pixel(u, v);
                    let (r, g, b) = Self::depth_to_rgb(z);

                    *image_rgba.mut_pixel(u, v) = SVec::<u8, 4>::new(r, g, b, 255);
                }
            }

            let image_rgba = ArcImage4U8::from(image_rgba);

            self.state.queue.write_texture(
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

        OffscreenRenderResult {
            depth: depth_image,
            rgba_egui_tex_id: self.textures.rgba.egui_tex_id,
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
        compute_visual_depth: bool,
        backface_culling: bool,
    ) -> OffscreenRenderResult {
        self.render_impl(
            view_port_size,
            zoom,
            scene_from_camera,
            Some(interaction_enum),
            compute_visual_depth,
            backface_culling,
        )
    }

    /// render
    pub fn render(
        &mut self,
        view_port_size: &ImageSize,
        zoom: TranslationAndScaling,
        scene_from_camera: Isometry3F64,
        compute_visual_depth: bool,
        backface_culling: bool,
    ) -> OffscreenRenderResult {
        self.render_impl(
            view_port_size,
            zoom,
            scene_from_camera,
            None,
            compute_visual_depth,
            backface_culling,
        )
    }
}
