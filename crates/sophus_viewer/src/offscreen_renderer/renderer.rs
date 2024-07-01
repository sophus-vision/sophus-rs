use eframe::egui;
use sophus_core::linalg::VecF64;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::arc_image::GenArcImage;
use sophus_image::ImageSize;
use sophus_lie::Isometry3;
use sophus_sensor::DynCamera;

use crate::offscreen_renderer::pixel_renderer::LineVertex2;
use crate::offscreen_renderer::pixel_renderer::PixelRenderer;
use crate::offscreen_renderer::pixel_renderer::PointVertex2;
use crate::offscreen_renderer::scene_renderer::line::LineVertex3;
use crate::offscreen_renderer::scene_renderer::mesh::MeshVertex3;
use crate::offscreen_renderer::scene_renderer::point::PointVertex3;
use crate::offscreen_renderer::scene_renderer::textured_mesh::TexturedMeshVertex3;
use crate::offscreen_renderer::scene_renderer::SceneRenderer;
use crate::offscreen_renderer::textures::OffscreenTextures;
use crate::renderables::renderable2d::Renderable2d;
use crate::renderables::renderable3d::Renderable3d;
use crate::renderables::renderable3d::TexturedMesh3;
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
    pub const DEFAULT_NEAR: f64 = 0.1;
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
    /// rgba texture id
    pub rgba_tex_id: egui::TextureId,
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
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
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

            let tex_mesh = TexturedMesh3::make(&[
                [(p0, [0.0, 0.0]), (p1, [1.0, 0.0]), (p2, [0.0, 1.0])],
                [(p1, [1.0, 0.0]), (p2, [0.0, 1.0]), (p3, [1.0, 1.0])],
            ]);
            self.scene
                .textured_mesh_renderer
                .mesh_table
                .insert("background".to_owned(), tex_mesh.mesh);
            self.maybe_background_image = Some(background_image.clone());
        }
    }

    /// update 2d renderables
    pub fn update_2d_renderables(&mut self, renderables: Vec<Renderable2d>) {
        for m in renderables {
            match m {
                Renderable2d::Lines2(lines) => {
                    self.pixel
                        .line_renderer
                        .lines_table
                        .insert(lines.name, lines.lines);
                }
                Renderable2d::Points2(points) => {
                    self.pixel
                        .point_renderer
                        .points_table
                        .insert(points.name, points.points);
                }
            }
        }

        self.pixel.clear_vertex_data();

        for (_, points) in self.pixel.point_renderer.points_table.iter() {
            for point in points.iter() {
                let v = PointVertex2 {
                    _pos: [point.p[0], point.p[1]],
                    _color: [point.color.r, point.color.g, point.color.b, point.color.a],
                    _point_size: point.point_size,
                };
                for _i in 0..6 {
                    self.pixel.point_renderer.vertex_data.push(v);
                }
            }
        }
        for (_, lines) in self.pixel.line_renderer.lines_table.iter() {
            for line in lines.iter() {
                let p0 = line.p0;
                let p1 = line.p1;
                let d = (p0 - p1).normalize();
                let normal = [d[1], -d[0]];

                let v0 = LineVertex2 {
                    _pos: [p0[0], p0[1]],
                    _normal: normal,
                    _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                    _line_width: line.line_width,
                };
                let v1 = LineVertex2 {
                    _pos: [p1[0], p1[1]],
                    _normal: normal,
                    _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                    _line_width: line.line_width,
                };
                self.pixel.line_renderer.vertex_data.push(v0);
                self.pixel.line_renderer.vertex_data.push(v0);
                self.pixel.line_renderer.vertex_data.push(v1);
                self.pixel.line_renderer.vertex_data.push(v0);
                self.pixel.line_renderer.vertex_data.push(v1);
                self.pixel.line_renderer.vertex_data.push(v1);
            }
        }
        for (_, points) in self.scene.point_renderer.point_table.iter() {
            for point in points.iter() {
                let v = PointVertex3 {
                    _pos: [point.p[0], point.p[1], point.p[2]],
                    _color: [point.color.r, point.color.g, point.color.b, point.color.a],
                    _point_size: point.point_size,
                };
                for _i in 0..6 {
                    self.scene.point_renderer.vertex_data.push(v);
                }
            }
        }
        for (_, lines) in self.scene.line_renderer.line_table.iter() {
            for line in lines.iter() {
                let p0 = line.p0;
                let p1 = line.p1;

                let v0 = LineVertex3 {
                    _p0: [p0[0], p0[1], p0[2]],
                    _p1: [p1[0], p1[1], p1[2]],
                    _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                    _line_width: line.line_width,
                };
                let v1 = LineVertex3 {
                    _p0: [p0[0], p0[1], p0[2]],
                    _p1: [p1[0], p1[1], p1[2]],
                    _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                    _line_width: line.line_width,
                };
                self.scene.line_renderer.vertex_data.push(v0);
                self.scene.line_renderer.vertex_data.push(v0);
                self.scene.line_renderer.vertex_data.push(v1);
                self.scene.line_renderer.vertex_data.push(v0);
                self.scene.line_renderer.vertex_data.push(v1);
                self.scene.line_renderer.vertex_data.push(v1);
            }
        }
        for (_, mesh) in self.scene.mesh_renderer.mesh_table.iter() {
            for trig in mesh.iter() {
                let v0 = MeshVertex3 {
                    _pos: [trig.p0[0], trig.p0[1], trig.p0[2]],
                    _color: [trig.color.r, trig.color.g, trig.color.b, trig.color.a],
                };
                let v1 = MeshVertex3 {
                    _pos: [trig.p1[0], trig.p1[1], trig.p1[2]],
                    _color: [trig.color.r, trig.color.g, trig.color.b, trig.color.a],
                };
                let v2 = MeshVertex3 {
                    _pos: [trig.p2[0], trig.p2[1], trig.p2[2]],
                    _color: [trig.color.r, trig.color.g, trig.color.b, trig.color.a],
                };
                self.scene.mesh_renderer.vertices.push(v0);
                self.scene.mesh_renderer.vertices.push(v1);
                self.scene.mesh_renderer.vertices.push(v2);
            }
        }
        for (_, mesh) in self.scene.textured_mesh_renderer.mesh_table.iter() {
            for trig in mesh.iter() {
                let v0 = TexturedMeshVertex3 {
                    _pos: [trig.p0[0], trig.p0[1], trig.p0[2]],
                    _tex: [trig.tex0[0], trig.tex0[1]],
                };
                let v1 = TexturedMeshVertex3 {
                    _pos: [trig.p1[0], trig.p1[1], trig.p1[2]],
                    _tex: [trig.tex1[0], trig.tex1[1]],
                };
                let v2 = TexturedMeshVertex3 {
                    _pos: [trig.p2[0], trig.p2[1], trig.p2[2]],
                    _tex: [trig.tex2[0], trig.tex2[1]],
                };
                self.scene.textured_mesh_renderer.vertices.push(v0);
                self.scene.textured_mesh_renderer.vertices.push(v1);
                self.scene.textured_mesh_renderer.vertices.push(v2);
            }
        }
    }

    /// update 3d renerables
    pub fn update_3d_renderables(&mut self, renderables: Vec<Renderable3d>) {
        for m in renderables {
            match m {
                Renderable3d::Lines3(lines3) => {
                    self.scene
                        .line_renderer
                        .line_table
                        .insert(lines3.name, lines3.lines);
                }
                Renderable3d::Points3(points3) => {
                    self.scene
                        .point_renderer
                        .point_table
                        .insert(points3.name, points3.points);
                }
                Renderable3d::Mesh3(mesh) => {
                    self.scene
                        .mesh_renderer
                        .mesh_table
                        .insert(mesh.name, mesh.mesh);
                }
            }
        }

        // maybe separate call needed

        self.scene.clear_vertex_data();

        for (_, points) in self.scene.point_renderer.point_table.iter() {
            for point in points.iter() {
                let v = PointVertex3 {
                    _pos: [point.p[0], point.p[1], point.p[2]],
                    _color: [point.color.r, point.color.g, point.color.b, point.color.a],
                    _point_size: point.point_size,
                };
                for _i in 0..6 {
                    self.scene.point_renderer.vertex_data.push(v);
                }
            }
        }
        for (_, lines) in self.scene.line_renderer.line_table.iter() {
            for line in lines.iter() {
                let p0 = line.p0;
                let p1 = line.p1;

                let v0 = LineVertex3 {
                    _p0: [p0[0], p0[1], p0[2]],
                    _p1: [p1[0], p1[1], p1[2]],
                    _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                    _line_width: line.line_width,
                };
                let v1 = LineVertex3 {
                    _p0: [p0[0], p0[1], p0[2]],
                    _p1: [p1[0], p1[1], p1[2]],
                    _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                    _line_width: line.line_width,
                };
                self.scene.line_renderer.vertex_data.push(v0);
                self.scene.line_renderer.vertex_data.push(v0);
                self.scene.line_renderer.vertex_data.push(v1);
                self.scene.line_renderer.vertex_data.push(v0);
                self.scene.line_renderer.vertex_data.push(v1);
                self.scene.line_renderer.vertex_data.push(v1);
            }
        }
        for (_, mesh) in self.scene.mesh_renderer.mesh_table.iter() {
            for trig in mesh.iter() {
                let v0 = MeshVertex3 {
                    _pos: [trig.p0[0], trig.p0[1], trig.p0[2]],
                    _color: [trig.color.r, trig.color.g, trig.color.b, trig.color.a],
                };
                let v1 = MeshVertex3 {
                    _pos: [trig.p1[0], trig.p1[1], trig.p1[2]],
                    _color: [trig.color.r, trig.color.g, trig.color.b, trig.color.a],
                };
                let v2 = MeshVertex3 {
                    _pos: [trig.p2[0], trig.p2[1], trig.p2[2]],
                    _color: [trig.color.r, trig.color.g, trig.color.b, trig.color.a],
                };
                self.scene.mesh_renderer.vertices.push(v0);
                self.scene.mesh_renderer.vertices.push(v1);
                self.scene.mesh_renderer.vertices.push(v2);
            }
        }
        for (_, mesh) in self.scene.textured_mesh_renderer.mesh_table.iter() {
            for trig in mesh.iter() {
                let v0 = TexturedMeshVertex3 {
                    _pos: [trig.p0[0], trig.p0[1], trig.p0[2]],
                    _tex: [trig.tex0[0], trig.tex0[1]],
                };
                let v1 = TexturedMeshVertex3 {
                    _pos: [trig.p1[0], trig.p1[1], trig.p1[2]],
                    _tex: [trig.tex1[0], trig.tex1[1]],
                };
                let v2 = TexturedMeshVertex3 {
                    _pos: [trig.p2[0], trig.p2[1], trig.p2[2]],
                    _tex: [trig.tex2[0], trig.tex2[1]],
                };
                self.scene.textured_mesh_renderer.vertices.push(v0);
                self.scene.textured_mesh_renderer.vertices.push(v1);
                self.scene.textured_mesh_renderer.vertices.push(v2);
            }
        }
    }

    fn render_impl(
        &mut self,
        view_port_size: &ImageSize,
        zoom: TranslationAndScaling,
        scene_from_camera: Isometry3<f64, 1>,
        maybe_interaction_enum: Option<&InteractionEnum>,
    ) -> OffscreenRenderResult {
        if self.textures.view_port_size != *view_port_size {
            self.textures = OffscreenTextures::new(&self.state, view_port_size);
        }

        self.scene.prepare(
            &self.state,
            zoom,
            &self.intrinsics,
            &scene_from_camera,
            &self.maybe_background_image,
        );

        self.maybe_background_image = None;
        self.pixel
            .prepare(&self.state, view_port_size, &self.intrinsics, zoom);

        let mut command_encoder = self
            .state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.scene.paint(
            &mut command_encoder,
            &self.textures.rgba.rgba_texture_view,
            &self.textures.z_buffer,
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
            &mut command_encoder,
            &self.textures.depth.depth_texture_view_f32,
            &self.textures.z_buffer,
        );

        let depth_image =
            self.textures
                .depth
                .download_image(&self.state, command_encoder, view_port_size);

        OffscreenRenderResult {
            depth: depth_image,
            rgba_tex_id: self.textures.rgba.rgba_tex_id,
        }
    }

    /// render with interaction marker
    pub fn render_with_interaction_marker(
        &mut self,
        view_port_size: &ImageSize,
        zoom: TranslationAndScaling,
        scene_from_camera: Isometry3<f64, 1>,
        interaction_enum: &InteractionEnum,
    ) -> OffscreenRenderResult {
        self.render_impl(
            view_port_size,
            zoom,
            scene_from_camera,
            Some(interaction_enum),
        )
    }

    /// render
    pub fn render(
        &mut self,
        view_port_size: &ImageSize,
        zoom: TranslationAndScaling,
        scene_from_camera: Isometry3<f64, 1>,
    ) -> OffscreenRenderResult {
        self.render_impl(view_port_size, zoom, scene_from_camera, None)
    }
}
