mod distortion;
mod line;
mod mesh;
mod point;
mod textured_mesh;
mod traced;

pub use distortion::*;
use eframe::wgpu;
pub use line::*;
pub use mesh::*;
pub use point::*;
use sophus_autodiff::linalg::VecF64;
use sophus_lie::Isometry3F64;
pub use textured_mesh::*;
pub use traced::*;
use wgpu::DepthStencilState;

use crate::{
    RenderContext,
    pipeline_builder::{
        PipelineBuilder,
        TargetTexture,
    },
    prelude::*,
    scene_renderer::{
        mesh::MeshRenderer,
        point::ScenePointRenderer,
    },
    textures::{
        DepthTextures,
        RgbdTexture,
    },
    uniform_buffers::VertexShaderUniformBuffers,
};

/// The direction the light shines, in camera coordinates - fixed to the camera, so it needs no
/// uniform of its own and never leaves anything unlit. It is held a little off the optical axis:
/// a light exactly on it gives a sphere no terminator, which reads as a disc rather than a ball.
///
/// About 12 degrees off, which is enough to shape a sphere without moving the brightest point far
/// from the middle of the view - a wider offset leaves a surface square on to the camera visibly
/// short of its own colour. The camera looks along +z with +y down, so this is over the viewer's
/// left shoulder.
pub(crate) const LIGHT_IN_CAMERA: VecF64<3> = VecF64::<3>::new(0.13, -0.17, 1.0);

/// Where the scene is seen from, for one pass.
///
/// The pose is the one the *intermediate* is rendered from - the camera for a single plane, and a
/// frustum face otherwise - while the light is fixed to the camera, so the two are not related by
/// anything the shader could work out for itself.
pub(crate) struct SceneView {
    pub(crate) world_from_camera: Isometry3F64,
    pub(crate) light_in_world: VecF64<3>,
}

/// Scene renderer
pub struct SceneRenderer {
    /// uniforms
    pub uniforms: Arc<VertexShaderUniformBuffers>,
    /// Mesh renderer
    pub mesh_renderer: MeshRenderer,
    /// Textured mesh renderer
    pub textured_mesh_renderer: TexturedMeshRenderer,
    /// Point renderer
    pub point_renderer: ScenePointRenderer,
    /// Line renderer
    pub line_renderer: line::SceneLineRenderer,
    /// World from scene
    pub world_from_scene: Isometry3F64,
}

impl SceneRenderer {
    /// Create a new scene renderer
    pub fn new(
        render_context: &RenderContext,
        depth_stencil: Option<DepthStencilState>,
        uniforms: Arc<VertexShaderUniformBuffers>,
    ) -> Self {
        let scene_pipeline_builder = PipelineBuilder::new_scene(
            render_context,
            Arc::new(TargetTexture {
                rgba_output_format: wgpu::TextureFormat::Rgba8Unorm,
            }),
            uniforms.clone(),
            depth_stencil,
        );

        Self {
            uniforms,
            mesh_renderer: MeshRenderer::new(render_context, &scene_pipeline_builder),
            line_renderer: line::SceneLineRenderer::new(render_context, &scene_pipeline_builder),
            point_renderer: ScenePointRenderer::new(render_context, &scene_pipeline_builder),
            textured_mesh_renderer: TexturedMeshRenderer::new(
                render_context,
                &scene_pipeline_builder,
            ),
            world_from_scene: Isometry3F64::identity(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn paint<'rp>(
        &'rp self,
        context: &RenderContext,
        scene_from_camera: &Isometry3F64,
        light_in_world: VecF64<3>,
        command_encoder: &'rp mut wgpu::CommandEncoder,
        rgba: &'rp RgbdTexture,
        depth: &DepthTextures,
        backface_culling: bool,
    ) {
        let mut entity_slot = 0;
        self.paint_into(
            context,
            scene_from_camera,
            light_in_world,
            command_encoder,
            &rgba.multisample_texture_view,
            &rgba.resolved_texture_view,
            &depth.main_render_ndc_z_texture.multisample_texture_view,
            &mut entity_slot,
            backface_culling,
        );
    }

    /// Renders the scene into the given attachments.
    ///
    /// `entity_slot` is threaded through rather than owned, because a multi-frustum intermediate
    /// paints several faces into one command buffer, and every entity of every face needs a pose
    /// slot of its own - see `CameraFromEntityPoseUniform::update_given_camera_and_entity`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn paint_into<'rp>(
        &'rp self,
        context: &RenderContext,
        scene_from_camera: &Isometry3F64,
        light_in_world: VecF64<3>,
        command_encoder: &'rp mut wgpu::CommandEncoder,
        multisample_view: &wgpu::TextureView,
        resolve_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        entity_slot: &mut u32,
        backface_culling: bool,
    ) {
        let view = SceneView {
            world_from_camera: self.world_from_scene * scene_from_camera,
            light_in_world,
        };

        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: multisample_view,
                resolve_target: Some(resolve_view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        self.mesh_renderer.paint(
            context,
            &view,
            &self.uniforms,
            &mut render_pass,
            entity_slot,
            backface_culling,
        );
        self.point_renderer.paint(
            context,
            &view,
            &self.uniforms,
            &mut render_pass,
            entity_slot,
        );
        self.line_renderer.paint(
            context,
            &view,
            &self.uniforms,
            &mut render_pass,
            entity_slot,
        );
        self.textured_mesh_renderer.paint(
            context,
            &view,
            &self.uniforms,
            &mut render_pass,
            entity_slot,
            backface_culling,
        );
    }
}
