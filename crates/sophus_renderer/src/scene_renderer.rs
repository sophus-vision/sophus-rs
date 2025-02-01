/// distortion renderer
pub mod distortion;
/// line renderer
pub mod line;
/// mesh renderer
pub mod mesh;
/// point renderer
pub mod point;
/// textured mesh renderer
pub mod textured_mesh;

use sophus_lie::Isometry3F64;
use wgpu::DepthStencilState;

use crate::{
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
        depth::DepthTextures,
        rgba::RgbdTexture,
    },
    uniform_buffers::VertexShaderUniformBuffers,
    RenderContext,
};

/// Scene renderer
pub struct SceneRenderer {
    /// uniforms
    pub uniforms: Arc<VertexShaderUniformBuffers>,
    /// Mesh renderer
    pub mesh_renderer: MeshRenderer,
    // /// Textured mesh renderer
    // pub textured_mesh_renderer: textured_mesh::TexturedMeshRenderer,
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
            // textured_mesh_renderer: TexturedMeshRenderer::new(
            //     render_context,
            //     &scene_pipeline_builder,
            // ),
            world_from_scene: Isometry3F64::identity(),
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        state: &RenderContext,
        scene_from_camera: &Isometry3F64,
        command_encoder: &'rp mut wgpu::CommandEncoder,
        rgba: &'rp RgbdTexture,
        depth: &DepthTextures,
        backface_culling: bool,
    ) {
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &rgba.multisample_texture_view,
                resolve_target: Some(&rgba.resolved_texture_view),
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
                view: &depth.main_render_ndc_z_texture.multisample_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_bind_group(0, &self.uniforms.render_bind_group, &[]);

        self.mesh_renderer.paint(
            state,
            scene_from_camera,
            &self.world_from_scene,
            &self.uniforms,
            &mut render_pass,
            backface_culling,
        );
        self.point_renderer.paint(
            state,
            scene_from_camera,
            &self.world_from_scene,
            &self.uniforms,
            &mut render_pass,
        );
        self.line_renderer.paint(
            state,
            scene_from_camera,
            &self.world_from_scene,
            &self.uniforms,
            &mut render_pass,
        );
        // self.textured_mesh_renderer.paint(
        //     state,
        //     scene_from_camera,
        //     &self.uniforms,
        //     &mut render_pass,
        // );
    }
}
