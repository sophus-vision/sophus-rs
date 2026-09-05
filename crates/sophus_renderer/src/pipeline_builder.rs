use bytemuck::{
    Pod,
    Zeroable,
};
use eframe::wgpu;
use wgpu::DepthStencilState;

use crate::{
    RenderContext,
    prelude::*,
    types::SOPHUS_RENDER_MULTISAMPLE_COUNT,
    uniform_buffers::VertexShaderUniformBuffers,
};

pub(crate) struct TargetTexture {
    pub(crate) rgba_output_format: wgpu::TextureFormat,
}

/// pipeline type
#[derive(Debug)]
pub enum PipelineType {
    /// 2d pixel pipeline
    Pixel,
    /// 3d scene pipeline
    Scene,
}

/// builder
pub struct PipelineBuilder {
    context: RenderContext,
    rgba_target: Arc<TargetTexture>,
    pub(crate) uniforms: Arc<VertexShaderUniformBuffers>,
    pub(crate) pipeline_type: PipelineType,
    depth_stencil: Option<DepthStencilState>,
}

pub(crate) trait IsVertex {
    fn array_stride() -> wgpu::BufferAddress;

    fn step_mode() -> wgpu::VertexStepMode {
        wgpu::VertexStepMode::Vertex
    }

    fn attr() -> Vec<wgpu::VertexAttribute>;
}

/// 2d line vertex - one per segment, expanded into a quad by the vertex shader
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex2 {
    pub(crate) _p0: [f32; 2],
    pub(crate) _p1: [f32; 2],
    pub(crate) _color: [f32; 4],
    pub(crate) _normal: [f32; 2],
    pub(crate) _line_width: f32,
}

impl IsVertex for LineVertex2 {
    fn array_stride() -> wgpu::BufferAddress {
        core::mem::size_of::<LineVertex2>() as wgpu::BufferAddress
    }

    fn step_mode() -> wgpu::VertexStepMode {
        wgpu::VertexStepMode::Instance
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![
            0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Float32x2, 4 => Float32
        ]
        .to_vec()
    }
}

/// 2d point vertex - one per point, expanded into a quad by the vertex shader
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PointVertex2 {
    pub(crate) _pos: [f32; 2],
    pub(crate) _point_size: f32,
    pub(crate) _color: [f32; 4],
}

impl IsVertex for PointVertex2 {
    fn array_stride() -> wgpu::BufferAddress {
        core::mem::size_of::<PointVertex2>() as wgpu::BufferAddress
    }

    fn step_mode() -> wgpu::VertexStepMode {
        wgpu::VertexStepMode::Instance
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x2, 1=>Float32, 2 => Float32x4].to_vec()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MeshVertex3 {
    pub(crate) _pos: [f32; 3],
    pub(crate) _normal: [f32; 3],
    pub(crate) _color: [f32; 4],
}

impl IsVertex for MeshVertex3 {
    fn array_stride() -> wgpu::BufferAddress {
        core::mem::size_of::<MeshVertex3>() as wgpu::BufferAddress
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4].to_vec()
    }
}

/// 3d line vertex - one per segment, expanded into a quad by the vertex shader
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct LineVertex3 {
    pub(crate) _p0: [f32; 3],
    pub(crate) _p1: [f32; 3],
    pub(crate) _color: [f32; 4],
    pub(crate) _line_width: f32,
}

impl IsVertex for LineVertex3 {
    fn array_stride() -> wgpu::BufferAddress {
        core::mem::size_of::<LineVertex3>() as wgpu::BufferAddress
    }

    fn step_mode() -> wgpu::VertexStepMode {
        wgpu::VertexStepMode::Instance
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x3, 1=>Float32x3, 2 => Float32x4, 3 => Float32]
            .to_vec()
    }
}

/// 3d point vertex - one per point, expanded into a quad by the vertex shader
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct PointVertex3 {
    pub(crate) _pos: [f32; 3],
    pub(crate) _point_size: f32,
    pub(crate) _color: [f32; 4],
}

impl IsVertex for PointVertex3 {
    fn array_stride() -> wgpu::BufferAddress {
        core::mem::size_of::<PointVertex3>() as wgpu::BufferAddress
    }

    fn step_mode() -> wgpu::VertexStepMode {
        wgpu::VertexStepMode::Instance
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x3, 1=>Float32, 2 => Float32x4].to_vec()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct TexturedMeshVertex3 {
    pub(crate) _pos: [f32; 3],
    pub(crate) _normal: [f32; 3],
    pub(crate) _tex: [f32; 2],
}

impl IsVertex for TexturedMeshVertex3 {
    fn array_stride() -> wgpu::BufferAddress {
        core::mem::size_of::<TexturedMeshVertex3>() as wgpu::BufferAddress
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2].to_vec()
    }
}

impl PipelineBuilder {
    pub(crate) fn new_pixel(
        context: &RenderContext,
        target: Arc<TargetTexture>,
        uniforms: Arc<VertexShaderUniformBuffers>,
    ) -> Self {
        PipelineBuilder {
            context: context.clone(),
            rgba_target: target.clone(),
            uniforms: uniforms.clone(),
            pipeline_type: PipelineType::Pixel,
            depth_stencil: None,
        }
    }

    pub(crate) fn new_scene(
        context: &RenderContext,
        target: Arc<TargetTexture>,
        uniforms: Arc<VertexShaderUniformBuffers>,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        PipelineBuilder {
            context: context.clone(),
            rgba_target: target.clone(),
            uniforms: uniforms.clone(),
            pipeline_type: PipelineType::Scene,
            depth_stencil,
        }
    }

    pub(crate) fn create<Vertex: IsVertex>(
        &self,
        name: String,
        shader: &wgpu::ShaderModule,
        cull_mode: Option<wgpu::Face>,
    ) -> wgpu::RenderPipeline {
        self.create_with_bind_group_layouts::<Vertex>(name, shader, cull_mode, &[])
    }

    /// Like [Self::create], but with additional bind group layouts bound after the uniforms -
    /// used by pipelines with per-entity resources, such as the textured mesh renderer.
    pub(crate) fn create_with_bind_group_layouts<Vertex: IsVertex>(
        &self,
        name: String,
        shader: &wgpu::ShaderModule,
        cull_mode: Option<wgpu::Face>,
        extra_bind_group_layouts: &[&wgpu::BindGroupLayout],
    ) -> wgpu::RenderPipeline {
        let device = self.context.wgpu_device.clone();

        let mut bind_group_layouts = vec![&self.uniforms.render_bind_group_layout];
        bind_group_layouts.extend_from_slice(extra_bind_group_layouts);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!(
                "`{}` `{:?}` pipeline layout",
                name, self.pipeline_type
            )),
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            cache: None,
            label: Some(&format!("`{}` `{:?}` pipeline", name, self.pipeline_type)),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: Vertex::array_stride(),
                    step_mode: Vertex::step_mode(),
                    attributes: &Vertex::attr(),
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.rgba_target.rgba_output_format,
                    blend: match self.pipeline_type {
                        // 2d renderables are drawn on top of the finished image. Without
                        // blending, a color with alpha < 1 would be written straight into the
                        // target's alpha channel and punch a hole into the view instead of
                        // being composited onto the image.
                        PipelineType::Pixel => Some(wgpu::BlendState::ALPHA_BLENDING),
                        // The scene is rendered into its own texture whose alpha channel *is*
                        // the opacity mask which the distortion pass blends with the background
                        // (`mix(background, foreground, foreground.a)`), so it must be written
                        // through unmodified here.
                        PipelineType::Scene => None,
                    },
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                ..Default::default()
            },
            depth_stencil: self.depth_stencil.clone(),
            multisample: match self.pipeline_type {
                PipelineType::Scene => wgpu::MultisampleState {
                    count: SOPHUS_RENDER_MULTISAMPLE_COUNT,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                PipelineType::Pixel => Default::default(),
            },
            multiview: None,
        })
    }
}
