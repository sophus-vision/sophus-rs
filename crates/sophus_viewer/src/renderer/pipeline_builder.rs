use crate::renderer::uniform_buffers::VertexShaderUniformBuffers;
use crate::RenderContext;
use bytemuck::Pod;
use bytemuck::Zeroable;
use std::sync::Arc;
use wgpu::DepthStencilState;

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

/// 2d line vertex
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex2 {
    pub(crate) _pos: [f32; 2],
    pub(crate) _color: [f32; 4],
    pub(crate) _normal: [f32; 2],
    pub(crate) _line_width: f32,
}

impl IsVertex for LineVertex2 {
    fn array_stride() -> wgpu::BufferAddress {
        std::mem::size_of::<LineVertex2>() as wgpu::BufferAddress
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x2, 1=>Float32x4, 2 => Float32x2, 3 => Float32]
            .to_vec()
    }
}

/// 2d point vertex
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PointVertex2 {
    pub(crate) _pos: [f32; 2],
    pub(crate) _point_size: f32,
    pub(crate) _color: [f32; 4],
}

impl IsVertex for PointVertex2 {
    fn array_stride() -> wgpu::BufferAddress {
        std::mem::size_of::<PointVertex2>() as wgpu::BufferAddress
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x2, 1=>Float32, 2 => Float32x4].to_vec()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MeshVertex3 {
    pub(crate) _pos: [f32; 3],
    pub(crate) _color: [f32; 4],
}

impl IsVertex for MeshVertex3 {
    fn array_stride() -> wgpu::BufferAddress {
        std::mem::size_of::<MeshVertex3>() as wgpu::BufferAddress
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4].to_vec()
    }
}

/// 3d line vertex
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
        std::mem::size_of::<LineVertex3>() as wgpu::BufferAddress
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x3, 1=>Float32x3, 2 => Float32x4, 3 => Float32]
            .to_vec()
    }
}

/// 3d point vertex
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct PointVertex3 {
    pub(crate) _pos: [f32; 3],
    pub(crate) _point_size: f32,
    pub(crate) _color: [f32; 4],
}

impl IsVertex for PointVertex3 {
    fn array_stride() -> wgpu::BufferAddress {
        std::mem::size_of::<PointVertex3>() as wgpu::BufferAddress
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x3, 1=>Float32, 2 => Float32x4].to_vec()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct TexturedMeshVertex3 {
    pub(crate) _pos: [f32; 3],
    pub(crate) _tex: [f32; 2],
}

impl IsVertex for TexturedMeshVertex3 {
    fn array_stride() -> wgpu::BufferAddress {
        std::mem::size_of::<TexturedMeshVertex3>() as wgpu::BufferAddress
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x3, 1=>Float32x2].to_vec()
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
        let device = self.context.wgpu_device.clone();

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!(
                "`{}` `{:?}` pipeline layout",
                name, self.pipeline_type
            )),
            bind_group_layouts: &[&self.uniforms.render_bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("`{}` `{:?}` pipeline", name, self.pipeline_type)),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: Vertex::array_stride(),
                    step_mode: Vertex::step_mode(),
                    attributes: &Vertex::attr(),
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[Some(self.rgba_target.rgba_output_format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                ..Default::default()
            },
            depth_stencil: self.depth_stencil.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }
}
