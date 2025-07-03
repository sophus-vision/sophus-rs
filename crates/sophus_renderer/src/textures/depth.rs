use eframe::wgpu;
use sophus_image::{
    ArcImageF32,
    ImageSize,
    ImageViewF32,
};
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

use crate::{
    FinalRenderResult,
    RenderContext,
    RenderResult,
    camera::ClippingPlanesF32,
    prelude::*,
    textures::{
        depth_image::DepthImage,
        ndc_z_buffer::NdcZBuffer,
        visual_depth::VisualDepthTexture,
    },
};

/// d
#[derive(Debug)]
pub struct DepthTextures {
    pub(crate) main_render_ndc_z_texture: NdcZBuffer,
    pub(crate) staging_buffer: wgpu::Buffer,
    pub(crate) visual_depth_texture: VisualDepthTexture,
}

impl DepthTextures {
    /// There are 4 bytes in a depth pixel.
    pub const BYTES_PER_PIXEL: u32 = 4;

    /// Calculates the number of bytes per image row. This takes wgpu alignment requirements into
    /// account.
    pub fn bytes_per_row(width: u32) -> u32 {
        let unaligned_bytes_per_row = width * Self::BYTES_PER_PIXEL;
        let align = COPY_BYTES_PER_ROW_ALIGNMENT;

        if unaligned_bytes_per_row % align == 0 {
            // already aligned
            unaligned_bytes_per_row
        } else {
            // align to the next multiple of `align`
            (unaligned_bytes_per_row / align + 1) * align
        }
    }

    pub(crate) fn new(render_state: &RenderContext, view_port_size: &ImageSize) -> Self {
        let depth_buffer_size = DepthTextures::bytes_per_row(view_port_size.width as u32)
            * view_port_size.height as u32;
        let staging_buffer = render_state
            .wgpu_device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Depth Buffer Staging"),
                size: depth_buffer_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        DepthTextures {
            staging_buffer,
            main_render_ndc_z_texture: NdcZBuffer::new(render_state, view_port_size),
            visual_depth_texture: VisualDepthTexture::new(render_state, view_port_size),
        }
    }
}

/// Downloads renderer depth texture on the GPU to the CPU. It takes in the intermediate
/// "render_result" and returns the "final_render_result" which includes the downloaded depth image.
///
/// This function is async, since some architectures such as wasm require that.
pub async fn download_depth(
    show_depth: bool,
    clipping_planes: ClippingPlanesF32,
    context: RenderContext,
    view_port_size: &ImageSize,
    render_result: &RenderResult,
) -> FinalRenderResult {
    let bytes_per_row = DepthTextures::bytes_per_row(view_port_size.width as u32);

    let mut command_encoder = context
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    command_encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &render_result.depth_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &render_result.depth_staging_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(view_port_size.height as u32),
            },
        },
        wgpu::Extent3d {
            width: view_port_size.width as u32,
            height: view_port_size.height as u32,
            depth_or_array_layers: 1,
        },
    );

    let device = context.wgpu_device.clone();
    context.wgpu_queue.submit(Some(command_encoder.finish()));

    let buffer_slice = render_result.depth_staging_buffer.slice(..);

    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);

    // Here we await until the image is downloaded.
    rx.receive().await.unwrap().unwrap();
    let depth_image;

    #[allow(unused_assignments)]
    {
        let data = buffer_slice.get_mapped_range();
        let view = ImageViewF32::from_stride_and_slice(
            ImageSize {
                width: view_port_size.width,
                height: view_port_size.height,
            },
            (bytes_per_row / DepthTextures::BYTES_PER_PIXEL) as usize,
            bytemuck::cast_slice(&data[..]),
        );
        depth_image = ArcImageF32::make_copy_from(&view);
    }
    render_result.depth_staging_buffer.unmap();
    let depth_image = DepthImage::new(depth_image, clipping_planes);

    if show_depth {
        let image_rgba = depth_image.color_mapped();

        context.wgpu_queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &render_result.visual_depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(image_rgba.tensor.scalar_view().as_slice().unwrap()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * image_rgba.image_size().width as u32),
                rows_per_image: Some(image_rgba.image_size().height as u32),
            },
            render_result.visual_depth_texture.size(),
        );
    }

    FinalRenderResult {
        rgba_image: render_result.rgba_image.clone(),
        rgba_egui_tex_id: render_result.rgba_egui_tex_id,
        depth_egui_tex_id: render_result.depth_egui_tex_id,
        depth_image,
    }
}
