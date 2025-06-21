use eframe::wgpu;
use log::info;
use sophus_image::{
    ArcImageF32,
    ImageSize,
    ImageViewF32,
};
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

use crate::{
    RenderContext,
    camera::ClippingPlanesF64,
    prelude::*,
    textures::{
        depth_image::DepthImage,
        ndc_z_buffer::NdcZBuffer,
        visual_depth::VisualDepthTexture,
    },
};

#[derive(Debug)]
pub(crate) struct DepthTextures {
    pub(crate) main_render_ndc_z_texture: NdcZBuffer,
    pub(crate) staging_buffer: wgpu::Buffer,
    pub(crate) visual_depth_texture: VisualDepthTexture,
}

impl DepthTextures {
    pub const BYTES_PER_PIXEL: u32 = 4;
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

        info!("depth_buffer_size!!! {depth_buffer_size}");
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

    /// Uploads the depth image to the visual depth texture.
    pub fn compute_visual_depth_texture(&self, state: &RenderContext, depth_image: &DepthImage) {
        let image_rgba = depth_image.color_mapped();

        state.wgpu_queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.visual_depth_texture.visual_texture,
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
            self.visual_depth_texture.visual_texture.size(),
        );
    }

    /// Downloads the depth image from the main render z buffer texture.
    pub fn download_depth_image(
        &self,
        state: &RenderContext,
        mut command_encoder: wgpu::CommandEncoder,
        view_port_size: &ImageSize,
        clipping_planes: &ClippingPlanesF64,
    ) -> DepthImage {
        info!("start depth!!!");

        let w = view_port_size.width;
        let h = view_port_size.height;

        let bytes_per_row = DepthTextures::bytes_per_row(view_port_size.width as u32);

        // Copy depth texture to staging buffer
        command_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.main_render_ndc_z_texture.final_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging_buffer,
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

        // Submit command encoder and wait for GPU
        let device = state.wgpu_device.clone();
        state.wgpu_queue.submit(Some(command_encoder.finish()));

        // Read staging buffer
        let buffer_slice = self.staging_buffer.slice(..);
        log::info!("111");

        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            if r.is_err() {
                log::info!("map_async failed – most likely the copy was out-of-bounds");
            }
        });
        log::info!("222");

        device.poll(wgpu::Maintain::Wait);
        log::info!("333");

        let depth_image;

        #[allow(unused_assignments)]
        {
            let data = buffer_slice.get_mapped_range();
            log::info!("444");

            let view = ImageViewF32::from_stride_and_slice(
                ImageSize {
                    width: w,
                    height: h,
                },
                (bytes_per_row / DepthTextures::BYTES_PER_PIXEL) as usize,
                bytemuck::cast_slice(&data[..]),
            );
            depth_image = ArcImageF32::make_copy_from(&view);
        }
        self.staging_buffer.unmap();
        info!("end depth!!!");

        DepthImage::new(depth_image, clipping_planes.cast())
    }
}
