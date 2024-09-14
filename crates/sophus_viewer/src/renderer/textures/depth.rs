use crate::renderer::textures::main_render_z_buffer::MainRenderZBuffer;
use crate::renderer::textures::visual_depth::VisualDepthTexture;
use crate::RenderContext;
use sophus_image::arc_image::ArcImageF32;
use sophus_image::image_view::ImageViewF32;
use sophus_image::ImageSize;
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

#[derive(Debug)]
pub(crate) struct DepthTextures {
    pub(crate) main_render_ndc_z_texture: MainRenderZBuffer,
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
            main_render_ndc_z_texture: MainRenderZBuffer::new(render_state, view_port_size),
            visual_depth_texture: VisualDepthTexture::new(render_state, view_port_size),
        }
    }

    pub fn download_depth_image(
        &self,
        state: &RenderContext,
        mut command_encoder: wgpu::CommandEncoder,
        view_port_size: &ImageSize,
    ) -> ArcImageF32 {
        let w = view_port_size.width;
        let h = view_port_size.height;

        let bytes_per_row = DepthTextures::bytes_per_row(view_port_size.width as u32);

        // Copy depth texture to staging buffer
        command_encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.main_render_ndc_z_texture.ndc_z_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.staging_buffer,
                layout: wgpu::ImageDataLayout {
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
        buffer_slice.map_async(wgpu::MapMode::Read, move |_result| {});
        device.poll(wgpu::Maintain::Wait);

        let depth_image;

        #[allow(unused_assignments)]
        {
            let data = buffer_slice.get_mapped_range();

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

        depth_image
    }
}
