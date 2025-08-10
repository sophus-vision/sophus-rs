use core::f32;

use eframe::{
    egui::{
        self,
    },
    wgpu,
};
use sophus_image::{
    ArcImage4U8,
    ArcImage4U16,
    ArcImageF32,
    ImageSize,
    ImageView4U8,
};
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

use crate::{
    RenderContext,
    types::SOPHUS_RENDER_MULTISAMPLE_COUNT,
};

/// rgba texture
#[derive(Debug)]
pub struct RgbdTexture {
    /// multisample render target - first render pass
    pub multisample_texture: wgpu::Texture,
    /// multisample texture view - first render pass
    pub multisample_texture_view: wgpu::TextureView,
    /// rgba texture - first render pass
    pub resolved_texture: wgpu::Texture,
    /// rgba texture view - first render pass
    pub resolved_texture_view: wgpu::TextureView,
    /// rgba texture
    pub final_texture: wgpu::Texture,
    /// rgba texture view distortedProcess
    pub final_texture_view: wgpu::TextureView,
    pub(crate) egui_tex_id: egui::TextureId,
    pub(crate) render_context: RenderContext,
}

impl Drop for RgbdTexture {
    fn drop(&mut self) {
        self.render_context
            .egui_wgpu_renderer
            .write()
            .free_texture(&self.egui_tex_id);
    }
}

/// rgbd render result
pub struct RgbdRenderResult {
    /// rgba image
    pub rgba_u16: ArcImage4U16,
    /// depth image
    pub depth: ArcImageF32,
    /// min depth
    pub min_depth: f32,
    /// max depth
    pub max_depth: f32,
}

impl RgbdTexture {
    const BYTES_PER_PIXEL_U8: u32 = 4; // r, g, b, a as u8

    pub(crate) fn bytes_per_row_u8(width: u32) -> u32 {
        let unaligned_bytes_per_row = width * Self::BYTES_PER_PIXEL_U8;
        let align = COPY_BYTES_PER_ROW_ALIGNMENT;

        if unaligned_bytes_per_row.is_multiple_of(align) {
            // already aligned
            unaligned_bytes_per_row
        } else {
            // align to the next multiple of `align`Process
            (unaligned_bytes_per_row / align + 1) * align
        }
    }

    pub(crate) fn new(render_state: &RenderContext, view_port_size: &ImageSize) -> Self {
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;

        let multisample_texture =
            render_state
                .wgpu_device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("rgba multisample texture"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: SOPHUS_RENDER_MULTISAMPLE_COUNT,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
                });

        let multisample_texture_view =
            multisample_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let resolved_texture = render_state
            .wgpu_device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("rgba resolved texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
            });

        let resolved_texture_view =
            resolved_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let final_texture = render_state
            .wgpu_device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("rgba final texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
            });

        let final_texture_view = final_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let egui_tex_id = render_state
            .egui_wgpu_renderer
            .write()
            .register_native_texture(
                render_state.wgpu_device.as_ref(),
                &final_texture_view,
                wgpu::FilterMode::Linear,
            );

        RgbdTexture {
            multisample_texture,
            multisample_texture_view,
            final_texture,
            final_texture_view,
            egui_tex_id,
            resolved_texture,
            resolved_texture_view,
            render_context: render_state.clone(),
        }
    }

    /// Method to download ArcImage4U8
    pub fn download(
        &self,
        context: &RenderContext,
        mut command_encoder: wgpu::CommandEncoder,
        view_port_size: &ImageSize,
    ) -> ArcImage4U8 {
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;
        let bytes_per_row = RgbdTexture::bytes_per_row_u8(w);

        let buffer = context.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes_per_row * h) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        command_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.final_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        context.wgpu_queue.submit(Some(command_encoder.finish()));

        #[allow(unused_assignments)]
        let rgba_image;
        {
            // Wait for buffer to be mapped and retrieve data
            let buffer_slice = buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, move |_result| {});
            context.wgpu_device.poll(wgpu::PollType::Wait).unwrap();

            let data = buffer_slice.get_mapped_range();

            let view = ImageView4U8::from_stride_and_slice(
                ImageSize {
                    width: w as usize,
                    height: h as usize,
                },
                (bytes_per_row / Self::BYTES_PER_PIXEL_U8) as usize,
                bytemuck::cast_slice(&data[..]),
            );
            rgba_image = ArcImage4U8::make_copy_from(&view);
        }

        rgba_image
    }
}
