use core::f32;

use crate::RenderContext;
use eframe::egui::{self};
use sophus_image::arc_image::ArcImage4U16;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::arc_image::ArcImageF32;
use sophus_image::image_view::ImageView4U8;
use sophus_image::ImageSize;
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// rgba texture
#[derive(Debug)]
pub struct RgbdTexture {
    /// rgba texture
    pub rgba_texture: wgpu::Texture,
    /// rgba texture view
    pub rgba_texture_view: wgpu::TextureView,
    /// rgba texture
    pub rgba_texture_distorted: wgpu::Texture,
    /// rgba texture view distorted
    pub rgba_texture_view_distorted: wgpu::TextureView,
    pub(crate) egui_tex_id: egui::TextureId,
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

        if unaligned_bytes_per_row % align == 0 {
            // already aligned
            unaligned_bytes_per_row
        } else {
            // align to the next multiple of `align`
            (unaligned_bytes_per_row / align + 1) * align
        }
    }

    pub(crate) fn new(render_state: &RenderContext, view_port_size: &ImageSize) -> Self {
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;

        let render_target = render_state
            .wgpu_device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[wgpu::TextureFormat::Rgba32Float],
            });

        let texture_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

        let render_target_distorted =
            render_state
                .wgpu_device
                .create_texture(&wgpu::TextureDescriptor {
                    label: None,
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

        let texture_view_distorted =
            render_target_distorted.create_view(&wgpu::TextureViewDescriptor::default());

        let tex_id = render_state
            .egui_wgpu_renderer
            .write()
            .register_native_texture(
                render_state.wgpu_device.as_ref(),
                &texture_view_distorted,
                wgpu::FilterMode::Linear,
            );

        RgbdTexture {
            rgba_texture: render_target,
            rgba_texture_view: texture_view,
            rgba_texture_distorted: render_target_distorted,
            rgba_texture_view_distorted: texture_view_distorted,
            egui_tex_id: tex_id,
        }
    }

    /// Method to download ArcImage4U8
    pub fn download(
        &self,
        state: &RenderContext,
        mut command_encoder: wgpu::CommandEncoder,
        view_port_size: &ImageSize,
    ) -> ArcImage4U8 {
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;
        let bytes_per_row = RgbdTexture::bytes_per_row_u8(w);

        let buffer = state.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes_per_row * h) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        command_encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.rgba_texture_distorted,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
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

        state.wgpu_queue.submit(Some(command_encoder.finish()));

        #[allow(unused_assignments)]
        let rgba_image;
        {
            // Wait for buffer to be mapped and retrieve data
            let buffer_slice = buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            state.wgpu_device.poll(wgpu::Maintain::Wait);
            rx.try_recv().unwrap().unwrap();

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
