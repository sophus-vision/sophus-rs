use std::num::NonZeroU32;

use crate::RenderContext;
use eframe::egui::{self};
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::arc_image::ArcImageF32;
use sophus_image::image_view::ImageView4U8;
use sophus_image::image_view::ImageViewF32;
use sophus_image::ImageSize;
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// rgba texture
#[derive(Debug)]
pub struct RgbaTexture {
    /// rgba texture
    pub rgba_texture: wgpu::Texture,
    pub(crate) rgba_texture_view: wgpu::TextureView,
    pub(crate) egui_tex_id: egui::TextureId,
}

impl RgbaTexture {
    const BYTES_PER_PIXEL: u32 = 4; // r, g, b, a as u8
    fn bytes_per_row(width: u32) -> u32 {
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
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
            });

        let texture_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());
        let tex_id = render_state
            .egui_wgpu_renderer
            .write()
            .register_native_texture(
                render_state.wgpu_device.as_ref(),
                &texture_view,
                wgpu::FilterMode::Linear,
            );

        RgbaTexture {
            rgba_texture: render_target,
            rgba_texture_view: texture_view,
            egui_tex_id: tex_id,
        }
    }

    /// Method to download ArcImage4U8
    pub fn download_rgba_image(
        &self,
        state: &RenderContext,
        mut command_encoder: wgpu::CommandEncoder,
        view_port_size: &ImageSize,
    ) -> ArcImage4U8 {
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;
        let bytes_per_row = RgbaTexture::bytes_per_row(w);

        let buffer = state.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes_per_row * h) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        command_encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.rgba_texture,
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
        let mut maybe_rgba_image = None;
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
                (bytes_per_row / Self::BYTES_PER_PIXEL) as usize,
                bytemuck::cast_slice(&data[..]),
            );
            maybe_rgba_image = Some(ArcImage4U8::make_copy_from(&view));
        }

        buffer.unmap();

        maybe_rgba_image.unwrap()
    }
}

#[derive(Debug)]
pub(crate) struct ZBufferTexture {
    pub(crate) render_path_ndc_z_texture: wgpu::Texture,
    pub(crate) render_path_ndc_z_texture_view: wgpu::TextureView,
    pub(crate) staging_buffer: wgpu::Buffer,
}

impl ZBufferTexture {
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
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;
        let bytes_per_row = ZBufferTexture::bytes_per_row(w);

        pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

        let size = wgpu::Extent3d {
            width: view_port_size.width as u32,
            height: view_port_size.height as u32,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = render_state.wgpu_device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_buffer_size = ZBufferTexture::bytes_per_row(view_port_size.width as u32)
            * view_port_size.height as u32;
        let staging_buffer = render_state
            .wgpu_device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Depth Buffer Staging"),
                size: depth_buffer_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        ZBufferTexture {
            staging_buffer,
            render_path_ndc_z_texture: texture,
            render_path_ndc_z_texture_view: view,
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

        let bytes_per_row = ZBufferTexture::bytes_per_row(view_port_size.width as u32);

        // Copy depth texture to staging buffer
        command_encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.render_path_ndc_z_texture,
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
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {});
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
                (bytes_per_row / ZBufferTexture::BYTES_PER_PIXEL) as usize,
                bytemuck::cast_slice(&data[..]),
            );
            depth_image = ArcImageF32::make_copy_from(&view);
        }
        self.staging_buffer.unmap();

        depth_image
    }
}

#[derive(Debug)]
pub(crate) struct DepthTexture {
    pub visual_texture: wgpu::Texture,
    pub(crate) egui_tex_id: egui::TextureId,
}

impl DepthTexture {
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

    /// Create a new depth renderer.
    pub fn new(render_state: &RenderContext, view_port_size: &ImageSize) -> Self {
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;
        let bytes_per_row = DepthTexture::bytes_per_row(w);

        let required_buffer_size = bytes_per_row * h; // Total bytes needed in the buffer

        let visual_texture = render_state
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
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
            });

        let visual_texture_view =
            visual_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let egui_tex_id = render_state
            .egui_wgpu_renderer
            .write()
            .register_native_texture(
                render_state.wgpu_device.as_ref(),
                &visual_texture_view,
                wgpu::FilterMode::Linear,
            );
        Self {
            visual_texture,
            egui_tex_id,
        }
    }
}

#[derive(Debug)]
pub(crate) struct OffscreenTextures {
    pub(crate) view_port_size: ImageSize,
    pub(crate) rgba: RgbaTexture,
    pub z_buffer: ZBufferTexture,
    pub(crate) depth: DepthTexture,
}

impl OffscreenTextures {
    pub(crate) fn new(render_state: &RenderContext, view_port_size: &ImageSize) -> Self {
        Self {
            view_port_size: *view_port_size,
            rgba: RgbaTexture::new(render_state, view_port_size),
            z_buffer: ZBufferTexture::new(render_state, view_port_size),
            depth: DepthTexture::new(render_state, view_port_size),
        }
    }
}
