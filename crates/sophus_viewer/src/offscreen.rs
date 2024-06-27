use std::num::NonZeroU32;

use crate::ViewerRenderState;
use eframe::egui::{self};
use sophus_image::arc_image::ArcImageF32;
use sophus_image::image_view::ImageViewF32;
use sophus_image::ImageSize;
use wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

#[derive(Debug)]
pub(crate) struct RgbaTexture {
    pub(crate) rgba_texture_view: wgpu::TextureView,
    pub(crate) rgba_tex_id: egui::TextureId,
}

impl RgbaTexture {
    pub(crate) fn new(render_state: &ViewerRenderState, view_port_size: &ImageSize) -> Self {
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;

        let render_target = render_state
            .device
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
        let tex_id = render_state.wgpu_state.write().register_native_texture(
            render_state.device.as_ref(),
            &texture_view,
            wgpu::FilterMode::Linear,
        );

        RgbaTexture {
            rgba_texture_view: texture_view,
            rgba_tex_id: tex_id,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ZBufferTexture {
    pub(crate) _depth_texture: wgpu::Texture,
    pub(crate) depth_texture_view: wgpu::TextureView,
    pub(crate) _depth_texture_sampler: wgpu::Sampler,
}

impl ZBufferTexture {
    pub(crate) fn new(render_state: &ViewerRenderState, view_port_size: &ImageSize) -> Self {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT // 3.
                        | wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let texture = render_state.device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = render_state
            .device
            .create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                lod_min_clamp: 0.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            });

        ZBufferTexture {
            _depth_texture: texture,
            depth_texture_view: view,
            _depth_texture_sampler: sampler,
        }
    }
}

#[derive(Debug)]
pub(crate) struct DepthTexture {
    pub(crate) depth_output_staging_buffer: wgpu::Buffer,
    pub(crate) depth_render_target_f32: wgpu::Texture,
    pub(crate) depth_texture_view_f32: wgpu::TextureView,
}

impl DepthTexture {
    const BYTES_PER_PIXEL: u32 = 4;
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

    /// Create a new depth renderer.
    pub fn new(render_state: &ViewerRenderState, view_port_size: &ImageSize) -> Self {
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;
        let bytes_per_row = DepthTexture::bytes_per_row(w);

        let required_buffer_size = bytes_per_row * h; // Total bytes needed in the buffer

        let depth_output_staging_buffer =
            render_state.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: required_buffer_size as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

        let depth_render_target_f32 =
            render_state
                .device
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
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[wgpu::TextureFormat::R32Float],
                });

        let depth_texture_view_f32 =
            depth_render_target_f32.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: Some(wgpu::TextureFormat::R32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
            });
        Self {
            depth_output_staging_buffer,
            depth_texture_view_f32,
            depth_render_target_f32,
        }
    }

    // download the depth image from the GPU to ArcImageF32
    //
    // Note: The depth image return might have a greater width than the requested width, to ensure
    //       that the bytes per row is a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (i.e. 256).
    pub fn download_image(
        &self,
        state: &ViewerRenderState,
        mut command_encoder: wgpu::CommandEncoder,
        view_port_size: &ImageSize,
    ) -> ArcImageF32 {
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;

        let bytes_per_row = DepthTexture::bytes_per_row(w);

        command_encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.depth_render_target_f32,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.depth_output_staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(bytes_per_row).unwrap().get()),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        state.queue.submit(Some(command_encoder.finish()));

        #[allow(unused_assignments)]
        let mut maybe_depth_image = None;
        {
            let buffer_slice = self.depth_output_staging_buffer.slice(..);

            let (tx, mut rx) = tokio::sync::oneshot::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            state.device.poll(wgpu::Maintain::Wait);
            rx.try_recv().unwrap().unwrap();

            let data = buffer_slice.get_mapped_range();

            let view = ImageViewF32::from_size_and_slice(
                ImageSize {
                    width: (bytes_per_row / Self::BYTES_PER_PIXEL) as usize,
                    height: h as usize,
                },
                bytemuck::cast_slice(&data[..]),
            );
            let img = ArcImageF32::make_copy_from(&view);
            maybe_depth_image = Some(img);
        }
        self.depth_output_staging_buffer.unmap();

        maybe_depth_image.unwrap()
    }
}

#[derive(Debug)]
pub(crate) struct OffscreenTextures {
    view_port_size: ImageSize,
    pub(crate) rgba: RgbaTexture,
    pub(crate) z_buffer: ZBufferTexture,
    pub(crate) depth: DepthTexture,
}

impl OffscreenTextures {
    pub(crate) fn view_port_size(&self) -> &ImageSize {
        &self.view_port_size
    }
    pub(crate) fn new(render_state: &ViewerRenderState, view_port_size: &ImageSize) -> Self {
        Self {
            view_port_size: *view_port_size,
            rgba: RgbaTexture::new(render_state, view_port_size),
            z_buffer: ZBufferTexture::new(render_state, view_port_size),
            depth: DepthTexture::new(render_state, view_port_size),
        }
    }
}
