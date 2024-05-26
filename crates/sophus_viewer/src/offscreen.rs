use crate::ViewerRenderState;
use eframe::egui::{self};
use sophus_image::arc_image::ArcImageF32;
use sophus_image::image_view::ImageViewF32;
use sophus_image::ImageSize;
use sophus_sensor::DynCamera;

#[derive(Debug)]
pub(crate) struct OffscreenTexture {
    pub(crate) rgba_texture_view: wgpu::TextureView,
    pub(crate) rgba_tex_id: egui::TextureId,
    pub(crate) depth_render_target: wgpu::Texture,
    pub(crate) depth_output_staging_buffer: wgpu::Buffer,
    pub(crate) depth_texture_view: wgpu::TextureView,
}

impl OffscreenTexture {
    pub(crate) fn new(render_state: &ViewerRenderState, intrinsics: &DynCamera<f64, 1>) -> Self {
        let w = intrinsics.image_size().width as f32;
        let h = intrinsics.image_size().height as f32;

        let render_target = render_state
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: w as u32,
                    height: h as u32,
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

        let depth_texture_data = Vec::<u8>::with_capacity(w as usize * h as usize * 4);

        let depth_render_target = render_state
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: w as u32,
                    height: h as u32,
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
        let depth_output_staging_buffer =
            render_state.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: depth_texture_data.capacity() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

        let depth_texture_view = depth_render_target.create_view(&wgpu::TextureViewDescriptor {
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
            rgba_texture_view: texture_view,
            rgba_tex_id: tex_id,
            depth_render_target,
            depth_output_staging_buffer,
            depth_texture_view,
        }
    }

    // download the depth image from the GPU to ArcImageF32 and copy the rgba texture to the egui texture
    pub fn download_images(
        &self,
        state: &ViewerRenderState,
        mut command_encoder: wgpu::CommandEncoder,
        image_size: &ImageSize,
    ) -> ArcImageF32 {
        let w = image_size.width as f32;
        let h = image_size.height as f32;
        command_encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.depth_render_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.depth_output_staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some((w as u32) * 4),
                    rows_per_image: Some(h as u32),
                },
            },
            wgpu::Extent3d {
                width: w as u32,
                height: h as u32,
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
                    width: w as usize,
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
