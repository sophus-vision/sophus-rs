use core::f32;

use crate::RenderContext;
use eframe::egui::{self};
use sophus_core::linalg::SVec;
use sophus_image::arc_image::ArcImage4U16;
use sophus_image::arc_image::ArcImageF32;
use sophus_image::mut_image::MutImage4U16;
use sophus_image::mut_image::MutImageF32;
use sophus_image::prelude::IsImageView;
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
    const BYTES_PER_PIXEL_F32: u32 = 4 * 4; // r, g, b, a as f32
    const BYTES_PER_PIXEL_U8: u32 = 4; // r, g, b, a as u8
    fn bytes_per_row_f32(width: u32) -> u32 {
        let unaligned_bytes_per_row = width * Self::BYTES_PER_PIXEL_F32;
        let align = COPY_BYTES_PER_ROW_ALIGNMENT;

        if unaligned_bytes_per_row % align == 0 {
            // already aligned
            unaligned_bytes_per_row
        } else {
            // align to the next multiple of `align`
            (unaligned_bytes_per_row / align + 1) * align
        }
    }

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
    pub fn download_rgb_and_depth(
        &self,
        state: &RenderContext,
        mut command_encoder: wgpu::CommandEncoder,
        view_port_size: &ImageSize,
    ) -> RgbdRenderResult {
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;
        let bytes_per_row = RgbdTexture::bytes_per_row_f32(w);

        let buffer = state.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes_per_row * h) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // state.wgpu_queue.rite_texture(
        //     wgpu::ImageCopyTexture {
        //         texture: &diffuse_texture,
        //         mip_level: 0,
        //         origin: wgpu::Origin3d::ZERO,
        //         aspect: wgpu::TextureAspect::All,
        //     },
        //     &self.rgba_texture,
        //     wgpu::ImageCopyBuffer {
        //         buffer: &buffer,
        //         layout: wgpu::ImageDataLayout {
        //             offset: 0,
        //             bytes_per_row: Some(bytes_per_row),
        //             rows_per_image: Some(h),
        //         },
        //     },
        //     wgpu::Extent3d {w
        //         width: w,
        //         height: h,
        //         depth_or_array_layers: 1,
        //     },
        // );

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

        
        use sophus_image::image_view::ImageView4F32;
        use sophus_image::mut_image::MutImage4F32;
        use sophus_image::mut_image_view::IsMutImageView;

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

            let view = ImageView4F32::from_stride_and_slice(
                ImageSize {
                    width: w as usize,
                    height: h as usize,
                },
                (bytes_per_row / Self::BYTES_PER_PIXEL_F32) as usize,
                bytemuck::cast_slice(&data[..]),
            );
            maybe_rgba_image = Some(MutImage4F32::make_copy_from(&view));
        }
        

        

        let mut min_depth = f32::MAX;
        let max_depth = 0.0;
        let img = maybe_rgba_image.unwrap();

        let max = 65535;
        let factor = max as f32;
        let mut rgba_u16 = MutImage4U16::from_image_size(ImageSize {
            width: w as usize,
            height: h as usize,
        });
        let mut depth = MutImageF32::from_image_size(ImageSize {
            width: w as usize,
            height: h as usize,
        });
        for v in 0..img.image_size().height {
            for u in 0..img.image_size().width {
                let rgbd = img.pixel(u, v);
                let d = rgbd[3];
                *depth.mut_pixel(u, v) = d;
                min_depth = d.min(min_depth);
                min_depth = d.max(max_depth);
                *rgba_u16.mut_pixel(u, v) = SVec::<u16, 4>::new(
                    (rgbd[0] * factor) as u16,
                    (rgbd[1] * factor) as u16,
                    (rgbd[2] * factor) as u16,
                    max,
                );
            }
        }

        buffer.unmap();

        // command_encoder.copy_texture_to_buffer(
        //     wgpu::ImageCopyTexture {
        //         texture: &self.rgba_texture,
        //         mip_level: 0,
        //         origin: wgpu::Origin3d::ZERO,
        //         aspect: wgpu::TextureAspect::All,
        //     },
        //     wgpu::ImageCopyBuffer {
        //         buffer: &buffer,
        //         layout: wgpu::ImageDataLayout {
        //             offset: 0,
        //             bytes_per_row: Some(bytes_per_row),
        //             rows_per_image: Some(h),
        //         },
        //     },
        //     wgpu::Extent3d {
        //         width: w,
        //         height: h,
        //         depth_or_array_layers: 1,
        //     },
        // );

        // //let foo: ArcImage4U8 = img.to_shared().convert_to::<u8>();
        // println!("downloaded rgba image: {:?}", rgba_u16);

        // let foo = rgba_u16.to_shared();
        // let foo_u8 = foo.clone().convert_to::<u8>();

        // save_as_png(&foo.image_view(), "foo.png");

        // save_as_png(&foo_u8.image_view(), "foo_u8.png");

        RgbdRenderResult {
            rgba_u16: rgba_u16.to_shared(),
            depth: depth.to_shared(),
            min_depth: f32::MAX,
            max_depth: 0.0,
        }
    }
}
