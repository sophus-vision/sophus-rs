use std::fs::File;
use std::io::BufWriter;

use crate::image::mut_image::MutImage2U16;

use super::mut_image::MutImage2U8;
use super::mut_image::MutImage3U16;
use super::mut_image::MutImage3U8;
use super::mut_image::MutImage4U16;
use super::mut_image::MutImage4U8;
use super::mut_image::MutImageU16;
use super::mut_image::MutImageU8;
use super::percentage_image::DynPercentageMutImage;
use super::percentage_image::PercentageViewImageU;
use super::view::ImageSize;

pub fn save_as_png<'a, GenImageView: PercentageViewImageU<'a>>(
    image_u8: &'a GenImageView,
    path: &std::path::Path,
) {
    let file = File::create(path).unwrap();
    let ref mut writer = BufWriter::new(file);

    let mut encoder = png::Encoder::new(
        writer,
        image_u8.size().width as u32,
        image_u8.size().height as u32,
    );
    encoder.set_color(GenImageView::COLOR_TYPE);
    encoder.set_depth(GenImageView::BIT_DEPTH);

    let mut writer = encoder.write_header().unwrap();

    writer.write_image_data(&image_u8.u8_slice()).unwrap();
}

pub fn load_png(path: &std::path::Path) -> Result<DynPercentageMutImage, String> {
    let file = File::open(path).or_else(|e| Err(e.to_string()))?;
    let decoder = png::Decoder::new(file);
    // let (info, mut reader) = decoder.read_info().or_else(|e| Err(e.to_string()))?;
    let mut reader = decoder.read_info().or_else(|e| Err(e.to_string()))?;

    let mut buf = vec![0; reader.output_buffer_size()];
    // Read the next frame. An APNG might contain multiple frames.
    let info = reader
        .next_frame(&mut buf)
        .or_else(|e| Err(e.to_string()))?;
    // Grab the bytes of the image.
    let bytes = &buf[..info.buffer_size()];

    match (info.color_type, info.bit_depth) {
        (png::ColorType::Grayscale, png::BitDepth::Eight) => {
            let image = MutImageU8::make_copy_from_size_and_slice(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytes,
            );
            Ok(DynPercentageMutImage::GrayscaleU8(image))
        }
        (png::ColorType::Grayscale, png::BitDepth::Sixteen) => {
            let image = MutImageU16::make_copy_from_size_and_slice(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytemuck::cast_slice(bytes),
            );
            Ok(DynPercentageMutImage::GrayscaleU16(image))
        }
        (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => {
            let image = MutImage2U8::try_make_copy_from_make_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytes,
            )?;
            Ok(DynPercentageMutImage::GrayscaleAlphaU8(image))
        }
        (png::ColorType::GrayscaleAlpha, png::BitDepth::Sixteen) => {
            let image = MutImage2U16::try_make_copy_from_make_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytes,
            )?;
            Ok(DynPercentageMutImage::GrayscaleAlphaU16(image))
        }
        (png::ColorType::Rgb, png::BitDepth::Eight) => {
            let image = MutImage3U8::try_make_copy_from_make_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytes,
            )?;
            Ok(DynPercentageMutImage::RgbU8(image))
        }
        (png::ColorType::Rgb, png::BitDepth::Sixteen) => {
            let image = MutImage3U16::try_make_copy_from_make_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytemuck::cast_slice(bytes),
            )?;
            Ok(DynPercentageMutImage::RgbU16(image))
        }
        (png::ColorType::Rgba, png::BitDepth::Eight) => {
            let image = MutImage4U8::try_make_copy_from_make_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytes,
            )?;
            Ok(DynPercentageMutImage::RgbaU8(image))
        }
        (png::ColorType::Rgba, png::BitDepth::Sixteen) => {
            let image = MutImage4U16::try_make_copy_from_make_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytemuck::cast_slice(bytes),
            )?;
            Ok(DynPercentageMutImage::RgbaU16(image))
        }
        _ => Err(format!(
            "Unsupported color type and bit depth: {:?}, {:?}",
            info.color_type, info.bit_depth
        )),
    }
    .or_else(|e| Err(e.to_string()))
}
