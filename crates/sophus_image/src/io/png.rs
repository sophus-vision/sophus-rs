use crate::intensity_image::dyn_intensity_image::DynIntensityMutImage;
use crate::intensity_image::intensity_image_view::IsIntensityViewImageU;
use crate::mut_image::MutImage2U16;
use crate::mut_image::MutImage2U8;
use crate::mut_image::MutImage3U16;
use crate::mut_image::MutImage3U8;
use crate::mut_image::MutImage4U16;
use crate::mut_image::MutImage4U8;
use crate::mut_image::MutImageU16;
use crate::mut_image::MutImageU8;
use crate::ImageSize;
use std::fs::File;
use std::io::BufWriter;

/// Save an image of unsigned integers as a PNG file
pub fn save_as_png<'a, ImageView: IsIntensityViewImageU<'a>>(
    image_u: &'a ImageView,
    path: impl ToString,
) -> std::io::Result<()> {
    let file = File::create(path.to_string())?;
    let writer = &mut BufWriter::new(file);

    let mut encoder = png::Encoder::new(
        writer,
        image_u.size().width as u32,
        image_u.size().height as u32,
    );
    encoder.set_color(ImageView::PNG_COLOR_TYPE);
    encoder.set_depth(ImageView::BIT_DEPTH);

    let mut writer = encoder.write_header()?;

    writer.write_image_data(image_u.raw_u8_slice())?;

    Ok(())
}

/// Load an image of unsigned integers from a PNG file
///
/// Returns a `DynIntensityMutImage` with can be cheaply converted to a `DynIntensityImage`,
/// or an error if the file could not be read or the image format is not supported.
pub fn load_png(path: impl ToString) -> std::io::Result<DynIntensityMutImage> {
    let file = File::open(path.to_string())?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()?;

    let mut buf = vec![0; reader.output_buffer_size()];
    // Read the next frame. An APNG might contain multiple frames.
    let info = reader.next_frame(&mut buf)?;
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
            Ok(DynIntensityMutImage::GrayscaleU8(image))
        }
        (png::ColorType::Grayscale, png::BitDepth::Sixteen) => {
            let image = MutImageU16::make_copy_from_size_and_slice(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytemuck::cast_slice(bytes),
            );
            Ok(DynIntensityMutImage::GrayscaleU16(image))
        }
        (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => {
            let image = MutImage2U8::try_make_copy_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytes,
            )
            .map_err(|e| std::io::Error::other(e))?;
            Ok(DynIntensityMutImage::GrayscaleAlphaU8(image))
        }
        (png::ColorType::GrayscaleAlpha, png::BitDepth::Sixteen) => {
            let image = MutImage2U16::try_make_copy_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytes,
            )
            .map_err(|e| std::io::Error::other(e))?;
            Ok(DynIntensityMutImage::GrayscaleAlphaU16(image))
        }
        (png::ColorType::Rgb, png::BitDepth::Eight) => {
            let image = MutImage3U8::try_make_copy_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytes,
            )
            .map_err(|e| std::io::Error::other(e))?;
            Ok(DynIntensityMutImage::RgbU8(image))
        }
        (png::ColorType::Rgb, png::BitDepth::Sixteen) => {
            let image = MutImage3U16::try_make_copy_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytemuck::cast_slice(bytes),
            )
            .map_err(|e| std::io::Error::other(e))?;
            Ok(DynIntensityMutImage::RgbU16(image))
        }
        (png::ColorType::Rgba, png::BitDepth::Eight) => {
            let image = MutImage4U8::try_make_copy_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytes,
            )
            .map_err(|e| std::io::Error::other(e))?;
            Ok(DynIntensityMutImage::RgbaU8(image))
        }
        (png::ColorType::Rgba, png::BitDepth::Sixteen) => {
            let image = MutImage4U16::try_make_copy_from_size_and_bytes(
                ImageSize {
                    width: info.width as usize,
                    height: info.height as usize,
                },
                bytemuck::cast_slice(bytes),
            )
            .map_err(|e| std::io::Error::other(e))?;
            Ok(DynIntensityMutImage::RgbaU16(image))
        }
        _ => Err(std::io::Error::other(format!(
            "Unsupported color type and bit depth: {:?}, {:?}",
            info.color_type, info.bit_depth
        ))),
    }
}
