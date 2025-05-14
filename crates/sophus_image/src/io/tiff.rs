use std::{
    format,
    fs::File,
    string::ToString,
};

use tiff::{
    TiffError,
    decoder::{
        Decoder,
        DecodingResult,
    },
    encoder::{
        TiffEncoder,
        TiffValue,
        colortype::ColorType,
    },
};

use crate::{
    DynIntensityMutImage,
    ImageSize,
    MutImage2F32,
    MutImage2U8,
    MutImage2U16,
    MutImage3F32,
    MutImage3U8,
    MutImage3U16,
    MutImage4F32,
    MutImage4U8,
    MutImage4U16,
    MutImageF32,
    MutImageU8,
    MutImageU16,
    intensity_image::intensity_image_view::IsIntensityViewImageF32,
};

fn err_conv(e: TiffError) -> std::io::Error {
    match e {
        TiffError::FormatError(e) => std::io::Error::other(format!("tiff-format: {e}")),
        TiffError::UnsupportedError(e) => std::io::Error::other(format!("tiff-unsupported: {e}")),
        TiffError::IoError(io) => io,
        TiffError::LimitsExceeded => std::io::Error::other("tiff-limits-exceeded"),
        TiffError::IntSizeError => std::io::Error::other("tiff-int-size"),
        TiffError::UsageError(e) => std::io::Error::other(format!("tiff-usage: {e}")),
    }
}

/// Save an f32 image as a TIFF file
pub fn save_as_tiff<'a, ImageView: IsIntensityViewImageF32<'a>>(
    image_f32: &'a ImageView,
    path: impl ToString,
) -> std::io::Result<()>
where
    [<<ImageView as IsIntensityViewImageF32<'a>>::TiffColorType as ColorType>::Inner]: TiffValue,
{
    let mut file = File::create(path.to_string())?;
    let mut tiff = TiffEncoder::new(&mut file).map_err(err_conv)?;
    tiff.write_image::<ImageView::TiffColorType>(
        image_f32.size().width as u32,
        image_f32.size().height as u32,
        image_f32.raw_f32_slice(),
    )
    .map_err(err_conv)?;
    Ok(())
}

/// Load an f32 image from a TIFF file
///
/// Returns a `DynIntensityMutImage` with can be cheaply converted to a `DynIntensityImage`,
/// or an error if the file could not be read or the image format is not supported.
pub fn load_tiff(path: impl ToString) -> std::io::Result<DynIntensityMutImage> {
    let file = File::open(path.to_string())?;

    let mut decoder = Decoder::new(file).map_err(err_conv)?;
    let dims = decoder.dimensions().map_err(err_conv)?;

    let image_size = ImageSize {
        width: dims.0 as usize,
        height: dims.1 as usize,
    };

    let color_type = decoder.colortype().map_err(err_conv)?;

    let num_channels = match color_type {
        tiff::ColorType::Gray(_) => 1,
        tiff::ColorType::GrayA(_) => 2,
        tiff::ColorType::RGB(_) => 3,
        tiff::ColorType::RGBA(_) => 4,
        _ => {
            return Err(std::io::Error::other(format!(
                "unsupported color type: {color_type:?}"
            )));
        }
    };

    let decoding_result: DecodingResult = decoder.read_image().map_err(err_conv)?;

    match decoding_result {
        DecodingResult::U8(u8_img) => {
            if num_channels == 1 {
                return Ok(DynIntensityMutImage::GrayscaleU8(
                    MutImageU8::make_copy_from_size_and_slice(image_size, &u8_img),
                ));
            }
            if num_channels == 2 {
                return Ok(DynIntensityMutImage::GrayscaleAlphaU8(
                    MutImage2U8::make_copy_from_size_and_slice(
                        image_size,
                        bytemuck::cast_slice(&u8_img),
                    ),
                ));
            }
            if num_channels == 3 {
                return Ok(DynIntensityMutImage::RgbU8(
                    MutImage3U8::make_copy_from_size_and_slice(
                        image_size,
                        bytemuck::cast_slice(&u8_img),
                    ),
                ));
            }
            // num_channels == 4
            Ok(DynIntensityMutImage::RgbaU8(
                MutImage4U8::make_copy_from_size_and_slice(
                    image_size,
                    bytemuck::cast_slice(&u8_img),
                ),
            ))
        }
        DecodingResult::U16(u16_img) => {
            if num_channels == 1 {
                return Ok(DynIntensityMutImage::GrayscaleU16(
                    MutImageU16::make_copy_from_size_and_slice(image_size, &u16_img),
                ));
            }
            if num_channels == 2 {
                return Ok(DynIntensityMutImage::GrayscaleAlphaU16(
                    MutImage2U16::make_copy_from_size_and_slice(
                        image_size,
                        bytemuck::cast_slice(&u16_img),
                    ),
                ));
            }
            if num_channels == 3 {
                return Ok(DynIntensityMutImage::RgbU16(
                    MutImage3U16::make_copy_from_size_and_slice(
                        image_size,
                        bytemuck::cast_slice(&u16_img),
                    ),
                ));
            }
            // num_channels == 4
            Ok(DynIntensityMutImage::RgbaU16(
                MutImage4U16::make_copy_from_size_and_slice(
                    image_size,
                    bytemuck::cast_slice(&u16_img),
                ),
            ))
        }
        DecodingResult::F32(f32_img) => {
            if num_channels == 1 {
                return Ok(DynIntensityMutImage::GrayscaleF32(
                    MutImageF32::make_copy_from_size_and_slice(image_size, &f32_img),
                ));
            }
            if num_channels == 2 {
                return Ok(DynIntensityMutImage::GrayscaleAlphaF32(
                    MutImage2F32::make_copy_from_size_and_slice(
                        image_size,
                        bytemuck::cast_slice(&f32_img),
                    ),
                ));
            }
            if num_channels == 3 {
                return Ok(DynIntensityMutImage::RgbF32(
                    MutImage3F32::make_copy_from_size_and_slice(
                        image_size,
                        bytemuck::cast_slice(&f32_img),
                    ),
                ));
            }
            // num_channels == 4
            Ok(DynIntensityMutImage::RgbaF32(
                MutImage4F32::make_copy_from_size_and_slice(
                    image_size,
                    bytemuck::cast_slice(&f32_img),
                ),
            ))
        }
        _ => Err(std::io::Error::other(format!(
            "unsupported decoding: {decoding_result:?}"
        ))),
    }
}
