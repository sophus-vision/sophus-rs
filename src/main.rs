fn main() {
    let _str = "https://www.wwf.org.uk/sites/default/files/styles/gallery_image/public/2019-09/pangolin_with_tongue_out.jpg".to_owned();
    let image_size = sophus_rs::image::ImageSize {
        width: 640,
        height: 480,
    };
    let pixel_format = sophus_rs::image::PixelFormat {
        is_floating_point: false,
        num_channels: 1,
        num_bytes_per_pixel_channel: 1,
    };
    let _image = sophus_rs::glue::ffi::FfiIntensityImage::from_size_and_pixel_format(
        image_size,
        pixel_format,
    );
}
