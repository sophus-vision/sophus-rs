use sophus_image::arc_image::ArcImage4U8;
use sophus_image::mut_image::MutImage4U8;
use sophus_image::mut_image_view::IsMutImageView;
use sophus_image::ImageSize;

/// Makes example image of image-size
pub fn make_example_image(image_size: ImageSize) -> ArcImage4U8 {
    let mut img = MutImage4U8::from_image_size_and_val(
        image_size,
        nalgebra::SVector::<u8, 4>::new(255, 0, 255, 255),
    );

    let w = image_size.width;
    let h = image_size.height;

    for i in 0..10 {
        for j in 0..10 {
            img.mut_pixel(i, j).copy_from_slice(&[0, 0, 0, 255]);
            img.mut_pixel(i, h - j - 1)
                .copy_from_slice(&[255, 255, 255, 255]);
            img.mut_pixel(w - i - 1, h - j - 1)
                .copy_from_slice(&[0, 0, 255, 255]);
        }
    }
    img.to_shared()
}
