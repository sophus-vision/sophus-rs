use sophus_renderer::camera::properties::RenderCameraProperties;
use sophus_renderer::renderables::frame::ImageFrame;
use sophus_core::linalg::SVec;
use sophus_core::linalg::VecF64;
use sophus_core::prelude::IsVector;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::mut_image::MutImage4U8;
use sophus_image::mut_image_view::IsMutImageView;
use sophus_image::ImageSize;
use sophus_sensor::dyn_camera::DynCameraF64;

/// Makes example image of image-size
pub fn make_example_image(image_size: ImageSize) -> ArcImage4U8 {
    let mut img =
        MutImage4U8::from_image_size_and_val(image_size, SVec::<u8, 4>::new(255, 255, 255, 255));

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

pub fn make_distorted_frame() -> ImageFrame {
    let focal_length = 500.0;

    let image_size = ImageSize::new(638, 479);
    let cx = 320.0;
    let cy = 240.0;

    let unified_cam = DynCameraF64::new_unified(
        &VecF64::from_array([focal_length, focal_length, cx, cy, 0.629, 1.22]),
        image_size,
    );

    let mut img =
        MutImage4U8::from_image_size_and_val(image_size, SVec::<u8, 4>::new(255, 255, 255, 255));

    for v in 0..image_size.height {
        for u in 0..image_size.width {
            let uv = VecF64::<2>::new(u as f64, v as f64);
            let p_on_z1 = unified_cam.cam_unproj(&uv);

            if p_on_z1[0].abs() < 0.5 {
                *img.mut_pixel(u, v) = SVec::<u8, 4>::new(255, 0, 0, 255);

                if p_on_z1[1].abs() < 0.3 {
                    *img.mut_pixel(u, v) = SVec::<u8, 4>::new(0, 0, 255, 255);
                }
            }
        }
    }

    ImageFrame {
        image: Some(img.to_shared()),
        camera_properties: RenderCameraProperties::from_intrinsics(&unified_cam),
    }
}
