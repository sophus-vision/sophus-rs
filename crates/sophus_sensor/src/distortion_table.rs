use crate::dyn_camera::DynCameraF64;
use crate::prelude::*;
use nalgebra::SVector;
use sophus_autodiff::linalg::VecF64;
use sophus_geo::region::Region;
use sophus_image::arc_image::ArcImage2F32;
use sophus_image::image_view::IsImageView;
use sophus_image::interpolation::interpolate;
use sophus_image::mut_image::MutImage2F32;
use sophus_image::mut_image_view::IsMutImageView;

extern crate alloc;

/// A table of distortion values.
#[derive(Debug, Clone)]
pub struct DistortTable {
    /// The table of distortion values.
    pub table: ArcImage2F32,
    /// The x and y region of the values the table represents.
    pub region: Region<2>,
}

impl DistortTable {
    /// Returns the increment represented by one bin/pixel in the table.
    pub fn incr(&self) -> VecF64<2> {
        VecF64::<2>::new(
            self.region.range().x / ((self.table.image_size().width - 1) as f64),
            self.region.range().y / ((self.table.image_size().height - 1) as f64),
        )
    }

    /// Returns the offset of the table.
    pub fn offset(&self) -> VecF64<2> {
        self.region.min()
    }

    /// Looks up the distortion value for a given point - using bilinear interpolation.
    pub fn lookup(&self, point: &VecF64<2>) -> VecF64<2> {
        let mut norm_point = point - self.offset();
        norm_point.x /= self.incr().x;
        norm_point.y /= self.incr().y;

        let p2 = interpolate(&self.table, norm_point.cast());
        VecF64::<2>::new(p2[0] as f64, p2[1] as f64)
    }
}

/// Returns the distortion lookup table
pub fn distort_table(cam: &DynCameraF64) -> DistortTable {
    // First we find min and max values in the proj plane.
    // Just test the 4 corners might not be enough, so we will test the image boundary.

    let mut region = Region::<2>::empty();

    let v_top = -0.5;
    let v_bottom = cam.image_size().height as f64 - 0.5;

    let u_left = -0.5;
    let u_right = cam.image_size().width as f64 - 0.5;

    for i in 0..=cam.image_size().width {
        let u = i as f64 - 0.5;
        // top border
        region.extend(&cam.undistort(VecF64::<2>::new(u, v_top)));
        // bottom border
        region.extend(&cam.undistort(VecF64::<2>::new(u, v_bottom)));
    }
    for i in 0..=cam.image_size().height {
        let v = i as f64 - 0.5;
        // left border
        region.extend(&cam.undistort(VecF64::<2>::new(u_left, v)));
        // right border
        region.extend(&cam.undistort(VecF64::<2>::new(u_right, v)));
    }

    let mid = region.mid();
    let range = region.range();

    let larger_region = Region::<2>::from_min_max(mid - range, mid + range);

    let mut distort_table = DistortTable {
        table: ArcImage2F32::from_image_size_and_val(cam.image_size(), SVector::<f32, 2>::zeros()),
        region: larger_region,
    };

    let mut table = MutImage2F32::from_image_size(cam.image_size());

    for v in 0..cam.image_size().height {
        for u in 0..cam.image_size().width {
            let point_proj = VecF64::<2>::new(
                distort_table.offset().x + (u as f64) * distort_table.incr().x,
                distort_table.offset().y + (v as f64) * distort_table.incr().y,
            );
            let pixel = cam.distort(point_proj);
            *table.mut_pixel(u, v) = SVector::<f32, 2>::new(pixel.cast().x, pixel.cast().y);
        }
    }
    distort_table.table = table.into();
    distort_table
}

#[test]
fn camera_distortion_table_tests() {
    use crate::distortion_table::distort_table;
    use crate::distortion_table::DistortTable;
    use approx::assert_abs_diff_eq;
    use approx::assert_relative_eq;
    use sophus_autodiff::linalg::VecF64;
    use sophus_autodiff::linalg::EPS_F64;
    use sophus_autodiff::maps::vector_valued_maps::VectorValuedVectorMap;
    use sophus_image::ImageSize;

    {
        let mut cameras: alloc::vec::Vec<DynCameraF64> = alloc::vec![];
        cameras.push(DynCameraF64::new_pinhole(
            VecF64::<4>::new(600.0, 600.0, 1.0, 0.5),
            ImageSize {
                width: 3,
                height: 2,
            },
        ));

        cameras.push(DynCameraF64::new_kannala_brandt(
            VecF64::<8>::from_vec(alloc::vec![
                1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001
            ]),
            ImageSize {
                width: 640,
                height: 480,
            },
        ));

        for camera in cameras {
            let pixels_in_image = alloc::vec![
                VecF64::<2>::new(0.0, 0.0),
                VecF64::<2>::new(2.0, 1.0),
                VecF64::<2>::new(2.9, 1.9),
                VecF64::<2>::new(2.5, 1.5),
                VecF64::<2>::new(3.0, 2.0),
                VecF64::<2>::new(-0.5, -0.5),
                VecF64::<2>::new(1.0, 400.0),
                VecF64::<2>::new(320.0, 240.0),
                VecF64::<2>::new(319.5, 239.5),
                VecF64::<2>::new(100.0, 40.0),
                VecF64::<2>::new(639.0, 479.0),
            ];

            for pixel in pixels_in_image.clone() {
                if pixel[0] > camera.image_size().width as f64
                    || pixel[1] > camera.image_size().height as f64
                {
                    continue;
                }

                for d in [1.0, 0.1, 0.5, 1.1, 3.0, 15.0] {
                    let point_in_camera = camera.cam_unproj_with_z(pixel, d);
                    assert_relative_eq!(point_in_camera[2], d, epsilon = EPS_F64);

                    let pixel_in_image2 = camera.cam_proj(point_in_camera);
                    assert_relative_eq!(pixel_in_image2, pixel, epsilon = EPS_F64);
                }
                let ab_in_z1plane = camera.undistort(pixel);

                let pixel_in_image3 = camera.distort(ab_in_z1plane);
                assert_relative_eq!(pixel_in_image3, pixel, epsilon = EPS_F64);

                let dx = camera.dx_distort_x(pixel);
                let numeric_dx = VectorValuedVectorMap::sym_diff_quotient_jacobian(
                    |x: VecF64<2>| camera.distort(x),
                    pixel,
                    EPS_F64,
                );

                assert_relative_eq!(dx, numeric_dx, epsilon = 1e-4);
            }

            let table: DistortTable = distort_table(&camera);

            for pixel in pixels_in_image {
                if pixel[0] >= camera.image_size().width as f64
                    || pixel[1] >= camera.image_size().height as f64
                {
                    continue;
                }
                let proj = camera.undistort(pixel);
                let analytic_pixel = camera.distort(proj);
                let lut_pixel = table.lookup(&proj);

                assert_abs_diff_eq!(analytic_pixel, pixel, epsilon = 1e-3);
                assert_abs_diff_eq!(analytic_pixel, lut_pixel, epsilon = 1e-3);
            }
        }
    }
}
