use crate::dyn_camera::DynCamera;
use nalgebra::SVector;
use sophus_core::calculus::region::IsRegion;
use sophus_core::calculus::region::Region;
use sophus_core::linalg::vector::IsVector;
use sophus_core::linalg::VecF64;
use sophus_image::arc_image::ArcImage2F32;
use sophus_image::image_view::IsImageView;
use sophus_image::interpolation::interpolate;
use sophus_image::mut_image::MutImage2F32;
use sophus_image::mut_image_view::IsMutImageView;

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

/// Returns the undistortion lookup table
pub fn undistort_table(cam: &DynCamera<f64, 1>) -> MutImage2F32 {
    let mut table = MutImage2F32::from_image_size(cam.image_size());
    let w = cam.image_size().width;
    let h = cam.image_size().height;
    for v in 0..h {
        for u in 0..w {
            let pixel = cam.undistort(&VecF64::<2>::from_f64_array([u as f64, v as f64]));
            *table.mut_pixel(u, v) = pixel.cast();
        }
    }
    table
}

/// Returns the distortion lookup table
pub fn distort_table(cam: &DynCamera<f64, 1>) -> DistortTable {
    // first we find min and max values in the proj plane
    // just test the 4 corners might not be enough
    // so we will test the image boundary

    let mut region = Region::<2>::empty();

    let w = cam.image_size().width;
    let h = cam.image_size().height;

    for u in 0..cam.image_size().width {
        // top border
        let v = 0;
        let point_in_proj = cam.undistort(&VecF64::<2>::from_f64_array([u as f64, v as f64]));
        region.extend(&point_in_proj);
        // bottom border
        let v = cam.image_size().height - 1;
        let point_in_proj = cam.undistort(&VecF64::<2>::new(u as f64, v as f64));
        region.extend(&point_in_proj);
    }
    for v in 0..cam.image_size().height {
        // left border
        let u = 0;
        let point_in_proj = cam.undistort(&VecF64::<2>::new(u as f64, v as f64));
        region.extend(&point_in_proj);
        // right border
        let u = cam.image_size().width - 1;
        let point_in_proj = cam.undistort(&VecF64::<2>::new(u as f64, v as f64));
        region.extend(&point_in_proj);
    }
    let region = Region::<2>::from_min_max(region.min().cast() * 2.0, region.max().cast() * 2.0);

    let mut distort_table = DistortTable {
        table: ArcImage2F32::from_image_size_and_val(cam.image_size(), SVector::<f32, 2>::zeros()),
        region,
    };

    let mut table = MutImage2F32::from_image_size(cam.image_size());

    for v in 0..h {
        for u in 0..w {
            let point_proj = VecF64::<2>::new(
                distort_table.offset().x + (u as f64) * distort_table.incr().x,
                distort_table.offset().y + (v as f64) * distort_table.incr().y,
            );
            let pixel = cam.distort(&point_proj);
            *table.mut_pixel(u, v) = SVector::<f32, 2>::new(pixel.cast().x, pixel.cast().y);
        }
    }
    distort_table.table = table.into();
    distort_table
}
