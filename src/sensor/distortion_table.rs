use crate::calculus::region::IsRegion;
use crate::calculus::region::Region;
use crate::calculus::types::VecF64;
use crate::image::arc_image::ArcImage2F32;
use crate::image::image_view::IsImageView;
use crate::image::interpolation::interpolate;

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
