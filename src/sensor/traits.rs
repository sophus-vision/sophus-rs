use nalgebra::SMatrix;
use nalgebra::SVector;

use crate::calculus::region::Region;
use crate::calculus::region::RegionTraits;
use crate::calculus::types::params::ParamsImpl;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::vector::IsVectorLike;
use crate::image::arc_image::ArcImage2F32;
use crate::image::interpolation::interpolate;
use crate::image::mut_image::MutImage2F32;
use crate::image::view::ImageSize;
use crate::image::view::IsImageView;

type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

pub trait IsCameraDistortionImpl<S: IsScalar, const DISTORT: usize, const PARAMS: usize>:
    ParamsImpl<S, PARAMS>
{
    fn identity_params() -> S::Vector<PARAMS> {
        let mut params = S::Vector::<PARAMS>::zero();
        params.set_c(0, 1.0);
        params.set_c(1, 1.0);
        params
    }

    fn distort(
        params: &S::Vector<PARAMS>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Vector<2>;

    fn undistort(params: &S::Vector<PARAMS>, distorted_point: &S::Vector<2>) -> S::Vector<2>;

    fn dx_distort_x(params: &V<PARAMS>, proj_point_in_camera_z1_plane: &V<2>) -> M<2, 2>;
}

pub trait IsProjection<S: IsScalar> {
    fn proj(point_in_camera: &S::Vector<3>) -> S::Vector<2>;

    fn unproj(point_in_camera: &S::Vector<2>, extension: S) -> S::Vector<3>;

    fn dx_proj_x(point_in_camera: &V<3>) -> M<2, 3>;
}

#[derive(Debug, Copy, Clone)]
pub struct WrongParamsDim;

#[derive(Debug, Clone)]
pub struct DistortTable {
    pub table: ArcImage2F32,
    pub region: Region<2>,
}

impl DistortTable {
    pub fn incr(&self) -> V<2> {
        V::<2>::new(
            self.region.range().x / ((self.table.image_size().width - 1) as f64),
            self.region.range().y / ((self.table.image_size().height - 1) as f64),
        )
    }

    pub fn offset(&self) -> V<2> {
        self.region.min()
    }

    pub fn lookup(&self, point: &V<2>) -> V<2> {
        let mut norm_point = point - self.offset();
        norm_point.x /= self.incr().x;
        norm_point.y /= self.incr().y;

        let p2 = interpolate(&self.table, norm_point.cast());
        V::<2>::new(p2[0] as f64, p2[1] as f64)
    }
}

pub trait IsCameraEnum {
    fn new_pinhole(params: &V<4>, image_size: ImageSize) -> Self;
    fn new_kannala_brandt(params: &V<8>, image_size: ImageSize) -> Self;

    fn image_size(&self) -> ImageSize;

    fn cam_proj(&self, point_in_camera: &V<3>) -> V<2>;
    fn cam_unproj_with_z(&self, point_in_camera: &V<2>, z: f64) -> V<3>;
    fn distort(&self, point_in_camera: &V<2>) -> V<2>;
    fn undistort(&self, point_in_camera: &V<2>) -> V<2>;
    fn undistort_table(&self) -> MutImage2F32;
    fn distort_table(&self) -> DistortTable;

    fn dx_distort_x(&self, point_in_camera: &V<2>) -> M<2, 2>;
}
