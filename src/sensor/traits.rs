use crate::calculus::types::params::ParamsImpl;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::vector::IsVectorLike;
use crate::calculus::types::MatF64;
use crate::calculus::types::VecF64;
use crate::image::image_view::ImageSize;
use crate::image::mut_image::MutImage2F32;

use super::distortion_table::DistortTable;

/// Camera distortion implementation trait
pub trait IsCameraDistortionImpl<S: IsScalar<1>, const DISTORT: usize, const PARAMS: usize>:
    ParamsImpl<S, PARAMS, 1>
{
    /// identity parameters
    fn identity_params() -> S::Vector<PARAMS> {
        let mut params = S::Vector::<PARAMS>::zero();
        params.set_c(0, 1.0);
        params.set_c(1, 1.0);
        params
    }

    /// Distortion - maps a point in the camera z=1 plane to a distorted point
    fn distort(
        params: &S::Vector<PARAMS>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Vector<2>;

    /// Undistortion - maps a distorted pixel to a point in the camera z=1 plane
    fn undistort(params: &S::Vector<PARAMS>, distorted_point: &S::Vector<2>) -> S::Vector<2>;

    /// Derivative of the distortion w.r.t. the point in the camera z=1 plane
    fn dx_distort_x(
        params: &VecF64<PARAMS>,
        proj_point_in_camera_z1_plane: &VecF64<2>,
    ) -> MatF64<2, 2>;
}

/// Camera projection implementation trait
pub trait IsProjection<S: IsScalar<1>> {
    /// Projects a 3D point in the camera frame to a 2D point in the z=1 plane
    fn proj(point_in_camera: &S::Vector<3>) -> S::Vector<2>;

    /// Unprojects a 2D point in the z=1 plane to a 3D point in the camera frame
    fn unproj(point_in_camera: &S::Vector<2>, extension: S) -> S::Vector<3>;

    /// Derivative of the projection w.r.t. the point in the camera frame
    fn dx_proj_x(point_in_camera: &VecF64<3>) -> MatF64<2, 3>;
}

/// Camera trait
pub trait IsCameraEnum {
    /// Creates a new pinhole camera
    fn new_pinhole(params: &VecF64<4>, image_size: ImageSize) -> Self;
    /// Creates a new Kannala-Brandt camera
    fn new_kannala_brandt(params: &VecF64<8>, image_size: ImageSize) -> Self;

    /// Returns the image size
    fn image_size(&self) -> ImageSize;

    /// Projects a 3D point in the camera frame to a pixel in the image
    fn cam_proj(&self, point_in_camera: &VecF64<3>) -> VecF64<2>;
    /// Unprojects a pixel in the image to a 3D point in the camera frame
    fn cam_unproj_with_z(&self, pixel: &VecF64<2>, z: f64) -> VecF64<3>;
    /// Distortion - maps a point in the camera z=1 plane to a distorted point
    fn distort(&self, proj_point_in_camera_z1_plane: &VecF64<2>) -> VecF64<2>;
    /// Undistortion - maps a distorted pixel to a point in the camera z=1 plane
    fn undistort(&self, pixel: &VecF64<2>) -> VecF64<2>;
    /// Returns the undistortion lookup table
    fn undistort_table(&self) -> MutImage2F32;
    /// Returns the distortion lookup table
    fn distort_table(&self) -> DistortTable;
    /// Derivative of the distortion w.r.t. the point in the camera z=1 plane
    fn dx_distort_x(&self, proj_point_in_camera_z1_plane: &VecF64<2>) -> MatF64<2, 2>;
}
