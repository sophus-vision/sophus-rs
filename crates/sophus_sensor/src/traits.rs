use crate::camera_enum::perspective_camera::UnifiedCamera;
use crate::prelude::*;
use crate::BrownConradyCamera;
use crate::KannalaBrandtCamera;
use crate::PinholeCamera;
use sophus_core::params::ParamsImpl;
use sophus_image::ImageSize;

/// Camera distortion implementation trait
pub trait IsCameraDistortionImpl<
    S: IsScalar<BATCH>,
    const DISTORT: usize,
    const PARAMS: usize,
    const BATCH: usize,
>: ParamsImpl<S, PARAMS, BATCH>
{
    /// identity parameters
    fn identity_params() -> S::Vector<PARAMS> {
        let mut params = S::Vector::<PARAMS>::zeros();
        params.set_elem(0, S::ones());
        params.set_elem(1, S::ones());
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
        params: &S::Vector<PARAMS>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Matrix<2, 2>;

    /// Derivative of the distortion w.r.t. the parameters
    fn dx_distort_params(
        params: &S::Vector<PARAMS>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Matrix<2, PARAMS>;
}

/// Camera projection implementation trait
pub trait IsProjection<S: IsScalar<BATCH>, const BATCH: usize> {
    /// Projects a 3D point in the camera frame to a 2D point in the z=1 plane
    fn proj(point_in_camera: &S::Vector<3>) -> S::Vector<2>;

    /// Unprojects a 2D point in the z=1 plane to a 3D point in the camera frame
    fn unproj(point_in_camera: &S::Vector<2>, extension: S) -> S::Vector<3>;

    /// Derivative of the projection w.r.t. the point in the camera frame
    fn dx_proj_x(point_in_camera: &S::Vector<3>) -> S::Matrix<2, 3>;
}

/// Camera trait
pub trait IsCameraEnum<S: IsScalar<BATCH>, const BATCH: usize> {
    /// Creates a new pinhole camera
    fn new_pinhole(params: &S::Vector<4>, image_size: ImageSize) -> Self;
    /// Creates a new Kannala-Brandt camera
    fn new_kannala_brandt(params: &S::Vector<8>, image_size: ImageSize) -> Self;
    /// Creates a new Brown-Conrady camera
    fn new_brown_conrady(params: &S::Vector<12>, image_size: ImageSize) -> Self;
    /// Creates a new Unified Extended camera
    fn new_unified(params: &S::Vector<6>, image_size: ImageSize) -> Self;

    /// Returns the image size
    fn image_size(&self) -> ImageSize;

    /// Projects a 3D point in the camera frame to a pixel in the image
    fn cam_proj(&self, point_in_camera: &S::Vector<3>) -> S::Vector<2>;
    /// Unprojects a pixel in the image to a 3D point in the camera frame
    fn cam_unproj_with_z(&self, pixel: &S::Vector<2>, z: S) -> S::Vector<3>;
    /// Distortion - maps a point in the camera z=1 plane to a distorted point
    fn distort(&self, proj_point_in_camera_z1_plane: &S::Vector<2>) -> S::Vector<2>;
    /// Undistortion - maps a distorted pixel to a point in the camera z=1 plane
    fn undistort(&self, pixel: &S::Vector<2>) -> S::Vector<2>;
    /// Derivative of the distortion w.r.t. the point in the camera z=1 plane
    fn dx_distort_x(&self, proj_point_in_camera_z1_plane: &S::Vector<2>) -> S::Matrix<2, 2>;

    /// Returns the brown-conrady camera
    fn try_get_brown_conrady(self) -> Option<BrownConradyCamera<S, BATCH>>;

    /// Returns the kannala-brandt camera
    fn try_get_kannala_brandt(self) -> Option<KannalaBrandtCamera<S, BATCH>>;

    /// Returns the pinhole camera
    fn try_get_pinhole(self) -> Option<PinholeCamera<S, BATCH>>;

    /// Returns the unified extended camera
    fn try_get_unified_extended(self) -> Option<UnifiedCamera<S, BATCH>>;
}

/// Dynamic camera trait
pub trait IsPerspectiveCameraEnum<S: IsScalar<BATCH>, const BATCH: usize> {
    /// Return the first four parameters: fx, fy, cx, cy
    fn pinhole_params(&self) -> S::Vector<4>;
}
