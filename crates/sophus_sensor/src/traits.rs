use core::{
    borrow::Borrow,
    fmt::Debug,
};

use sophus_autodiff::params::IsParamsImpl;
use sophus_image::ImageSize;

use crate::{
    camera_enum::perspective_camera::UnifiedCamera,
    prelude::*,
    BrownConradyCamera,
    KannalaBrandtCamera,
    PinholeCamera,
};

/// Camera distortion implementation trait
pub trait IsCameraDistortionImpl<
    S: IsScalar<BATCH, DM, DN>,
    const DISTORT: usize,
    const PARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: IsParamsImpl<S, PARAMS, BATCH, DM, DN> + Debug + Clone + Send + Sync + 'static
{
    /// identity parameters
    fn identity_params() -> S::Vector<PARAMS> {
        let mut params = S::Vector::<PARAMS>::zeros();
        params.set_elem(0, S::ones());
        params.set_elem(1, S::ones());
        params
    }

    /// Distortion - maps a point in the camera z=1 plane to a distorted point
    fn distort<PA, PO>(params: PA, proj_point_in_camera_z1_plane: PO) -> S::Vector<2>
    where
        PA: Borrow<S::Vector<PARAMS>>,
        PO: Borrow<S::Vector<2>>;

    /// Undistortion - maps a distorted pixel to a point in the camera z=1 plane
    fn undistort<PA, PO>(params: PA, distorted_point: PO) -> S::Vector<2>
    where
        PA: Borrow<S::Vector<PARAMS>>,
        PO: Borrow<S::Vector<2>>;

    /// Derivative of the distortion w.r.t. the point in the camera z=1 plane
    fn dx_distort_x<PA, PO>(params: PA, proj_point_in_camera_z1_plane: PO) -> S::Matrix<2, 2>
    where
        PA: Borrow<S::Vector<PARAMS>>,
        PO: Borrow<S::Vector<2>>;

    /// Derivative of the distortion w.r.t. the parameters
    fn dx_distort_params<PA, PO>(
        params: PA,
        proj_point_in_camera_z1_plane: PO,
    ) -> S::Matrix<2, PARAMS>
    where
        PA: Borrow<S::Vector<PARAMS>>,
        PO: Borrow<S::Vector<2>>;
}

/// Camera projection implementation trait
pub trait IsProjection<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: Debug + Clone + Send + Sync + 'static
{
    /// Projects a 3D point in the camera frame to a 2D point in the z=1 plane
    fn proj<P>(point_in_camera: P) -> S::Vector<2>
    where
        P: Borrow<S::Vector<3>>;

    /// Unprojects a 2D point in the z=1 plane to a 3D point in the camera frame
    fn unproj<P>(point_in_camera: P, extension: S) -> S::Vector<3>
    where
        P: Borrow<S::Vector<2>>;

    /// Derivative of the projection w.r.t. the point in the camera frame
    fn dx_proj_x<P>(point_in_camera: P) -> S::Matrix<2, 3>
    where
        P: Borrow<S::Vector<3>>;
}

/// Camera trait
pub trait IsCamera<
    S: IsScalar<BATCH, DM, DN> + 'static + Send + Sync,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>
{
    /// Creates a new pinhole camera
    fn new_pinhole<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<4>>;
    /// Creates a new Kannala-Brandt camera
    fn new_kannala_brandt<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<8>>;
    /// Creates a new Brown-Conrady camera
    fn new_brown_conrady<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<12>>;
    /// Creates a new Unified Extended camera
    fn new_unified<P>(params: P, image_size: ImageSize) -> Self
    where
        P: Borrow<S::Vector<6>>;

    /// Returns the image size
    fn image_size(&self) -> ImageSize;

    /// Projects a 3D point in the camera frame to a pixel in the image
    fn cam_proj<P>(&self, point_in_camera: P) -> S::Vector<2>
    where
        P: Borrow<S::Vector<3>>;
    /// Unprojects a pixel in the image to a 3D point in the camera frame
    fn cam_unproj_with_z<P>(&self, pixel: P, z: S) -> S::Vector<3>
    where
        P: Borrow<S::Vector<2>>;
    /// Distortion - maps a point in the camera z=1 plane to a distorted point
    fn distort<P>(&self, proj_point_in_camera_z1_plane: P) -> S::Vector<2>
    where
        P: Borrow<S::Vector<2>>;
    /// Undistortion - maps a distorted pixel to a point in the camera z=1 plane
    fn undistort<P>(&self, pixel: P) -> S::Vector<2>
    where
        P: Borrow<S::Vector<2>>;
    /// Derivative of the distortion w.r.t. the point in the camera z=1 plane
    fn dx_distort_x<P>(&self, proj_point_in_camera_z1_plane: P) -> S::Matrix<2, 2>
    where
        P: Borrow<S::Vector<2>>;

    /// Returns the brown-conrady camera
    fn try_get_brown_conrady(self) -> Option<BrownConradyCamera<S, BATCH, DM, DN>>;

    /// Returns the kannala-brandt camera
    fn try_get_kannala_brandt(self) -> Option<KannalaBrandtCamera<S, BATCH, DM, DN>>;

    /// Returns the pinhole camera
    fn try_get_pinhole(self) -> Option<PinholeCamera<S, BATCH, DM, DN>>;

    /// Returns the unified extended camera
    fn try_get_unified_extended(self) -> Option<UnifiedCamera<S, BATCH, DM, DN>>;
}

/// Dynamic camera trait
pub trait IsPerspectiveCamera<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>
{
    /// Return the first four parameters: fx, fy, cx, cy
    fn pinhole_params(&self) -> S::Vector<4>;
}
