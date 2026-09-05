use core::{
    borrow::Borrow,
    fmt::Debug,
};

use sophus_autodiff::params::IsParamsImpl;
use sophus_geo::UnitVector3;
use sophus_image::ImageSize;

use crate::{
    BrownConradyCamera,
    KannalaBrandtCamera,
    PinholeCamera,
    camera_enum::EnhancedUnifiedCamera,
    prelude::*,
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
        *params.elem_mut(0) = S::ones();
        *params.elem_mut(1) = S::ones();
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

    /// Unprojection - maps a distorted pixel to the direction it observes.
    ///
    /// Unlike [Self::undistort], this remains well defined at and beyond 90 degrees off axis,
    /// where the observed ray is parallel to - or behind - the z = 1 plane and therefore has no
    /// point on it. A model whose field of view can reach that far, such as the enhanced unified
    /// one, should override this; the default is only correct below 180 degrees.
    fn undistort_to_unit_vector<PA, PO>(
        params: PA,
        distorted_point: PO,
    ) -> UnitVector3<S, BATCH, DM, DN>
    where
        PA: Borrow<S::Vector<PARAMS>>,
        PO: Borrow<S::Vector<2>>,
    {
        let point_in_z1_plane = Self::undistort(params, distorted_point);
        UnitVector3::from_vector_and_normalize(&S::Vector::<3>::from_array([
            point_in_z1_plane.elem(0),
            point_in_z1_plane.elem(1),
            S::ones(),
        ]))
    }

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
    fn new_pinhole(params: S::Vector<4>, image_size: ImageSize) -> Self;
    /// Creates a new Kannala-Brandt camera
    fn new_kannala_brandt(params: S::Vector<8>, image_size: ImageSize) -> Self;
    /// Creates a new Brown-Conrady camera
    fn new_brown_conrady(params: S::Vector<12>, image_size: ImageSize) -> Self;
    /// Creates a new Enhanced Unified camera
    fn new_enhanced_unified(params: S::Vector<6>, image_size: ImageSize) -> Self;

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
    /// Unprojects a pixel to the unit-length direction it observes
    fn cam_unproj_to_unit_vector<P>(&self, pixel: P) -> UnitVector3<S, BATCH, DM, DN>
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

    /// Returns the enhanced unified camera
    fn try_get_enhanced_unified(self) -> Option<EnhancedUnifiedCamera<S, BATCH, DM, DN>>;
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
