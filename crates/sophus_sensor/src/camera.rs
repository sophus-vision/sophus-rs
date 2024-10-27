use super::traits::IsCameraDistortionImpl;
use crate::prelude::*;
use sophus_image::ImageSize;

extern crate alloc;

/// A generic camera model
#[derive(Debug, Copy, Clone)]
pub struct Camera<
    S: IsScalar<BATCH>,
    const DISTORT: usize,
    const PARAMS: usize,
    const BATCH: usize,
    Distort: IsCameraDistortionImpl<S, DISTORT, PARAMS, BATCH>,
    Proj: IsProjection<S, BATCH>,
> {
    params: S::Vector<PARAMS>,
    phantom: core::marker::PhantomData<(Distort, Proj)>,
    image_size: ImageSize,
}

impl<
        S: IsScalar<BATCH>,
        const DISTORT: usize,
        const PARAMS: usize,
        const BATCH: usize,
        Distort: IsCameraDistortionImpl<S, DISTORT, PARAMS, BATCH>,
        Proj: IsProjection<S, BATCH>,
    > Camera<S, DISTORT, PARAMS, BATCH, Distort, Proj>
{
    /// Creates a new camera
    pub fn new(params: &S::Vector<PARAMS>, image_size: ImageSize) -> Self {
        Self::from_params_and_size(params, image_size)
    }

    /// Creates a new camera from parameters and image size
    pub fn from_params_and_size(params: &S::Vector<PARAMS>, size: ImageSize) -> Self {
        assert!(
            Distort::are_params_valid(params).all(),
            "Invalid parameters for {:?}",
            params
        );
        Self {
            params: params.clone(),
            phantom: core::marker::PhantomData,
            image_size: size,
        }
    }

    /// Returns the image size
    pub fn image_size(&self) -> ImageSize {
        self.image_size
    }

    /// Distortion - maps a point in the camera z=1 plane to a distorted point
    pub fn distort(&self, proj_point_in_camera_z1_plane: &S::Vector<2>) -> S::Vector<2> {
        Distort::distort(&self.params, proj_point_in_camera_z1_plane)
    }

    /// Undistortion - maps a distorted pixel to a point in the camera z=1 plane
    pub fn undistort(&self, pixel: &S::Vector<2>) -> S::Vector<2> {
        Distort::undistort(&self.params, pixel)
    }

    /// Derivative of the distortion w.r.t. the point in the camera z=1 plane
    pub fn dx_distort_x(&self, proj_point_in_camera_z1_plane: &S::Vector<2>) -> S::Matrix<2, 2> {
        Distort::dx_distort_x(&self.params, proj_point_in_camera_z1_plane)
    }

    /// Derivative of the distortion w.r.t. the parameters
    pub fn dx_distort_params(
        &self,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Matrix<2, PARAMS> {
        Distort::dx_distort_params(&self.params, proj_point_in_camera_z1_plane)
    }

    /// Projects a 3D point in the camera frame to a pixel in the image
    pub fn cam_proj(&self, point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        self.distort(&Proj::proj(point_in_camera))
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame - assuming z=1
    pub fn cam_unproj(&self, point_in_camera: &S::Vector<2>) -> S::Vector<3> {
        self.cam_unproj_with_z(point_in_camera, S::ones())
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame
    pub fn cam_unproj_with_z(&self, point_in_camera: &S::Vector<2>, z: S) -> S::Vector<3> {
        Proj::unproj(&self.undistort(point_in_camera), z)
    }

    /// Sets the camera parameters
    pub fn set_params(&mut self, params: &S::Vector<PARAMS>) {
        self.params = params.clone();
    }

    /// Returns the camera parameters
    pub fn params(&self) -> &S::Vector<PARAMS> {
        &self.params
    }

    /// Returns true if the camera is empty
    pub fn is_empty(&self) -> bool {
        self.image_size.width == 0 || self.image_size.height == 0
    }

    /// Examples of valid parameters
    pub fn params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>> {
        Distort::params_examples()
    }

    /// Examples of invalid parameters
    pub fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>> {
        Distort::invalid_params_examples()
    }
}

impl<
        S: IsScalar<BATCH>,
        const DISTORT: usize,
        const PARAMS: usize,
        const BATCH: usize,
        Distort: IsCameraDistortionImpl<S, DISTORT, PARAMS, BATCH>,
        Proj: IsProjection<S, BATCH>,
    > Default for Camera<S, DISTORT, PARAMS, BATCH, Distort, Proj>
{
    fn default() -> Self {
        Self::from_params_and_size(&Distort::identity_params(), ImageSize::default())
    }
}
