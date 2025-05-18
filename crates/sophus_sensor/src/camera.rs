use core::borrow::Borrow;

use sophus_image::ImageSize;

use super::traits::IsCameraDistortionImpl;
use crate::prelude::*;

extern crate alloc;

/// A generic camera model
#[derive(Debug, Copy, Clone)]
pub struct Camera<
    S: IsScalar<BATCH, DM, DN>,
    const DISTORT: usize,
    const PARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    Distort: IsCameraDistortionImpl<S, DISTORT, PARAMS, BATCH, DM, DN>,
    Proj: IsProjection<S, BATCH, DM, DN>,
> {
    params: S::Vector<PARAMS>,
    phantom: core::marker::PhantomData<(Distort, Proj)>,
    image_size: ImageSize,
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DISTORT: usize,
    const PARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    Distort: IsCameraDistortionImpl<S, DISTORT, PARAMS, BATCH, DM, DN>,
    Proj: IsProjection<S, BATCH, DM, DN>,
> Camera<S, DISTORT, PARAMS, BATCH, DM, DN, Distort, Proj>
{
    /// Creates a new camera
    pub fn new(params: S::Vector<PARAMS>, image_size: ImageSize) -> Self {
        Self::from_params_and_size(params, image_size)
    }

    /// Creates a new camera from parameters and image size
    pub fn from_params_and_size(params: S::Vector<PARAMS>, size: ImageSize) -> Self {
        assert!(
            Distort::are_params_valid(params).all(),
            "Invalid parameters for {params:?}"
        );
        Self {
            params,
            phantom: core::marker::PhantomData,
            image_size: size,
        }
    }

    /// Returns the image size
    pub fn image_size(&self) -> ImageSize {
        self.image_size
    }

    /// Distortion - maps a point in the camera z=1 plane to a distorted point
    pub fn distort<Q>(&self, proj_point_in_camera_z1_plane: Q) -> S::Vector<2>
    where
        Q: Borrow<S::Vector<2>>,
    {
        Distort::distort(self.params, proj_point_in_camera_z1_plane)
    }

    /// Undistortion - maps a distorted pixel to a point in the camera z=1 plane
    pub fn undistort<Q>(&self, pixel: Q) -> S::Vector<2>
    where
        Q: Borrow<S::Vector<2>>,
    {
        Distort::undistort(self.params, pixel)
    }

    /// Derivative of the distortion w.r.t. the point in the camera z=1 plane
    pub fn dx_distort_x<Q>(&self, proj_point_in_camera_z1_plane: Q) -> S::Matrix<2, 2>
    where
        Q: Borrow<S::Vector<2>>,
    {
        Distort::dx_distort_x(self.params, proj_point_in_camera_z1_plane)
    }

    /// Derivative of the distortion w.r.t. the parameters
    pub fn dx_distort_params<Q>(&self, proj_point_in_camera_z1_plane: Q) -> S::Matrix<2, PARAMS>
    where
        Q: Borrow<S::Vector<2>>,
    {
        Distort::dx_distort_params(self.params, proj_point_in_camera_z1_plane)
    }

    /// Projects a 3D point in the camera frame to a pixel in the image
    pub fn cam_proj<Q>(&self, point_in_camera: Q) -> S::Vector<2>
    where
        Q: Borrow<S::Vector<3>>,
    {
        self.distort(Proj::proj(point_in_camera))
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame - assuming z=1
    pub fn cam_unproj<Q>(&self, point_in_camera: Q) -> S::Vector<3>
    where
        Q: Borrow<S::Vector<2>>,
    {
        self.cam_unproj_with_z(point_in_camera, S::ones())
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame
    pub fn cam_unproj_with_z<Q>(&self, point_in_camera: Q, z: S) -> S::Vector<3>
    where
        Q: Borrow<S::Vector<2>>,
    {
        Proj::unproj(self.undistort(point_in_camera), z)
    }

    /// Sets the camera parameters
    pub fn set_params<Q>(&mut self, params: Q)
    where
        Q: Borrow<S::Vector<PARAMS>>,
    {
        self.params = *params.borrow();
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
    S: IsScalar<BATCH, DM, DN>,
    const DISTORT: usize,
    const PARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    Distort: IsCameraDistortionImpl<S, DISTORT, PARAMS, BATCH, DM, DN>,
    Proj: IsProjection<S, BATCH, DM, DN>,
> Default for Camera<S, DISTORT, PARAMS, BATCH, DM, DN, Distort, Proj>
{
    fn default() -> Self {
        Self::from_params_and_size(Distort::identity_params(), ImageSize::default())
    }
}
