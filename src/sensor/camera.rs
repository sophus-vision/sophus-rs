use nalgebra::{SMatrix, SVector};

use crate::{calculus, image::layout::ImageSize};

use super::affine::AffineDistortionImpl;
use super::kannala_brandt::KannalaBrandtDistortionImpl;
use super::projections::ProjectionZ1;
use super::traits::{CameraDistortionImpl, Projection};
type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

#[derive(Debug, Copy, Clone)]
pub struct Camera<
    const DISTORT: usize,
    const PARAMS: usize,
    Distort: CameraDistortionImpl<DISTORT, PARAMS>,
    Proj: Projection,
> {
    params: V<PARAMS>,
    phantom: std::marker::PhantomData<(Distort, Proj)>,
    image_size: ImageSize,
}

#[derive(Debug, Copy, Clone)]
pub struct WrongParamsDim;

impl<
        const DISTORT: usize,
        const PARAMS: usize,
        Distort: CameraDistortionImpl<DISTORT, PARAMS>,
        Proj: Projection,
    > Camera<DISTORT, PARAMS, Distort, Proj>
{
    pub fn new(params: &V<PARAMS>, image_size: ImageSize) -> Self {
        Self::from_params_and_size(params, image_size)
    }

    pub fn from_params_and_size(params: &V<PARAMS>, size: ImageSize) -> Self {
        assert!(
            Distort::are_params_valid(params),
            "Invalid parameters for {}",
            params
        );
        Self {
            params: params.clone(),
            phantom: std::marker::PhantomData,
            image_size: size,
        }
    }

    pub fn distort(&self, proj_point_in_camera_z1_plane: &V<2>) -> V<2> {
        Distort::distort(&self.params, proj_point_in_camera_z1_plane)
    }

    pub fn undistort(&self, distorted_point: &V<2>) -> V<2> {
        Distort::undistort(&self.params, distorted_point)
    }

    pub fn dx_distort_x(&self, proj_point_in_camera_z1_plane: &V<2>) -> M<2, 2> {
        Distort::dx_distort_x(&self.params, proj_point_in_camera_z1_plane)
    }

    pub fn cam_proj(&self, point_in_camera: &V<3>) -> V<2> {
        self.distort(&Proj::proj(&point_in_camera))
    }

    pub fn cam_unproj(&self, point_in_camera: &V<2>) -> V<3> {
        self.cam_unproj_with_z(point_in_camera, 1.0)
    }

    pub fn cam_unproj_with_z(&self, point_in_camera: &V<2>, z: f64) -> V<3> {
        Proj::unproj(&self.undistort(&point_in_camera), z)
    }

    pub fn set_params(&mut self, params: &V<PARAMS>) {
        self.params = params.clone();
    }

    pub fn try_set_dyn_params(
        &mut self,
        params: &nalgebra::DVector<f64>,
    ) -> Result<(), WrongParamsDim> {
        if params.len() != PARAMS {
            return Err(WrongParamsDim);
        }
        self.params = V::<PARAMS>::from_iterator(params.iter().cloned());
        Ok(())
    }

    pub fn params(&self) -> &V<PARAMS> {
        &self.params
    }

    pub fn is_empty(&self) -> bool {
        self.image_size.width == 0 && self.image_size.height == 0
    }

    pub fn params_examples() -> Vec<V<PARAMS>> {
        Distort::params_examples()
    }

    pub fn invalid_params_examples() -> Vec<V<PARAMS>> {
        Distort::invalid_params_examples()
    }
}

impl<
        const DISTORT: usize,
        const PARAMS: usize,
        Distort: CameraDistortionImpl<DISTORT, PARAMS>,
        Proj: Projection,
    > Default for Camera<DISTORT, PARAMS, Distort, Proj>
{
    fn default() -> Self {
        Self::from_params_and_size(&Distort::identity_params(), ImageSize::default())
    }
}

type PinholeCamera = Camera<0, 4, AffineDistortionImpl, ProjectionZ1>;
type OrthoCamera = Camera<0, 4, AffineDistortionImpl, ProjectionZ1>;
type KannalaBrandtCamera = Camera<4, 8, KannalaBrandtDistortionImpl, ProjectionZ1>;

#[derive(Debug, Clone)]
pub enum CameraType {
    Pinhole(PinholeCamera),
    Ortho(OrthoCamera),
    KannalaBrandt(KannalaBrandtCamera),
}

#[derive(Debug, Clone)]
pub struct DynCamera {
    camera_type: CameraType,
}

impl DynCamera {
    pub fn from_model(camera_type: CameraType) -> Self {
        Self { camera_type }
    }

    pub fn new_pinhole(params: &V<4>, image_size: ImageSize) -> Self {
        Self {
            camera_type: CameraType::Pinhole(PinholeCamera::new(params, image_size)),
        }
    }

    pub fn new_ortho(params: &V<4>, image_size: ImageSize) -> Self {
        Self {
            camera_type: CameraType::Ortho(OrthoCamera::new(params, image_size)),
        }
    }

    pub fn cam_proj(&self, point_in_camera: &V<3>) -> V<2> {
        match &self.camera_type {
            CameraType::Pinhole(model) => model.cam_proj(point_in_camera),
            CameraType::Ortho(model) => model.cam_proj(point_in_camera),
            CameraType::KannalaBrandt(model) => model.cam_proj(point_in_camera),
            
        }
    }

    pub fn cam_unproj(&self, point_in_camera: &V<2>) -> V<3> {
        match &self.camera_type {
            CameraType::Pinhole(model) => model.cam_unproj(point_in_camera),
            CameraType::Ortho(model) => model.cam_unproj(point_in_camera),
            CameraType::KannalaBrandt(model) => model.cam_unproj(point_in_camera),
        }
    }

    pub fn cam_unproj_with_z(&self, point_in_camera: &V<2>, z: f64) -> V<3> {
        match &self.camera_type {
            CameraType::Pinhole(model) => model.cam_unproj_with_z(point_in_camera, z),
            CameraType::Ortho(model) => model.cam_unproj_with_z(point_in_camera, z),
            CameraType::KannalaBrandt(model) => model.cam_unproj_with_z(point_in_camera, z),
        }
    }

    pub fn try_set_params(
        &mut self,
        params: &nalgebra::DVector<f64>,
    ) -> Result<(), WrongParamsDim> {
        match &mut self.camera_type {
            CameraType::Pinhole(model) => model.try_set_dyn_params(params),
            CameraType::Ortho(model) => model.try_set_dyn_params(params),
            CameraType::KannalaBrandt(model) => model.try_set_dyn_params(params),
        }
    }
}
