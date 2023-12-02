use crate::image::view::ImageSize;
use crate::image::mut_image::MutImage2F32;

use super::affine::AffineDistortionImpl;
use super::generic_camera::Camera;
use super::kannala_brandt::KannalaBrandtDistortionImpl;
use super::traits::CameraEnum;
use super::traits::Projection;
use super::traits::WrongParamsDim;

use nalgebra::SMatrix;
use nalgebra::SVector;

type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

#[derive(Debug, Clone, Copy)]
pub struct ProjectionZ1;

impl Projection for ProjectionZ1 {
    fn proj(point_in_camera: &V<3>) -> V<2> {
        V::<2>::new(
            point_in_camera[0] / point_in_camera[2],
            point_in_camera[1] / point_in_camera[2],
        )
    }

    fn unproj(point_in_camera: &V<2>, extension: f64) -> V<3> {
        V::<3>::new(
            point_in_camera[0] * extension,
            point_in_camera[1] * extension,
            extension,
        )
    }

    fn dx_proj_x(point_in_camera: &V<3>) -> M<2, 3> {
        M::<2, 3>::new(
            1.0 / point_in_camera[2],
            0.0,
            -point_in_camera[0] / (point_in_camera[2] * point_in_camera[2]),
            0.0,
            1.0 / point_in_camera[2],
            -point_in_camera[1] / (point_in_camera[2] * point_in_camera[2]),
        )
    }
}

pub type PinholeCamera = Camera<0, 4, AffineDistortionImpl, ProjectionZ1>;
pub type OrthoCamera = Camera<0, 4, AffineDistortionImpl, ProjectionZ1>;
pub type KannalaBrandtCamera = Camera<4, 8, KannalaBrandtDistortionImpl, ProjectionZ1>;

#[derive(Debug, Clone, Copy)]
pub enum PerspectiveCameraType {
    Pinhole(PinholeCamera),
    KannalaBrandt(KannalaBrandtCamera),
}

impl CameraEnum for PerspectiveCameraType {
    fn new_pinhole(params: &V<4>, image_size: ImageSize) -> Self {
        Self::Pinhole(PinholeCamera::from_params_and_size(params, image_size))
    }

    fn new_kannala_brandt(params: &V<8>, image_size: ImageSize) -> Self {
        Self::KannalaBrandt(KannalaBrandtCamera::from_params_and_size(
            params, image_size,
        ))
    }

    fn cam_proj(&self, point_in_camera: &V<3>) -> V<2> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.cam_proj(point_in_camera),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.cam_proj(point_in_camera),
        }
    }

    fn cam_unproj_with_z(&self, point_in_camera: &V<2>, z: f64) -> V<3> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.cam_unproj_with_z(point_in_camera, z),
            PerspectiveCameraType::KannalaBrandt(camera) => {
                camera.cam_unproj_with_z(point_in_camera, z)
            }
        }
    }

    fn distort(&self, point_in_camera: &V<2>) -> V<2> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.distort(point_in_camera),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.distort(point_in_camera),
        }
    }

    fn undistort(&self, point_in_camera: &V<2>) -> V<2> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.undistort(point_in_camera),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.undistort(point_in_camera),
        }
    }

    fn dx_distort_x(&self, point_in_camera: &V<2>) -> M<2, 2> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.dx_distort_x(point_in_camera),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.dx_distort_x(point_in_camera),
        }
    }

    fn try_set_params(&mut self, params: &nalgebra::DVector<f64>) -> Result<(), WrongParamsDim> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.try_set_params(params),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.try_set_params(params),
        }
    }

    fn undistort_table(&self) -> MutImage2F32 {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.undistort_table(),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.undistort_table(),
        }
    }
}
