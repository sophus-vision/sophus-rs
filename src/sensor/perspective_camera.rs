use crate::calculus::types::matrix::IsMatrix;
use crate::calculus::types::scalar::IsScalar;
use crate::image::mut_image::MutImage2F32;
use crate::image::view::ImageSize;

use super::affine::AffineDistortionImpl;
use super::generic_camera::Camera;
use super::kannala_brandt::KannalaBrandtDistortionImpl;
use super::traits::DistortTable;
use super::traits::IsCameraEnum;
use super::traits::IsProjection;
use crate::calculus::types::vector::IsVector;

use nalgebra::SMatrix;
use nalgebra::SVector;

type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

#[derive(Debug, Clone, Copy)]
pub struct ProjectionZ1;

impl<S: IsScalar> IsProjection<S> for ProjectionZ1 {
    fn proj(point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        S::Vector::<2>::from_array([
            point_in_camera.get(0) / point_in_camera.get(2),
            point_in_camera.get(1) / point_in_camera.get(2),
        ])
    }

    fn unproj(point_in_camera: &S::Vector<2>, extension: S) -> S::Vector<3> {
        S::Vector::<3>::from_array([
            point_in_camera.get(0) * extension.clone(),
            point_in_camera.get(1) * extension.clone(),
            extension,
        ])
    }

    fn dx_proj_x(point_in_camera: &V<3>) -> M<2, 3> {
        M::<2, 3>::from_array2([
            [
                1.0 / point_in_camera[2],
                0.0,
                -point_in_camera[0] / (point_in_camera[2] * point_in_camera[2]),
            ],
            [
                0.0,
                1.0 / point_in_camera[2],
                -point_in_camera[1] / (point_in_camera[2] * point_in_camera[2]),
            ],
        ])
    }
}

pub type PinholeCamera<S> = Camera<S, 0, 4, AffineDistortionImpl<S>, ProjectionZ1>;
pub type OrthoCamera<S> = Camera<S, 0, 4, AffineDistortionImpl<S>, ProjectionZ1>;
pub type KannalaBrandtCamera<S> = Camera<S, 4, 8, KannalaBrandtDistortionImpl<S>, ProjectionZ1>;

#[derive(Debug, Clone, Copy)]
pub enum PerspectiveCameraType {
    Pinhole(PinholeCamera<f64>),
    KannalaBrandt(KannalaBrandtCamera<f64>),
}

impl IsCameraEnum for PerspectiveCameraType {
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

    fn undistort_table(&self) -> MutImage2F32 {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.undistort_table(),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.undistort_table(),
        }
    }

    fn image_size(&self) -> ImageSize {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.image_size(),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.image_size(),
        }
    }

    fn distort_table(&self) -> DistortTable {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.distort_table(),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.distort_table(),
        }
    }
}
