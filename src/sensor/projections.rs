use super::traits::Projection;

use nalgebra::{SMatrix, SVector};

type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct ProjectionOrtho;

impl Projection for ProjectionOrtho {
    fn proj(point_in_camera: &V<3>) -> V<2> {
        V::<2>::new(point_in_camera[0], point_in_camera[1])
    }

    fn unproj(point_in_camera: &V<2>, extension: f64) -> V<3> {
        V::<3>::new(point_in_camera[0], point_in_camera[1], extension)
    }

    fn dx_proj_x(_point_in_camera: &V<3>) -> M<2, 3> {
        unimplemented!("dx_proj_x not implemented for ProjectionOrtho")
    }
}
