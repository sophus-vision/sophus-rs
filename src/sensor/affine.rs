use crate::manifold::traits::ParamsImpl;

use super::traits::CameraDistortionImpl;
type V<const N: usize> = nalgebra::SVector<f64, N>;
type M<const N: usize, const O: usize> = nalgebra::SMatrix<f64, N, O>;

#[derive(Debug, Clone, Copy)]
pub struct AffineDistortionImpl;

impl ParamsImpl<f64, 4> for AffineDistortionImpl {
    fn are_params_valid(params: &V<4>) -> bool {
        params[0] != 0.0 && params[1] != 0.0
    }

    fn params_examples() -> Vec<V<4>> {
        vec![V::<4>::new(1.0, 1.0, 0.0, 0.0)]
    }

    fn invalid_params_examples() -> Vec<V<4>> {
        vec![
            V::<4>::new(0.0, 1.0, 0.0, 0.0),
            V::<4>::new(1.0, 0.0, 0.0, 0.0),
        ]
    }
}

impl CameraDistortionImpl<0, 4> for AffineDistortionImpl {
    fn distort(params: &V<4>, proj_point_in_camera_z1_plane: &V<2>) -> V<2> {
        V::<2>::new(
            proj_point_in_camera_z1_plane[0] * params[0] + params[2],
            proj_point_in_camera_z1_plane[1] * params[1] + params[3],
        )
    }

    fn undistort(params: &V<4>, distorted_point: &V<2>) -> V<2> {
        V::<2>::new(
            (distorted_point[0] - params[2]) / params[0],
            (distorted_point[1] - params[3]) / params[1],
        )
    }

    fn dx_distort_x(params: &V<4>, _proj_point_in_camera_z1_plane: &V<2>) -> M<2, 2> {
        M::<2, 2>::new(params[0], 0.0, 0.0, params[1])
    }
}
