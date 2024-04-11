use crate::traits::IsProjection;
use sophus_core::linalg::matrix::IsMatrix;
use sophus_core::linalg::scalar::IsScalar;
use sophus_core::linalg::vector::IsVector;

/// Perspective camera projection - using z=1 plane
///
/// Projects a 3D point in the camera frame to a 2D point in the z=1 plane
#[derive(Debug, Clone, Copy)]
pub struct PerspectiveProjectionImpl;

impl<S: IsScalar<BATCH>, const BATCH: usize> IsProjection<S, BATCH> for PerspectiveProjectionImpl {
    fn proj(point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        S::Vector::<2>::from_array([
            point_in_camera.get_elem(0) / point_in_camera.get_elem(2),
            point_in_camera.get_elem(1) / point_in_camera.get_elem(2),
        ])
    }

    fn unproj(point_in_camera: &S::Vector<2>, extension: S) -> S::Vector<3> {
        S::Vector::<3>::from_array([
            point_in_camera.get_elem(0) * extension.clone(),
            point_in_camera.get_elem(1) * extension.clone(),
            extension,
        ])
    }

    fn dx_proj_x(point_in_camera: &S::Vector<3>) -> S::Matrix<2, 3> {
        S::Matrix::<2, 3>::from_array2([
            [
                S::ones() / point_in_camera.get_elem(2),
                S::zeros(),
                -point_in_camera.get_elem(0)
                    / (point_in_camera.get_elem(2) * point_in_camera.get_elem(2)),
            ],
            [
                S::zeros(),
                S::ones() / point_in_camera.get_elem(2),
                -point_in_camera.get_elem(1)
                    / (point_in_camera.get_elem(2) * point_in_camera.get_elem(2)),
            ],
        ])
    }
}
