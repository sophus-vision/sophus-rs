use crate::prelude::*;
use crate::traits::IsCameraDistortionImpl;
use sophus_core::params::ParamsImpl;
use std::marker::PhantomData;

/// Affine "distortion" implementation
///
/// This is not a distortion in the traditional sense, but rather a simple affine transformation.
#[derive(Debug, Clone, Copy)]
pub struct AffineDistortionImpl<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH>, const BATCH: usize> ParamsImpl<S, 4, BATCH>
    for AffineDistortionImpl<S, BATCH>
{
    fn are_params_valid(_params: &S::Vector<4>) -> S::Mask {
        S::Mask::all_true()
    }

    fn params_examples() -> Vec<S::Vector<4>> {
        vec![S::Vector::<4>::from_f64_array([1.0, 1.0, 0.0, 0.0])]
    }

    fn invalid_params_examples() -> Vec<S::Vector<4>> {
        vec![
            S::Vector::<4>::from_f64_array([0.0, 1.0, 0.0, 0.0]),
            S::Vector::<4>::from_f64_array([1.0, 0.0, 0.0, 0.0]),
        ]
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> IsCameraDistortionImpl<S, 0, 4, BATCH>
    for AffineDistortionImpl<S, BATCH>
{
    fn distort(
        params: &S::Vector<4>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Vector<2> {
        S::Vector::<2>::from_array([
            proj_point_in_camera_z1_plane.get_elem(0) * params.get_elem(0) + params.get_elem(2),
            proj_point_in_camera_z1_plane.get_elem(1) * params.get_elem(1) + params.get_elem(3),
        ])
    }

    fn undistort(params: &S::Vector<4>, distorted_point: &S::Vector<2>) -> S::Vector<2> {
        S::Vector::<2>::from_array([
            (distorted_point.get_elem(0) - params.get_elem(2)) / params.get_elem(0),
            (distorted_point.get_elem(1) - params.get_elem(3)) / params.get_elem(1),
        ])
    }

    fn dx_distort_x(
        params: &S::Vector<4>,
        _proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Matrix<2, 2> {
        S::Matrix::<2, 2>::from_array2([
            [params.get_elem(0), S::zeros()],
            [S::zeros(), params.get_elem(1)],
        ])
    }
}
