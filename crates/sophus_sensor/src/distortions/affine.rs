use core::{
    borrow::Borrow,
    marker::PhantomData,
};

use sophus_autodiff::params::IsParamsImpl;

use crate::{
    prelude::*,
    traits::IsCameraDistortionImpl,
};

extern crate alloc;

/// Affine "distortion" implementation
///
/// This is not a distortion in the traditional sense, but rather a simple affine transformation.
#[derive(Debug, Clone, Copy)]
pub struct AffineDistortionImpl<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsParamsImpl<S, 4, BATCH, DM, DN> for AffineDistortionImpl<S, BATCH, DM, DN>
{
    fn are_params_valid(_params: S::Vector<4>) -> S::Mask
    {
        S::Mask::all_true()
    }

    fn params_examples() -> alloc::vec::Vec<S::Vector<4>> {
        alloc::vec![S::Vector::<4>::from_f64_array([1.0, 1.0, 0.0, 0.0])]
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<4>> {
        alloc::vec![
            S::Vector::<4>::from_f64_array([0.0, 1.0, 0.0, 0.0]),
            S::Vector::<4>::from_f64_array([1.0, 0.0, 0.0, 0.0]),
        ]
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsCameraDistortionImpl<S, 0, 4, BATCH, DM, DN> for AffineDistortionImpl<S, BATCH, DM, DN>
{
    fn distort<PA, PO>(params: PA, proj_point_in_camera_z1_plane: PO) -> S::Vector<2>
    where
        PA: Borrow<S::Vector<4>>,
        PO: Borrow<S::Vector<2>>,
    {
        let params = params.borrow();
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();
        S::Vector::<2>::from_array([
            proj_point_in_camera_z1_plane.elem(0) * params.elem(0) + params.elem(2),
            proj_point_in_camera_z1_plane.elem(1) * params.elem(1) + params.elem(3),
        ])
    }

    fn undistort<PA, PO>(params: PA, distorted_point: PO) -> S::Vector<2>
    where
        PA: Borrow<S::Vector<4>>,
        PO: Borrow<S::Vector<2>>,
    {
        let params = params.borrow();
        let distorted_point = distorted_point.borrow();

        S::Vector::<2>::from_array([
            (distorted_point.elem(0) - params.elem(2)) / params.elem(0),
            (distorted_point.elem(1) - params.elem(3)) / params.elem(1),
        ])
    }

    fn dx_distort_x<PA, PO>(params: PA, _proj_point_in_camera_z1_plane: PO) -> S::Matrix<2, 2>
    where
        PA: Borrow<S::Vector<4>>,
        PO: Borrow<S::Vector<2>>,
    {
        let params = params.borrow();

        S::Matrix::<2, 2>::from_array2([[params.elem(0), S::zeros()], [S::zeros(), params.elem(1)]])
    }

    fn dx_distort_params<PA, PO>(_params: PA, proj_point_in_camera_z1_plane: PO) -> S::Matrix<2, 4>
    where
        PA: Borrow<S::Vector<4>>,
        PO: Borrow<S::Vector<2>>,
    {
        let proj_point_in_camera_z1_plane = proj_point_in_camera_z1_plane.borrow();

        S::Matrix::<2, 4>::from_array2([
            [
                proj_point_in_camera_z1_plane.elem(0),
                S::zeros(),
                S::ones(),
                S::zeros(),
            ],
            [
                S::zeros(),
                proj_point_in_camera_z1_plane.elem(1),
                S::zeros(),
                S::ones(),
            ],
        ])
    }
}
