use core::borrow::Borrow;

use sophus_autodiff::{
    linalg::{
        matrix::IsMatrix,
        scalar::IsScalar,
        vector::IsVector,
    },
    prelude::IsSingleScalar,
};

use crate::traits::IsProjection;

/// Perspective camera projection - using z=1 plane
///
/// Projects a 3D point in the camera frame to a 2D point in the z=1 plane
#[derive(Debug, Clone, Copy)]
pub struct PerspectiveProjectionImpl<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsProjection<S, BATCH, DM, DN> for PerspectiveProjectionImpl<S, BATCH, DM, DN>
{
    fn proj<P>(point_in_camera: P) -> S::Vector<2>
    where
        P: Borrow<S::Vector<3>>,
    {
        let point_in_camera = point_in_camera.borrow();
        S::Vector::<2>::from_array([
            point_in_camera.get_elem(0) / point_in_camera.get_elem(2),
            point_in_camera.get_elem(1) / point_in_camera.get_elem(2),
        ])
    }

    fn unproj<P>(point_in_camera: P, extension: S) -> S::Vector<3>
    where
        P: Borrow<S::Vector<2>>,
    {
        let point_in_camera = point_in_camera.borrow();

        S::Vector::<3>::from_array([
            point_in_camera.get_elem(0) * extension,
            point_in_camera.get_elem(1) * extension,
            extension,
        ])
    }

    fn dx_proj_x<P>(point_in_camera: P) -> S::Matrix<2, 3>
    where
        P: Borrow<S::Vector<3>>,
    {
        let point_in_camera = point_in_camera.borrow();

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

/// Perspective projection for single scalar
pub fn perspect_proj<
    S: IsSingleScalar<DM, DN> + 'static + Send + Sync,
    const DM: usize,
    const DN: usize,
>(
    point_in_camera: &S::Vector<3>,
) -> S::Vector<2> {
    PerspectiveProjectionImpl::<S, 1, DM, DN>::proj(point_in_camera)
}
