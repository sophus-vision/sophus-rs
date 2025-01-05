use crate::camera::Camera;
use crate::distortions::affine::AffineDistortionImpl;
use crate::traits::IsProjection;
use core::borrow::Borrow;
use core::marker::PhantomData;
use sophus_core::linalg::scalar::IsScalar;
use sophus_core::linalg::vector::IsVector;

/// Orthographic projection implementation
#[derive(Debug, Clone)]
pub struct OrthographisProjectionImpl<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsProjection<S, BATCH, DM, DN> for OrthographisProjectionImpl<S, BATCH, DM, DN>
{
    fn proj<P>(point_in_camera: P) -> S::Vector<2>
    where
        P: Borrow<S::Vector<3>>,
    {
        point_in_camera.borrow().get_fixed_subvec::<2>(0)
    }

    fn unproj<P>(point_in_camera: P, extension: S) -> S::Vector<3>
    where
        P: Borrow<S::Vector<2>>,
    {
        let point_in_camera = point_in_camera.borrow();
        S::Vector::<3>::from_array([
            point_in_camera.get_elem(0),
            point_in_camera.get_elem(1),
            extension,
        ])
    }

    fn dx_proj_x<P>(_point_in_camera: P) -> S::Matrix<2, 3>
    where
        P: Borrow<S::Vector<3>>,
    {
        unimplemented!("dx_proj_x not implemented for ProjectionOrtho")
    }
}

/// Orthographic camera
pub type OrthographicCamera<S, const BATCH: usize, const DM: usize, const DN: usize> = Camera<
    S,
    0,
    4,
    BATCH,
    DM,
    DN,
    AffineDistortionImpl<S, BATCH, DM, DN>,
    OrthographisProjectionImpl<S, BATCH, DM, DN>,
>;
