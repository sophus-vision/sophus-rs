use crate::camera::Camera;
use crate::distortions::affine::AffineDistortionImpl;
use crate::traits::IsProjection;
use sophus_core::linalg::scalar::IsScalar;
use sophus_core::linalg::vector::IsVector;
use std::marker::PhantomData;

/// Orthographic projection implementation
#[derive(Debug, Clone)]
pub struct OrthographisProjectionImpl<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH>, const BATCH: usize> IsProjection<S, BATCH>
    for OrthographisProjectionImpl<S, BATCH>
{
    fn proj(point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        point_in_camera.get_fixed_subvec::<2>(0)
    }

    fn unproj(point_in_camera: &S::Vector<2>, extension: S) -> S::Vector<3> {
        S::Vector::<3>::from_array([
            point_in_camera.get_elem(0),
            point_in_camera.get_elem(1),
            extension,
        ])
    }

    fn dx_proj_x(_point_in_camera: &S::Vector<3>) -> S::Matrix<2, 3> {
        unimplemented!("dx_proj_x not implemented for ProjectionOrtho")
    }
}

/// Orthographic camera
pub type OrthographicCamera<S, const BATCH: usize> =
    Camera<S, 0, 4, BATCH, AffineDistortionImpl<S, BATCH>, OrthographisProjectionImpl<S, BATCH>>;
