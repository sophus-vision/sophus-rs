use std::marker::PhantomData;

use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::M;
use crate::calculus::types::V;

use super::affine::AffineDistortionImpl;
use super::generic_camera::Camera;
use super::traits::IsProjection;

#[derive(Debug, Clone)]
pub struct ProjectionOrtho<S: IsScalar> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar> IsProjection<S> for ProjectionOrtho<S> {
    fn proj(point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        point_in_camera.get_fixed_rows::<2>(0)
    }

    fn unproj(point_in_camera: &S::Vector<2>, extension: S) -> S::Vector<3> {
        S::Vector::<3>::from_array([point_in_camera.get(0), point_in_camera.get(1), extension])
    }

    fn dx_proj_x(_point_in_camera: &V<3>) -> M<2, 3> {
        unimplemented!("dx_proj_x not implemented for ProjectionOrtho")
    }
}

pub type OrthoCamera<S> = Camera<S, 0, 4, AffineDistortionImpl<S>, ProjectionOrtho<S>>;
