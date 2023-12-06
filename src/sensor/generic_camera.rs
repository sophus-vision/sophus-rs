use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::M;
use crate::calculus::types::V;
use crate::image::mut_image::MutImage2F32;
use crate::image::mut_view::IsMutImageView;
use crate::image::view::ImageSize;

use super::traits::IsCameraDistortionImpl;
use super::traits::IsProjection;

#[derive(Debug, Copy, Clone)]
pub struct Camera<
    S: IsScalar,
    const DISTORT: usize,
    const PARAMS: usize,
    Distort: IsCameraDistortionImpl<S, DISTORT, PARAMS>,
    Proj: IsProjection<S>,
> {
    params: S::Vector<PARAMS>,
    phantom: std::marker::PhantomData<(Distort, Proj)>,
    image_size: ImageSize,
}

impl<
        S: IsScalar,
        const DISTORT: usize,
        const PARAMS: usize,
        Distort: IsCameraDistortionImpl<S, DISTORT, PARAMS>,
        Proj: IsProjection<S>,
    > Camera<S, DISTORT, PARAMS, Distort, Proj>
{
    pub fn new(params: &S::Vector<PARAMS>, image_size: ImageSize) -> Self {
        Self::from_params_and_size(params, image_size)
    }

    pub fn from_params_and_size(params: &S::Vector<PARAMS>, size: ImageSize) -> Self {
        assert!(
            Distort::are_params_valid(params),
            "Invalid parameters for {:?}",
            params
        );
        Self {
            params: params.clone(),
            phantom: std::marker::PhantomData,
            image_size: size,
        }
    }

    pub fn image_size(&self) -> ImageSize {
        self.image_size
    }

    pub fn distort(&self, proj_point_in_camera_z1_plane: &S::Vector<2>) -> S::Vector<2> {
        Distort::distort(&self.params, proj_point_in_camera_z1_plane)
    }

    pub fn undistort(&self, distorted_point: &S::Vector<2>) -> S::Vector<2> {
        Distort::undistort(&self.params, distorted_point)
    }

    pub fn dx_distort_x(&self, proj_point_in_camera_z1_plane: &V<2>) -> M<2, 2> {
        Distort::dx_distort_x(&self.params.real(), proj_point_in_camera_z1_plane)
    }

    pub fn cam_proj(&self, point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        self.distort(&Proj::proj(point_in_camera))
    }

    pub fn cam_unproj(&self, point_in_camera: &S::Vector<2>) -> S::Vector<3> {
        self.cam_unproj_with_z(point_in_camera, 1.0.into())
    }

    pub fn cam_unproj_with_z(&self, point_in_camera: &S::Vector<2>, z: S) -> S::Vector<3> {
        Proj::unproj(&self.undistort(point_in_camera), z)
    }

    pub fn set_params(&mut self, params: &S::Vector<PARAMS>) {
        self.params = params.clone();
    }

    pub fn params(&self) -> &S::Vector<PARAMS> {
        &self.params
    }

    pub fn is_empty(&self) -> bool {
        self.image_size.width == 0 || self.image_size.height == 0
    }

    pub fn params_examples() -> Vec<S::Vector<PARAMS>> {
        Distort::params_examples()
    }

    pub fn invalid_params_examples() -> Vec<S::Vector<PARAMS>> {
        Distort::invalid_params_examples()
    }
}

impl<
        const DISTORT: usize,
        const PARAMS: usize,
        Distort: IsCameraDistortionImpl<f64, DISTORT, PARAMS>,
        Proj: IsProjection<f64>,
    > Camera<f64, DISTORT, PARAMS, Distort, Proj>
{
    pub fn undistort_table(&self) -> MutImage2F32 {
        let mut table = MutImage2F32::from_image_size(self.image_size);
        let w = self.image_size.width;
        let h = self.image_size.height;
        for v in 0..h {
            for u in 0..w {
                let pixel = self.undistort(&V::<2>::new(u as f64, v as f64));
                *table.mut_pixel(u, v) = pixel.cast();
            }
        }
        table
    }
}

impl<
        S: IsScalar,
        const DISTORT: usize,
        const PARAMS: usize,
        Distort: IsCameraDistortionImpl<S, DISTORT, PARAMS>,
        Proj: IsProjection<S>,
    > Default for Camera<S, DISTORT, PARAMS, Distort, Proj>
{
    fn default() -> Self {
        Self::from_params_and_size(&Distort::identity_params(), ImageSize::default())
    }
}
