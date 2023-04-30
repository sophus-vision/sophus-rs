use crate::image::layout::ImageSize;
use crate::image::layout::ImageSizeTrait;
use crate::image::mut_image::MutImage;
use crate::image::mut_view::MutImageViewTrait;

use super::traits::WrongParamsDim;
use super::traits::{CameraDistortionImpl, Projection};
type V<const N: usize> = nalgebra::SVector<f64, N>;
type M<const N: usize, const O: usize> = nalgebra::SMatrix<f64, N, O>;

#[derive(Debug, Copy, Clone)]
pub struct Camera<
    const DISTORT: usize,
    const PARAMS: usize,
    Distort: CameraDistortionImpl<DISTORT, PARAMS>,
    Proj: Projection,
> {
    params: V<PARAMS>,
    phantom: std::marker::PhantomData<(Distort, Proj)>,
    image_size: ImageSize,
}

impl<
        const DISTORT: usize,
        const PARAMS: usize,
        Distort: CameraDistortionImpl<DISTORT, PARAMS>,
        Proj: Projection,
    > Camera<DISTORT, PARAMS, Distort, Proj>
{
    pub fn new(params: &V<PARAMS>, image_size: ImageSize) -> Self {
        Self::from_params_and_size(params, image_size)
    }

    pub fn from_params_and_size(params: &V<PARAMS>, size: ImageSize) -> Self {
        assert!(
            Distort::are_params_valid(params),
            "Invalid parameters for {}",
            params
        );
        Self {
            params: params.clone(),
            phantom: std::marker::PhantomData,
            image_size: size,
        }
    }

    pub fn distort(&self, proj_point_in_camera_z1_plane: &V<2>) -> V<2> {
        Distort::distort(&self.params, proj_point_in_camera_z1_plane)
    }

    pub fn undistort(&self, distorted_point: &V<2>) -> V<2> {
        Distort::undistort(&self.params, distorted_point)
    }

    pub fn undistort_table(&self) -> MutImage<2, f32> {
        let mut table = MutImage::<2, f32>::with_size(self.image_size);
        let w = self.image_size.width();
        let h = self.image_size.height();
        for v in 0..h {
            let row_slice = table.mut_row_slice(v);
            for u in 0..w {
                let pixel = self.undistort(&V::<2>::new(u as f64, v as f64));
                row_slice[u] = pixel.cast();
            }
        }
        table
    }

    pub fn dx_distort_x(&self, proj_point_in_camera_z1_plane: &V<2>) -> M<2, 2> {
        Distort::dx_distort_x(&self.params, proj_point_in_camera_z1_plane)
    }

    pub fn cam_proj(&self, point_in_camera: &V<3>) -> V<2> {
        self.distort(&Proj::proj(&point_in_camera))
    }

    pub fn cam_unproj(&self, point_in_camera: &V<2>) -> V<3> {
        self.cam_unproj_with_z(point_in_camera, 1.0)
    }

    pub fn cam_unproj_with_z(&self, point_in_camera: &V<2>, z: f64) -> V<3> {
        Proj::unproj(&self.undistort(&point_in_camera), z)
    }

    pub fn set_params(&mut self, params: &V<PARAMS>) {
        self.params = params.clone();
    }

    pub fn try_set_params(
        &mut self,
        params: &nalgebra::DVector<f64>,
    ) -> Result<(), WrongParamsDim> {
        if params.len() != PARAMS {
            return Err(WrongParamsDim);
        }
        self.params = V::<PARAMS>::from_iterator(params.iter().cloned());
        Ok(())
    }

    pub fn params(&self) -> &V<PARAMS> {
        &self.params
    }

    pub fn is_empty(&self) -> bool {
        self.image_size.width == 0 && self.image_size.height == 0
    }

    pub fn params_examples() -> Vec<V<PARAMS>> {
        Distort::params_examples()
    }

    pub fn invalid_params_examples() -> Vec<V<PARAMS>> {
        Distort::invalid_params_examples()
    }
}

impl<
        const DISTORT: usize,
        const PARAMS: usize,
        Distort: CameraDistortionImpl<DISTORT, PARAMS>,
        Proj: Projection,
    > Default for Camera<DISTORT, PARAMS, Distort, Proj>
{
    fn default() -> Self {
        Self::from_params_and_size(&Distort::identity_params(), ImageSize::default())
    }
}
