use approx::assert_relative_eq;
use nalgebra::{SMatrix, SVector};

use crate::image::layout::ImageSize;
use crate::image::layout::ImageSizeTrait;
use crate::image::mut_image::MutImage;
use crate::image::mut_view::MutImageViewTrait;

use super::affine::AffineDistortionImpl;
use super::kannala_brandt::KannalaBrandtDistortionImpl;
use super::projections::ProjectionZ1;
use super::traits::{CameraDistortionImpl, Projection};
type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

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

#[derive(Debug, Copy, Clone)]
pub struct WrongParamsDim;

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

type PinholeCamera = Camera<0, 4, AffineDistortionImpl, ProjectionZ1>;
type OrthoCamera = Camera<0, 4, AffineDistortionImpl, ProjectionZ1>;
type KannalaBrandtCamera = Camera<4, 8, KannalaBrandtDistortionImpl, ProjectionZ1>;

#[derive(Debug, Clone)]
pub enum PerspectiveCameraType {
    Pinhole(PinholeCamera),
    KannalaBrandt(KannalaBrandtCamera),
}

#[derive(Debug, Clone)]
pub enum AnyProjCameraType {
    Perspective(PerspectiveCameraType),
    Ortho(OrthoCamera),
}

impl AnyProjCameraType {
    fn new_perspective(model: PerspectiveCameraType) -> Self {
        Self::Perspective(model)
    }
}

pub trait CameraEnum {
    fn new_pinhole(params: &V<4>, image_size: ImageSize) -> Self;
    fn new_kannala_brandt(params: &V<8>, image_size: ImageSize) -> Self;

    fn cam_proj(&self, point_in_camera: &V<3>) -> V<2>;
    fn cam_unproj_with_z(&self, point_in_camera: &V<2>, z: f64) -> V<3>;
    fn distort(&self, point_in_camera: &V<2>) -> V<2>;
    fn undistort(&self, point_in_camera: &V<2>) -> V<2>;
    fn undistort_table(&self) -> MutImage<2, f32>;

    fn dx_distort_x(&self, point_in_camera: &V<2>) -> M<2, 2>;

    fn try_set_params(&mut self, params: &nalgebra::DVector<f64>) -> Result<(), WrongParamsDim>;
}

impl CameraEnum for PerspectiveCameraType {
    fn new_pinhole(params: &V<4>, image_size: ImageSize) -> Self {
        Self::Pinhole(PinholeCamera::from_params_and_size(params, image_size))
    }

    fn new_kannala_brandt(params: &V<8>, image_size: ImageSize) -> Self {
        Self::KannalaBrandt(KannalaBrandtCamera::from_params_and_size(
            params, image_size,
        ))
    }

    fn cam_proj(&self, point_in_camera: &V<3>) -> V<2> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.cam_proj(point_in_camera),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.cam_proj(point_in_camera),
        }
    }

    fn cam_unproj_with_z(&self, point_in_camera: &V<2>, z: f64) -> V<3> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.cam_unproj_with_z(point_in_camera, z),
            PerspectiveCameraType::KannalaBrandt(camera) => {
                camera.cam_unproj_with_z(point_in_camera, z)
            }
        }
    }

    fn distort(&self, point_in_camera: &V<2>) -> V<2> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.distort(point_in_camera),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.distort(point_in_camera),
        }
    }

    fn undistort(&self, point_in_camera: &V<2>) -> V<2> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.undistort(point_in_camera),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.undistort(point_in_camera),
        }
    }

    fn dx_distort_x(&self, point_in_camera: &V<2>) -> M<2, 2> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.dx_distort_x(point_in_camera),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.dx_distort_x(point_in_camera),
        }
    }

    fn try_set_params(&mut self, params: &nalgebra::DVector<f64>) -> Result<(), WrongParamsDim> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.try_set_params(params),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.try_set_params(params),
        }
    }

    fn undistort_table(&self) -> MutImage<2, f32> {
        match self {
            PerspectiveCameraType::Pinhole(camera) => camera.undistort_table(),
            PerspectiveCameraType::KannalaBrandt(camera) => camera.undistort_table(),
        }
    }
}

impl CameraEnum for AnyProjCameraType {
    fn new_pinhole(params: &V<4>, image_size: ImageSize) -> Self {
        Self::Perspective(PerspectiveCameraType::new_pinhole(params, image_size))
    }

    fn new_kannala_brandt(params: &V<8>, image_size: ImageSize) -> Self {
        Self::Perspective(PerspectiveCameraType::new_kannala_brandt(
            params, image_size,
        ))
    }

    fn cam_proj(&self, point_in_camera: &V<3>) -> V<2> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.cam_proj(point_in_camera),
            AnyProjCameraType::Ortho(camera) => camera.cam_proj(point_in_camera),
        }
    }

    fn cam_unproj_with_z(&self, point_in_camera: &V<2>, z: f64) -> V<3> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.cam_unproj_with_z(point_in_camera, z),
            AnyProjCameraType::Ortho(camera) => camera.cam_unproj_with_z(point_in_camera, z),
        }
    }

    fn distort(&self, point_in_camera: &V<2>) -> V<2> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.distort(point_in_camera),
            AnyProjCameraType::Ortho(camera) => camera.distort(point_in_camera),
        }
    }

    fn undistort(&self, point_in_camera: &V<2>) -> V<2> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.undistort(point_in_camera),
            AnyProjCameraType::Ortho(camera) => camera.undistort(point_in_camera),
        }
    }

    fn try_set_params(&mut self, params: &nalgebra::DVector<f64>) -> Result<(), WrongParamsDim> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.try_set_params(params),
            AnyProjCameraType::Ortho(camera) => camera.try_set_params(params),
        }
    }

    fn dx_distort_x(&self, point_in_camera: &V<2>) -> M<2, 2> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.dx_distort_x(point_in_camera),
            AnyProjCameraType::Ortho(camera) => camera.dx_distort_x(point_in_camera),
        }
    }

    fn undistort_table(&self) -> MutImage<2, f32> {
        match self {
            AnyProjCameraType::Perspective(camera) => camera.undistort_table(),
            AnyProjCameraType::Ortho(camera) => camera.undistort_table(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynCameraFacade<CameraType: CameraEnum> {
    camera_type: CameraType,
}

type DynAnyProjCamera = DynCameraFacade<AnyProjCameraType>;
type DynCamera = DynCameraFacade<PerspectiveCameraType>;

impl<CameraType: CameraEnum> DynCameraFacade<CameraType> {
    pub fn from_model(camera_type: CameraType) -> Self {
        Self { camera_type }
    }

    pub fn new_pinhole(params: &V<4>, image_size: ImageSize) -> Self {
        Self::from_model(CameraType::new_pinhole(params, image_size))
    }

    pub fn new_kannala_brandt(params: &V<8>, image_size: ImageSize) -> Self {
        Self::from_model(CameraType::new_kannala_brandt(params, image_size))
    }

    pub fn cam_proj(&self, point_in_camera: &V<3>) -> V<2> {
        self.camera_type.cam_proj(point_in_camera)
    }

    pub fn cam_unproj(&self, point_in_camera: &V<2>) -> V<3> {
        self.cam_unproj_with_z(point_in_camera, 1.0)
    }

    pub fn cam_unproj_with_z(&self, point_in_camera: &V<2>, z: f64) -> V<3> {
        self.camera_type.cam_unproj_with_z(point_in_camera, z)
    }

    pub fn distort(&self, point_in_camera: &V<2>) -> V<2> {
        self.camera_type.distort(point_in_camera)
    }

    pub fn undistort(&self, point_in_camera: &V<2>) -> V<2> {
        self.camera_type.undistort(point_in_camera)
    }

    pub fn dx_distort_x(&self, point_in_camera: &V<2>) -> M<2, 2> {
        self.camera_type.dx_distort_x(point_in_camera)
    }

    pub fn try_set_params(
        &mut self,
        params: &nalgebra::DVector<f64>,
    ) -> Result<(), WrongParamsDim> {
        self.camera_type.try_set_params(params)
    }

    pub fn undistort_table(&self) -> MutImage<2, f32> {
        self.camera_type.undistort_table()
    }
}

mod tests {

    use crate::{calculus::numeric_diff::VectorField, image::{view::ImageViewTrait, interpolation::interpolate}};

    use super::*;

    #[test]
    fn camera_prop_tests() {
        let mut cameras: Vec<DynCamera> = vec![];
        cameras.push(DynCamera::new_pinhole(
            &V::<4>::new(600.0, 600.0, 319.5, 239.5),
            ImageSize::from_width_and_height(640, 480),
        ));

        cameras.push(DynCamera::new_kannala_brandt(
            &V::<8>::from_vec(vec![1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001]),
            ImageSize::from_width_and_height(640, 480),
        ));

        for camera in cameras {
            let pixels_in_image = vec![
                V::<2>::new(0.0, 0.0),
                V::<2>::new(1.0, 400.0),
                V::<2>::new(320.0, 240.0),
                V::<2>::new(319.5, 239.5),
                V::<2>::new(100.0, 40.0),
                V::<2>::new(639.0, 479.0),
            ];

            let table = camera.undistort_table();

            for pixel in pixels_in_image {
                for d in [1.0, 0.1, 0.5, 1.1, 3.0, 15.0] {
                    let point_in_camera = camera.cam_unproj_with_z(&pixel, d);
                    assert_relative_eq!(point_in_camera[2], d, epsilon = 1e-6);

                    let pixel_in_image2 = camera.cam_proj(&point_in_camera);
                    assert_relative_eq!(pixel_in_image2, pixel, epsilon = 1e-6);
                }
                let ab_in_z1plane = camera.undistort(&pixel);
                let ab_in_z1plane2_f32 =  interpolate(&table, pixel.cast());
                let ab_in_z1plane2 = ab_in_z1plane2_f32.cast();
                assert_relative_eq!(ab_in_z1plane, ab_in_z1plane2, epsilon = 0.000001);

                let pixel_in_image3 = camera.distort(&ab_in_z1plane);
                assert_relative_eq!(pixel_in_image3, pixel, epsilon = 1e-6);

                let dx = camera.dx_distort_x(&pixel);
                let numeric_dx =
                    VectorField::numeric_diff(|x: &V<2>| camera.distort(x), pixel, 1e-6);

                assert_relative_eq!(dx, numeric_dx, epsilon = 1e-4);
            }
        }
    }
}
