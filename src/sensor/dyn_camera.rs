use crate::image::layout::ImageSize;
use crate::image::mut_image::MutImage;

type V<const N: usize> = nalgebra::SVector<f64, N>;
type M<const N: usize, const O: usize> = nalgebra::SMatrix<f64, N, O>;
use super::any_camera::AnyProjCameraType;
use super::perspective_camera::PerspectiveCameraType;
use super::traits::CameraEnum;
use super::traits::WrongParamsDim;

#[derive(Debug, Clone)]
pub struct DynCameraFacade<CameraType: CameraEnum> {
    camera_type: CameraType,
}

pub type DynAnyProjCamera = DynCameraFacade<AnyProjCameraType>;
pub type DynCamera = DynCameraFacade<PerspectiveCameraType>;

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
    use approx::assert_relative_eq;

    use crate::{
        calculus::numeric_diff::VectorField,
        image::{interpolation::interpolate, layout::ImageSize},
    };

    use super::{DynCamera, V};

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
                let ab_in_z1plane2_f32 = interpolate(&table, pixel.cast());
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
