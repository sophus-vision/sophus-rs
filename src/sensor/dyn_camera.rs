use crate::image::image_view::ImageSize;
use crate::image::mut_image::MutImage2F32;

type V<const N: usize> = nalgebra::SVector<f64, N>;
type M<const N: usize, const O: usize> = nalgebra::SMatrix<f64, N, O>;
use super::distortion_table::DistortTable;
use super::general_camera::GeneralCameraEnum;
use super::perspective_camera::PerspectiveCameraEnum;
use super::traits::IsCameraEnum;

/// Dynamic camera facade
#[derive(Debug, Clone)]
pub struct DynCameraFacade<CameraType: IsCameraEnum> {
    camera_type: CameraType,
}

/// Dynamic generalized camera (perspective or orthographic)
pub type DynGeneralCamera = DynCameraFacade<GeneralCameraEnum>;
/// Dynamic perspective camera
pub type DynCamera = DynCameraFacade<PerspectiveCameraEnum>;

impl<CameraType: IsCameraEnum> DynCameraFacade<CameraType> {
    /// Create a new dynamic camera facade from a camera model
    pub fn from_model(camera_type: CameraType) -> Self {
        Self { camera_type }
    }

    /// Create a pinhole camera instance
    pub fn new_pinhole(params: &V<4>, image_size: ImageSize) -> Self {
        Self::from_model(CameraType::new_pinhole(params, image_size))
    }

    /// Create a Kannala-Brandt camera instance
    pub fn new_kannala_brandt(params: &V<8>, image_size: ImageSize) -> Self {
        Self::from_model(CameraType::new_kannala_brandt(params, image_size))
    }

    /// Projects a 3D point in the camera frame to a pixel in the image
    pub fn cam_proj(&self, point_in_camera: &V<3>) -> V<2> {
        self.camera_type.cam_proj(point_in_camera)
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame - assuming z=1
    pub fn cam_unproj(&self, pixel: &V<2>) -> V<3> {
        self.cam_unproj_with_z(pixel, 1.0)
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame
    pub fn cam_unproj_with_z(&self, pixel: &V<2>, z: f64) -> V<3> {
        self.camera_type.cam_unproj_with_z(pixel, z)
    }

    /// Distortion - maps a point in the camera z=1 plane to a distorted point
    pub fn distort(&self, proj_point_in_camera_z1_plane: &V<2>) -> V<2> {
        self.camera_type.distort(proj_point_in_camera_z1_plane)
    }

    /// Undistortion - maps a distorted pixel to a point in the camera z=1 plane
    pub fn undistort(&self, pixel: &V<2>) -> V<2> {
        self.camera_type.undistort(pixel)
    }

    /// Derivative of the distortion w.r.t. the point in the camera z=1 plane
    pub fn dx_distort_x(&self, point_in_camera: &V<2>) -> M<2, 2> {
        self.camera_type.dx_distort_x(point_in_camera)
    }

    /// Returns undistortion lookup table
    pub fn undistort_table(&self) -> MutImage2F32 {
        self.camera_type.undistort_table()
    }

    /// Returns distortion lookup table
    pub fn distort_table(&self) -> DistortTable {
        self.camera_type.distort_table()
    }
}

mod tests {

    #[test]
    fn camera_prop_tests() {
        use approx::assert_abs_diff_eq;
        use approx::assert_relative_eq;

        use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
        use crate::image::image_view::ImageSize;
        use crate::image::image_view::IsImageView;
        use crate::image::interpolation::interpolate;
        use crate::image::mut_image::MutImage2F32;
        use crate::sensor::distortion_table::DistortTable;

        use super::DynCamera;
        use super::V;
        let mut cameras: Vec<DynCamera> = vec![];
        cameras.push(DynCamera::new_pinhole(
            &V::<4>::new(600.0, 600.0, 319.5, 239.5),
            ImageSize {
                width: 640,
                height: 480,
            },
        ));

        cameras.push(DynCamera::new_kannala_brandt(
            &V::<8>::from_vec(vec![1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001]),
            ImageSize {
                width: 640,
                height: 480,
            },
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

            let table: MutImage2F32 = camera.undistort_table();

            for pixel in pixels_in_image.clone() {
                for d in [1.0, 0.1, 0.5, 1.1, 3.0, 15.0] {
                    let point_in_camera = camera.cam_unproj_with_z(&pixel, d);
                    assert_relative_eq!(point_in_camera[2], d, epsilon = 1e-6);

                    let pixel_in_image2 = camera.cam_proj(&point_in_camera);
                    assert_relative_eq!(pixel_in_image2, pixel, epsilon = 1e-6);
                }
                let ab_in_z1plane = camera.undistort(&pixel);
                let ab_in_z1plane2_f32 = interpolate(&table.image_view(), pixel.cast());
                let ab_in_z1plane2 = ab_in_z1plane2_f32.cast();
                assert_relative_eq!(ab_in_z1plane, ab_in_z1plane2, epsilon = 0.000001);

                let pixel_in_image3 = camera.distort(&ab_in_z1plane);
                assert_relative_eq!(pixel_in_image3, pixel, epsilon = 1e-6);

                let dx = camera.dx_distort_x(&pixel);
                let numeric_dx = VectorValuedMapFromVector::static_sym_diff_quotient(
                    |x: V<2>| camera.distort(&x),
                    pixel,
                    1e-6,
                );

                assert_relative_eq!(dx, numeric_dx, epsilon = 1e-4);
            }

            let table: DistortTable = camera.distort_table();

            for pixel in pixels_in_image {
                let proj = camera.undistort(&pixel);
                println!("proj: {:?}", proj);
                println!("region: {:?}", table.region);
                let analytic_pixel = camera.distort(&proj);
                let lut_pixel = table.lookup(&proj);

                assert_abs_diff_eq!(analytic_pixel, pixel, epsilon = 1e-3);
                assert_abs_diff_eq!(analytic_pixel, lut_pixel, epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn camera_pinhole_distort_test() {
        use crate::image::view::ImageSize;
        use approx::assert_relative_eq;

        use super::DynCamera;
        use super::V;
        let camera: DynCamera = DynCamera::new_pinhole(
            &V::<4>::new(600.0, 600.0, 319.5, 239.5),
            ImageSize {
                width: 640,
                height: 480,
            },
        );

        let pixels_in_image = vec![
            V::<2>::new(0.0, 0.0),
            V::<2>::new(1.0, 400.0),
            V::<2>::new(320.0, 240.0),
            V::<2>::new(319.5, 239.5),
            V::<2>::new(100.0, 40.0),
            V::<2>::new(639.0, 479.0),
        ];

        let expected_distorted_pixels = vec![
            V::<2>::new(319.5, 239.5),
            V::<2>::new(919.5, 240239.5),
            V::<2>::new(192319.5, 144239.5),
            V::<2>::new(192019.5, 143939.5),
            V::<2>::new(60319.5, 24239.5),
            V::<2>::new(383719.5, 287639.5),
        ];

        for (pixel, expected_distorted_pixel) in
            pixels_in_image.iter().zip(expected_distorted_pixels.iter())
        {
            let distorted_pixel = camera.distort(&pixel);
            // println!("distorted_pixel: {:?}", distorted_pixel);
            assert_relative_eq!(distorted_pixel, *expected_distorted_pixel, epsilon = 1e-6);

            let undistorted_pixel = camera.undistort(&distorted_pixel);
            assert_relative_eq!(undistorted_pixel, *pixel, epsilon = 1e-6);
        }
    }

    #[test]
    fn camera_kannala_brandt_distort_test() {
        use crate::image::view::ImageSize;
        use approx::assert_relative_eq;

        use super::DynCamera;
        use super::V;
        let camera: DynCamera = DynCamera::new_kannala_brandt(
            &V::<8>::from_vec(vec![1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001]),
            ImageSize {
                width: 640,
                height: 480,
            },
        );

        let pixels_in_image = vec![
            V::<2>::new(0.0, 0.0),
            V::<2>::new(1.0, 400.0),
            V::<2>::new(320.0, 240.0),
            V::<2>::new(319.5, 239.5),
            V::<2>::new(100.0, 40.0),
            V::<2>::new(639.0, 479.0),
        ];

        let expected_distorted_pixels = vec![
            V::<2>::new(320.0, 280.0),
            V::<2>::new(325.1949172763466, 2357.966910538644),
            V::<2>::new(1982.378709731326, 1526.7840322984944),
            V::<2>::new(1982.6832644475849, 1526.3619462760455),
            V::<2>::new(2235.6822069661744, 1046.2728827864696),
            V::<2>::new(1984.8663275417607, 1527.9983895031353),
        ];

        for (pixel, expected_distorted_pixel) in
            pixels_in_image.iter().zip(expected_distorted_pixels.iter())
        {
            let distorted_pixel = camera.distort(&pixel);
            // println!("distorted_pixel: {:?}", distorted_pixel);
            assert_relative_eq!(distorted_pixel, *expected_distorted_pixel, epsilon = 1e-6);

            let undistorted_pixel = camera.undistort(&distorted_pixel);
            assert_relative_eq!(undistorted_pixel, *pixel, epsilon = 1e-6);
        }
    }
}
