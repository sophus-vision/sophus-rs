use crate::camera_enum::GeneralCameraEnum;
use crate::camera_enum::PerspectiveCameraEnum;
use crate::prelude::*;
use sophus_image::ImageSize;

/// Dynamic camera facade
#[derive(Debug, Clone)]
pub struct DynCameraFacade<
    S: IsScalar<BATCH>,
    const BATCH: usize,
    CameraType: IsCameraEnum<S, BATCH>,
> {
    camera_type: CameraType,
    phantom: std::marker::PhantomData<S>,
}

/// Dynamic generalized camera (perspective or orthographic)
pub type DynGeneralCamera<S, const BATCH: usize> =
    DynCameraFacade<S, BATCH, GeneralCameraEnum<S, BATCH>>;
/// Dynamic perspective camera
pub type DynCamera<S, const BATCH: usize> =
    DynCameraFacade<S, BATCH, PerspectiveCameraEnum<S, BATCH>>;

impl<S: IsScalar<BATCH>, const BATCH: usize, CameraType: IsCameraEnum<S, BATCH>>
    DynCameraFacade<S, BATCH, CameraType>
{
    /// Create default pnhole from Image Size
    pub fn default_pinhole(image_size: ImageSize) -> Self {
        let w = image_size.width as f64;
        let h = image_size.height as f64;

        let focal_length = (w + h) * 0.5;
        Self {
            camera_type: CameraType::new_pinhole(
                &S::Vector::<4>::from_f64_array([
                    focal_length,
                    focal_length,
                    0.5 * w - 0.5,
                    0.5 * h - 0.5,
                ]),
                image_size,
            ),
            phantom: std::marker::PhantomData,
        }
    }

    /// Create a new dynamic camera facade from a camera model
    pub fn from_model(camera_type: CameraType) -> Self {
        Self {
            camera_type,
            phantom: std::marker::PhantomData,
        }
    }

    /// Create a pinhole camera instance
    pub fn new_pinhole(params: &S::Vector<4>, image_size: ImageSize) -> Self {
        Self::from_model(CameraType::new_pinhole(params, image_size))
    }

    /// Create a Kannala-Brandt camera instance
    pub fn new_kannala_brandt(params: &S::Vector<8>, image_size: ImageSize) -> Self {
        Self::from_model(CameraType::new_kannala_brandt(params, image_size))
    }

    /// Projects a 3D point in the camera frame to a pixel in the image
    pub fn cam_proj(&self, point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        self.camera_type.cam_proj(point_in_camera)
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame - assuming z=1
    pub fn cam_unproj(&self, pixel: &S::Vector<2>) -> S::Vector<3> {
        self.cam_unproj_with_z(pixel, S::ones())
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame
    pub fn cam_unproj_with_z(&self, pixel: &S::Vector<2>, z: S) -> S::Vector<3> {
        self.camera_type.cam_unproj_with_z(pixel, z)
    }

    /// Distortion - maps a point in the camera z=1 plane to a distorted point
    pub fn distort(&self, proj_point_in_camera_z1_plane: &S::Vector<2>) -> S::Vector<2> {
        self.camera_type.distort(proj_point_in_camera_z1_plane)
    }

    /// Undistortion - maps a distorted pixel to a point in the camera z=1 plane
    pub fn undistort(&self, pixel: &S::Vector<2>) -> S::Vector<2> {
        self.camera_type.undistort(pixel)
    }

    /// Derivative of the distortion w.r.t. the point in the camera z=1 plane
    pub fn dx_distort_x(&self, point_in_camera: &S::Vector<2>) -> S::Matrix<2, 2> {
        self.camera_type.dx_distort_x(point_in_camera)
    }

    /// Returns the image size
    pub fn image_size(&self) -> ImageSize {
        self.camera_type.image_size()
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> DynCamera<S, BATCH> {
    /// Returns the pinhole parameters
    pub fn pinhole_params(&self) -> S::Vector<4> {
        self.camera_type.pinhole_params()
    }
}

#[test]
fn dyn_camera_tests() {
    use crate::distortion_table::distort_table;
    use crate::distortion_table::undistort_table;
    use crate::distortion_table::DistortTable;
    use approx::assert_abs_diff_eq;
    use approx::assert_relative_eq;
    use sophus_core::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
    use sophus_core::linalg::VecF64;
    use sophus_image::image_view::IsImageView;
    use sophus_image::interpolation::interpolate;
    use sophus_image::mut_image::MutImage2F32;
    use sophus_image::ImageSize;

    type DynCameraF64 = DynCamera<f64, 1>;

    {
        let mut cameras: Vec<DynCameraF64> = vec![];
        cameras.push(DynCameraF64::new_pinhole(
            &VecF64::<4>::new(600.0, 600.0, 319.5, 239.5),
            ImageSize {
                width: 640,
                height: 480,
            },
        ));

        cameras.push(DynCamera::new_kannala_brandt(
            &VecF64::<8>::from_vec(vec![1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001]),
            ImageSize {
                width: 640,
                height: 480,
            },
        ));

        for camera in cameras {
            let pixels_in_image = vec![
                VecF64::<2>::new(0.0, 0.0),
                VecF64::<2>::new(1.0, 400.0),
                VecF64::<2>::new(320.0, 240.0),
                VecF64::<2>::new(319.5, 239.5),
                VecF64::<2>::new(100.0, 40.0),
                VecF64::<2>::new(639.0, 479.0),
            ];

            let table: MutImage2F32 = undistort_table(&camera);

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
                    |x: VecF64<2>| camera.distort(&x),
                    pixel,
                    1e-6,
                );

                assert_relative_eq!(dx, numeric_dx, epsilon = 1e-4);
            }

            let table: DistortTable = distort_table(&camera);

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

    {
        let camera: DynCameraF64 = DynCameraF64::new_pinhole(
            &VecF64::<4>::new(600.0, 600.0, 319.5, 239.5),
            ImageSize {
                width: 640,
                height: 480,
            },
        );

        let point_in_z1_plane = [
            VecF64::<2>::new(0.0, 0.0),
            VecF64::<2>::new(1.0, 400.0),
            VecF64::<2>::new(320.0, 240.0),
            VecF64::<2>::new(319.5, 239.5),
            VecF64::<2>::new(100.0, 40.0),
            VecF64::<2>::new(639.0, 479.0),
        ];

        let expected_distorted_pixels = [
            VecF64::<2>::new(319.5, 239.5),
            VecF64::<2>::new(919.5, 240239.5),
            VecF64::<2>::new(192319.5, 144239.5),
            VecF64::<2>::new(192019.5, 143939.5),
            VecF64::<2>::new(60319.5, 24239.5),
            VecF64::<2>::new(383719.5, 287639.5),
        ];

        for (pixel, expected_distorted_pixel) in point_in_z1_plane
            .iter()
            .zip(expected_distorted_pixels.iter())
        {
            let distorted_pixel = camera.distort(pixel);
            // println!("distorted_pixel: {:?}", distorted_pixel);
            assert_relative_eq!(distorted_pixel, *expected_distorted_pixel, epsilon = 1e-6);

            let undistorted_pixel = camera.undistort(&distorted_pixel);
            assert_relative_eq!(undistorted_pixel, *pixel, epsilon = 1e-6);
        }
    }

    {
        let camera: DynCameraF64 = DynCameraF64::new_kannala_brandt(
            &VecF64::<8>::from_vec(vec![1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001]),
            ImageSize {
                width: 640,
                height: 480,
            },
        );

        let point_in_z1_plane = [
            VecF64::<2>::new(0.0, 0.0),
            VecF64::<2>::new(1.0, 400.0),
            VecF64::<2>::new(320.0, 240.0),
            VecF64::<2>::new(319.5, 239.5),
            VecF64::<2>::new(100.0, 40.0),
            VecF64::<2>::new(639.0, 479.0),
        ];

        let expected_distorted_pixels = [
            VecF64::<2>::new(320.0, 280.0),
            VecF64::<2>::new(325.1949172763466, 2357.966910538644),
            VecF64::<2>::new(1982.378709731326, 1526.7840322984944),
            VecF64::<2>::new(1982.6832644475849, 1526.3619462760455),
            VecF64::<2>::new(2235.6822069661744, 1046.2728827864696),
            VecF64::<2>::new(1984.8663275417607, 1527.9983895031353),
        ];

        for (pixel, expected_distorted_pixel) in point_in_z1_plane
            .iter()
            .zip(expected_distorted_pixels.iter())
        {
            let distorted_pixel = camera.distort(pixel);
            // println!("distorted_pixel: {:?}", distorted_pixel);
            assert_relative_eq!(distorted_pixel, *expected_distorted_pixel, epsilon = 1e-6);

            let undistorted_pixel = camera.undistort(&distorted_pixel);
            assert_relative_eq!(undistorted_pixel, *pixel, epsilon = 1e-6);
        }
    }
}
