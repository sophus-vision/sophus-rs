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
    /// Create default pinhole from Image Size
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

    /// Create default distorted from Image Size
    pub fn default_distorted(image_size: ImageSize) -> Self {
        let w = image_size.width as f64;
        let h = image_size.height as f64;

        let focal_length = (w + h) * 0.5;
        Self {
            camera_type: CameraType::new_kannala_brandt(
                &S::Vector::<8>::from_f64_array([
                    focal_length,
                    focal_length,
                    0.5 * w - 0.5,
                    0.5 * h - 0.5,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
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
    use approx::assert_relative_eq;
    use sophus_core::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
    use sophus_core::linalg::VecF64;
    use sophus_image::ImageSize;

    type DynCameraF64 = DynCamera<f64, 1>;

    {
        let mut cameras: Vec<DynCameraF64> = vec![];
        cameras.push(DynCameraF64::new_pinhole(
            &VecF64::<4>::new(600.0, 600.0, 1.0, 0.5),
            ImageSize {
                width: 3,
                height: 2,
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
                VecF64::<2>::new(2.0, 1.0),
                VecF64::<2>::new(2.9, 1.9),
                VecF64::<2>::new(2.5, 1.5),
                VecF64::<2>::new(3.0, 2.0),
                VecF64::<2>::new(-0.5, -0.5),
                VecF64::<2>::new(1.0, 400.0),
                VecF64::<2>::new(320.0, 240.0),
                VecF64::<2>::new(319.5, 239.5),
                VecF64::<2>::new(100.0, 40.0),
                VecF64::<2>::new(639.0, 479.0),
            ];

            for pixel in pixels_in_image.clone() {
                if pixel[0] > camera.image_size().width as f64
                    || pixel[1] > camera.image_size().height as f64
                {
                    continue;
                }

                for d in [1.0, 0.1, 0.5, 1.1, 3.0, 15.0] {
                    let point_in_camera = camera.cam_unproj_with_z(&pixel, d);
                    assert_relative_eq!(point_in_camera[2], d, epsilon = 1e-6);

                    let pixel_in_image2 = camera.cam_proj(&point_in_camera);
                    assert_relative_eq!(pixel_in_image2, pixel, epsilon = 1e-6);
                }
                let ab_in_z1plane = camera.undistort(&pixel);

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
        }
    }
}
