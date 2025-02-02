use core::borrow::Borrow;

use sophus_image::ImageSize;

use crate::{
    camera_enum::{
        perspective_camera::EnhancedUnifiedCamera,
        GeneralCameraEnum,
        PerspectiveCameraEnum,
    },
    prelude::*,
    BrownConradyCamera,
    KannalaBrandtCamera,
    PinholeCamera,
};

extern crate alloc;

/// Dynamic camera facade
#[derive(Debug, Clone)]
pub struct DynCameraFacade<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    CameraType: IsPerspectiveCamera<S, BATCH, DM, DN>,
> {
    camera_type: CameraType,
    phantom: core::marker::PhantomData<S>,
}

/// Dynamic generalized camera (perspective or orthographic)
pub type DynGeneralCamera<S, const BATCH: usize, const DM: usize, const DN: usize> =
    DynCameraFacade<S, BATCH, DM, DN, GeneralCameraEnum<S, BATCH, DM, DN>>;
/// Dynamic perspective camera
pub type DynCamera<S, const BATCH: usize, const DM: usize, const DN: usize> =
    DynCameraFacade<S, BATCH, DM, DN, PerspectiveCameraEnum<S, BATCH, DM, DN>>;

/// Create a new dynamic camera facade from a camera model
pub type DynCameraF64 = DynCamera<f64, 1, 0, 0>;

impl<
        S: IsScalar<BATCH, DM, DN> + 'static + Send + Sync,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
        CameraType: IsPerspectiveCamera<S, BATCH, DM, DN> + IsCamera<S, BATCH, DM, DN>,
    > DynCameraFacade<S, BATCH, DM, DN, CameraType>
{
    /// Create default pinhole from Image Size
    pub fn default_pinhole(image_size: ImageSize) -> Self {
        let w = image_size.width as f64;
        let h = image_size.height as f64;

        let focal_length = (w + h) * 0.5;
        Self {
            camera_type: CameraType::new_pinhole(
                S::Vector::<4>::from_f64_array([
                    focal_length,
                    focal_length,
                    0.5 * w - 0.5,
                    0.5 * h - 0.5,
                ]),
                image_size,
            ),
            phantom: core::marker::PhantomData,
        }
    }

    /// Create default distorted from Image Size
    pub fn default_distorted(image_size: ImageSize) -> Self {
        let w = image_size.width as f64;
        let h = image_size.height as f64;

        let focal_length = (w + h) * 0.5;
        Self {
            camera_type: CameraType::new_kannala_brandt(
                S::Vector::<8>::from_f64_array([
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
            phantom: core::marker::PhantomData,
        }
    }

    /// Create a new dynamic camera facade from a camera model
    pub fn from_model(camera_type: CameraType) -> Self {
        Self {
            camera_type,
            phantom: core::marker::PhantomData,
        }
    }

    /// Create a pinhole camera instance
    pub fn new_pinhole(params: impl Borrow<S::Vector<4>>, image_size: ImageSize) -> Self {
        Self::from_model(CameraType::new_pinhole(params, image_size))
    }

    /// Create a Kannala-Brandt camera instance
    pub fn new_kannala_brandt(params: impl Borrow<S::Vector<8>>, image_size: ImageSize) -> Self {
        Self::from_model(CameraType::new_kannala_brandt(params, image_size))
    }

    /// Create a Brown-Conrady camera instance
    pub fn new_brown_conrady(params: impl Borrow<S::Vector<12>>, image_size: ImageSize) -> Self {
        Self::from_model(CameraType::new_brown_conrady(params, image_size))
    }

    /// Create a enhanced unified camera instance
    pub fn new_enhanced_unified(params: impl Borrow<S::Vector<6>>, image_size: ImageSize) -> Self {
        Self::from_model(CameraType::new_enhanced_unified(params, image_size))
    }

    /// Projects a 3D point in the camera frame to a pixel in the image
    pub fn cam_proj(&self, point_in_camera: impl Borrow<S::Vector<3>>) -> S::Vector<2> {
        self.camera_type.cam_proj(point_in_camera.borrow())
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame - assuming z=1
    pub fn cam_unproj(&self, pixel: impl Borrow<S::Vector<2>>) -> S::Vector<3> {
        self.cam_unproj_with_z(pixel.borrow(), S::ones())
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame
    pub fn cam_unproj_with_z(&self, pixel: impl Borrow<S::Vector<2>>, z: S) -> S::Vector<3> {
        self.camera_type.cam_unproj_with_z(pixel.borrow(), z)
    }

    /// Distortion - maps a point in the camera z=1 plane to a distorted point
    pub fn distort(
        &self,
        proj_point_in_camera_z1_plane: impl Borrow<S::Vector<2>>,
    ) -> S::Vector<2> {
        self.camera_type
            .distort(proj_point_in_camera_z1_plane.borrow())
    }

    /// Undistortion - maps a distorted pixel to a point in the camera z=1 plane
    pub fn undistort(&self, pixel: impl Borrow<S::Vector<2>>) -> S::Vector<2> {
        self.camera_type.undistort(pixel.borrow())
    }

    /// Derivative of the distortion w.r.t. the point in the camera z=1 plane
    pub fn dx_distort_x(&self, point_in_camera: impl Borrow<S::Vector<2>>) -> S::Matrix<2, 2> {
        self.camera_type.dx_distort_x(point_in_camera.borrow())
    }

    /// Returns the image size
    pub fn image_size(&self) -> ImageSize {
        self.camera_type.image_size()
    }

    /// Returns the Kannala-Brandt camera
    pub fn try_get_brown_conrady(self) -> Option<BrownConradyCamera<S, BATCH, DM, DN>> {
        self.camera_type.try_get_brown_conrady()
    }

    /// Returns the Kannala-Brandt camera
    pub fn try_get_kannala_brandt(self) -> Option<KannalaBrandtCamera<S, BATCH, DM, DN>> {
        self.camera_type.try_get_kannala_brandt()
    }

    /// Returns the pinhole parameters
    pub fn try_get_pinhole(self) -> Option<PinholeCamera<S, BATCH, DM, DN>> {
        self.camera_type.try_get_pinhole()
    }

    /// Returns the enhanced unified camera
    pub fn try_get_enhanced_unified(self) -> Option<EnhancedUnifiedCamera<S, BATCH, DM, DN>> {
        self.camera_type.try_get_enhanced_unified()
    }

    /// Returns the camera model enum
    pub fn model_enum(&self) -> &CameraType {
        &self.camera_type
    }
}

impl<
        S: IsScalar<BATCH, DM, DN> + 'static + Send + Sync,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > DynCamera<S, BATCH, DM, DN>
{
    /// Returns the pinhole parameters
    pub fn pinhole_params(&self) -> S::Vector<4> {
        self.camera_type.pinhole_params()
    }
}

#[test]
fn dyn_camera_tests() {
    use approx::assert_relative_eq;
    use sophus_autodiff::{
        linalg::{
            VecF64,
            EPS_F64,
        },
        maps::vector_valued_maps::VectorValuedVectorMap,
    };
    use sophus_image::ImageSize;

    use crate::{
        distortions::{
            affine::AffineDistortionImpl,
            enhanced_unified::EnhancedUnifiedDistortionImpl,
            kannala_brandt::KannalaBrandtDistortionImpl,
        },
        traits::IsCameraDistortionImpl,
    };

    {
        let mut cameras: alloc::vec::Vec<DynCameraF64> = alloc::vec![];
        cameras.push(DynCameraF64::new_pinhole(
            VecF64::<4>::new(600.0, 599.0, 1.0, 0.5),
            ImageSize {
                width: 3,
                height: 2,
            },
        ));

        cameras.push(DynCamera::new_enhanced_unified(
            VecF64::<6>::from_vec(alloc::vec![998.0, 1000.0, 320.0, 280.0, 0.5, 1.2]),
            ImageSize {
                width: 640,
                height: 480,
            },
        ));

        cameras.push(DynCamera::new_kannala_brandt(
            VecF64::<8>::from_vec(alloc::vec![
                999.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001
            ]),
            ImageSize {
                width: 640,
                height: 480,
            },
        ));

        cameras.push(DynCamera::new_brown_conrady(
            VecF64::<12>::from_vec(alloc::vec![
                288.0,
                284.0,
                0.5 * (424.0 - 1.0),
                0.5 * (400.0 - 1.0),
                0.726405,
                -0.0148413,
                1.38447e-05,
                0.000419742,
                -0.00514224,
                1.06774,
                0.128429,
                -0.019901,
            ]),
            ImageSize {
                width: 640,
                height: 480,
            },
        ));

        for camera in cameras {
            let pixels_in_image = alloc::vec![
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
                VecF64::<2>::new(0.0, 1.0),
                VecF64::<2>::new(1.0, 0.0),
                VecF64::<2>::new(0.0, 0.0),
            ];

            for pixel in pixels_in_image.clone() {
                if pixel[0] > camera.image_size().width as f64
                    || pixel[1] > camera.image_size().height as f64
                {
                    continue;
                }

                for d in [1.0, 0.1, 0.5, 1.1, 3.0, 15.0] {
                    let point_in_camera = camera.cam_unproj_with_z(pixel, d);
                    assert_relative_eq!(point_in_camera[2], d, epsilon = EPS_F64);

                    let pixel_in_image2 = camera.cam_proj(point_in_camera);
                    assert_relative_eq!(pixel_in_image2, pixel, epsilon = 1e-4);
                }
                let ab_in_z1plane = camera.undistort(pixel);

                let pixel_in_image3 = camera.distort(ab_in_z1plane);
                assert_relative_eq!(pixel_in_image3, pixel, epsilon = 1e-4);

                let dx = camera.dx_distort_x(pixel);
                let numeric_dx = VectorValuedVectorMap::sym_diff_quotient_jacobian(
                    |x: VecF64<2>| camera.distort(x),
                    pixel,
                    EPS_F64,
                );

                assert_relative_eq!(dx, numeric_dx, epsilon = 1e-4);

                match camera.camera_type {
                    PerspectiveCameraEnum::Pinhole(camera) => {
                        let dx_params = camera.dx_distort_params(pixel);

                        let params = *camera.params();
                        let numeric_dx_params = VectorValuedVectorMap::sym_diff_quotient_jacobian(
                            |p: VecF64<4>| AffineDistortionImpl::<f64, 1, 0, 0>::distort(p, pixel),
                            params,
                            EPS_F64,
                        );
                        assert_relative_eq!(dx_params, numeric_dx_params, epsilon = 1e-4);
                    }
                    PerspectiveCameraEnum::KannalaBrandt(camera) => {
                        let dx_params = camera.dx_distort_params(pixel);

                        let params = *camera.params();
                        let numeric_dx_params = VectorValuedVectorMap::sym_diff_quotient_jacobian(
                            |p: VecF64<8>| {
                                KannalaBrandtDistortionImpl::<f64, 1, 0, 0>::distort(p, pixel)
                            },
                            params,
                            EPS_F64,
                        );
                        assert_relative_eq!(dx_params, numeric_dx_params, epsilon = 1e-4);
                    }
                    PerspectiveCameraEnum::BrownConrady(_camera) => {
                        // let dx_params = camera.dx_distort_params(&pixel);

                        // let params = camera.params();
                        // let numeric_dx_params = VectorValuedVectorMap::static_sym_diff_quotient(
                        //     |p: VecF64<12>| {
                        //         BrownConradyDistortionImpl::<f64, 1>::distort(&p, &pixel)
                        //     },
                        //     params.clone(),
                        //     EPS_F64,
                        // );
                        // assert_relative_eq!(dx_params, numeric_dx_params, epsilon = 1e-4);
                    }
                    PerspectiveCameraEnum::EnhancedUnified(camera) => {
                        let dx_params = camera.dx_distort_params(pixel);

                        let params = camera.params();
                        let numeric_dx_params = VectorValuedVectorMap::sym_diff_quotient_jacobian(
                            |p: VecF64<6>| {
                                EnhancedUnifiedDistortionImpl::<f64, 1, 0, 0>::distort(p, pixel)
                            },
                            *params,
                            EPS_F64,
                        );

                        assert_relative_eq!(dx_params, numeric_dx_params, epsilon = 1e-4);
                    }
                }
            }
        }
    }
}
