use nalgebra::SVector;

use super::distortion_table::DistortTable;
use super::traits::IsCameraDistortionImpl;
use super::traits::IsProjection;
use sophus_calculus::region;
use sophus_calculus::region::IsRegion;
use sophus_calculus::region::Region;
use sophus_calculus::types::scalar::IsScalar;
use sophus_calculus::types::vector::IsVector;
use sophus_calculus::types::MatF64;
use sophus_calculus::types::VecF64;
use sophus_image::arc_image::ArcImage2F32;
use sophus_image::image_view::ImageSize;
use sophus_image::mut_image::MutImage2F32;
use sophus_image::mut_image_view::IsMutImageView;

/// A generic camera model
#[derive(Debug, Copy, Clone)]
pub struct Camera<
    S: IsScalar<1>,
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
        S: IsScalar<1>,
        const DISTORT: usize,
        const PARAMS: usize,
        Distort: IsCameraDistortionImpl<S, DISTORT, PARAMS>,
        Proj: IsProjection<S>,
    > Camera<S, DISTORT, PARAMS, Distort, Proj>
{
    /// Creates a new camera
    pub fn new(params: &S::Vector<PARAMS>, image_size: ImageSize) -> Self {
        Self::from_params_and_size(params, image_size)
    }

    /// Creates a new camera from parameters and image size
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

    /// Returns the image size
    pub fn image_size(&self) -> ImageSize {
        self.image_size
    }

    /// Distortion - maps a point in the camera z=1 plane to a distorted point
    pub fn distort(&self, proj_point_in_camera_z1_plane: &S::Vector<2>) -> S::Vector<2> {
        Distort::distort(&self.params, proj_point_in_camera_z1_plane)
    }

    /// Undistortion - maps a distorted pixel to a point in the camera z=1 plane
    pub fn undistort(&self, pixel: &S::Vector<2>) -> S::Vector<2> {
        Distort::undistort(&self.params, pixel)
    }

    /// Derivative of the distortion w.r.t. the point in the camera z=1 plane
    pub fn dx_distort_x(&self, proj_point_in_camera_z1_plane: &VecF64<2>) -> MatF64<2, 2> {
        Distort::dx_distort_x(self.params.real(), proj_point_in_camera_z1_plane)
    }

    /// Projects a 3D point in the camera frame to a pixel in the image
    pub fn cam_proj(&self, point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        self.distort(&Proj::proj(point_in_camera))
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame - assuming z=1
    pub fn cam_unproj(&self, point_in_camera: &S::Vector<2>) -> S::Vector<3> {
        self.cam_unproj_with_z(point_in_camera, 1.0.into())
    }

    /// Unprojects a pixel in the image to a 3D point in the camera frame
    pub fn cam_unproj_with_z(&self, point_in_camera: &S::Vector<2>, z: S) -> S::Vector<3> {
        Proj::unproj(&self.undistort(point_in_camera), z)
    }

    /// Sets the camera parameters
    pub fn set_params(&mut self, params: &S::Vector<PARAMS>) {
        self.params = params.clone();
    }

    /// Returns the camera parameters
    pub fn params(&self) -> &S::Vector<PARAMS> {
        &self.params
    }

    /// Returns true if the camera is empty
    pub fn is_empty(&self) -> bool {
        self.image_size.width == 0 || self.image_size.height == 0
    }

    /// Examples of valid parameters
    pub fn params_examples() -> Vec<S::Vector<PARAMS>> {
        Distort::params_examples()
    }

    /// Examples of invalid parameters
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
    /// Returns the undistortion lookup table
    pub fn undistort_table(&self) -> MutImage2F32 {
        let mut table = MutImage2F32::from_image_size(self.image_size);
        let w = self.image_size.width;
        let h = self.image_size.height;
        for v in 0..h {
            for u in 0..w {
                let pixel = self.undistort(&VecF64::<2>::new(u as f64, v as f64));
                *table.mut_pixel(u, v) = pixel.cast();
            }
        }
        table
    }

    /// Returns the distortion lookup table
    pub fn distort_table(&self) -> DistortTable {
        // first we find min and max values in the proj plane
        // just test the 4 corners might not be enough
        // so we will test the image boundary

        let mut region = Region::<2>::empty();

        let w = self.image_size.width;
        let h = self.image_size.height;

        for u in 0..self.image_size.width {
            // top border
            let v = 0;
            let point_in_proj = self.undistort(&VecF64::<2>::new(u as f64, v as f64));
            region.extend(&point_in_proj);
            // bottom border
            let v = self.image_size.height - 1;
            let point_in_proj = self.undistort(&VecF64::<2>::new(u as f64, v as f64));
            region.extend(&point_in_proj);
        }
        for v in 0..self.image_size.height {
            // left border
            let u = 0;
            let point_in_proj = self.undistort(&VecF64::<2>::new(u as f64, v as f64));
            region.extend(&point_in_proj);
            // right border
            let u = self.image_size.width - 1;
            let point_in_proj = self.undistort(&VecF64::<2>::new(u as f64, v as f64));
            region.extend(&point_in_proj);
        }
        let region =
            region::Region::<2>::from_min_max(region.min().cast() * 2.0, region.max().cast() * 2.0);

        let mut distort_table = DistortTable {
            table: ArcImage2F32::from_image_size_and_val(
                self.image_size,
                SVector::<f32, 2>::zeros(),
            ),
            region,
        };

        let mut table = MutImage2F32::from_image_size(self.image_size);

        for v in 0..h {
            for u in 0..w {
                let point_proj = VecF64::<2>::new(
                    distort_table.offset().x + (u as f64) * distort_table.incr().x,
                    distort_table.offset().y + (v as f64) * distort_table.incr().y,
                );
                let pixel = self.distort(&point_proj);
                *table.mut_pixel(u, v) = SVector::<f32, 2>::new(pixel.cast().x, pixel.cast().y);
            }
        }
        distort_table.table = table.into();
        distort_table
    }
}

impl<
        S: IsScalar<1>,
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
