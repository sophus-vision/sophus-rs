use num_traits::cast;
use sophus_core::linalg::VecF64;
use sophus_image::ImageSize;
use sophus_lie::Isometry3;
use sophus_lie::Isometry3F64;
use sophus_sensor::camera_enum::perspective_camera::PinholeCameraF64;
use sophus_sensor::camera_enum::perspective_camera::UnifiedCameraF64;
use sophus_sensor::camera_enum::PerspectiveCameraEnum;
use sophus_sensor::dyn_camera::DynCameraF64;

use crate::renderer::scene_renderer::buffers::Frustum;

/// floating point: f32 or f64
pub trait FloatingPointNumber:
    num_traits::Float + num_traits::FromPrimitive + num_traits::NumCast
{
}
impl FloatingPointNumber for f32 {}
impl FloatingPointNumber for f64 {}

/// Clipping planes for the Wgpu renderer
#[derive(Clone, Copy, Debug)]
pub struct ClippingPlanes<S: FloatingPointNumber> {
    /// Near clipping plane
    pub near: S,
    /// Far clipping plane
    pub far: S,
}

impl Default for ClippingPlanesF64 {
    fn default() -> Self {
        ClippingPlanes {
            near: ClippingPlanes::DEFAULT_NEAR,
            far: ClippingPlanes::DEFAULT_FAR,
        }
    }
}

impl<S: FloatingPointNumber> ClippingPlanes<S> {
    /// metric z from ndc z
    pub fn metric_z_from_ndc_z(&self, ndc_z: S) -> S {
        -(self.far * self.near) / (-self.far + ndc_z * self.far - ndc_z * self.near)
    }

    /// ndc z from metric z
    pub fn ndc_z_from_metric_z(&self, z: S) -> S {
        (self.far * (z - self.near)) / (z * (self.far - self.near))
    }

    /// cast
    pub fn cast<Other: FloatingPointNumber>(self) -> ClippingPlanes<Other> {
        ClippingPlanes {
            near: cast(self.near).unwrap(),
            far: cast(self.far).unwrap(),
        }
    }
}

/// f32 clipping planes
pub type ClippingPlanesF32 = ClippingPlanes<f32>;
/// f64 clipping planes
pub type ClippingPlanesF64 = ClippingPlanes<f64>;

impl ClippingPlanesF64 {
    /// default near clipping plane
    pub const DEFAULT_NEAR: f64 = 1.0;
    /// default far clipping plane
    pub const DEFAULT_FAR: f64 = 1000.0;
}

/// Camera intrinsics
#[derive(Clone, Debug)]
pub enum RenderIntrinsics {
    /// Pinhole camera model
    Pinhole(PinholeCameraF64),
    /// Unified camera model
    UnifiedExtended(UnifiedCameraF64),
}

impl RenderIntrinsics {
    /// Get the image size
    pub fn image_size(&self) -> ImageSize {
        match self {
            RenderIntrinsics::Pinhole(pinhole) => pinhole.image_size(),
            RenderIntrinsics::UnifiedExtended(unified) => unified.image_size(),
        }
    }

    /// Unproject a point from the image plane with a given depth
    pub fn cam_unproj_with_z(&self, uv: &VecF64<2>, z: f64) -> VecF64<3> {
        match self {
            RenderIntrinsics::Pinhole(pinhole) => pinhole.cam_unproj_with_z(uv, z),
            RenderIntrinsics::UnifiedExtended(unified) => unified.cam_unproj_with_z(uv, z),
        }
    }

    /// Create a new RenderIntrinsics from a DynCameraF64
    pub fn new(camera: &DynCameraF64) -> RenderIntrinsics {
        match camera.model_enum() {
            PerspectiveCameraEnum::Pinhole(pinhole) => RenderIntrinsics::Pinhole(*pinhole),
            PerspectiveCameraEnum::KannalaBrandt(_camera) => todo!(),
            PerspectiveCameraEnum::BrownConrady(_camera) => todo!(),
            PerspectiveCameraEnum::UnifiedExtended(camera) => {
                RenderIntrinsics::UnifiedExtended(*camera)
            }
        }
    }
}

/// Camera properties
#[derive(Clone, Debug)]
pub struct RenderCameraProperties {
    /// Camera intrinsics
    pub intrinsics: RenderIntrinsics,
    /// Clipping planes
    pub clipping_planes: ClippingPlanesF64,
}

impl Default for RenderCameraProperties {
    fn default() -> Self {
        RenderCameraProperties::default_from(ImageSize::new(639, 479))
    }
}

impl RenderCameraProperties {
    /// Create a new RenderCameraProperties from a DynCameraF64
    pub fn new(intrinsics: DynCameraF64, clipping_planes: ClippingPlanesF64) -> Self {
        RenderCameraProperties {
            intrinsics: RenderIntrinsics::new(&intrinsics),
            clipping_planes,
        }
    }

    /// Create default viewer camera from image size
    pub fn default_from(image_size: ImageSize) -> RenderCameraProperties {
        RenderCameraProperties {
            intrinsics: RenderIntrinsics::Pinhole(
                DynCameraF64::default_pinhole(image_size)
                    .try_get_pinhole()
                    .unwrap(),
            ),
            clipping_planes: ClippingPlanes::default(),
        }
    }

    pub(crate) fn to_frustum(&self) -> Frustum {
        match self.intrinsics {
            RenderIntrinsics::Pinhole(pinhole) => Frustum {
                camera_image_width: pinhole.image_size().width as f32,
                camera_image_height: pinhole.image_size().height as f32,
                near: self.clipping_planes.near as f32,
                far: self.clipping_planes.far as f32,
                fx: pinhole.params()[0] as f32,
                fy: pinhole.params()[1] as f32,
                px: pinhole.params()[2] as f32,
                py: pinhole.params()[3] as f32,
                alpha: 0.0,
                beta: 0.0,
            },
            RenderIntrinsics::UnifiedExtended(unified) => Frustum {
                camera_image_width: unified.image_size().width as f32,
                camera_image_height: unified.image_size().height as f32,
                near: self.clipping_planes.near as f32,
                far: self.clipping_planes.far as f32,
                fx: unified.params()[0] as f32,
                fy: unified.params()[1] as f32,
                px: unified.params()[2] as f32,
                py: unified.params()[3] as f32,
                alpha: unified.params()[4] as f32,
                beta: unified.params()[5] as f32,
            },
        }
    }
}

/// Render camera configuration.
#[derive(Clone, Debug)]
pub struct RenderCamera {
    /// Scene from camera pose
    pub scene_from_camera: Isometry3F64,
    /// Camera properties
    pub properties: RenderCameraProperties,
}

impl Default for RenderCamera {
    fn default() -> Self {
        RenderCamera::default_from(ImageSize::new(639, 479))
    }
}

impl RenderCamera {
    /// Create default viewer camera from image size
    pub fn default_from(image_size: ImageSize) -> RenderCamera {
        RenderCamera {
            properties: RenderCameraProperties::default_from(image_size),
            scene_from_camera: Isometry3::from_translation(&VecF64::<3>::new(0.0, 0.0, -5.0)),
        }
    }
}

#[test]
fn clipping_plane_tests() {
    for near in [0.1, 0.5, 1.0, 7.0] {
        for far in [10.0, 5.0, 100.0, 1000.0] {
            for ndc_z in [0.0, 0.1, 0.5, 0.7, 0.99, 1.0] {
                let clipping_planes = ClippingPlanesF64 { near, far };

                let metric_z = clipping_planes.metric_z_from_ndc_z(ndc_z);
                let roundtrip_ndc_z = clipping_planes.ndc_z_from_metric_z(metric_z);

                approx::assert_abs_diff_eq!(roundtrip_ndc_z, ndc_z, epsilon = 0.0001);
            }
        }
    }
}
