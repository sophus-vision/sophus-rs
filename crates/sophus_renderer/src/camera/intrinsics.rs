use sophus_autodiff::linalg::VecF64;
use sophus_image::ImageSize;
use sophus_lie::prelude::IsVector;
use sophus_sensor::{
    DynCameraF64,
    EnhancedUnifiedCameraF64,
    PerspectiveCameraEnum,
    PinholeCameraF64,
};

/// Camera intrinsics
#[derive(Clone, Debug)]
pub enum RenderIntrinsics {
    /// Pinhole camera model
    Pinhole(PinholeCameraF64),
    /// Unified camera model
    UnifiedExtended(EnhancedUnifiedCameraF64),
}

impl RenderIntrinsics {
    /// Create a new RenderIntrinsics from a DynCameraF64
    pub fn new(camera: &DynCameraF64) -> RenderIntrinsics {
        match camera.model_enum() {
            PerspectiveCameraEnum::Pinhole(pinhole) => RenderIntrinsics::Pinhole(*pinhole),
            PerspectiveCameraEnum::KannalaBrandt(_camera) => todo!(),
            PerspectiveCameraEnum::BrownConrady(_camera) => todo!(),
            PerspectiveCameraEnum::EnhancedUnified(camera) => {
                RenderIntrinsics::UnifiedExtended(*camera)
            }
        }
    }

    /// Get the image size
    pub fn image_size(&self) -> ImageSize {
        match self {
            RenderIntrinsics::Pinhole(pinhole) => pinhole.image_size(),
            RenderIntrinsics::UnifiedExtended(unified) => unified.image_size(),
        }
    }

    /// Unprojects a pixel to the unit-length direction it observes.
    ///
    /// Unlike [Self::cam_unproj_with_z], this remains well defined at and beyond 90 degrees off
    /// axis, where the observed ray has no point on the z = 1 plane at all.
    pub fn cam_unproj_to_unit_vector(&self, uv: &VecF64<2>) -> VecF64<3> {
        match self {
            RenderIntrinsics::Pinhole(pinhole) => pinhole.cam_unproj_to_unit_vector(uv).vector(),
            RenderIntrinsics::UnifiedExtended(unified) => {
                unified.cam_unproj_to_unit_vector(uv).vector()
            }
        }
    }

    /// Projects a point in the camera frame to the pixel which observes it.
    pub fn cam_proj(&self, point_in_camera: &VecF64<3>) -> VecF64<2> {
        match self {
            RenderIntrinsics::Pinhole(pinhole) => pinhole.cam_proj(point_in_camera),
            RenderIntrinsics::UnifiedExtended(unified) => unified.cam_proj(point_in_camera),
        }
    }

    /// Unproject a point from the image plane with a given depth
    pub fn cam_unproj_with_z(&self, uv: &VecF64<2>, z: f64) -> VecF64<3> {
        match self {
            RenderIntrinsics::Pinhole(pinhole) => pinhole.cam_unproj_with_z(uv, z),
            RenderIntrinsics::UnifiedExtended(unified) => unified.cam_unproj_with_z(uv, z),
        }
    }

    /// Return pinhole model
    pub fn pinhole_model(&self) -> PinholeCameraF64 {
        match self {
            RenderIntrinsics::Pinhole(camera) => *camera,
            RenderIntrinsics::UnifiedExtended(camera) => PinholeCameraF64::new(
                VecF64::<4>::from_array([
                    0.5 * camera.params()[0],
                    0.5 * camera.params()[1],
                    camera.params()[2],
                    camera.params()[3],
                ]),
                camera.image_size(),
            ),
        }
    }
}
