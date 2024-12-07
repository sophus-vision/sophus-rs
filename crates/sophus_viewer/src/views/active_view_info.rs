use crate::preludes::*;
use sophus_image::ImageSize;
use sophus_lie::Isometry3F64;
use sophus_renderer::camera::properties::RenderCameraProperties;
use sophus_renderer::types::SceneFocusMarker;

/// active view info
pub struct ActiveViewInfo {
    /// name of active view
    pub active_view: String,
    /// scene-from-camera pose
    pub scene_from_camera: Isometry3F64,
    /// camere properties
    pub camera_properties: Option<RenderCameraProperties>,
    /// scene focus
    pub scene_focus: SceneFocusMarker,
    /// type
    pub view_type: String,
    /// view-port size
    pub view_port_size: ImageSize,
    /// xy-locked
    pub locked_to_birds_eye_orientation: bool,
}
