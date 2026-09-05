use sophus_lie::Isometry3F64;
use sophus_renderer::{
    camera::{
        RenderCamera,
        RenderCameraProperties,
    },
    renderables::SceneRenderable,
};

use crate::{
    packets::Packet,
    prelude::*,
};

/// Content of a scene view packet
#[derive(Clone, Debug)]
pub enum SceneViewPacketContent {
    /// List of 3d renderables
    Renderables(Vec<SceneRenderable>),
    /// create a new view
    Creation(SceneViewCreation),
    /// delete the scene view
    Delete,
    /// world-from-scene pose update
    WorldFromSceneUpdate(Isometry3F64),
    /// intrinsics and clipping planes of the view's camera
    ///
    /// The camera model is uniform data which is uploaded every frame, so this is cheap - only a
    /// change of the *image size* makes the view rebuild its textures and pipelines.
    CameraPropertiesUpdate(RenderCameraProperties),
}

/// Creation of a scene view
#[derive(Clone, Debug)]
pub struct SceneViewCreation {
    /// Initial camera, ignored if not the first packet for this view
    pub initial_camera: RenderCamera,
    /// lock xy plane
    pub locked_to_birds_eye_orientation: bool,
}

/// Packet to populate a scene view
#[derive(Clone, Debug)]
pub struct SceneViewPacket {
    /// Name of the view
    pub view_label: String,
    /// Content of the packet
    pub content: SceneViewPacketContent,
}

/// Packet to change the camera properties - intrinsics and clipping planes - of a scene view.
pub fn update_scene_camera_properties(
    view_label: &str,
    camera_properties: RenderCameraProperties,
) -> Packet {
    Packet::Scene(SceneViewPacket {
        view_label: view_label.to_owned(),
        content: SceneViewPacketContent::CameraPropertiesUpdate(camera_properties),
    })
}
