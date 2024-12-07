use crate::preludes::*;
use sophus_lie::Isometry3F64;
use sophus_renderer::camera::RenderCamera;
use sophus_renderer::renderables::scene_renderable::SceneRenderable;

/// Content of a scene view packet
#[derive(Clone, Debug)]
pub enum SceneViewPacketContent {
    /// List of 3d renderables
    Renderables(Vec<SceneRenderable>),
    /// create a new view
    Creation(SceneViewCreation),
    /// world-from-scene pose update
    WorldFromSceneUpdate(Isometry3F64),
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
