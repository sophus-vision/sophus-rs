use sophus_lie::Isometry3F64;
use sophus_renderer::{
    camera::RenderCamera,
    renderables::{
        ImageFrame,
        PixelRenderable,
        SceneRenderable,
    },
};

use crate::prelude::*;

mod image_view_packet;
mod plot_view_packet;
mod scene_view_packet;

pub use image_view_packet::*;
pub use plot_view_packet::*;
pub use scene_view_packet::*;

/// Packet of renderables
#[derive(Clone, Debug)]
pub enum Packet {
    /// scene view packet
    Scene(SceneViewPacket),
    /// image view packet
    Image(ImageViewPacket),
    /// plot view packet
    Plot(Vec<PlotViewPacket>),
}

/// Create a image packet
pub fn make_image_packet(
    view_label: &str,
    frame: Option<ImageFrame>,
    pixel_renderables: Vec<PixelRenderable>,
    scene_renderables: Vec<SceneRenderable>,
) -> Packet {
    Packet::Image(ImageViewPacket {
        frame,
        pixel_renderables,
        scene_renderables,
        view_label: view_label.to_string(),
        delete: false,
    })
}

/// Delete an image packet
pub fn delete_image_packet(view_label: &str) -> Packet {
    Packet::Image(ImageViewPacket {
        frame: None,
        pixel_renderables: vec![],
        scene_renderables: vec![],
        view_label: view_label.to_string(),
        delete: true,
    })
}

/// Create a scene packet
pub fn create_scene_packet(
    view_label: &str,
    initial_camera: RenderCamera,
    locked_to_birds_eye_orientation: bool,
) -> Packet {
    Packet::Scene(SceneViewPacket {
        view_label: view_label.to_string(),
        content: SceneViewPacketContent::Creation(SceneViewCreation {
            initial_camera,
            locked_to_birds_eye_orientation,
        }),
    })
}

/// Delete a scene packet
pub fn delete_scene_packet(view_label: &str) -> Packet {
    Packet::Scene(SceneViewPacket {
        view_label: view_label.to_string(),
        content: SceneViewPacketContent::Delete,
    })
}

/// Append to scene packet
pub fn append_to_scene_packet(view_label: &str, scene_renderables: Vec<SceneRenderable>) -> Packet {
    Packet::Scene(SceneViewPacket {
        view_label: view_label.to_string(),
        content: SceneViewPacketContent::Renderables(scene_renderables),
    })
}

/// Create world-from-scene update, scene packet
pub fn world_from_scene_update_packet(
    view_label: &str,
    world_from_scene_update: Isometry3F64,
) -> Packet {
    Packet::Scene(SceneViewPacket {
        view_label: view_label.to_string(),
        content: SceneViewPacketContent::WorldFromSceneUpdate(world_from_scene_update),
    })
}
