/// color
pub mod color;
/// frame
pub mod frame;
/// 2d renderable
pub mod renderable2d;
/// 3d rendeable
pub mod renderable3d;

use sophus_image::arc_image::ArcImage4U8;

use crate::renderables::frame::Frame;
use crate::renderables::renderable2d::Renderable2d;
use crate::renderables::renderable2d::View2dPacket;
use crate::renderables::renderable3d::Renderable3d;
use crate::renderables::renderable3d::View3dPacket;
use crate::simple_viewer::ViewerCamera;

/// Image view renderable
#[derive(Clone, Debug)]
pub enum ImageViewRenderable {
    /// Background image
    BackgroundImage(ArcImage4U8),
}

/// Packet of renderables
#[derive(Clone, Debug)]
pub enum Packet {
    /// View3d packet
    View3d(View3dPacket),
    /// View2d packet
    View2d(View2dPacket),
}

/// Packet of renderables
#[derive(Clone, Debug, Default)]
pub struct Packets {
    /// List of packets
    pub packets: Vec<Packet>,
}

/// Create a view3d packet
pub fn make_view2d_packet(
    view_label: &str,
    frame: Option<Frame>,
    renderables2d: Vec<Renderable2d>,
    renderables3d: Vec<Renderable3d>,
) -> Packet {
    Packet::View2d(View2dPacket {
        frame,
        renderables2d,
        renderables3d,
        view_label: view_label.to_owned(),
    })
}

/// Create a view3d packet
pub fn make_view3d_packet(
    view_label: &str,
    initial_camera: ViewerCamera,
    renderables3d: Vec<Renderable3d>,
) -> Packet {
    Packet::View3d(View3dPacket {
        initial_camera,
        view_label: view_label.to_owned(),
        renderables3d,
    })
}
