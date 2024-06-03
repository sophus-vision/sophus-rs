/// color
pub mod color;
/// frame
pub mod frame;
/// 2d renderable
pub mod renderable2d;
/// 3d rendeable
pub mod renderable3d;

use sophus_image::arc_image::ArcImage4U8;

use crate::renderables::renderable2d::View2dPacket;
use crate::renderables::renderable3d::View3dPacket;

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
