use crate::prelude::*;
use sophus_renderer::renderables::frame::ImageFrame;
use sophus_renderer::renderables::pixel_renderable::PixelRenderable;
use sophus_renderer::renderables::scene_renderable::SceneRenderable;

/// Packet to populate an image view
#[derive(Clone, Debug)]
pub struct ImageViewPacket {
    /// Frame to hold content
    ///
    ///  1. For each `view_label`, content (i.e. pixel_renderables, scene_renderables) will be added to
    ///     the existing frame. If no frame exists yet, e.g. frame was always None for `view_label`,
    ///     the content is ignored.
    ///  2. If we have a new frame, that is `frame == Some(...)`, all previous content is deleted, but
    ///     content from this packet will be added.
    pub frame: Option<ImageFrame>,
    /// List of 2d renderables
    pub pixel_renderables: Vec<PixelRenderable>,
    /// List of scene renderables
    pub scene_renderables: Vec<SceneRenderable>,
    /// Name of the view
    pub view_label: String,
}
