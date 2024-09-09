/// eframea app impl
pub mod app;
/// aspect ratio
pub mod aspect_ratio;
/// builder
pub mod builder;
/// Interactions
pub mod interactions;
/// plugin
pub mod plugin;
/// Types used in the API.
pub mod types;
/// The view struct.
pub mod views;

use linked_hash_map::LinkedHashMap;

use crate::renderables::Packet;
use crate::renderables::Packets;
use crate::viewer::builder::ViewerBuilder;
use crate::viewer::plugin::IsUiPlugin;
use crate::viewer::plugin::NullPlugin;
use crate::viewer::types::CancelRequest;
use crate::viewer::views::view2d::View2d;
use crate::viewer::views::view3d::View3d;
use crate::viewer::views::View;
use crate::RenderContext;

/// Viewer top-level struct.
pub struct Viewer<Plugin: IsUiPlugin> {
    state: RenderContext,
    views: LinkedHashMap<String, View>,
    message_recv: std::sync::mpsc::Receiver<Packets>,
    cancel_request_sender: std::sync::mpsc::Sender<CancelRequest>,
    show_depth: bool,
    backface_culling: bool,
    plugin: Plugin,
}

impl<Plugin: IsUiPlugin> Viewer<Plugin> {
    /// Create a new viewer.
    pub fn new(builder: ViewerBuilder<Plugin>, render_state: RenderContext) -> Box<Viewer<Plugin>> {
        Box::new(Viewer::<Plugin> {
            state: render_state.clone(),
            views: LinkedHashMap::new(),
            message_recv: builder.message_recv,
            cancel_request_sender: builder.cancel_request_sender,
            show_depth: false,
            backface_culling: false,
            plugin: builder.plugin,
        })
    }

    pub(crate) fn add_renderables_to_tables(&mut self) {
        loop {
            let maybe_stream = self.message_recv.try_recv();
            if maybe_stream.is_err() {
                break;
            }
            let stream = maybe_stream.unwrap();
            for packet in stream.packets {
                match packet {
                    Packet::View3d(packet) => View3d::update(&mut self.views, packet, &self.state),
                    Packet::View2d(packet) => View2d::update(&mut self.views, packet, &self.state),
                }
            }
        }
    }
}

/// simple viewer
pub type SimpleViewer = Viewer<NullPlugin>;
/// simple viewer builder
pub type SimpleViewerBuilder = ViewerBuilder<NullPlugin>;
