use crate::renderables::Packets;
use crate::viewer::plugin::IsUiPlugin;
use crate::viewer::types::CancelRequest;

/// Simple viewer builder.
pub struct ViewerBuilder<Plugin: IsUiPlugin> {
    /// Message receiver.
    pub message_recv: std::sync::mpsc::Receiver<Packets>,
    /// Cancel request sender.
    pub cancel_request_sender: std::sync::mpsc::Sender<CancelRequest>,
    /// Plugin
    pub plugin: Plugin,
}
