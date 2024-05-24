use crate::scene_renderer::interaction::WgpuClippingPlanes;
use crate::Renderable;
use crate::ViewerRenderState;
use eframe::egui;
use hollywood::actors::egui::EguiAppFromBuilder;
use hollywood::actors::egui::GenericEguiBuilder;
use hollywood::RequestWithReplyChannel;
use sophus_lie::Isometry3;
use sophus_sensor::dyn_camera::DynCamera;

/// Viewer camera configuration.
pub struct ViewerCamera {
    /// Camera intrinsics
    pub intrinsics: DynCamera<f64, 1>,
    /// Clipping planes
    pub clipping_planes: WgpuClippingPlanes,
    /// Scene from camera pose
    pub scene_from_camera: Isometry3<f64, 1>,
}

/// Viewer configuration.
pub struct ViewerConfig {
    /// Camera configuration
    pub camera: ViewerCamera,
}

/// Inbound message for the Viewer actor.
#[derive(Clone, Debug)]
pub enum ViewerMessage {
    /// Packets to render
    Packets(Vec<Renderable>),
}

/// Inbound message for the Viewer actor.
#[derive(Debug)]
pub enum ViewerInRequestMessage {
    /// Request the view pose
    RequestViewPose(RequestWithReplyChannel<(), Isometry3<f64, 1>>),
}

impl Default for ViewerMessage {
    fn default() -> Self {
        ViewerMessage::Packets(Vec::default())
    }
}

/// Builder for the Viewer actor.
pub type ViewerBuilder = GenericEguiBuilder<
    Vec<Renderable>,
    RequestWithReplyChannel<(), Isometry3<f64, 1>>,
    (),
    (),
    ViewerConfig,
>;

/// Execute the viewer actor on the main thread.
pub fn run_viewer_on_main_thread<
    Builder: 'static,
    V: EguiAppFromBuilder<Builder, State = ViewerRenderState>,
>(
    builder: Builder,
) {
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([640.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,

        ..Default::default()
    };
    eframe::run_native(
        "Egui actor",
        options,
        Box::new(|cc| {
            V::new(
                builder,
                ViewerRenderState {
                    _adapter: cc.wgpu_render_state.as_ref().unwrap().adapter.clone(),
                    device: cc.wgpu_render_state.as_ref().unwrap().device.clone(),
                    queue: cc.wgpu_render_state.as_ref().unwrap().queue.clone(),
                    wgpu_state: cc.wgpu_render_state.as_ref().unwrap().renderer.clone(),
                },
            )
        }),
    )
    .unwrap();
}
