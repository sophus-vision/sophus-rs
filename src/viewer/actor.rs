use eframe::egui;

use hollywood::actors::egui::EguiAppFromBuilder;
use hollywood::actors::egui::GenericEguiBuilder;
use hollywood::core::request::RequestMessage;

use crate::lie::rotation3::Isometry3;
use crate::sensor::perspective_camera::KannalaBrandtCamera;

use super::scene_renderer::interaction::WgpuClippingPlanes;
use super::Renderable;
use super::ViewerRenderState;

pub struct ViewerCamera {
    pub intrinsics: KannalaBrandtCamera<f64>,
    pub clipping_planes: WgpuClippingPlanes,
    pub scene_from_camera: Isometry3<f64>,
}

pub struct ViewerConfig {
    pub camera: ViewerCamera,
}

/// Inbound message for the Viewer actor.
#[derive(Clone, Debug)]
pub enum ViewerMessage {
    Packets(Vec<Renderable>),
    RequestViewPose(RequestMessage<(), Isometry3<f64>>),
}

impl Default for ViewerMessage {
    fn default() -> Self {
        ViewerMessage::Packets(Vec::default())
    }
}

pub type ViewerBuilder =
    GenericEguiBuilder<Vec<Renderable>, RequestMessage<(), Isometry3<f64>>, ViewerConfig>;

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
                    adapter: cc.wgpu_render_state.as_ref().unwrap().adapter.clone(),
                    device: cc.wgpu_render_state.as_ref().unwrap().device.clone(),
                    queue: cc.wgpu_render_state.as_ref().unwrap().queue.clone(),
                    wgpu_state: cc.wgpu_render_state.as_ref().unwrap().renderer.clone(),
                },
            )
        }),
    )
    .unwrap();
}
