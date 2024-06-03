use eframe::egui;
use hollywood::actors::egui::EguiAppFromBuilder;
use hollywood::actors::egui::GenericEguiBuilder;
use hollywood::RequestWithReplyChannel;
use sophus_core::linalg::VecF64;
use sophus_image::ImageSize;
use sophus_lie::traits::IsTranslationProductGroup;
use sophus_lie::Isometry3;
use sophus_sensor::dyn_camera::DynCamera;

use crate::interactions::WgpuClippingPlanes;
use crate::renderables::Packets;
use crate::ViewerRenderState;

/// Viewer camera configuration.
#[derive(Clone, Debug)]
pub struct ViewerCamera {
    /// Camera intrinsics
    pub intrinsics: DynCamera<f64, 1>,
    /// Clipping planes
    pub clipping_planes: WgpuClippingPlanes,
    /// Scene from camera pose
    pub scene_from_camera: Isometry3<f64, 1>,
}

impl Default for ViewerCamera {
    fn default() -> Self {
        ViewerCamera::default_from(ImageSize::new(640, 480))
    }
}

impl ViewerCamera {
    /// Create default viewer camera from image size
    pub fn default_from(image_size: ImageSize) -> ViewerCamera {
        ViewerCamera {
            intrinsics: DynCamera::default_pinhole(image_size),
            clipping_planes: WgpuClippingPlanes::default(),
            scene_from_camera: Isometry3::from_t(&VecF64::<3>::new(0.0, 0.0, -5.0)),
        }
    }
}

/// Viewer configuration.
pub struct ViewerConfig {}

/// Inbound message for the Viewer actor.
#[derive(Clone, Debug)]
pub enum ViewerMessage {
    /// Packets to render
    Packets(Packets),
}

/// Inbound message for the Viewer actor.
#[derive(Debug)]
pub enum ViewerInRequestMessage {
    /// Request the view pose
    RequestViewPose(RequestWithReplyChannel<String, Isometry3<f64, 1>>),
}

impl Default for ViewerMessage {
    fn default() -> Self {
        ViewerMessage::Packets(Packets::default())
    }
}

/// Builder for the Viewer actor.
pub type ViewerBuilder = GenericEguiBuilder<
    Packets,
    RequestWithReplyChannel<String, Isometry3<f64, 1>>,
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
