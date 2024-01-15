use async_trait::async_trait;
use eframe::egui;
use hollywood::core::actor::ActorNode;
use hollywood::core::request::NullRequest;
use hollywood::core::request::RequestMessage;
use hollywood::core::runner::Runner;
use hollywood::core::*;
use hollywood::macros::actor_inputs;
use std::sync::Arc;
use tokio::select;

use super::scene_renderer::interaction::WgpuClippingPlanes;
use super::Renderable;
use super::SimpleViewer;
use super::ViewerBuilder;
use super::ViewerRenderState;
use crate::lie::rotation3::Isometry3;
use crate::sensor::perspective_camera::KannalaBrandtCamera;

#[derive(Clone, Debug, Default)]
pub struct ViewerProp {}

/// Inbound message for the Viewer actor.
#[derive(Clone, Debug)]
#[actor_inputs(ViewerInbound, {ViewerProp, ViewerState, NullOutbound, NullRequest})]
pub enum ViewerMessage {
    Packets(Vec<Renderable>),
    RequestViewPose(RequestMessage<(), Isometry3<f64>>),
}

/// State of the actor.
#[derive(Clone, Debug, Default)]
pub struct ViewerState {
    pub sender: Option<std::sync::mpsc::Sender<ViewerMessage>>,
    pub view_pose_receiver: Option<Arc<tokio::sync::mpsc::Receiver<Isometry3<f64>>>>,
}

impl ViewerState {
    pub fn new(
        sender: std::sync::mpsc::Sender<ViewerMessage>,
        view_pose_receiver: Arc<tokio::sync::mpsc::Receiver<Isometry3<f64>>>,
    ) -> Self {
        Self {
            sender: Some(sender),
            view_pose_receiver: Some(view_pose_receiver),
        }
    }
}

impl OnMessage for ViewerMessage {
    fn on_message(
        self,
        _prop: &Self::Prop,
        _state: &mut Self::State,
        _outbound: &NullOutbound,
        _request: &NullRequest,
    ) {
        panic!("ViewerMessage::on_message() should never be called")
    }
}

impl InboundMessageNew<Vec<Renderable>> for ViewerMessage {
    fn new(_inbound_name: String, p: Vec<Renderable>) -> Self {
        ViewerMessage::Packets(p)
    }
}

impl InboundMessageNew<RequestMessage<(), Isometry3<f64>>> for ViewerMessage {
    fn new(_inbound_name: String, p: RequestMessage<(), Isometry3<f64>>) -> Self {
        ViewerMessage::RequestViewPose(p)
    }
}

pub struct ViewerActor {
    pub actor_name: String,
    pub inbound: ViewerInbound,
    pub view_pose_receiver: Option<std::sync::mpsc::Receiver<Isometry3<f64>>>,
}

impl
    FromPropState<
        ViewerProp,
        ViewerInbound,
        ViewerState,
        NullOutbound,
        ViewerMessage,
        NullRequest,
        ViewerRunner,
    > for ViewerActor
{
    fn name_hint(_prop: &ViewerProp) -> String {
        "Viewer".to_owned()
    }
}

/// The custom runner for the viewer  actor.
pub struct ViewerRunner {}

impl Runner<ViewerProp, ViewerInbound, ViewerState, NullOutbound, NullRequest, ViewerMessage>
    for ViewerRunner
{
    fn new_actor_node(
        name: String,
        prop: ViewerProp,
        state: ViewerState,
        receiver: tokio::sync::mpsc::Receiver<ViewerMessage>,
        _forward: actor::ForwardTable<
            ViewerProp,
            ViewerState,
            NullOutbound,
            NullRequest,
            ViewerMessage,
        >,
        outbound: NullOutbound,
        _request: NullRequest,
    ) -> Box<dyn ActorNode + Send + Sync> {
        Box::new(ViewerNodeImpl {
            name: name.clone(),
            _prop: prop.clone(),
            state,
            receiver: Some(receiver),
            outbound,
        })
    }
}

pub(crate) struct ViewerNodeImpl {
    pub(crate) name: String,
    pub(crate) _prop: ViewerProp,
    pub(crate) state: ViewerState,
    pub(crate) receiver: Option<tokio::sync::mpsc::Receiver<ViewerMessage>>,
    #[allow(dead_code)]
    pub(crate) outbound: NullOutbound,
}

#[async_trait]
impl ActorNode for ViewerNodeImpl {
    fn name(&self) -> &String {
        &self.name
    }

    fn reset(&mut self) {
        // no-op
    }

    async fn run(&mut self, mut kill: tokio::sync::broadcast::Receiver<()>) {
        let sender = self.state.sender.take().unwrap();
        let mut receiver = self.receiver.take().unwrap();

        let on_viewer_message = |m: ViewerMessage| match sender.send(m) {
            Ok(_) => {}
            Err(w) => {
                println!("{}", w)
            }
        };

        loop {
            select! {
                _ = kill.recv() => {
                    while receiver.try_recv().is_ok(){}
                    return;
                },
                m = receiver.recv() => {
                    if m.is_none() {
                        let _ = kill.try_recv();
                        return;
                    }
                   on_viewer_message(m.unwrap());
                }
            }
        }
    }
}

pub struct ViewerCamera {
    pub intrinsics: KannalaBrandtCamera<f64>,
    pub clipping_planes: WgpuClippingPlanes,
    pub scene_from_camera: Isometry3<f64>,
}

pub fn run_viewer_on_man_thread(builder: ViewerBuilder) {
    // Important for wpgu debugging
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        renderer: eframe::Renderer::Wgpu,

        ..Default::default()
    };
    eframe::run_native(
        "Sophus viewer",
        options,
        Box::new(|cc| {
            let device = cc.wgpu_render_state.as_ref().unwrap().device.clone();
            let queue = cc.wgpu_render_state.as_ref().unwrap().queue.clone();

            let wgpu_state = cc.wgpu_render_state.as_ref().unwrap().renderer.clone();

            Box::<SimpleViewer>::new(SimpleViewer::new(
                builder,
                &ViewerRenderState {
                    wgpu_state,
                    device,
                    queue,
                    adapter: cc.wgpu_render_state.as_ref().unwrap().adapter.clone(),
                },
            ))
        }),
    )
    .unwrap();
}
