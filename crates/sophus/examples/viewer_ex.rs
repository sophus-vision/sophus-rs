use hollywood::actors::egui::EguiActor;
use hollywood::actors::egui::Stream;
use hollywood::prelude::*;
use sophus::examples::viewer_example::make_example_image;
use sophus::image::ImageSize;
use sophus::viewer::actor::run_viewer_on_main_thread;
use sophus::viewer::actor::ViewerBuilder;
use sophus::viewer::actor::ViewerCamera;
use sophus::viewer::actor::ViewerConfig;
use sophus::viewer::renderables::*;
use sophus_core::linalg::VecF64;
use sophus_image::intensity_image::intensity_arc_image::IsIntensityArcImage;
use sophus_image::mut_image::MutImageF32;
use sophus_image::mut_image_view::IsMutImageView;
use sophus_lie::traits::IsTranslationProductGroup;
use sophus_lie::Isometry3;
use sophus_sensor::DynCamera;
use sophus_viewer::offscreen_renderer::renderer::ClippingPlanes;
use sophus_viewer::renderables::color::Color;
use sophus_viewer::renderables::renderable2d::Points2;
use sophus_viewer::renderables::renderable2d::Renderable2d;
use sophus_viewer::renderables::renderable2d::View2dPacket;
use sophus_viewer::renderables::renderable3d::View3dPacket;
use sophus_viewer::simple_viewer::SimpleViewer;

use crate::frame::Frame;
use crate::renderable2d::Lines2;
use crate::renderable3d::Lines3;
use crate::renderable3d::Mesh3;
use crate::renderable3d::Points3;
use crate::renderable3d::Renderable3d;

#[actor(ContentGeneratorMessage, NullInRequestMessage)]
type ContentGenerator = Actor<
    NullProp,
    ContentGeneratorInbound,
    NullInRequests,
    ContentGeneratorState,
    ContentGeneratorOutbound,
    ContentGeneratorOutRequest,
>;

/// Inbound message for the ContentGenerator actor.
#[derive(Clone, Debug)]
#[actor_inputs(
    ContentGeneratorInbound,
    {
        NullProp,
        ContentGeneratorState,
        ContentGeneratorOutbound,
        ContentGeneratorOutRequest,
        NullInRequestMessage
    })]
pub enum ContentGeneratorMessage {
    /// in seconds
    ClockTick(f64),
    SceneFromCamera(ReplyMessage<Isometry3<f64, 1>>),
}

/// Request of the simulation actor.
pub struct ContentGeneratorOutRequest {
    /// Check time-stamp of receiver
    pub scene_from_camera_request:
        OutRequestChannel<String, Isometry3<f64, 1>, ContentGeneratorMessage>,
}

impl IsOutRequestHub<ContentGeneratorMessage> for ContentGeneratorOutRequest {
    fn from_parent_and_sender(
        actor_name: &str,
        sender: &tokio::sync::mpsc::UnboundedSender<ContentGeneratorMessage>,
    ) -> Self {
        Self {
            scene_from_camera_request: OutRequestChannel::new(
                actor_name.to_owned(),
                "scene_from_camera_request",
                sender,
            ),
        }
    }
}

impl HasActivate for ContentGeneratorOutRequest {
    fn extract(&mut self) -> Self {
        Self {
            scene_from_camera_request: self.scene_from_camera_request.extract(),
        }
    }

    fn activate(&mut self) {
        self.scene_from_camera_request.activate();
    }
}

#[derive(Clone, Debug)]
pub struct ContentGeneratorState {
    pub counter: u32,
}

/// Outbound hub for the ContentGenerator.
#[actor_outputs]
pub struct ContentGeneratorOutbound {
    /// curves
    pub packets: OutboundChannel<Stream<Packets>>,
}

fn create_view2_packet() -> Packet {
    let img = make_example_image(ImageSize {
        width: 300,
        height: 100,
    });

    let mut packet_2d = View2dPacket {
        view_label: "view_2d".to_owned(),
        renderables3d: vec![],
        renderables2d: vec![],
        frame: Some(Frame::from_image(&img)),
    };

    let points2 = Points2::make("points2", &[[16.0, 12.0], [32.0, 24.0]], &Color::red(), 5.0);
    packet_2d.renderables2d.push(Renderable2d::Points2(points2));

    let lines2 = Lines2::make("lines2", &[[[0.0, 0.0], [20.0, 20.0]]], &Color::red(), 5.0);
    packet_2d.renderables2d.push(Renderable2d::Lines2(lines2));

    Packet::View2d(packet_2d)
}

fn create_tiny_view2_packet() -> Packet {
    let mut img = MutImageF32::from_image_size_and_val(ImageSize::new(3, 2), 1.0);

    *img.mut_pixel(0, 0) = 0.0;
    *img.mut_pixel(0, 1) = 0.5;

    *img.mut_pixel(1, 1) = 0.0;

    *img.mut_pixel(2, 0) = 0.3;
    *img.mut_pixel(2, 1) = 0.6;

    let mut packet_2d = View2dPacket {
        view_label: "tiny".to_owned(),
        renderables3d: vec![],
        renderables2d: vec![],
        frame: Some(Frame::from_image(&img.to_shared().to_rgba())),
    };

    let lines2 = Lines2::make(
        "lines2",
        &[[[-0.5, -0.5], [0.5, 0.5]], [[0.5, -0.5], [-0.5, 0.5]]],
        &Color::red(),
        2.0,
    );
    packet_2d.renderables2d.push(Renderable2d::Lines2(lines2));

    Packet::View2d(packet_2d)
}

fn create_view3_packet() -> Packet {
    let initial_camera = ViewerCamera {
        intrinsics: DynCamera::default_distorted(ImageSize::new(639, 477)),
        clipping_planes: ClippingPlanes::default(),
        scene_from_camera: Isometry3::from_t(&VecF64::<3>::new(0.0, 0.0, -5.0)),
    };
    let mut packet_3d = View3dPacket {
        view_label: "view_3d".to_owned(),
        renderables3d: vec![],
        initial_camera: initial_camera.clone(),
    };

    let trig_points = [[0.0, 0.0, -0.1], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];

    let points3 = Points3::make("points3", &trig_points, &Color::red(), 5.0);
    packet_3d.renderables3d.push(Renderable3d::Points3(points3));

    let lines3 = Lines3::make(
        "lines3",
        &[
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        &Color::green(),
        5.0,
    );
    packet_3d.renderables3d.push(Renderable3d::Lines3(lines3));

    let blue = Color::blue();
    let mesh = Mesh3::make("mesh", &[(trig_points, blue)]);
    packet_3d
        .renderables3d
        .push(renderable3d::Renderable3d::Mesh3(mesh));

    Packet::View3d(packet_3d)
}

impl HasOnMessage for ContentGeneratorMessage {
    /// Process the inbound time_stamp message.
    fn on_message(
        self,
        _prop: &Self::Prop,
        state: &mut Self::State,
        outbound: &Self::OutboundHub,
        _request: &ContentGeneratorOutRequest,
    ) {
        match &self {
            ContentGeneratorMessage::ClockTick(_time_in_seconds) => {
                let mut packets = Packets { packets: vec![] };

                if state.counter == 0 {
                    packets.packets.push(create_view2_packet());
                    packets.packets.push(create_tiny_view2_packet());
                    packets.packets.push(create_view3_packet());
                }

                state.counter += 1;

                outbound.packets.send(Stream { msg: packets });

                // request
                //     .scene_from_camera_request
                //     .send_request("view_2d".to_owned());
            }
            ContentGeneratorMessage::SceneFromCamera(reply) => {
                println!("{}", reply.reply);
            }
        }
    }
}

impl IsInboundMessageNew<f64> for ContentGeneratorMessage {
    fn new(_inbound_name: String, msg: f64) -> Self {
        ContentGeneratorMessage::ClockTick(msg)
    }
}

impl IsInboundMessageNew<ReplyMessage<Isometry3<f64, 1>>> for ContentGeneratorMessage {
    fn new(_inbound_name: String, scene_from_camera: ReplyMessage<Isometry3<f64, 1>>) -> Self {
        ContentGeneratorMessage::SceneFromCamera(scene_from_camera)
    }
}

pub async fn run_viewer_example() {
    let mut builder = ViewerBuilder::from_config(ViewerConfig {});

    // Pipeline configuration
    let pipeline = Hollywood::configure(&mut |context| {
        // Actor creation:
        // 1. Periodic timer to drive the simulation
        let mut timer = hollywood::actors::Periodic::new_with_period(context, 0.01);
        // 2. The content generator of the example
        let mut content_generator = ContentGenerator::from_prop_and_state(
            context,
            NullProp {},
            ContentGeneratorState { counter: 0 },
        );
        // 3. The viewer actor
        let mut viewer = EguiActor::<Packets, String, Isometry3<f64, 1>, (), ()>::from_builder(
            context, &builder,
        );

        // Pipeline connections:
        timer
            .outbound
            .time_stamp
            .connect(context, &mut content_generator.inbound.clock_tick);
        content_generator
            .outbound
            .packets
            .connect(context, &mut viewer.inbound.stream);
        content_generator
            .out_requests
            .scene_from_camera_request
            .connect(context, &mut viewer.in_requests.request);
    });

    // The cancel_requester is used to cancel the pipeline.
    builder
        .cancel_request_sender
        .clone_from(&pipeline.cancel_request_sender_template);

    // Plot the pipeline graph to the console.
    pipeline.print_flow_graph();

    // Pipeline execution:

    // 1. Run the pipeline on a separate thread.
    let pipeline_handle = tokio::spawn(pipeline.run());
    // 2. Run the viewer on the main thread. This is a blocking call.
    run_viewer_on_main_thread::<ViewerBuilder, SimpleViewer>(builder);
    // 3. Wait for the pipeline to finish.
    pipeline_handle.await.unwrap();
}

fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            run_viewer_example().await;
        })
}
