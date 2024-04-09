use hollywood::actors::egui::EguiActor;
use hollywood::actors::egui::Stream;
pub use hollywood::compute::Context;
pub use hollywood::core::*;
use hollywood::macros::*;
use sophus::image::image_view::ImageSize;
use sophus::lie::traits::IsTranslationProductGroup;
use sophus::opt::example_problems::pose_circle::PoseCircleProblem;
use sophus::viewer::actor::run_viewer_on_main_thread;
use sophus::viewer::actor::ViewerBuilder;
use sophus::viewer::actor::ViewerCamera;
use sophus::viewer::actor::ViewerConfig;
use sophus::viewer::renderable::*;
use sophus::viewer::scene_renderer::interaction::WgpuClippingPlanes;
use sophus::viewer::SimpleViewer;
use sophus_core::linalg::vector::IsVector;
use sophus_core::linalg::VecF64;
use sophus_lie::groups::isometry2::Isometry2;
use sophus_lie::groups::isometry3::Isometry3;
use sophus_sensor::camera_enum::perspective_camera::KannalaBrandtCamera;
use sophus_sensor::camera_enum::perspective_camera::PerspectiveCameraEnum;
use sophus_sensor::dyn_camera::DynCamera;

#[actor(ContentGeneratorMessage)]
type ContentGenerator = Actor<
    NullProp,
    ContentGeneratorInbound,
    ContentGeneratorState,
    ContentGeneratorOutbound,
    NullRequest,
>;

/// Inbound message for the ContentGenerator actor.
#[derive(Clone, Debug)]
#[actor_inputs(ContentGeneratorInbound, {NullProp, ContentGeneratorState, ContentGeneratorOutbound, NullRequest})]
pub enum ContentGeneratorMessage {
    /// in seconds
    ClockTick(f64),
}

#[derive(Clone, Debug, Default)]
pub struct ContentGeneratorState {}

/// Outbound hub for the ContentGenerator.
#[actor_outputs]
pub struct ContentGeneratorOutbound {
    /// curves
    pub packets: OutboundChannel<Stream<Vec<Renderable>>>,
}

fn make_axes(world_from_local_poses: Vec<Isometry2<f64, 1>>) -> Vec<Line3> {
    let zero_in_local = VecF64::<2>::zeros();
    let x_axis_local = VecF64::<2>::new(1.0, 0.0);
    let y_axis_local = VecF64::<2>::new(0.0, 1.0);

    let mut lines = vec![];

    for world_from_local in world_from_local_poses.iter() {
        let zero_in_world = world_from_local.transform(&zero_in_local);
        let axis_x_in_world = world_from_local.transform(&x_axis_local);
        let axis_y_in_world = world_from_local.transform(&y_axis_local);

        lines.push(Line3 {
            p0: VecF64::<3>::new(zero_in_world.x, zero_in_world.y, 0.0).cast(),
            p1: VecF64::<3>::new(axis_x_in_world.x, axis_x_in_world.y, 0.0).cast(),
            color: Color {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            line_width: 2.0,
        });
        lines.push(Line3 {
            p0: VecF64::<3>::new(zero_in_world.x, zero_in_world.y, 0.0).cast(),
            p1: VecF64::<3>::new(axis_y_in_world.x, axis_y_in_world.y, 0.0).cast(),
            color: Color {
                r: 0.0,
                g: 1.0,
                b: 0.0,
                a: 1.0,
            },
            line_width: 2.0,
        });
    }

    lines
}

impl OnMessage for ContentGeneratorMessage {
    /// Process the inbound time_stamp message.
    fn on_message(
        self,
        _prop: &Self::Prop,
        _state: &mut Self::State,
        outbound: &Self::OutboundHub,
        _request: &NullRequest,
    ) {
        match &self {
            ContentGeneratorMessage::ClockTick(_time_in_seconds) => {
                let pose_graph = PoseCircleProblem::new(25);

                let mut renderables = vec![];

                renderables.push(Renderable::Lines3(Lines3 {
                    name: "true".into(),
                    lines: make_axes(pose_graph.true_world_from_robot.clone()),
                }));
                renderables.push(Renderable::Lines3(Lines3 {
                    name: "est".into(),
                    lines: make_axes(pose_graph.est_world_from_robot.clone()),
                }));

                outbound.packets.send(Stream { msg: renderables });
            }
        }
    }
}

impl InboundMessageNew<f64> for ContentGeneratorMessage {
    fn new(_inbound_name: String, msg: f64) -> Self {
        ContentGeneratorMessage::ClockTick(msg)
    }
}

pub async fn run_viewer_example() {
    // Camera / view pose parameters
    let intrinsics = KannalaBrandtCamera::<f64, 1>::new(
        &VecF64::<8>::from_array([600.0, 600.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0]),
        ImageSize {
            width: 640,
            height: 480,
        },
    );
    let scene_from_camera = Isometry3::from_t(&VecF64::<3>::new(0.0, 0.0, -50.0));
    let clipping_planes = WgpuClippingPlanes {
        near: 0.1,
        far: 1000.0,
    };
    let camera = ViewerCamera {
        intrinsics: DynCamera::<f64, 1>::from_model(PerspectiveCameraEnum::KannalaBrandt(
            intrinsics,
        )),
        clipping_planes,
        scene_from_camera,
    };

    let mut builder = ViewerBuilder::from_config(ViewerConfig { camera });

    // Pipeline configuration
    let pipeline = hollywood::compute::Context::configure(&mut |context| {
        // Actor creation:
        // 1. Periodic timer to drive the simulation
        let mut timer = hollywood::actors::Periodic::new_with_period(context, 0.01);
        // 2. The content generator of the example
        let mut content_generator = ContentGenerator::from_prop_and_state(
            context,
            NullProp {},
            ContentGeneratorState::default(),
        );
        // 3. The viewer actor
        let mut viewer =
            EguiActor::<Vec<Renderable>, (), Isometry3<f64, 1>>::from_builder(context, &builder);

        // Pipeline connections:
        timer
            .outbound
            .time_stamp
            .connect(context, &mut content_generator.inbound.clock_tick);
        content_generator
            .outbound
            .packets
            .connect(context, &mut viewer.inbound.stream);
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
