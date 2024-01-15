

pub use hollywood::compute::Context;
use hollywood::core::request::ReplyMessage;
use hollywood::core::request::RequestChannel;
use hollywood::core::request::RequestHub;
pub use hollywood::core::*;
use hollywood::macros::*;
use nalgebra::SVector;
use sophus_rs::calculus::types::vector::IsVector;
use sophus_rs::calculus::types::V;
use sophus_rs::image::view::ImageSize;
use sophus_rs::lie::rotation3::Isometry3;
use sophus_rs::lie::traits::IsTranslationProductGroup;
use sophus_rs::sensor::perspective_camera::KannalaBrandtCamera;
use sophus_rs::viewer::actor::run_viewer_on_man_thread;
use sophus_rs::viewer::actor::ViewerActor;
use sophus_rs::viewer::actor::ViewerCamera;
use sophus_rs::viewer::actor::ViewerProp;
use sophus_rs::viewer::Color;
use sophus_rs::viewer::Line2;
use sophus_rs::viewer::Line3;
use sophus_rs::viewer::Lines2;
use sophus_rs::viewer::Lines3;
use sophus_rs::viewer::Point2;
use sophus_rs::viewer::Point3;
use sophus_rs::viewer::Points2;
use sophus_rs::viewer::Points3;
use sophus_rs::viewer::Renderable;
use sophus_rs::viewer::Triangle3;
use sophus_rs::viewer::Triangles3;
use sophus_rs::viewer::ViewerBuilder;
use sophus_rs::viewer::scene_renderer::interaction::WgpuClippingPlanes;

#[actor(ContentGeneratorMessage)]
type ContentGenerator = Actor<
    NullProp,
    ContentGeneratorInbound,
    ContentGeneratorState,
    ContentGeneratorOutbound,
    ContentGeneratorRequest,
>;

/// Inbound message for the ContentGenerator actor.
#[derive(Clone, Debug)]
#[actor_inputs(ContentGeneratorInbound, {NullProp, ContentGeneratorState, ContentGeneratorOutbound, ContentGeneratorRequest})]
pub enum ContentGeneratorMessage {
    /// in seconds
    ClockTick(f64),
    SceneFromCamera(ReplyMessage<Isometry3<f64>>),
}

/// Request of the simulation actor.
pub struct ContentGeneratorRequest {
    /// Check time-stamp of receiver
    pub scene_from_camera_request: RequestChannel<(), Isometry3<f64>, ContentGeneratorMessage>,
}

impl RequestHub<ContentGeneratorMessage> for ContentGeneratorRequest {
    fn from_context_and_parent(
        actor_name: &str,
        sender: &tokio::sync::mpsc::Sender<ContentGeneratorMessage>,
    ) -> Self {
        Self {
            scene_from_camera_request: RequestChannel::new(
                actor_name.to_owned(),
                "scene_from_camera_request",
                sender,
            ),
        }
    }
}

impl Morph for ContentGeneratorRequest {
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
    pub show: bool,
    pub intrinsics: KannalaBrandtCamera<f64>,
    pub scene_from_camera: Isometry3<f64>,
}

impl Default for ContentGeneratorState {
    fn default() -> Self {
        ContentGeneratorState {
            counter: 0,
            show: false,
            intrinsics: KannalaBrandtCamera::<f64>::new(
                &V::<8>::from_array([600.0, 600.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0]),
                ImageSize {
                    width: 640,
                    height: 480,
                },
            ),
            scene_from_camera: Isometry3::from_t(&V::<3>::new(0.0, 0.0, -5.0)),
        }
    }
}

/// Outbound hub for the ContentGenerator.
#[actor_outputs]
pub struct ContentGeneratorOutbound {
    /// curves
    pub packets: OutboundChannel<Vec<Renderable>>,
}

impl OnMessage for ContentGeneratorMessage {
    /// Process the inbound time_stamp message.
    fn on_message(
        self,
        _prop: &Self::Prop,
        state: &mut Self::State,
        outbound: &Self::OutboundHub,
        request: &ContentGeneratorRequest,
    ) {
        match &self {
            ContentGeneratorMessage::ClockTick(_time_in_seconds) => {
                let mut renderables = vec![];

                let trig_points = [
                    SVector::<f32, 3>::new(0.0, 0.0, -0.1),
                    SVector::<f32, 3>::new(0.0, 1.0, 0.0),
                    SVector::<f32, 3>::new(1.0, 0.0, 0.0),
                ];

                if state.counter == 0 {
                    let points2 = vec![
                        Point2 {
                            p: SVector::<f32, 2>::new(16.0, 12.0),
                            color: Color {
                                r: 1.0,
                                g: 1.0,
                                b: 0.0,
                                a: 1.0,
                            },
                            point_size: 5.0,
                        },
                        Point2 {
                            p: SVector::<f32, 2>::new(32.0, 24.0),
                            color: Color {
                                r: 1.0,
                                g: 1.0,
                                b: 0.0,
                                a: 1.0,
                            },
                            point_size: 5.0,
                        },
                    ];

                    let points3 = vec![
                        Point3 {
                            p: trig_points[0],
                            color: Color {
                                r: 0.0,
                                g: 1.0,
                                b: 0.0,
                                a: 1.0,
                            },
                            point_size: 5.0,
                        },
                        Point3 {
                            p: trig_points[1],
                            color: Color {
                                r: 0.0,
                                g: 1.0,
                                b: 0.0,
                                a: 1.0,
                            },
                            point_size: 5.0,
                        },
                        Point3 {
                            p: trig_points[2],
                            color: Color {
                                r: 0.0,
                                g: 1.0,
                                b: 0.0,
                                a: 1.0,
                            },
                            point_size: 5.0,
                        },
                    ];

                    let mesh = vec![Triangle3 {
                        p0: trig_points[0],
                        p1: trig_points[1],
                        p2: trig_points[2],
                        color: Color {
                            r: 1.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.5,
                        },
                    }];

                    let line3 = vec![
                        Line3 {
                            p0: SVector::<f32, 3>::new(0.0, 0.0, 0.0),
                            p1: SVector::<f32, 3>::new(1.0, 0.0, 0.0),
                            color: Color {
                                r: 1.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            },
                            line_width: 5.0,
                        },
                        Line3 {
                            p0: SVector::<f32, 3>::new(0.0, 0.0, 0.0),
                            p1: SVector::<f32, 3>::new(0.0, 1.0, 0.0),
                            color: Color {
                                r: 0.0,
                                g: 1.0,
                                b: 0.0,
                                a: 1.0,
                            },
                            line_width: 5.0,
                        },
                        Line3 {
                            p0: SVector::<f32, 3>::new(0.0, 0.0, 0.0),
                            p1: SVector::<f32, 3>::new(0.0, 0.0, 1.0),
                            color: Color {
                                r: 0.0,
                                g: 0.0,
                                b: 1.0,
                                a: 1.0,
                            },
                            line_width: 5.0,
                        },
                    ];

                    renderables.push(Renderable::Points2(Points2 {
                        name: "points2".to_owned(),
                        points: points2,
                    }));
                    renderables.push(Renderable::Points3(Points3 {
                        name: "points23".to_owned(),
                        points: points3,
                    }));
                    renderables.push(Renderable::Triangles3(Triangles3 {
                        name: "points2".to_owned(),
                        mesh,
                    }));

                    renderables.push(Renderable::Lines3(Lines3 {
                        name: "line3".to_owned(),
                        lines: line3,
                    }));
                }

                let proj_line2 = vec![
                    Line2 {
                        p0: state
                            .intrinsics
                            .cam_proj(
                                &state
                                    .scene_from_camera
                                    .inverse()
                                    .transform(&trig_points[0].cast()),
                            )
                            .cast(),
                        p1: state
                            .intrinsics
                            .cam_proj(
                                &state
                                    .scene_from_camera
                                    .inverse()
                                    .transform(&trig_points[1].cast()),
                            )
                            .cast(),
                        color: Color {
                            r: 0.0,
                            g: 1.0,
                            b: 0.0,
                            a: 0.5,
                        },
                        line_width: 2.0,
                    },
                    Line2 {
                        p0: state
                            .intrinsics
                            .cam_proj(
                                &state
                                    .scene_from_camera
                                    .inverse()
                                    .transform(&trig_points[1].cast()),
                            )
                            .cast(),
                        p1: state
                            .intrinsics
                            .cam_proj(
                                &state
                                    .scene_from_camera
                                    .inverse()
                                    .transform(&trig_points[2].cast()),
                            )
                            .cast(),
                        color: Color {
                            r: 0.0,
                            g: 1.0,
                            b: 0.0,
                            a: 0.5,
                        },
                        line_width: 2.0,
                    },
                    Line2 {
                        p0: state
                            .intrinsics
                            .cam_proj(
                                &state
                                    .scene_from_camera
                                    .inverse()
                                    .transform(&trig_points[2].cast()),
                            )
                            .cast(),
                        p1: state
                            .intrinsics
                            .cam_proj(
                                &state
                                    .scene_from_camera
                                    .inverse()
                                    .transform(&trig_points[0].cast()),
                            )
                            .cast(),
                        color: Color {
                            r: 0.0,
                            g: 1.0,
                            b: 0.0,
                            a: 0.5,
                        },
                        line_width: 2.0,
                    },
                ];

                renderables.push(Renderable::Lines2(Lines2 {
                    name: "proj_line".to_owned(),
                    lines: proj_line2,
                }));
                state.counter += 1;

                if state.counter > 20 {
                    state.show = !state.show;
                    state.counter = 0;
                }

                let lines = if state.show {
                    vec![
                        Line2 {
                            p0: SVector::<f32, 2>::new(16.0, 12.0),
                            p1: SVector::<f32, 2>::new(32.0, 24.0),
                            color: Color {
                                r: 1.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            },
                            line_width: 5.0,
                        },
                        Line2 {
                            p0: SVector::<f32, 2>::new(-0.5, -0.5),
                            p1: SVector::<f32, 2>::new(0.5, 0.5),
                            color: Color {
                                r: 1.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            },
                            line_width: 5.0,
                        },
                    ]
                } else {
                    vec![]
                };

                renderables.push(Renderable::Lines2(Lines2 {
                    name: "l".to_owned(),
                    lines,
                }));

                outbound.packets.send(renderables);

                request.scene_from_camera_request.send_request(());
            }
            ContentGeneratorMessage::SceneFromCamera(reply) => {
                state.scene_from_camera = reply.reply;
            }
        }
    }
}

impl InboundMessageNew<f64> for ContentGeneratorMessage {
    fn new(_inbound_name: String, msg: f64) -> Self {
        ContentGeneratorMessage::ClockTick(msg)
    }
}

impl InboundMessageNew<ReplyMessage<Isometry3<f64>>> for ContentGeneratorMessage {
    fn new(_inbound_name: String, scene_from_camera: ReplyMessage<Isometry3<f64>>) -> Self {
        ContentGeneratorMessage::SceneFromCamera(scene_from_camera)
    }
}

pub async fn run_viewer_example() {
    // Camera / view pose parameters
    let intrinsics = KannalaBrandtCamera::<f64>::new(
        &V::<8>::from_array([600.0, 600.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0]),
        ImageSize {
            width: 640,
            height: 480,
        },
    );
    let scene_from_camera = Isometry3::from_t(&V::<3>::new(0.0, 0.0, -5.0));
    let clipping_planes = WgpuClippingPlanes {
        near: 0.1,
        far: 1000.0,
    };
    let camera = ViewerCamera {
        intrinsics,
        clipping_planes,
        scene_from_camera,
    };

    let mut builder = ViewerBuilder::new(camera);

    // Pipeline configuration
    let pipeline = hollywood::compute::Context::configure(&mut |context| {
        // Actor creation:
        // 1. Periodic timer to drive the simulation
        let mut timer = hollywood::actors::Periodic::new_with_period(context, 0.01);
        // 2. The content generator of the example
        let mut content_generator = ContentGenerator::from_prop_and_state(
            context,
            NullProp {},
            ContentGeneratorState {
                counter: 0,
                show: false,
                intrinsics,
                scene_from_camera,
            },
        );
        // 3. The viewer actor
        let mut viewer =
            ViewerActor::from_prop_and_state(context, ViewerProp {}, builder.viewer_state.clone());

        // Pipeline connections:
        timer
            .outbound
            .time_stamp
            .connect(context, &mut content_generator.inbound.clock_tick);
        content_generator
            .outbound
            .packets
            .connect(context, &mut viewer.inbound.packets);
        content_generator
            .request
            .scene_from_camera_request
            .connect(context, &mut viewer.inbound.request_view_pose);
    });

    // The cancel_requester is used to cancel the pipeline.
    builder.cancel_request_sender = pipeline.cancel_request_sender_template.clone();

    // Plot the pipeline graph to the console.
    pipeline.print_flow_graph();

    // Pipeline execution:

    // 1. Run the pipeline on a separate thread.
    let pipeline_handle = tokio::spawn(pipeline.run());
    // 2. Run the viewer on the main thread. This is a blocking call.
    run_viewer_on_man_thread(builder);
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
