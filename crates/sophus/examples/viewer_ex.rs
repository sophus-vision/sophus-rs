use sophus::examples::viewer_example::make_example_image;
use sophus::image::ImageSize;
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
use sophus_viewer::simple_viewer::SimplerViewerBuilder;
use sophus_viewer::simple_viewer::ViewerCamera;

use crate::frame::Frame;
use crate::renderable2d::Lines2;
use crate::renderable3d::Lines3;
use crate::renderable3d::Mesh3;
use crate::renderable3d::Points3;
use crate::renderable3d::Renderable3d;



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


pub async fn run_viewer_example() {
    let (message_tx, message_rx) = tokio::sync::mpsc::unbounded_channel();
    let (cancel_tx, _cancel_rx) = tokio::sync::mpsc::unbounded_channel();

    tokio::spawn(async move {
        let mut packets = Packets { packets: vec![] };
        packets.packets.push(create_view3_packet());
        packets.packets.push(create_view2_packet());
        packets.packets.push(create_tiny_view2_packet());
        message_tx.send(packets).unwrap();
    });

    let builder = SimplerViewerBuilder {
        message_recv: message_rx,
        cancel_request_sender: cancel_tx,
    };
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([640.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,

        ..Default::default()
    };
    eframe::run_native(
        "Egui actor",
        options,
        Box::new(|cc| {
            Ok(SimpleViewer::new(
                builder,
                sophus_viewer::ViewerRenderState::new(cc),
            ))
        }),
    )
    .unwrap();
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
