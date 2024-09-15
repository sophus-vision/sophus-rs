use crate::frame::Frame;
use crate::renderable2d::make_line2;
use crate::renderable2d::make_point2;
use crate::renderable3d::make_line3;
use crate::renderable3d::make_mesh3_at;
use crate::renderable3d::make_point3;
use sophus::examples::viewer_example::make_example_image;
use sophus::image::ImageSize;
use sophus::prelude::IsVector;
use sophus::viewer::renderables::*;
use sophus_core::linalg::VecF64;
use sophus_image::intensity_image::intensity_arc_image::IsIntensityArcImage;
use sophus_image::mut_image::MutImageF32;
use sophus_image::mut_image_view::IsMutImageView;
use sophus_lie::Isometry3;
use sophus_sensor::dyn_camera::DynCameraF64;
use sophus_viewer::renderables::color::Color;
use sophus_viewer::renderables::renderable2d::View2dPacket;
use sophus_viewer::renderables::renderable3d::View3dPacket;
use sophus_viewer::renderer::camera::clipping_planes::ClippingPlanes;
use sophus_viewer::renderer::camera::properties::RenderCameraProperties;
use sophus_viewer::renderer::camera::RenderCamera;
use sophus_viewer::viewer::simple_viewer::SimpleViewer;

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

    packet_2d.renderables2d.push(make_point2(
        "points2",
        &[[16.0, 12.0], [32.0, 24.0]],
        &Color::red(),
        5.0,
    ));
    packet_2d.renderables2d.push(make_line2(
        "lines2",
        &[[[0.0, 0.0], [20.0, 20.0]]],
        &Color::red(),
        5.0,
    ));

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

    packet_2d.renderables2d.push(make_line2(
        "lines2",
        &[[[-0.5, -0.5], [0.5, 0.5]], [[0.5, -0.5], [-0.5, 0.5]]],
        &Color::red(),
        2.0,
    ));

    Packet::View2d(packet_2d)
}

fn create_view3_packet() -> Packet {
    let initial_camera = RenderCamera {
        properties: RenderCameraProperties::new(
            DynCameraF64::new_unified(
                &VecF64::from_array([500.0, 500.0, 300.0, 200.0, 0.629, 1.02]),
                ImageSize::new(639, 479),
            ),
            ClippingPlanes::default(),
        ),
        scene_from_camera: Isometry3::trans_z(-5.0),
    };
    let mut packet_3d = View3dPacket {
        view_label: "view_3d".to_owned(),
        renderables3d: vec![],
        initial_camera: initial_camera.clone(),
    };

    let trig_points = [[0.0, 0.0, -0.1], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];

    packet_3d
        .renderables3d
        .push(make_point3("points3", &trig_points, &Color::red(), 5.0));

    packet_3d.renderables3d.push(make_line3(
        "lines3",
        &[
            [[0.0, 0.1, 0.0], [0.1, 0.2, 0.0]],
            [[0.1, 0.2, 0.0], [0.2, 0.3, 0.0]],
            [[0.2, 0.3, 0.0], [0.3, 0.4, 0.0]],
            [[0.3, 0.4, 0.0], [0.4, 0.5, 0.0]],
            [[0.4, 0.5, 0.0], [0.5, 0.6, 0.0]],
            [[0.5, 0.6, 0.0], [0.6, 0.7, 0.0]],
            [[0.6, 0.7, 0.0], [0.7, 0.8, 0.0]],
            [[0.7, 0.8, 0.0], [0.8, 0.9, 0.0]],
            [[0.8, 0.9, 0.0], [0.9, 1.0, 0.0]],
            [[0.9, 1.0, 0.0], [1.0, 1.1, 0.0]],
        ],
        &Color::green(),
        5.0,
    ));

    let blue = Color::blue();
    packet_3d.renderables3d.push(make_mesh3_at(
        "mesh",
        &[(trig_points, blue)],
        Isometry3::trans_z(3.0),
    ));

    Packet::View3d(packet_3d)
}

pub async fn run_viewer_example() {
    let (message_tx, message_rx) = std::sync::mpsc::channel();

    tokio::spawn(async move {
        let mut packets = Packets { packets: vec![] };
        packets.packets.push(create_view3_packet());
        packets.packets.push(create_view2_packet());
        packets.packets.push(create_tiny_view2_packet());
        message_tx.send(packets).unwrap();
    });

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([640.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,

        ..Default::default()
    };
    eframe::run_native(
        "Viewer Example",
        options,
        Box::new(|cc| {
            Ok(SimpleViewer::new(
                sophus_viewer::RenderContext::from_egui_cc(cc),
                message_rx,
            ))
        }),
    )
    .unwrap();
}

fn main() {
    env_logger::init();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            run_viewer_example().await;
        })
}
