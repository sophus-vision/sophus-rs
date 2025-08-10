use std::f64::consts::TAU;

use log::warn;
use sophus_autodiff::{
    linalg::{
        SVec,
        VecF64,
    },
    prelude::IsVector,
};
use sophus_image::{
    ArcImage4U8,
    ImageSize,
    MutImage4U8,
    MutImageF32,
    prelude::*,
};
use sophus_lie::Isometry3;
use sophus_renderer::{
    camera::{
        ClippingPlanes,
        RenderCamera,
        RenderCameraProperties,
    },
    renderables::{
        Color,
        ImageFrame,
        make_line2,
        make_line3,
        make_mesh3_at,
        make_point2,
        make_point3,
    },
};
use sophus_sensor::DynCameraF64;
use sophus_viewer::packets::{
    ClearCondition,
    CurveVecStyle,
    CurveVecWithConfStyle,
    ImageViewPacket,
    LineType,
    Packet,
    PlotViewPacket,
    ScalarCurveStyle,
    VerticalLine,
    append_to_scene_packet,
    create_scene_packet,
    delete_image_packet,
    delete_scene_packet,
};
use thingbuf::mpsc::blocking::Sender;

/// Makes example image of image-size
pub fn make_example_image(image_size: ImageSize) -> ArcImage4U8 {
    let mut img =
        MutImage4U8::from_image_size_and_val(image_size, SVec::<u8, 4>::new(255, 255, 255, 255));

    let w = image_size.width;
    let h = image_size.height;

    for i in 0..10 {
        for j in 0..10 {
            img.mut_pixel(i, j).copy_from_slice(&[0, 0, 0, 255]);
            img.mut_pixel(i, h - j - 1)
                .copy_from_slice(&[255, 255, 255, 255]);
            img.mut_pixel(w - i - 1, h - j - 1)
                .copy_from_slice(&[0, 0, 255, 255]);
        }
    }
    img.to_shared()
}

/// Creates a distorted image frame with a red and blue grid
pub fn make_distorted_frame() -> ImageFrame {
    let focal_length = 500.0;

    let image_size = ImageSize::new(638, 479);
    let cx = 320.0;
    let cy = 240.0;

    let unified_cam = DynCameraF64::new_enhanced_unified(
        VecF64::from_array([focal_length, focal_length, cx, cy, 0.629, 1.22]),
        image_size,
    );

    let mut img =
        MutImage4U8::from_image_size_and_val(image_size, SVec::<u8, 4>::new(255, 255, 255, 255));

    for v in 0..image_size.height {
        for u in 0..image_size.width {
            let uv = VecF64::<2>::new(u as f64, v as f64);
            let p_on_z1 = unified_cam.cam_unproj(uv);

            if p_on_z1[0].abs() < 0.5 {
                *img.mut_pixel(u, v) = SVec::<u8, 4>::new(255, 0, 0, 255);

                if p_on_z1[1].abs() < 0.3 {
                    *img.mut_pixel(u, v) = SVec::<u8, 4>::new(0, 0, 255, 255);
                }
            }
        }
    }

    ImageFrame {
        image: Some(img.to_shared()),
        camera_properties: RenderCameraProperties::from_intrinsics(&unified_cam),
    }
}

fn create_distorted_image_packet() -> Packet {
    let mut image_packet = ImageViewPacket {
        view_label: "distorted image".to_string(),
        scene_renderables: vec![],
        pixel_renderables: vec![],
        frame: Some(make_distorted_frame()),
        delete: false,
    };

    image_packet.pixel_renderables.push(make_point2(
        "points2",
        &[[16.0, 12.0], [32.0, 24.0]],
        &Color::red(),
        5.0,
    ));
    image_packet.pixel_renderables.push(make_line2(
        "lines2",
        &[[[0.0, 0.0], [20.0, 20.0]]],
        &Color::blue(),
        5.0,
    ));

    image_packet.scene_renderables.push(make_line3(
        "lines3",
        &[
            [[-0.5, -0.3, 1.0], [-0.5, 0.3, 1.0]],
            [[-0.5, 0.3, 1.0], [0.5, 0.3, 1.0]],
            [[0.5, 0.3, 1.0], [0.5, -0.3, 1.0]],
            [[0.5, -0.3, 1.0], [-0.5, -0.3, 1.0]],
        ],
        &Color::green(),
        5.0,
    ));

    Packet::Image(image_packet)
}

fn create_tiny_image_view_packet() -> Packet {
    let mut img = MutImageF32::from_image_size_and_val(ImageSize::new(3, 2), 1.0);

    *img.mut_pixel(0, 0) = 0.0;
    *img.mut_pixel(0, 1) = 0.5;

    *img.mut_pixel(1, 1) = 0.0;

    *img.mut_pixel(2, 0) = 0.3;
    *img.mut_pixel(2, 1) = 0.6;

    let mut image_packet = ImageViewPacket {
        view_label: "tiny image".to_string(),
        scene_renderables: vec![],
        pixel_renderables: vec![],
        frame: Some(ImageFrame::from_image(&img.to_shared().to_rgba())),
        delete: false,
    };

    image_packet.pixel_renderables.push(make_line2(
        "lines2",
        &[[[-0.5, -0.5], [0.5, 0.5]], [[0.5, -0.5], [-0.5, 0.5]]],
        &Color::red(),
        2.0,
    ));

    Packet::Image(image_packet)
}

fn create_scene(pinhole: bool) -> Vec<Packet> {
    let unified_cam = DynCameraF64::new_enhanced_unified(
        VecF64::from_array([500.0, 500.0, 320.0, 240.0, 0.629, 1.02]),
        ImageSize::new(639, 479),
    );
    let pinhole_cam = DynCameraF64::new_pinhole(
        VecF64::from_array([500.0, 500.0, 320.0, 240.0]),
        ImageSize::new(639, 479),
    );

    let initial_camera = RenderCamera {
        properties: RenderCameraProperties::new(
            match pinhole {
                true => pinhole_cam,
                false => unified_cam,
            },
            ClippingPlanes::default(),
        ),
        scene_from_camera: Isometry3::trans_z(-5.0),
    };

    let mut scene_renderables = vec![];
    let trig_points = [[0.0, 0.0, -0.1], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    scene_renderables.push(make_point3("points3", &trig_points, &Color::red(), 5.0));

    scene_renderables.push(make_line3(
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
    scene_renderables.push(make_mesh3_at(
        "mesh",
        &[(trig_points, blue)],
        Isometry3::trans_z(3.0),
    ));

    let label = match pinhole {
        false => "scene - distorted",
        true => "scene - bird's eye",
    };
    let packets = vec![
        create_scene_packet(label, initial_camera, pinhole),
        append_to_scene_packet(label, scene_renderables),
    ];

    packets
}

/// example of the Viewer
pub struct ViewerExampleWidget {
    /// visualization packet sender
    message_send: Sender<Vec<Packet>>,
    x: f64,
}

impl Drop for ViewerExampleWidget {
    fn drop(&mut self) {
        match self.message_send.send(vec![
            delete_scene_packet("scene - bird's eye"),
            delete_scene_packet("scene - distorted"),
            delete_image_packet("distorted image"),
            delete_image_packet("tiny image"),
            Packet::Plot(vec![PlotViewPacket::Delete("scalar-curve".to_owned())]),
            Packet::Plot(vec![PlotViewPacket::Delete("curve-vec".to_owned())]),
            Packet::Plot(vec![PlotViewPacket::Delete("curve-vec +- e".to_owned())]),
        ]) {
            Ok(_) => {}
            Err(_) => {
                warn!("Failed to send delete packets, viewer might not be running.");
            }
        }
    }
}

impl ViewerExampleWidget {
    /// Create a new simple viewer
    pub fn new(
        message_send: Sender<std::vec::Vec<sophus_viewer::packets::Packet>>,
    ) -> ViewerExampleWidget {
        let mut packets = vec![];
        packets.append(&mut create_scene(true));
        packets.append(&mut create_scene(false));
        packets.push(create_distorted_image_packet());
        packets.push(create_tiny_image_view_packet());
        message_send.send(packets).unwrap();

        ViewerExampleWidget {
            message_send,
            x: 0.0,
        }
    }

    /// Update the visualizations.
    pub fn update(&mut self) {
        let x = self.x;
        let sin_x = x.sin();
        let cos_x = x.cos();
        let tan_x = x.tan().clamp(-1.5, 1.5);

        let v_line = VerticalLine {
            x,
            name: "now".to_owned(),
        };

        let plot_packets = vec![
            PlotViewPacket::append_to_curve(
                ("scalar-curve", "sin"),
                vec![(x, sin_x)].into(),
                ScalarCurveStyle {
                    color: Color::orange(),
                    line_type: LineType::default(),
                },
                ClearCondition { max_x_range: TAU },
                Some(v_line.clone()),
            ),
            PlotViewPacket::append_to_curve_vec3(
                ("curve-vec", ("sin_cos_tan")),
                vec![(x, [sin_x, cos_x, tan_x])].into(),
                CurveVecStyle {
                    colors: [Color::red(), Color::green(), Color::blue()],
                    line_type: LineType::default(),
                },
                ClearCondition { max_x_range: TAU },
                Some(v_line.clone()),
            ),
            PlotViewPacket::append_to_curve_vec2_with_conf(
                ("curve-vec +- e", ("sin_cos")),
                vec![(x, ([sin_x, cos_x], [0.1 * sin_x, 0.1 * sin_x]))].into(),
                CurveVecWithConfStyle {
                    colors: [Color::red(), Color::green()],
                },
                ClearCondition { max_x_range: TAU },
                Some(v_line.clone()),
            ),
        ];

        let packets = vec![Packet::Plot(plot_packets)];
        self.message_send.send(packets).unwrap();

        self.x += 0.01;
    }
}
