use std::f64::consts::{
    PI,
    TAU,
};

use crossbeam_channel::Sender;
use log::warn;
use sophus_autodiff::{
    linalg::{
        MatF64,
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
use sophus_lie::{
    Isometry3,
    Isometry3F64,
    Rotation3,
};
use sophus_renderer::{
    camera::{
        ClippingPlanes,
        RenderCamera,
        RenderCameraProperties,
    },
    renderables::{
        Capsule3,
        Color,
        Cone3,
        Cylinder3,
        Ellipse2,
        Ellipsoid3,
        ImageFrame,
        Planar3,
        SceneRenderable,
        make_axes_arrows3,
        make_axes_arrows3_at,
        make_capsule3,
        make_capsule3_at,
        make_cone3,
        make_cylinder3,
        make_ellipsoid_rings3,
        make_ellipsoid3,
        make_line2,
        make_line3,
        make_mesh3_at,
        make_planar3,
        make_point2,
        make_point3,
        make_sphere3,
        make_textured_mesh3_at,
        named_ellipse2,
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
    update_scene_camera_properties,
};

/// Makes a small checkerboard texture, for the texture-mapped mesh of the example scene
pub fn make_checkerboard_texture(image_size: ImageSize) -> ArcImage4U8 {
    let mut img =
        MutImage4U8::from_image_size_and_val(image_size, SVec::<u8, 4>::new(255, 255, 255, 255));
    for v in 0..image_size.height {
        for u in 0..image_size.width {
            *img.mut_pixel(u, v) = match (u / 8 + v / 8) % 2 == 0 {
                true => SVec::<u8, 4>::new(40, 90, 200, 255),
                false => SVec::<u8, 4>::new(250, 200, 60, 255),
            };
        }
    }
    img.to_shared()
}

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
/// The camera the image view looks through: wide enough that a straight line in the world is
/// visibly a curve in the image, which is the point of showing it.
pub fn distorted_camera() -> DynCameraF64 {
    DynCameraF64::new_enhanced_unified(
        VecF64::from_array([500.0, 500.0, 320.0, 240.0, 0.629, 1.22]),
        ImageSize::new(638, 479),
    )
}

/// A frame of the park, seen through that camera - what the image view annotates.
///
/// This is the case an image view is for: marks made *on a picture*, over the thing the picture is
/// of. Rendering it wants a device of its own, and on the web a device cannot be waited for, so
/// there the ruled frame below stands in.
pub fn make_park_frame() -> ImageFrame {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(frame) = render_park_frame() {
        return frame;
    }
    make_distorted_frame()
}

/// Where the picture of the park is taken from, as `scene_from_camera`.
///
/// Worth keeping, and not only worth rendering from: an image view has no camera to move, so it
/// draws its 3d renderables in the camera's own frame. Anything meant to line up with what is in
/// the picture - a landmark, a box round a thing in it - is therefore given in the scene and
/// carried over by the inverse of this. Without it the picture says where everything is and the
/// annotations cannot say anything about it.
pub fn park_frame_pose() -> Isometry3F64 {
    look_at(
        VecF64::<3>::new(-1.3, -6.6, 1.7),
        VecF64::<3>::new(0.2, 1.4, 1.2),
    )
}

/// Renders that frame, if a device can be had and it comes out the size the camera says.
#[cfg(not(target_arch = "wasm32"))]
fn render_park_frame() -> Option<ImageFrame> {
    use sophus_renderer::RenderContext;
    use sophus_sim::camera_simulator::CameraSimulator;

    let properties = RenderCameraProperties::from_intrinsics(&distorted_camera());
    let mut simulator =
        CameraSimulator::new(&pollster::block_on(RenderContext::try_new())?, &properties);
    // the park without the survey: a camera standing in it sees the park, not the notes somebody
    // has made about it
    simulator.update_3d_renderables(make_park_scenery());
    let image = pollster::block_on(simulator.render(park_frame_pose())).rgba_image;
    ImageFrame::try_from(&image, &properties)
}

/// A frame ruled in red and blue, for when the park cannot be rendered.
pub fn make_distorted_frame() -> ImageFrame {
    let unified_cam = distorted_camera();
    let image_size = unified_cam.image_size();

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
        frame: Some(make_park_frame()),
        delete: false,
    };

    // Corners of that picture, from a Shi-Tomasi detector run over it once, off to the side: the
    // smaller eigenvalue of the structure tensor, suppressed to one maximum every ten pixels and
    // cut at a threshold which leaves a few dozen. The frame is rendered the same way every time,
    // so these are where the corners were found rather than where they would have looked good -
    // which is the whole point of marking them.
    //
    // A marker is measured in view-port pixels and anchored in image ones, so it stays on its
    // corner and keeps its size however far the view is zoomed in on it.
    let corners = [
        [152.0, 176.0],
        [533.0, 112.0],
        [490.0, 112.0],
        [335.0, 171.0],
        [558.0, 205.0],
        [189.0, 172.0],
        [378.0, 187.0],
        [183.0, 206.0],
        [198.0, 203.0],
        [344.0, 187.0],
        [221.0, 171.0],
        [595.0, 206.0],
        [328.0, 230.0],
        [70.0, 122.0],
        [511.0, 136.0],
        [576.0, 170.0],
        [168.0, 134.0],
        [507.0, 210.0],
        [470.0, 158.0],
        [361.0, 156.0],
        [83.0, 143.0],
        [11.0, 212.0],
        [217.0, 208.0],
        [520.0, 210.0],
        [324.0, 215.0],
        [338.0, 206.0],
        [511.0, 96.0],
        [419.0, 208.0],
        [218.0, 375.0],
        [232.0, 197.0],
        [273.0, 227.0],
        [233.0, 227.0],
        [324.0, 189.0],
        [232.0, 213.0],
        [65.0, 361.0],
        [90.0, 365.0],
    ];

    // What those corners say about the thing they sit on, summarised two ways.
    //
    // The cluster is the eight of them nearest the middle of the billboard, which is a thing in
    // the picture with corners of its own; the box and the ellipse below are both worked out from
    // that set rather than drawn around it by eye, so they say something about the data instead of
    // decorating it.
    let seed = [340.0f32, 195.0];
    let mut cluster = corners.to_vec();
    cluster.sort_by(|a, b| {
        let to_seed = |p: &[f32; 2]| (p[0] - seed[0]).powi(2) + (p[1] - seed[1]).powi(2);
        to_seed(a).total_cmp(&to_seed(b))
    });
    cluster.truncate(8);

    // The eight are drawn apart from the rest, so that what the box and the ellipse below were
    // worked out from can be told from what they were not.
    image_packet.pixel_renderables.push(make_point2(
        "corners",
        &corners
            .iter()
            .filter(|corner| !cluster.contains(corner))
            .copied()
            .collect::<Vec<_>>(),
        &Color::red(),
        5.0,
    ));
    image_packet
        .pixel_renderables
        .push(make_point2("cluster", &cluster, &Color::cyan(), 6.0));

    // the axis-aligned box which holds them, with a pixel of air around it
    let bound = |axis: usize, pick: fn(f32, f32) -> f32| {
        cluster.iter().fold(cluster[0][axis], |carried, point| {
            pick(carried, point[axis])
        })
    };
    let (low, high) = (
        [bound(0, f32::min) - 1.0, bound(1, f32::min) - 1.0],
        [bound(0, f32::max) + 1.0, bound(1, f32::max) + 1.0],
    );
    image_packet.pixel_renderables.push(make_line2(
        "bounding-box",
        &[
            [[low[0], low[1]], [high[0], low[1]]],
            [[high[0], low[1]], [high[0], high[1]]],
            [[high[0], high[1]], [low[0], high[1]]],
            [[low[0], high[1]], [low[0], low[1]]],
        ],
        &Color::yellow(),
        1.5,
    ));

    // and the ellipse of their spread: the mean and the sample covariance of the same eight
    // points, which says which way they lie as well as how far they reach - the box can only say
    // how far, and only along the two axes it happens to be aligned with
    let mean = cluster.iter().fold([0.0f64, 0.0], |carried, point| {
        [
            carried[0] + point[0] as f64 / cluster.len() as f64,
            carried[1] + point[1] as f64 / cluster.len() as f64,
        ]
    });
    let spread = cluster.iter().fold([0.0f64; 3], |carried, point| {
        let (du, dv) = (point[0] as f64 - mean[0], point[1] as f64 - mean[1]);
        [
            carried[0] + du * du,
            carried[1] + du * dv,
            carried[2] + dv * dv,
        ]
    });
    let normalise = (cluster.len() - 1) as f64;
    let covariance = MatF64::<2, 2>::new(
        spread[0] / normalise,
        spread[1] / normalise,
        spread[1] / normalise,
        spread[2] / normalise,
    );
    let centre = SVec::<f32, 2>::new(mean[0] as f32, mean[1] as f32);
    image_packet.pixel_renderables.push(named_ellipse2(
        "spread",
        Ellipse2::from_covariance(centre, covariance, 2.0, 1.5, Color::green())
            .into_iter()
            .collect(),
    ));
    // the same fit one sigma further out, filled rather than outlined - a line width of zero -
    // for when what matters is that the region is there rather than where it ends
    image_packet.pixel_renderables.push(named_ellipse2(
        "spread-outer",
        Ellipse2::from_covariance(
            centre,
            covariance,
            3.0,
            0.0,
            Color {
                r: 0.16,
                g: 0.72,
                b: 0.26,
                a: 0.16,
            },
        )
        .into_iter()
        .collect(),
    ));

    // A cylinder standing on the plaza: a 3d object placed over the picture, which is what an
    // image view's scene renderables are for. It is given where it stands in the *scene* and
    // carried into the camera's frame by the pose the picture was taken from - the frame an image
    // view draws them in, and what that pose is kept for.
    //
    // Traced, like the park it stands on: the same intersection a capsule uses, which is the
    // cylinder about the segment clipped to the segment's own extent, with the ends cut flat and
    // closed by a disk apiece rather than rounded over. So it is as round as its radius says at
    // any distance, and drawn as edges it shows its own silhouette rather than a tessellation.
    let camera_from_scene = park_frame_pose().inverse();
    let in_camera = |height: f64| {
        let point = camera_from_scene.transform(VecF64::<3>::new(0.0, 0.0, height));
        SVec::<f32, 3>::new(point[0] as f32, point[1] as f32, point[2] as f32)
    };
    image_packet.scene_renderables.append(
        &mut make_cylinder3(
            "column",
            // Short and wide, so that the disk closing the bottom sits on the ground and the one
            // closing the top is seen as the ellipse it projects to - which is what says the
            // thing has a round end and is standing there rather than pasted on.
            &[Cylinder3 {
                from: in_camera(0.0),
                to: in_camera(0.55),
                // as wide as the paved circle it stands on
                radius: 2.2,
                color: Color::green(),
            }],
        )
        .into_iter()
        .map(|renderable| renderable.wireframe(true))
        .collect(),
    );

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

/// The pose of a camera standing at `eye` and looking at `target`, in a scene where z is up.
fn look_at(eye: VecF64<3>, target: VecF64<3>) -> Isometry3<f64, 1, 0, 0> {
    let forward = (target - eye).normalize();
    let up = VecF64::<3>::new(0.0, 0.0, 1.0);
    // the camera's y is "down", which is the part of -z of the scene perpendicular to the view
    let down = (-up + up.dot(&forward) * forward).normalize();
    let right = down.cross(&forward);
    let rotation = Rotation3::try_from_mat(MatF64::<3, 3>::from_columns(&[right, down, forward]))
        .expect("the three axes are orthonormal");
    Isometry3::from_rotation_and_translation(rotation, eye)
}

/// A colour, opaque.
fn rgb(r: f32, g: f32, b: f32) -> Color {
    Color { r, g, b, a: 1.0 }
}

/// A conifer: a trunk with three tiers of foliage.
///
/// Each tier is a cone closed with a disk, since a traced cone is only the curved surface - and
/// the disk is what the tree stands on when it is looked at from below.
/// A cylinder, pushed into the clouds the park batches its primitives into: the side, and a disk
/// closing each end.
///
/// The same three pieces [make_cylinder3] builds, put where a scene which batches for itself can
/// use them - a trunk is a cylinder, and the flat ends are what tell it from a capsule.
fn cylinder(
    from: SVec<f32, 3>,
    to: SVec<f32, 3>,
    radius: f32,
    color: Color,
    capsules: &mut Vec<Capsule3>,
    planars: &mut Vec<Planar3>,
) {
    let along = to - from;
    if along.norm() < 1e-9 {
        return;
    }
    capsules.push(Capsule3 {
        from,
        to,
        radius,
        color,
        flat_ends: true,
    });
    let direction = along / along.norm();
    planars.push(Planar3::disk(to, direction, radius, color));
    planars.push(Planar3::disk(from, -direction, radius, color));
}

/// Where the tiers of a conifer sit, as fractions of its height: the foot of the tier, how tall
/// it is, and how wide at the bottom. Shared with [conifer_samples], so that points taken over a
/// tree are points on the tree which was drawn.
const CONIFER_TIERS: [(f32, f32, f32); 3] =
    [(0.26, 0.36, 0.30), (0.50, 0.32, 0.24), (0.72, 0.30, 0.17)];

fn conifer(
    at: [f32; 2],
    height: f32,
    capsules: &mut Vec<Capsule3>,
    cones: &mut Vec<Cone3>,
    planars: &mut Vec<Planar3>,
) {
    let [x, y] = at;
    cylinder(
        SVec::<f32, 3>::new(x, y, 0.0),
        SVec::<f32, 3>::new(x, y, 0.34 * height),
        0.045 * height,
        rgb(0.36, 0.25, 0.17),
        capsules,
        planars,
    );
    // three tiers, each starting inside the one below so no gap opens between them
    for ((base, tier_height, radius), color) in CONIFER_TIERS.into_iter().zip([
        rgb(0.12, 0.37, 0.20),
        rgb(0.15, 0.43, 0.23),
        rgb(0.19, 0.49, 0.26),
    ]) {
        let base_z = base * height;
        let tier = tier_height * height;
        let radius = radius * height;
        cones.push(Cone3 {
            apex: SVec::<f32, 3>::new(x, y, base_z + tier),
            axis: SVec::<f32, 3>::new(0.0, 0.0, -tier),
            radius,
            color,
        });
        planars.push(Planar3::disk(
            SVec::<f32, 3>::new(x, y, base_z),
            SVec::<f32, 3>::new(0.0, 0.0, -1.0),
            radius,
            color,
        ));
    }
}

/// Points taken over a conifer: around the rim of each tier, up its trunk, and at its tip.
///
/// Something to fit a shape to - the sort of cloud a survey of a tree would leave behind.
fn conifer_samples(at: [f32; 2], height: f32) -> Vec<[f32; 3]> {
    let [x, y] = at;
    let mut samples = vec![];
    for (base, tier_height, radius) in CONIFER_TIERS {
        for i in 0..12 {
            let angle = TAU as f32 * i as f32 / 12.0;
            samples.push([
                x + radius * height * angle.cos(),
                y + radius * height * angle.sin(),
                base * height,
            ]);
        }
        samples.push([x, y, (base + tier_height) * height]);
    }
    for i in 0..4 {
        samples.push([x, y, 0.34 * height * i as f32 / 4.0]);
    }
    samples
}

/// A broadleaf tree: a trunk with two lobes of canopy, each an ellipsoid.
fn broadleaf(
    at: [f32; 2],
    height: f32,
    capsules: &mut Vec<Capsule3>,
    planars: &mut Vec<Planar3>,
    ellipsoids: &mut Vec<Ellipsoid3>,
) {
    let [x, y] = at;
    cylinder(
        SVec::<f32, 3>::new(x, y, 0.0),
        SVec::<f32, 3>::new(x, y, 0.55 * height),
        0.04 * height,
        rgb(0.38, 0.27, 0.18),
        capsules,
        planars,
    );
    for (offset, semi_axes, color) in [
        ([0.0, 0.0, 0.72], [0.34, 0.30, 0.26], rgb(0.28, 0.53, 0.24)),
        (
            [0.16, -0.12, 0.56],
            [0.22, 0.20, 0.17],
            rgb(0.24, 0.47, 0.21),
        ),
    ] {
        ellipsoids.push(Ellipsoid3 {
            center: SVec::<f32, 3>::new(
                x + offset[0] * height,
                y + offset[1] * height,
                offset[2] * height,
            ),
            shape: MatF64::<3, 3>::from_diagonal(&VecF64::<3>::new(
                (semi_axes[0] * height) as f64,
                (semi_axes[1] * height) as f64,
                (semi_axes[2] * height) as f64,
            )),
            color,
        });
    }
}

/// A poster for the billboard: a radial star.
///
/// This is the pattern texture filtering is judged on. Its spokes converge until neighbouring
/// pixels straddle several of them at once, so a texture which is point sampled, or filtered only
/// along one axis, tears into moire rings right where the star is densest.
pub fn make_poster_texture(image_size: ImageSize) -> ArcImage4U8 {
    let paper = SVec::<u8, 4>::new(248, 244, 236, 255);
    let ink = SVec::<u8, 4>::new(38, 62, 110, 255);
    let band = SVec::<u8, 4>::new(214, 106, 44, 255);

    let mut img = MutImage4U8::from_image_size_and_val(image_size, paper);
    let w = image_size.width as f32;
    let h = image_size.height as f32;
    let header = 0.22 * h;
    let center = (0.5 * w, header + 0.5 * (h - header));

    for v in 0..image_size.height {
        for u in 0..image_size.width {
            let (x, y) = (u as f32 + 0.5, v as f32 + 0.5);
            let border = x < 0.03 * w || x > 0.97 * w || y < 0.03 * h || y > 0.97 * h;
            let pixel = if border {
                ink
            } else if y < header {
                band
            } else {
                // 28 spokes about the centre, and a plain disk at the middle where they would
                // otherwise be finer than the texture itself can hold
                let (dx, dy) = (x - center.0, y - center.1);
                let spoke = (dy.atan2(dx) / TAU as f32 * 28.0).floor() as i32;
                match dx.hypot(dy) < 0.06 * h || spoke.rem_euclid(2) == 0 {
                    true => ink,
                    false => paper,
                }
            };
            *img.mut_pixel(u, v) = pixel;
        }
    }
    img.to_shared()
}

/// The example scene: a small park, and a survey running through it.
///
/// Split out from the view it is shown in so that it can be built, and rendered, without a
/// window.
pub fn make_park_scene() -> Vec<SceneRenderable> {
    park(true)
}

/// The park itself, without the survey - what a camera standing in it would see.
pub fn make_park_scenery() -> Vec<SceneRenderable> {
    park(false)
}

fn park(with_survey: bool) -> Vec<SceneRenderable> {
    // A small park, and the point of it is that almost none of it is tessellated. A trunk is a
    // capsule, a tier of foliage a cone closed with a disk, a canopy an ellipsoid, the pond an
    // ellipse and the ground an unbounded plane: all of them are intersected with the ray through
    // each pixel by the distortion pass, so they stay exactly as round as their shape says at any
    // distance and under any distortion. Only the billboard and the bench are triangles, which is
    // what triangles are good at - a flat quad is two of them, exactly.
    let mut scene_renderables = vec![];
    let mut capsules = vec![];
    let mut cones = vec![];
    let mut planars = vec![];
    let mut ellipsoids = vec![];

    // The ground: an unbounded plane through the origin with z up, checkered. It cannot be
    // rasterized without picking a size, and its pattern is filtered over each pixel's footprint
    // rather than sampled, so it settles onto an even tone as it recedes instead of seething.
    // Drawn as a wireframe it is ruled instead, at whatever spacing the pixels there can hold.
    planars.push(
        Planar3::plane(
            SVec::<f32, 3>::new(0.0, 0.0, 0.0),
            SVec::<f32, 3>::new(0.0, 0.0, 1.0),
            rgb(0.96, 0.93, 0.85),
        )
        .with_checker(0.5, rgb(0.85, 0.67, 0.44)),
    );

    // A paved circle at the origin: the stone, then its paving lines, then a kerb around it. The
    // lines and the kerb are widths in *pixels* - they are drawn as lines rather than as thin
    // shapes, so they stay legible as the camera pulls away.
    planars.push(Planar3::disk(
        SVec::<f32, 3>::new(0.0, 0.0, 0.004),
        SVec::<f32, 3>::new(0.0, 0.0, 1.0),
        2.2,
        rgb(0.87, 0.84, 0.77),
    ));
    planars.push(
        Planar3::disk(
            SVec::<f32, 3>::new(0.0, 0.0, 0.006),
            SVec::<f32, 3>::new(0.0, 0.0, 1.0),
            2.2,
            rgb(0.66, 0.62, 0.55),
        )
        .with_grid(0.55, 1.5),
    );
    planars.push(
        Planar3::disk(
            SVec::<f32, 3>::new(0.0, 0.0, 0.008),
            SVec::<f32, 3>::new(0.0, 0.0, 1.0),
            2.32,
            rgb(0.55, 0.51, 0.45),
        )
        .outlined(2.5),
    );

    // A pond, which is what an ellipse on a plane is for, with a rim around it.
    let pond = SVec::<f32, 3>::new(3.6, -3.1, 0.004);
    let pond_axes = [
        SVec::<f32, 3>::new(1.7, 0.35, 0.0),
        SVec::<f32, 3>::new(-0.5, 1.05, 0.0),
    ];
    planars.push(Planar3::ellipse(pond, pond_axes, rgb(0.36, 0.62, 0.76)));
    planars.push(
        Planar3::ellipse(
            SVec::<f32, 3>::new(pond.x, pond.y, 0.006),
            [pond_axes[0] * 1.06, pond_axes[1] * 1.06],
            rgb(0.60, 0.55, 0.46),
        )
        .outlined(3.0),
    );

    let conifers = [
        ([-4.4f32, 2.6f32], 3.4f32),
        ([4.1, 3.6], 2.9),
        ([-3.6, -3.8], 3.1),
        ([5.0, 0.8], 2.5),
        ([-2.9, 6.4], 3.7),
        ([2.4, 6.9], 3.2),
    ];
    let broadleaves = [
        ([-3.1f32, -0.4f32], 2.6f32),
        ([3.0, 1.9], 2.3),
        ([-5.6, 4.6], 2.8),
    ];
    for (at, height) in conifers {
        conifer(at, height, &mut capsules, &mut cones, &mut planars);
    }
    for (at, height) in broadleaves {
        broadleaf(at, height, &mut capsules, &mut planars, &mut ellipsoids);
    }

    // Where the billboard stands, which the bench below is turned to face.
    let (board_w, board_h) = (2.6f32, 1.5f32);
    let (board_y, board_bottom) = (6.0f32, 1.15f32);

    // A lamp post: a pole, a globe, and a shade over it - the shade a cone, closed like the
    // conifers' tiers are.
    let lamp = SVec::<f32, 3>::new(1.85, -1.75, 0.0);
    let lamp_top = 2.6;
    capsules.push(Capsule3 {
        from: lamp,
        to: SVec::<f32, 3>::new(lamp.x, lamp.y, lamp_top),
        radius: 0.05,
        color: rgb(0.24, 0.24, 0.27),
        flat_ends: false,
    });
    ellipsoids.push(Ellipsoid3::sphere(
        SVec::<f32, 3>::new(lamp.x, lamp.y, lamp_top + 0.11),
        0.15,
        rgb(1.0, 0.94, 0.72),
    ));
    cones.push(Cone3 {
        apex: SVec::<f32, 3>::new(lamp.x, lamp.y, lamp_top + 0.46),
        axis: SVec::<f32, 3>::new(0.0, 0.0, -0.22),
        radius: 0.28,
        color: rgb(0.24, 0.24, 0.27),
    });
    planars.push(Planar3::disk(
        SVec::<f32, 3>::new(lamp.x, lamp.y, lamp_top + 0.24),
        SVec::<f32, 3>::new(0.0, 0.0, -1.0),
        0.28,
        rgb(0.24, 0.24, 0.27),
    ));

    // A bench, turned to face the billboard.
    //
    // It is built in a frame of its own - x across the seat, y the way it looks, z up - and put
    // where it goes by its `world_from_entity`, which is what aiming it at anything amounts to.
    // Doing it that way rather than in world coordinates is also the only way to keep the seat's
    // winding honest: a triangle is drawn from the side its normal - `(p1 - p0) x (p2 - p0)` -
    // points towards, so the seat is wound to face up, and a turn about z leaves that alone.
    let bench_at = VecF64::<3>::new(-2.05, -2.5, 0.0);
    let towards_board = VecF64::<3>::new(0.0, board_y as f64, 0.0) - bench_at;
    let bench_pose = Isometry3::from_rotation_and_translation(
        Rotation3::rot_z(towards_board.y.atan2(towards_board.x) - 0.5 * PI),
        bench_at,
    );
    let (bench_half, bench_depth, seat_z) = (0.75f32, 0.24f32, 0.45f32);
    let wood = rgb(0.58, 0.38, 0.22);
    let iron = rgb(0.28, 0.26, 0.25);
    scene_renderables.push(make_mesh3_at(
        "bench-seat",
        &[
            (
                [
                    [-bench_half, -bench_depth, seat_z],
                    [bench_half, -bench_depth, seat_z],
                    [bench_half, bench_depth, seat_z],
                ],
                wood,
            ),
            (
                [
                    [-bench_half, -bench_depth, seat_z],
                    [bench_half, bench_depth, seat_z],
                    [-bench_half, bench_depth, seat_z],
                ],
                wood,
            ),
        ],
        bench_pose,
    ));
    // a leg at each corner, and the back carried up from the two behind it, so that nothing is
    // left standing in the air
    let mut bench_frame = vec![];
    for side in [-1.0f32, 1.0] {
        let x = side * (bench_half - 0.08);
        for front in [-1.0f32, 1.0] {
            bench_frame.push(Capsule3 {
                from: SVec::<f32, 3>::new(x, front * (bench_depth - 0.05), 0.0),
                to: SVec::<f32, 3>::new(x, front * (bench_depth - 0.05), seat_z),
                radius: 0.028,
                color: iron,
                flat_ends: false,
            });
        }
        bench_frame.push(Capsule3 {
            from: SVec::<f32, 3>::new(x, -(bench_depth - 0.05), seat_z),
            to: SVec::<f32, 3>::new(x, -(bench_depth + 0.05), seat_z + 0.42),
            radius: 0.028,
            color: iron,
            flat_ends: false,
        });
    }
    for height in [0.24f32, 0.40] {
        let lean = 0.1 * height / 0.42;
        bench_frame.push(Capsule3 {
            from: SVec::<f32, 3>::new(-bench_half, -(bench_depth - 0.05) - lean, seat_z + height),
            to: SVec::<f32, 3>::new(bench_half, -(bench_depth - 0.05) - lean, seat_z + height),
            radius: 0.045,
            color: wood,
            flat_ends: false,
        });
    }
    scene_renderables.push(make_capsule3_at("bench-frame", bench_frame, bench_pose));

    // A billboard: a textured quad on two posts, with a frame behind it.
    //
    // The quad is a posed entity - it is built as a unit square in its own frame and stood
    // upright by its `world_from_entity`, which is what every entity is drawn at.
    for side in [-1.0f32, 1.0] {
        capsules.push(Capsule3 {
            from: SVec::<f32, 3>::new(side * 0.42 * board_w, board_y + 0.06, 0.0),
            to: SVec::<f32, 3>::new(
                side * 0.42 * board_w,
                board_y + 0.06,
                board_bottom + 0.3 * board_h,
            ),
            radius: 0.06,
            color: rgb(0.34, 0.31, 0.28),
            flat_ends: false,
        });
    }
    planars.push(Planar3::rectangle(
        SVec::<f32, 3>::new(0.0, board_y + 0.04, board_bottom + 0.5 * board_h),
        [
            SVec::<f32, 3>::new(0.55 * board_w + 0.08, 0.0, 0.0),
            SVec::<f32, 3>::new(0.0, 0.0, 0.5 * board_h + 0.08),
        ],
        rgb(0.30, 0.27, 0.24),
    ));
    // wound so that the normal points along the quad's own -z, which the pose below turns to face
    // the camera; the texture coordinates ride along with the corners they belong to, so the
    // poster reads the right way round rather than mirrored
    let poster = [
        [
            ([0.0, 0.0, 0.0], [0.0, 1.0]),
            ([board_w, 0.0, 0.0], [1.0, 1.0]),
            ([board_w, board_h, 0.0], [1.0, 0.0]),
        ],
        [
            ([0.0, 0.0, 0.0], [0.0, 1.0]),
            ([board_w, board_h, 0.0], [1.0, 0.0]),
            ([0.0, board_h, 0.0], [0.0, 0.0]),
        ],
    ];
    scene_renderables.push(make_textured_mesh3_at(
        "billboard",
        &poster,
        make_poster_texture(ImageSize::new(256, 256)),
        // stood up: the quad's own x runs along the world's x and its y up the world's z, which
        // leaves its -z - the way it was wound to face - pointing down the world's -y, where the
        // camera stands
        Isometry3::from_rotation_and_translation(
            Rotation3::try_from_mat(MatF64::<3, 3>::from_columns(&[
                VecF64::<3>::new(1.0, 0.0, 0.0),
                VecF64::<3>::new(0.0, 0.0, 1.0),
                VecF64::<3>::new(0.0, -1.0, 0.0),
            ]))
            .expect("the three axes are orthonormal"),
            VecF64::<3>::new(-0.5 * board_w as f64, board_y as f64, board_bottom as f64),
        ),
    ));

    if with_survey {
        // A survey through the park, which is what the rest of this library is for: a trajectory,
        // the poses along it, and the landmarks measured from them.
        //
        // The path is drawn as capsules with spheres at the joints rather than as line segments. A
        // line has a width in pixels and no place in the scene; a capsule is a shape in the world,
        // so it thickens as the camera comes closer and is occluded by what stands in front
        // of it.
        let path: Vec<SVec<f32, 3>> = (0..=26)
            .map(|i| {
                let s = i as f32 / 26.0;
                let angle = TAU as f32 * 0.40 * s - 2.35;
                let radius = 5.4 - 2.9 * s;
                SVec::<f32, 3>::new(
                    radius * angle.cos(),
                    radius * angle.sin(),
                    1.25 + 0.18 * (4.0 * s).sin(),
                )
            })
            .collect();
        let trajectory = rgb(0.95, 0.55, 0.15);
        for pair in path.windows(2) {
            capsules.push(Capsule3 {
                from: pair[0],
                to: pair[1],
                radius: 0.022,
                color: trajectory,
                flat_ends: false,
            });
        }
        scene_renderables.push(make_sphere3(
            "path-poses",
            &path
                .iter()
                .map(|p| ([p.x, p.y, p.z], 0.035f32))
                .collect::<Vec<_>>(),
            &trajectory,
        ));

        // Three of those poses in full, as axes: red, green and blue for the camera's own x, y and
        // z, which is x right, y down and z along the way it looks.
        let tip_of = |(at, height): ([f32; 2], f32)| [at[0], at[1], 1.02 * height];
        let landmark = SVec::<f32, 3>::from(tip_of(conifers[4]));
        for (i, index) in [3usize, 13, 24].iter().enumerate() {
            let eye = path[*index];
            scene_renderables.append(&mut make_axes_arrows3_at(
                format!("pose-{i}"),
                0.32,
                0.014,
                &rgb(0.15, 0.15, 0.18),
                look_at(
                    VecF64::<3>::new(eye.x as f64, eye.y as f64, eye.z as f64),
                    VecF64::<3>::new(landmark.x as f64, landmark.y as f64, landmark.z as f64),
                ),
            ));
        }

        // What the survey has actually measured: a sparse cloud of landmarks, and a few of the
        // bearings along which they were seen from the last pose.
        //
        // These are the two renderables with a width in *pixels* rather than in metres, and this is
        // what they are for. Everything else here is a thing in the park, and has a size; a
        // landmark is not a sphere of any radius and a bearing is not a rod. Drawn this way
        // they stay legible at any zoom, which is how the bundle adjustment and inverse
        // distance demos draw their points
        // - and, being annotations rather than surfaces, they are already edges, so the wireframe
        // toggle leaves them alone.
        let features: Vec<[f32; 3]> = conifers
            .iter()
            .map(|tree| tip_of(*tree))
            .chain(broadleaves.iter().map(|(at, height)| {
                // the top of the canopy: its centre, plus its semi-axis up
                [at[0], at[1], 0.98 * height]
            }))
            .chain([
                [lamp.x, lamp.y, lamp_top + 0.11],
                [-0.5 * board_w, board_y, board_bottom],
                [0.5 * board_w, board_y, board_bottom],
                [-0.5 * board_w, board_y, board_bottom + board_h],
                [0.5 * board_w, board_y, board_bottom + board_h],
                [pond.x + pond_axes[0].x, pond.y + pond_axes[0].y, 0.0],
                [pond.x - pond_axes[0].x, pond.y - pond_axes[0].y, 0.0],
            ])
            .collect();
        scene_renderables.push(make_point3(
            "landmarks",
            &features,
            &rgb(0.16, 0.30, 0.52),
            5.0,
        ));
        let eye = path[24];
        let sight_lines: Vec<[[f32; 3]; 2]> = features
            .iter()
            .step_by(2)
            .map(|feature| ([eye.x, eye.y, eye.z], *feature))
            .map(|(from, to)| [from, to])
            .collect();
        scene_renderables.push(make_line3(
            "sight-lines",
            &sight_lines,
            &rgb(0.45, 0.55, 0.72),
            1.0,
        ));

        // A tree, surveyed: points taken over it, and the shape those points fit.
        //
        // The mean and the covariance of the cloud are what the ellipsoid is drawn from, and the
        // covariance's eigenvectors are the cloud's principal components - the directions it lies
        // along, longest first. A conifer's are one upright and much longer than the two across
        // it, which is what the three rings show: two tall narrow ones and a small flat one.
        // Drawn as rings rather than as a surface, so the tree it is fitted to stays visible
        // inside it.
        let (surveyed_at, surveyed_height) = conifers[1];
        let samples = conifer_samples(surveyed_at, surveyed_height);
        scene_renderables.push(make_point3("tree-samples", &samples, &Color::yellow(), 4.0));

        let count = samples.len() as f64;
        let as_vector =
            |point: &[f32; 3]| VecF64::<3>::new(point[0] as f64, point[1] as f64, point[2] as f64);
        let mean = samples.iter().fold(VecF64::<3>::zeros(), |carried, point| {
            carried + as_vector(point)
        }) / count;
        let covariance = samples
            .iter()
            .fold(MatF64::<3, 3>::zeros(), |carried, point| {
                let offset = as_vector(point) - mean;
                carried + offset * offset.transpose()
            })
            / (count - 1.0);
        if let Some(fit) = Ellipsoid3::from_covariance(
            mean.cast(),
            covariance,
            2.0,
            Color {
                r: 0.85,
                g: 0.35,
                b: 0.85,
                a: 1.0,
            },
        ) {
            scene_renderables.push(make_ellipsoid_rings3("tree-fit", vec![fit], 2.0));
        }

        // What is known about where one of them is: a three-sigma covariance, which is the shape an
        // ellipsoid is here for. It is drawn see-through, so the landmark inside it stays visible.
        ellipsoids.push(Ellipsoid3::sphere(landmark, 0.06, rgb(0.85, 0.20, 0.35)));
        let spread = MatF64::<3, 3>::from_columns(&[
            VecF64::<3>::new(0.13, 0.04, 0.0),
            VecF64::<3>::new(0.0, 0.07, 0.03),
            VecF64::<3>::new(0.0, 0.0, 0.05),
        ]);
        ellipsoids.push(
            Ellipsoid3::from_covariance(
                landmark,
                spread * spread.transpose(),
                3.0,
                Color {
                    r: 0.85,
                    g: 0.20,
                    b: 0.35,
                    a: 0.35,
                },
            )
            .expect("a covariance built as `A A^T` is positive definite"),
        );
    }

    if with_survey {
        // The origin of the scene, as axes: black shafts with red, green and blue tips.
        scene_renderables.append(&mut make_axes_arrows3("origin", 0.8, Isometry3::identity()));
    }

    scene_renderables.push(make_capsule3("park-capsules", capsules));
    scene_renderables.push(make_cone3("park-cones", cones));
    scene_renderables.push(make_planar3("park-planars", planars));
    scene_renderables.push(make_ellipsoid3("park-ellipsoids", ellipsoids));

    scene_renderables
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
        scene_from_camera: match pinhole {
            // The bird's eye view looks straight down on the ground, along -z of the scene. A
            // camera's own frame is x right, y down, z forward, so pointing z down the world's
            // -z and y along the world's -y puts x to the right of the image and y up it.
            true => Isometry3::from_rotation_and_translation(
                Rotation3::try_from_mat(MatF64::<3, 3>::from_columns(&[
                    VecF64::<3>::new(1.0, 0.0, 0.0),
                    VecF64::<3>::new(0.0, -1.0, 0.0),
                    VecF64::<3>::new(0.0, 0.0, -1.0),
                ]))
                .expect("the three axes are orthonormal"),
                VecF64::<3>::new(0.0, 1.0, 13.0),
            ),
            // and the distorted view stands on the ground and looks across it
            false => look_at(
                VecF64::<3>::new(-0.6, -7.6, 2.3),
                VecF64::<3>::new(0.0, 1.2, 1.1),
            ),
        },
    };

    let scene_renderables = make_park_scene();
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
    /// Focal length of the distorted scene view's camera.
    ///
    /// The camera model is uniform data which is uploaded every frame, so this can be dragged
    /// around freely - shortening it widens the field of view, all the way past what a single
    /// undistorted intermediate can hold.
    pub focal_length: f64,
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
            focal_length: 500.0,
        }
    }

    /// Field of view across the image diagonal, in degrees, for the current focal length.
    pub fn field_of_view_degrees(&self) -> f64 {
        let (alpha, beta) = (0.629, 1.02);
        // half the image diagonal, for the 638 x 479 example camera
        let u = (319.0f64.powi(2) + 239.5f64.powi(2)).sqrt() / self.focal_length;
        let r2 = u * u;
        let gamma = 1.0 - alpha;
        let discriminant = 1.0 - (alpha - gamma) * beta * r2;
        if discriminant < 0.0 {
            return f64::NAN;
        }
        // `k` is the z component of the observed ray, and goes negative beyond 180 degrees
        let k = (1.0 - alpha * alpha * beta * r2) / (alpha * discriminant.sqrt() + gamma);
        2.0 * u.atan2(k).to_degrees()
    }

    /// The distorted scene view's camera, for the current focal length.
    fn distorted_camera_properties(&self) -> RenderCameraProperties {
        RenderCameraProperties::new(
            DynCameraF64::new_enhanced_unified(
                VecF64::from_array([
                    self.focal_length,
                    self.focal_length,
                    320.0,
                    240.0,
                    0.629,
                    1.02,
                ]),
                ImageSize::new(639, 479),
            ),
            ClippingPlanes::default(),
        )
    }

    /// Update the visualizations.
    pub fn update(&mut self) {
        // the camera model of the distorted scene view follows the focal length slider
        if self
            .message_send
            .send(vec![update_scene_camera_properties(
                "scene - distorted",
                self.distorted_camera_properties(),
            )])
            .is_err()
        {
            warn!("Failed to send the camera update, viewer might not be running.");
        }

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
