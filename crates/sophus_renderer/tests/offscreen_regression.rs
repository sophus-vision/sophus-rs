//! Regression tests for the offscreen renderer.
//!
//! These render small scenes head-lessly and assert on the resulting pixels - covering things
//! which are easy to break and impossible to catch by type checking alone: per-entity poses,
//! texture mapping, blending, culling, and the interaction of the 2d zoom with the distortion
//! pass.
//!
//! Tests are skipped (and say so) when no GPU is available, so they do not fail on a head-less
//! host without a software rasterizer.

use sophus_autodiff::linalg::{
    IsVector,
    MatF64,
    SVec,
    VecF64,
};
use sophus_image::{
    ArcImage4U8,
    ImageSize,
    MutImage4U8,
    prelude::{
        IsImageView,
        IsMutImageView,
    },
};
use sophus_lie::{
    IsAffineGroup,
    Isometry3,
    Isometry3F64,
    Rotation2,
    Rotation3,
};
use sophus_renderer::{
    OffscreenRenderer,
    RenderContext,
    TranslationAndScaling,
    camera::RenderCameraProperties,
    renderables::{
        Capsule3,
        Color,
        Cylinder3,
        Ellipse2,
        Ellipsoid3,
        Planar3,
        SceneRenderable,
        make_arrow3,
        make_axes_arrows3,
        make_capsule3,
        make_cylinder3,
        make_ellipsoid_rings3,
        make_ellipsoid3,
        make_ellipsoid3_at,
        make_line2,
        make_mesh3,
        make_mesh3_at,
        make_planar3,
        make_point2,
        make_point3_at,
        make_sphere3,
        make_sphere3_at,
        make_textured_mesh3,
        make_textured_mesh3_at,
        named_ellipse2,
    },
    textures::download_depth,
};
use sophus_sensor::DynCameraF64;

const W: usize = 256;
const H: usize = 256;

/// The renderer under test, or `None` when this host has no GPU.
fn renderer(camera: DynCameraF64) -> Option<(RenderContext, OffscreenRenderer)> {
    let context = pollster::block_on(RenderContext::try_new())?;
    let properties = RenderCameraProperties::from_intrinsics(&camera);
    let renderer = OffscreenRenderer::new(&context, &properties);
    Some((context, renderer))
}

fn pinhole() -> DynCameraF64 {
    DynCameraF64::new_pinhole(
        VecF64::from_array([200.0, 200.0, 127.5, 127.5]),
        ImageSize::new(W, H),
    )
}

/// Renders the current scene from `scene_from_camera` and downloads the result.
fn render(renderer: &mut OffscreenRenderer, scene_from_camera: Isometry3F64) -> ArcImage4U8 {
    render_with(
        renderer,
        scene_from_camera,
        false,
        TranslationAndScaling::identity(),
    )
}

fn render_with(
    renderer: &mut OffscreenRenderer,
    scene_from_camera: Isometry3F64,
    backface_culling: bool,
    zoom: TranslationAndScaling,
) -> ArcImage4U8 {
    renderer
        .render_params(&ImageSize::new(W, H), &scene_from_camera)
        .backface_culling(backface_culling)
        .zoom(zoom)
        .download_rgba(true)
        .render()
        .rgba_image
        .expect("`download_rgba` was requested")
}

/// Center of gravity of all pixels matching `pick`, and how many there were.
fn centroid(image: &ArcImage4U8, pick: impl Fn([u8; 4]) -> bool) -> Option<(VecF64<2>, usize)> {
    let (mut sum, mut count) = (VecF64::<2>::zeros(), 0usize);
    for v in 0..image.image_size().height {
        for u in 0..image.image_size().width {
            let p = image.pixel(u, v);
            if pick([p[0], p[1], p[2], p[3]]) {
                sum += VecF64::<2>::new(u as f64, v as f64);
                count += 1;
            }
        }
    }
    (count > 0).then(|| (sum / count as f64, count))
}

fn is_red(c: [u8; 4]) -> bool {
    c[0] > 200 && c[1] < 80 && c[2] < 80
}
fn is_blue(c: [u8; 4]) -> bool {
    c[2] > 200 && c[0] < 80 && c[1] < 80
}
fn is_green(c: [u8; 4]) -> bool {
    c[1] > 200 && c[0] < 80 && c[2] < 80
}

fn point_at(name: &str, pose: Isometry3F64, color: Color) -> SceneRenderable {
    make_point3_at(name, &[[0.0f32, 0.0, 0.0]], &color, 20.0, pose)
}

/// Each entity must be drawn at its *own* `world_from_entity`.
///
/// The pose of every entity is written into the same uniform buffer, and `Queue::write_buffer`
/// does not interleave with the draw calls of a render pass which is being encoded. Writing all
/// entities to one shared slot therefore drew every entity at the pose of whichever entity was
/// written last.
#[test]
fn each_entity_is_drawn_at_its_own_pose() {
    let camera = pinhole();
    let Some((_context, mut renderer)) = renderer(camera.clone()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let red_pose = Isometry3::from_translation(VecF64::<3>::new(0.0, 0.0, 2.0));
    let blue_pose = Isometry3::from_translation(VecF64::<3>::new(0.5, 0.0, 2.0));
    renderer.update_scene(vec![
        point_at("red", red_pose, Color::red()),
        point_at("blue", blue_pose, Color::blue()),
    ]);

    let image = render(&mut renderer, Isometry3F64::identity());

    let (red, _) = centroid(&image, is_red).expect("the red point must be visible");
    let (blue, _) = centroid(&image, is_blue).expect("the blue point must be visible");

    approx::assert_abs_diff_eq!(red, camera.cam_proj(red_pose.translation()), epsilon = 1.0);
    approx::assert_abs_diff_eq!(
        blue,
        camera.cam_proj(blue_pose.translation()),
        epsilon = 1.0
    );
}

/// A textured mesh must sample its own texture.
#[test]
fn textured_mesh_samples_its_texture() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // left half red, right half green
    let mut texture = MutImage4U8::from_image_size_and_val(
        ImageSize::new(64, 64),
        SVec::<u8, 4>::new(255, 0, 0, 255),
    );
    for v in 0..64 {
        for u in 32..64 {
            *texture.mut_pixel(u, v) = SVec::<u8, 4>::new(0, 255, 0, 255);
        }
    }
    let quad = [
        [
            ([-1.0, -1.0, 0.0], [0.0, 1.0]),
            ([1.0, -1.0, 0.0], [1.0, 1.0]),
            ([1.0, 1.0, 0.0], [1.0, 0.0]),
        ],
        [
            ([-1.0, -1.0, 0.0], [0.0, 1.0]),
            ([1.0, 1.0, 0.0], [1.0, 0.0]),
            ([-1.0, 1.0, 0.0], [0.0, 0.0]),
        ],
    ];
    renderer.update_scene(vec![make_textured_mesh3_at(
        "quad",
        &quad,
        texture.to_shared(),
        Isometry3::trans_z(3.0),
    )]);

    let image = render(&mut renderer, Isometry3F64::identity());

    let (red, red_count) = centroid(&image, is_red).expect("the red half must be visible");
    let (green, green_count) = centroid(&image, is_green).expect("the green half must be visible");

    // both halves cover a similar area, and red is left of green
    assert!(red[0] < green[0], "red half must be left of the green half");
    assert!(
        red_count.abs_diff(green_count) < red_count / 10,
        "both halves must cover a similar area, got {red_count} and {green_count}"
    );
}

/// A 2d renderable with alpha < 1 must be *blended onto* the image, rather than overwriting the
/// alpha channel of the view - which would punch a hole into it.
#[test]
fn semi_transparent_pixel_renderable_is_blended() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    renderer.update_pixels(vec![make_point2(
        "point",
        &[[127.0f32, 127.0]],
        &Color {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 0.5,
        },
        20.0,
    )]);

    // with an empty scene the distortion pass fills the view with opaque white
    let image = render(&mut renderer, Isometry3F64::identity());
    let center = image.pixel(127, 127);

    assert!(center[0] > 240, "red channel: {}", center[0]);
    assert!(
        (100..=160).contains(&center[1]) && (100..=160).contains(&center[2]),
        "half-transparent red over white must be blended, got {center:?}"
    );
    assert_eq!(center[3], 255, "the view itself must stay opaque");
}

/// Turning back-face culling off must actually show back-facing triangles.
#[test]
fn backface_culling_can_be_disabled() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // the same triangle twice, side by side, with opposite winding - so exactly one of them is
    // back-facing, whichever way round the winding rules work out
    let ccw = [[-1.0, -0.5, 0.0], [-0.2, -0.5, 0.0], [-0.6, 0.3, 0.0]];
    let cw = [[0.2, -0.5, 0.0], [0.6, 0.3, 0.0], [1.0, -0.5, 0.0]];
    renderer.update_scene(vec![
        make_mesh3_at("ccw", &[(ccw, Color::red())], Isometry3::trans_z(3.0)),
        make_mesh3_at("cw", &[(cw, Color::red())], Isometry3::trans_z(3.0)),
    ]);

    let culled = render_with(
        &mut renderer,
        Isometry3F64::identity(),
        true,
        TranslationAndScaling::identity(),
    );
    let not_culled = render_with(
        &mut renderer,
        Isometry3F64::identity(),
        false,
        TranslationAndScaling::identity(),
    );

    let culled_area = centroid(&culled, is_red).map(|(_, n)| n).unwrap_or(0);
    let not_culled_area = centroid(&not_culled, is_red).map(|(_, n)| n).unwrap_or(0);

    assert!(culled_area > 0, "culling must not remove *both* triangles");
    assert!(
        not_culled_area > culled_area,
        "with culling disabled both triangles must be visible: {not_culled_area} vs {culled_area}"
    );
}

/// Which side of a triangle is drawn, when back-face culling is on: the one its normal points
/// towards, the normal being `(p1 - p0) x (p2 - p0)`.
///
/// Nothing states this anywhere - it falls out of the winding rule, the projection, and the
/// camera's y running down the image, and it is easy to derive backwards. Getting it wrong makes
/// a mesh vanish, with nothing to say why, so it is pinned here.
#[test]
fn a_triangle_is_drawn_from_the_side_its_normal_points_towards() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Two triangles side by side, three metres down the camera's own z. The left one's normal
    // runs along +z, away from the camera; the right one's runs back at it.
    let away = [[-1.0, -0.5, 0.0], [-0.2, -0.5, 0.0], [-0.6, 0.3, 0.0]];
    let towards = [[0.2, -0.5, 0.0], [0.6, 0.3, 0.0], [1.0, -0.5, 0.0]];
    renderer.update_scene(vec![
        make_mesh3_at("away", &[(away, Color::red())], Isometry3::trans_z(3.0)),
        make_mesh3_at(
            "towards",
            &[(towards, Color::red())],
            Isometry3::trans_z(3.0),
        ),
    ]);

    let culled = render_with(
        &mut renderer,
        Isometry3F64::identity(),
        true,
        TranslationAndScaling::identity(),
    );
    let (center, count) = centroid(&culled, is_red).expect("one of the two must survive culling");
    assert!(
        center.x > (W / 2) as f64,
        "with culling on, {count} red pixels are drawn about u = {:.0} - the triangle facing away \
         from the camera is the one being kept, so a mesh has to be wound the other way round",
        center.x
    );
}

/// A cylinder is the capsule's own body with the ends cut flat rather than rounded over, closed
/// with a disk apiece - so it ends where its segment ends, where a capsule reaches a radius
/// further at each end.
#[test]
fn a_cylinder_ends_where_its_segment_does() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    const FOCAL: f64 = 200.0;
    let (from, to, radius) = (
        SVec::<f32, 3>::new(-0.6, 0.0, 4.0),
        SVec::<f32, 3>::new(0.6, 0.0, 4.0),
        0.2f32,
    );

    // how far the drawn shape reaches across the image
    let mut width_of = |renderables: Vec<SceneRenderable>| {
        renderer.update_scene(renderables);
        let image = render(&mut renderer, Isometry3F64::identity());
        let drawn: Vec<usize> = (0..W)
            .filter(|u| {
                (0..H).any(|v| {
                    is_red([
                        image.pixel(*u, v)[0],
                        image.pixel(*u, v)[1],
                        image.pixel(*u, v)[2],
                        255,
                    ])
                })
            })
            .collect();
        match (drawn.first(), drawn.last()) {
            (Some(low), Some(high)) => high - low + 1,
            _ => 0,
        }
    };

    let cylinder = width_of(make_cylinder3(
        "cylinder",
        &[Cylinder3 {
            from,
            to,
            radius,
            color: Color::red(),
        }],
    ));
    let capsule = width_of(vec![make_capsule3(
        "capsule",
        vec![Capsule3 {
            from,
            to,
            radius,
            color: Color::red(),
            flat_ends: false,
        }],
    )]);

    // The cylinder ends at its rim, and the widest part of a rim is the point of it nearest the
    // camera - a radius closer than the centre of the end is, so it projects a little wider than
    // the ends of the segment do.
    let expected = 2.0 * 0.6 * FOCAL / (4.0 - radius as f64);
    // The capsule ends at a hemisphere, whose silhouette stands a radius out from the segment,
    // seen at the angle the tangent from the camera makes.
    let expected_capsule =
        2.0 * (0.6 * FOCAL / 4.0 + radius as f64 * FOCAL / (16.0f64 - 0.04).sqrt());
    approx::assert_abs_diff_eq!(cylinder as f64, expected, epsilon = 2.0);
    approx::assert_abs_diff_eq!(capsule as f64, expected_capsule, epsilon = 2.0);
    assert!(
        capsule > cylinder,
        "the capsule reaches {capsule} px across and the cylinder {cylinder} - the ends are not \
         being cut"
    );
}

/// Under 2d zoom, the background image, the 2d pixel renderables and the 3d scene renderables
/// must stay aligned - through the (non-linear) distortion pass.
///
/// The zoom is folded into the intrinsics, so that the scene is rasterized at full view-port
/// resolution and the distortion warp stays exact; the background image and the 2d renderables
/// are zoomed separately. If those two paths ever disagree, the augmentations drift away from
/// the image as soon as the view is zoomed.
#[test]
fn zoom_keeps_image_overlay_and_augmentation_aligned() {
    let camera = DynCameraF64::new_enhanced_unified(
        VecF64::from_array([200.0, 200.0, 127.5, 127.5, 0.629, 1.22]),
        ImageSize::new(W, H),
    );
    let Some((_context, mut renderer)) = renderer(camera.clone()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // background image which encodes its own pixel coordinate as R = u, G = v, B = 0
    let mut background = MutImage4U8::from_image_size_and_val(
        ImageSize::new(W, H),
        SVec::<u8, 4>::new(0, 0, 0, 255),
    );
    for v in 0..H {
        for u in 0..W {
            *background.mut_pixel(u, v) = SVec::<u8, 4>::new(u as u8, v as u8, 0, 255);
        }
    }
    renderer.reset_2d_frame(&renderer.intrinsics(), Some(&background.to_shared()));

    // a 2d renderable anchored at an image pixel, and a 3d point - both blue-ish, so they can be
    // told apart from the background (whose blue channel is always zero)
    let overlay_uv = VecF64::<2>::new(120.0, 90.0);
    let point_in_camera = VecF64::<3>::new(0.1, -0.05, 1.0);
    renderer.update_pixels(vec![make_point2(
        "overlay",
        &[[overlay_uv[0] as f32, overlay_uv[1] as f32]],
        &Color::blue(),
        3.0,
    )]);
    renderer.update_scene(vec![make_point3_at(
        "augmentation",
        &[[
            point_in_camera[0] as f32,
            point_in_camera[1] as f32,
            point_in_camera[2] as f32,
        ]],
        &Color {
            r: 0.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        },
        3.0,
        Isometry3F64::identity(),
    )]);

    for zoom in [
        TranslationAndScaling::identity(),
        TranslationAndScaling {
            translation: VecF64::<2>::new(-300.0, -180.0),
            scaling: VecF64::<2>::new(3.0, 3.0),
        },
    ] {
        let image = render_with(&mut renderer, Isometry3F64::identity(), false, zoom);

        // (1) the background: which image pixel shows up where?
        for (u, v) in [(40, 40), (128, 96), (200, 170)] {
            let pixel = image.pixel(u, v);
            if pixel[2] > 0 {
                continue; // a marker (the only source of blue), not the background
            }
            let expected = zoom.apply_inverse(VecF64::<2>::new(u as f64, v as f64));
            if expected[0] < 0.0 || expected[1] < 0.0 {
                continue;
            }
            approx::assert_abs_diff_eq!(pixel[0] as f64, expected[0].floor(), epsilon = 1.0);
            approx::assert_abs_diff_eq!(pixel[1] as f64, expected[1].floor(), epsilon = 1.0);
        }

        // (2) the 2d overlay sits where the zoom puts its image pixel ...
        let (overlay, _) =
            centroid(&image, |c| c[2] > 200 && c[1] < 100).expect("the 2d overlay must be visible");
        approx::assert_abs_diff_eq!(overlay, zoom.apply(overlay_uv), epsilon = 1.5);

        // (3) ... and so does the 3d augmentation, at its *distorted* image pixel
        let (augmentation, _) = centroid(&image, |c| c[2] > 200 && c[1] > 200)
            .expect("the 3d augmentation must be visible");
        let uv = camera.cam_proj(point_in_camera);
        approx::assert_abs_diff_eq!(augmentation, zoom.apply(uv), epsilon = 1.5);
    }
}

/// Antialiased edges must not be darkened into a gray halo.
///
/// The scene is rendered with multisampling into a texture which the distortion pass composites
/// over the background. Multisample resolve mixes covered samples with the transparent clear
/// color, so partial coverage arrives as a *premultiplied* color. Compositing that as if it were
/// straight alpha darkens every edge - visible as a gray outline around every object.
#[test]
fn antialiased_edges_are_not_darkened() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // a green triangle with a diagonal edge, over the (white) background
    let triangle = [[-0.6, -0.6, 0.0], [0.6, -0.6, 0.0], [-0.6, 0.6, 0.0]];
    renderer.update_scene(vec![make_mesh3_at(
        "triangle",
        &[(triangle, Color::green())],
        Isometry3::trans_z(3.0),
    )]);

    let image = render(&mut renderer, Isometry3F64::identity());

    // The foreground is green, dimmed by the shading, and the background is white - so both have
    // a green channel of at least the foreground's, and no correct compositing of the two can
    // produce a green value below it. Read that value off the triangle rather than assuming a
    // saturated 255, which held only while surfaces were unlit.
    let interior_green = (0..H)
        .flat_map(|v| (0..W).map(move |u| (u, v)))
        .filter(|(u, v)| image.pixel(*u, *v)[0] == 0)
        .map(|(u, v)| image.pixel(u, v)[1])
        .max()
        .expect("the triangle covers some pixel completely");

    let mut edge_pixels = 0;
    for v in 0..H {
        for u in 0..W {
            let p = image.pixel(u, v);
            let partially_covered = p[0] > 0 && p[0] < 255;
            if partially_covered {
                edge_pixels += 1;
                assert!(
                    p[1] + 1 >= interior_green,
                    "edge pixel ({u}, {v}) is darkened: {:?} - green must not fall below the \
                     {interior_green} of the triangle between it and a white background",
                    [p[0], p[1], p[2]]
                );
            }
        }
    }
    assert!(
        edge_pixels > 10,
        "expected antialiased edges to inspect, found {edge_pixels}"
    );
}

/// The camera poses under test: pure translations as well as rotations, since `scene_from_camera`
/// enters the shader inverted and composed with the entity pose - a place where sign errors hide.
fn camera_poses() -> [(&'static str, Isometry3F64); 5] {
    [
        ("straight on, 3m back", Isometry3::trans_z(-3.0)),
        ("straight on, 5m back", Isometry3::trans_z(-5.0)),
        (
            "shifted right and down",
            Isometry3::from_translation(VecF64::<3>::new(0.6, 0.4, -3.0)),
        ),
        (
            "yawed 25 degrees",
            Isometry3::from_rotation_and_translation(
                Rotation3::rot_y(-0.4363),
                VecF64::<3>::new(1.5, 0.0, -3.0),
            ),
        ),
        (
            "pitched 20 degrees",
            Isometry3::from_rotation_and_translation(
                Rotation3::rot_x(0.3491),
                VecF64::<3>::new(0.0, -1.2, -3.0),
            ),
        ),
    ]
}

/// Scene renderables must be projected through `scene_from_camera`, not just through the
/// intrinsics - so the same scene, viewed from different camera poses, has to land exactly where
/// the CPU-side camera model says it does.
#[test]
fn camera_pose_is_applied_to_points() {
    let camera = pinhole();
    let Some((_context, mut renderer)) = renderer(camera.clone()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    const POINT_SIZE: f32 = 15.0;
    let places = [
        (
            VecF64::<3>::new(0.0, 0.0, 0.0),
            Color::red(),
            is_red as fn([u8; 4]) -> bool,
        ),
        (VecF64::<3>::new(0.6, 0.2, 0.5), Color::green(), is_green),
        (VecF64::<3>::new(-0.5, -0.3, 1.0), Color::blue(), is_blue),
    ];
    renderer.update_scene(
        places
            .iter()
            .enumerate()
            .map(|(i, (p, color, _))| {
                make_point3_at(
                    format!("point-{i}"),
                    &[[0.0f32, 0.0, 0.0]],
                    color,
                    POINT_SIZE,
                    Isometry3::from_translation(*p),
                )
            })
            .collect(),
    );

    let mut checked = 0;
    for (label, world_from_camera) in camera_poses() {
        let image = render(&mut renderer, world_from_camera);

        let expected: Vec<VecF64<2>> = places
            .iter()
            .map(|(p, _, _)| camera.cam_proj(world_from_camera.inverse().transform(*p)))
            .collect();

        for (i, (p_world, _, pick)) in places.iter().enumerate() {
            let uv = expected[i];
            let depth = world_from_camera.inverse().transform(*p_world)[2];

            // Only check points which are unambiguously measurable: in front of the camera, far
            // enough from the border not to be clipped, and far enough from the other points that
            // their quads cannot occlude each other and bias the centroid.
            let margin = POINT_SIZE as f64;
            let inside = depth > 0.5
                && uv[0] > margin
                && uv[1] > margin
                && uv[0] < W as f64 - margin
                && uv[1] < H as f64 - margin;
            let separated = expected.iter().enumerate().all(|(j, other)| {
                i == j
                    || (other[0] - uv[0]).abs().max((other[1] - uv[1]).abs())
                        > POINT_SIZE as f64 + 2.0
            });
            if !inside || !separated {
                continue;
            }

            let (measured, _) = centroid(&image, *pick)
                .unwrap_or_else(|| panic!("point {i} must be visible for pose `{label}`"));
            approx::assert_abs_diff_eq!(measured, uv, epsilon = 0.75);
            checked += 1;
        }
    }
    assert!(
        checked >= 6,
        "expected several measurable points, got {checked}"
    );
}

/// The same, for a mesh: the rendered triangle has to cover the projected triangle.
#[test]
fn camera_pose_is_applied_to_meshes() {
    let camera = pinhole();
    let Some((_context, mut renderer)) = renderer(camera.clone()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let world_from_entity = Isometry3::from_translation(VecF64::<3>::new(-0.2, -0.15, 0.4));
    let triangle: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]];
    renderer.update_scene(vec![make_mesh3_at(
        "triangle",
        &[(triangle, Color::red())],
        world_from_entity,
    )]);

    let mut checked = 0;
    for (label, world_from_camera) in camera_poses() {
        let image = render(&mut renderer, world_from_camera);
        let camera_from_entity = world_from_camera.inverse() * world_from_entity;

        let corners: Vec<VecF64<2>> = triangle
            .iter()
            .map(|p| {
                camera.cam_proj(camera_from_entity.transform(VecF64::<3>::new(
                    p[0] as f64,
                    p[1] as f64,
                    p[2] as f64,
                )))
            })
            .collect();
        let in_view = |uv: &VecF64<2>| {
            uv[0] > 2.0 && uv[1] > 2.0 && uv[0] < W as f64 - 2.0 && uv[1] < H as f64 - 2.0
        };
        let depths_ok = triangle.iter().all(|p| {
            camera_from_entity.transform(VecF64::<3>::new(p[0] as f64, p[1] as f64, p[2] as f64))[2]
                > 0.5
        });
        if !depths_ok || !corners.iter().all(in_view) {
            continue;
        }

        // the centroid of the projected corners lies inside the projected triangle
        let center = (corners[0] + corners[1] + corners[2]) / 3.0;
        let inside = image.pixel(center[0] as usize, center[1] as usize);
        assert!(
            is_red([inside[0], inside[1], inside[2], inside[3]]),
            "pose `{label}`: the triangle must cover its projected centroid {center:?}, got \
             {:?}",
            [inside[0], inside[1], inside[2]]
        );

        // ... and anything outside the bounding box of the corners is not covered
        let max_u = corners.iter().map(|c| c[0]).fold(f64::MIN, f64::max);
        let max_v = corners.iter().map(|c| c[1]).fold(f64::MIN, f64::max);
        let (out_u, out_v) = ((max_u + 6.0) as usize, (max_v + 6.0) as usize);
        if out_u < W && out_v < H {
            let outside = image.pixel(out_u, out_v);
            assert!(
                !is_red([outside[0], outside[1], outside[2], outside[3]]),
                "pose `{label}`: ({out_u}, {out_v}) is outside the projected triangle, but is \
                 covered"
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 3,
        "expected several measurable poses, got {checked}"
    );
}

/// A tiny image, magnified: every output pixel must show exactly the source pixel which the zoom
/// transform puts there - and a 2d renderable anchored at an image pixel must land exactly in the
/// middle of that pixel's block.
///
/// This pins down the sub-pixel convention: the image coordinate `k` denotes the *center* of
/// image pixel `k`, consistently for the background image, the 2d renderables and the 3d
/// augmentations. Mixing that up with a pixel-corner convention cancels out at zoom scale 1, but
/// makes the background slide away from the overlays as soon as the view is zoomed.
#[test]
fn tiny_image_zoom_is_pixel_exact() {
    const IW: usize = 3;
    const IH: usize = 2;
    const VW: usize = 300;
    const VH: usize = 200;

    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let properties = RenderCameraProperties::default_from(ImageSize::new(IW, IH));
    let mut renderer = OffscreenRenderer::new(&context, &properties);

    let colors = [
        SVec::<u8, 4>::new(255, 0, 0, 255),
        SVec::<u8, 4>::new(0, 255, 0, 255),
        SVec::<u8, 4>::new(0, 0, 255, 255),
        SVec::<u8, 4>::new(255, 255, 0, 255),
        SVec::<u8, 4>::new(0, 255, 255, 255),
        SVec::<u8, 4>::new(255, 0, 255, 255),
    ];
    let mut source = MutImage4U8::from_image_size_and_val(ImageSize::new(IW, IH), colors[0]);
    for v in 0..IH {
        for u in 0..IW {
            *source.mut_pixel(u, v) = colors[v * IW + u];
        }
    }
    renderer.reset_2d_frame(&properties.intrinsics, Some(&source.to_shared()));

    // a black 2d point anchored at the center of image pixel (1, 0)
    let anchor = VecF64::<2>::new(1.0, 0.0);
    renderer.update_pixels(vec![make_point2(
        "anchor",
        &[[anchor[0] as f32, anchor[1] as f32]],
        &Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
        4.0,
    )]);

    for zoom in [
        TranslationAndScaling::identity(),
        TranslationAndScaling {
            translation: VecF64::<2>::new(-0.85, 0.0),
            scaling: VecF64::<2>::new(1.7, 1.7),
        },
        TranslationAndScaling {
            translation: VecF64::<2>::new(-2.0, 0.0),
            scaling: VecF64::<2>::new(3.0, 3.0),
        },
    ] {
        // note: this view port is deliberately not the 256x256 of the other tests
        let image = renderer
            .render_params(&ImageSize::new(VW, VH), &Isometry3F64::identity())
            .zoom(zoom)
            .download_rgba(true)
            .render()
            .rgba_image
            .expect("`download_rgba` was requested");

        // view-port pixel -> image coordinate, in pixel-center convention
        let to_image = |u: usize, v: usize| {
            let uv_screen = VecF64::<2>::new(
                (u as f64 + 0.5) * IW as f64 / VW as f64 - 0.5,
                (v as f64 + 0.5) * IH as f64 / VH as f64 - 0.5,
            );
            zoom.apply_inverse(uv_screen)
        };

        let mut checked = 0;
        for v in 0..VH {
            for u in 0..VW {
                let uv = to_image(u, v);
                let (fx, fy) = (uv[0] + 0.5, uv[1] + 0.5);
                // skip pixels sitting exactly on a block boundary, where either neighbour is a
                // legitimate answer
                if (fx - fx.round()).abs() < 1e-4 || (fy - fy.round()).abs() < 1e-4 {
                    continue;
                }
                // skip the anchor marker, and the ring of pixels its round edge only partly
                // covers - the marker is drawn at the anchor, in view-port units
                let anchor_in_view_port = zoom.apply(anchor);
                let to_marker = VecF64::<2>::new(
                    (anchor_in_view_port[0] + 0.5) * VW as f64 / IW as f64 - 0.5 - u as f64,
                    (anchor_in_view_port[1] + 0.5) * VH as f64 / IH as f64 - 0.5 - v as f64,
                );
                if to_marker.norm() < 4.0 {
                    continue;
                }
                let rendered = image.pixel(u, v);
                let expected = if fx < 0.0 || fy < 0.0 || fx >= IW as f64 || fy >= IH as f64 {
                    SVec::<u8, 4>::new(255, 255, 255, 255) // outside the image: white
                } else {
                    colors[fy as usize * IW + fx as usize]
                };
                assert_eq!(
                    [rendered[0], rendered[1], rendered[2]],
                    [expected[0], expected[1], expected[2]],
                    "zoom {:?}: view-port pixel ({u}, {v}) maps to image {uv:?}",
                    zoom.scaling[0]
                );
                checked += 1;
            }
        }
        assert!(
            checked > VW * VH / 2,
            "expected most pixels checked, got {checked}"
        );

        // the 2d anchor must sit in the middle of its own block
        let screen = zoom.apply(anchor);
        let expected = VecF64::<2>::new(
            (screen[0] + 0.5) * VW as f64 / IW as f64 - 0.5,
            (screen[1] + 0.5) * VH as f64 / IH as f64 - 0.5,
        );
        // only when the whole marker is on screen, so the centroid is not clipped
        if expected[0] > 8.0
            && expected[1] > 8.0
            && expected[0] < VW as f64 - 8.0
            && expected[1] < VH as f64 - 8.0
        {
            let (marker, _) = centroid(&image, |c| c[0] < 40 && c[1] < 40 && c[2] < 40)
                .expect("the 2d anchor must be visible");
            approx::assert_abs_diff_eq!(marker, expected, epsilon = 0.6);
        }
    }
}

/// A 3d augmentation must survive being zoomed into, in a *distorted* image view.
///
/// The scene is rasterized through the undistorted pinhole model, and the distortion pass then
/// warps it. Folding the zoom of the distorted model into the pinhole model as well moves the
/// undistorted coordinates out of the scene texture - so the augmentation is simply not there
/// any more, while the background image around it zooms in just fine. See `pinhole_zoom`.
#[test]
fn zoomed_distorted_view_keeps_3d_augmentations() {
    let camera = DynCameraF64::new_enhanced_unified(
        VecF64::from_array([200.0, 200.0, 127.5, 127.5, 0.629, 1.22]),
        ImageSize::new(W, H),
    );
    let Some((_context, mut renderer)) = renderer(camera.clone()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // well off to the side, where the undistorted and the distorted coordinate differ a lot
    let point_in_camera = VecF64::<3>::new(-0.605, 0.0, 1.0);
    let uv = camera.cam_proj(point_in_camera);
    renderer.update_scene(vec![make_point3_at(
        "augmentation",
        &[[
            point_in_camera[0] as f32,
            point_in_camera[1] as f32,
            point_in_camera[2] as f32,
        ]],
        &Color::green(),
        7.0,
        Isometry3F64::identity(),
    )]);

    for scale in [1.0, 3.0, 6.0] {
        // zoom in on the augmentation, so it stays at its own image position
        let zoom = TranslationAndScaling {
            translation: VecF64::<2>::new(uv[0] - scale * uv[0], uv[1] - scale * uv[1]),
            scaling: VecF64::<2>::new(scale, scale),
        };
        let image = render_with(&mut renderer, Isometry3F64::identity(), false, zoom);

        let (measured, _) = centroid(&image, is_green)
            .unwrap_or_else(|| panic!("the augmentation vanished at zoom {scale}x"));
        approx::assert_abs_diff_eq!(measured, uv, epsilon = 1.5);
    }
}

/// A 2d line segment must cover exactly the band between its endpoints.
///
/// Points and line segments are expanded from a single vertex into a quad by the vertex shader,
/// one instance per item - so the vertex index selects both the endpoint and the side of the
/// line. Getting that mapping wrong yields degenerate or mirrored quads.
#[test]
fn pixel_line_covers_the_segment() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // a horizontal segment, 8 px wide, from x = 60 to x = 190 at y = 128
    const WIDTH: f32 = 8.0;
    let (x0, x1, y) = (60.0f32, 190.0f32, 128.0f32);
    renderer.update_pixels(vec![make_line2(
        "line",
        &[[[x0, y], [x1, y]]],
        &Color::red(),
        WIDTH,
    )]);

    let image = render(&mut renderer, Isometry3F64::identity());
    let (center, area) = centroid(&image, is_red).expect("the line must be visible");

    // centered on the middle of the segment ...
    approx::assert_abs_diff_eq!(
        center,
        VecF64::<2>::new((x0 + x1) as f64 / 2.0, y as f64),
        epsilon = 1.0
    );
    // ... and covering its length times its width
    let expected_area = (x1 - x0) as f64 * WIDTH as f64;
    assert!(
        (area as f64 - expected_area).abs() < 0.15 * expected_area,
        "expected about {expected_area:.0} covered pixels, got {area}"
    );

    // the band is bounded in y: nothing well above or below the segment
    for v in [(y - WIDTH) as usize, (y + WIDTH) as usize] {
        let pixel = image.pixel(((x0 + x1) / 2.0) as usize, v);
        assert!(
            !is_red([pixel[0], pixel[1], pixel[2], pixel[3]]),
            "row {v} is outside the line, but is covered"
        );
    }
}

/// Geometry behind the camera must not show up in front of it.
///
/// The projection divides by the camera-space depth, so a point behind the camera projects to a
/// mirrored position which is a perfectly plausible pixel - it has to be clipped rather than
/// drawn.
#[test]
fn geometry_behind_the_camera_is_not_drawn() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // 2 m behind the camera, and off-axis so that a mirrored projection would land well inside
    // the image rather than off its edge
    let behind = Isometry3::from_translation(VecF64::<3>::new(0.3, 0.2, -2.0));
    renderer.update_scene(vec![
        make_point3_at("behind", &[[0.0f32, 0.0, 0.0]], &Color::red(), 20.0, behind),
        make_mesh3_at(
            "behind-mesh",
            &[(
                [[0.0f32, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0]],
                Color::red(),
            )],
            behind,
        ),
    ]);

    let image = render(&mut renderer, Isometry3F64::identity());

    assert!(
        centroid(&image, is_red).is_none(),
        "geometry behind the camera must be clipped, not drawn"
    );
}

/// A wide field of view must not collapse the scene onto the principal point.
///
/// The scene is rasterized through an undistorted pinhole intermediate, and `pinhole_zoom` fits
/// that intermediate to the region which is actually visible - which gains resolution for a
/// camera of moderate field of view. As the field of view approaches 180 degrees, though, the
/// undistorted region grows without bound (`z1 = u / k`, with `k` going to zero at 90 degrees off
/// axis), and it stays *finite* in floating point - just enormous. Fitting that would scale the
/// whole scene down into a few pixels.
#[test]
fn a_wide_field_of_view_does_not_collapse_the_scene() {
    // focal lengths chosen for the corner of a 256 x 256 image, 180 px from the principal point
    for (fx, alpha, beta) in [
        // about 151 degrees across the diagonal, still within reach of a single plane
        (135.0, 0.629, 1.02),
        // 180 degrees: the rays at the rim have no place on a plane at all, but everything a
        // plane *can* reach still has to land where it belongs
        (108.0, 0.600, 1.00),
    ] {
        let angles = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let camera = DynCameraF64::new_enhanced_unified(
            VecF64::from_array([fx, fx, 127.5, 127.5, alpha, beta]),
            ImageSize::new(W, H),
        );
        let Some((_context, mut renderer)) = renderer(camera.clone()) else {
            eprintln!("skipping: no GPU available");
            return;
        };

        let mut checked = 0;
        for degrees in angles {
            let radians: f64 = degrees * core::f64::consts::PI / 180.0;
            let point = VecF64::<3>::new(5.0 * radians.sin(), 0.0, 5.0 * radians.cos());
            let uv = camera.cam_proj(point);
            // only what the camera actually images
            if uv[0] < 8.0 || uv[1] < 8.0 || uv[0] > W as f64 - 8.0 || uv[1] > H as f64 - 8.0 {
                continue;
            }

            renderer.update_scene(vec![make_point3_at(
                "p",
                &[[point[0] as f32, point[1] as f32, point[2] as f32]],
                &Color::red(),
                9.0,
                Isometry3F64::identity(),
            )]);
            let image = render(&mut renderer, Isometry3F64::identity());

            let (measured, _) = centroid(&image, is_red).unwrap_or_else(|| {
                panic!("fx={fx}: the point at {degrees} deg off axis is not visible at all")
            });
            approx::assert_abs_diff_eq!(measured, uv, epsilon = 1.5);
            checked += 1;
        }
        assert!(
            checked >= 4,
            "fx={fx}: expected several points in view, got {checked}"
        );
    }
}

/// A hemisphere is covered by rendering into several frusta.
///
/// The scene is rasterized through an undistorted pinhole intermediate. One plane runs out well
/// before 180 degrees - `pinhole_model` halves the focal length to widen it, which still only
/// reaches about 67 degrees off axis here - so beyond that the scene has to be rendered into
/// several frusta and sampled by ray direction.
#[test]
fn a_hemisphere_is_covered_by_several_frusta() {
    const WIDE: ImageSize = ImageSize {
        width: 640,
        height: 480,
    };
    // 180 degrees across the diagonal
    let camera = DynCameraF64::new_enhanced_unified(
        VecF64::from_array([240.0, 240.0, 320.0, 240.0, 0.6, 1.0]),
        WIDE,
    );
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let properties = RenderCameraProperties::from_intrinsics(&camera);
    let mut renderer = OffscreenRenderer::new(&context, &properties);

    let mut checked = 0;
    let mut beyond_a_single_plane = 0;
    for degrees in [0.0f64, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 65.0, 70.0] {
        let radians = degrees * core::f64::consts::PI / 180.0;
        let point = VecF64::<3>::new(5.0 * radians.sin(), 0.0, 5.0 * radians.cos());
        let uv = camera.cam_proj(point);
        if uv[0] < 8.0 || uv[1] < 8.0 || uv[0] > 632.0 || uv[1] > 472.0 {
            continue;
        }
        // where this ray would land on the single undistorted plane, whose focal length is half
        // the camera's and which is only `width` wide
        if 0.5 * 240.0 * radians.tan() + 320.0 > WIDE.width as f64 {
            beyond_a_single_plane += 1;
        }

        renderer.update_scene(vec![make_point3_at(
            "p",
            &[[point[0] as f32, point[1] as f32, point[2] as f32]],
            &Color::red(),
            9.0,
            Isometry3F64::identity(),
        )]);
        let image = renderer
            .render_params(&WIDE, &Isometry3F64::identity())
            .download_rgba(true)
            .render()
            .rgba_image
            .expect("`download_rgba` was requested");

        let (measured, _) = centroid(&image, is_red)
            .unwrap_or_else(|| panic!("the point at {degrees} deg off axis is not visible"));
        approx::assert_abs_diff_eq!(measured, uv, epsilon = 2.0);
        checked += 1;
    }

    assert!(
        checked >= 7,
        "expected several points in view, got {checked}"
    );
    assert!(
        beyond_a_single_plane > 0,
        "this test is pointless unless some of its points are out of reach of a single plane"
    );
}

/// Geometry appears only where it projects to - and nowhere else.
///
/// A multi-frustum intermediate renders the scene once per face, and a triangle well inside one
/// face lies close to the *plane* of its neighbour. Pinning such geometry to the near plane,
/// rather than letting it be clipped, placed it at a projected position which runs away as the
/// depth goes to zero - drawing a large wedge of it across a face it is not even in.
#[test]
fn geometry_is_not_smeared_across_neighbouring_frusta() {
    const WIDE: ImageSize = ImageSize {
        width: 639,
        height: 479,
    };
    // the widest the example's camera goes: 180 degrees across the diagonal
    let camera = DynCameraF64::new_enhanced_unified(
        VecF64::from_array([254.0, 254.0, 320.0, 240.0, 0.629, 1.02]),
        WIDE,
    );
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let properties = RenderCameraProperties::from_intrinsics(&camera);
    let mut renderer = OffscreenRenderer::new(&context, &properties);

    let triangle: [[f32; 3]; 3] = [[0.0, 0.0, -0.1], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    renderer.update_scene(vec![make_mesh3_at(
        "mesh",
        &[(triangle, Color::blue())],
        Isometry3F64::identity(),
    )]);

    // a pose which puts the triangle deep inside one face, close to the plane of the next
    let scene_from_camera = Isometry3::from_rotation_and_translation(
        Rotation3::exp(VecF64::<3>::new(0.0, 0.0, -3.075185239438908)),
        VecF64::<3>::new(
            0.3950064789524679,
            -0.025340511678425603,
            -3.918706677798824,
        ),
    );
    let image = renderer
        .render_params(&WIDE, &scene_from_camera)
        .download_rgba(true)
        .render()
        .rgba_image
        .expect("`download_rgba` was requested");

    // where the camera model says the triangle is
    let camera_from_scene = scene_from_camera.inverse();
    let (mut lo, mut hi) = (
        VecF64::<2>::new(f64::MAX, f64::MAX),
        VecF64::<2>::new(f64::MIN, f64::MIN),
    );
    for corner in triangle {
        let uv = camera.cam_proj(camera_from_scene.transform(VecF64::<3>::new(
            corner[0] as f64,
            corner[1] as f64,
            corner[2] as f64,
        )));
        lo = VecF64::<2>::new(lo[0].min(uv[0]), lo[1].min(uv[1]));
        hi = VecF64::<2>::new(hi[0].max(uv[0]), hi[1].max(uv[1]));
    }

    // nothing blue outside that, give or take the antialiased edge
    const MARGIN: f64 = 3.0;
    for v in 0..WIDE.height {
        for u in 0..WIDE.width {
            let pixel = image.pixel(u, v);
            if !is_blue([pixel[0], pixel[1], pixel[2], pixel[3]]) {
                continue;
            }
            assert!(
                (u as f64) >= lo[0] - MARGIN
                    && (u as f64) <= hi[0] + MARGIN
                    && (v as f64) >= lo[1] - MARGIN
                    && (v as f64) <= hi[1] + MARGIN,
                "the triangle projects into {lo:?}..{hi:?}, but is also drawn at ({u}, {v})"
            );
        }
    }
}

/// No part of the view is left uncovered, at any field of view.
///
/// The scene is rendered either into one plane or into several frusta. Deciding between them by
/// a threshold on how far off axis a ray may point leaves a band where the plane is kept but can
/// no longer reach the corners of the image - they come out as background, whatever is actually
/// there. The switch has to be made by measuring whether the plane covers the view.
///
/// Frustum-planes mode tints every pixel it composites, so a pixel left pure white is one the
/// distortion pass had nothing to offer.
#[test]
fn no_field_of_view_leaves_the_image_uncovered() {
    const WIDE: ImageSize = ImageSize {
        width: 639,
        height: 479,
    };
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // across the transition from one plane to the frusta, which for this model is around 330
    for focal_length in [
        254.0f64, 265.0, 280.0, 295.0, 310.0, 330.0, 366.0, 400.0, 500.0, 700.0,
    ] {
        let camera = DynCameraF64::new_enhanced_unified(
            VecF64::from_array([focal_length, focal_length, 320.0, 240.0, 0.629, 1.02]),
            WIDE,
        );
        let properties = RenderCameraProperties::from_intrinsics(&camera);
        let mut renderer = OffscreenRenderer::new(&context, &properties);
        renderer.update_scene(vec![make_point3_at(
            "p",
            &[[0.0f32, 0.0, 3.0]],
            &Color::red(),
            5.0,
            Isometry3F64::identity(),
        )]);

        let image = renderer
            .render_params(&WIDE, &Isometry3::trans_z(-5.0))
            .debug_frustum_planes(true)
            .download_rgba(true)
            .render()
            .rgba_image
            .expect("`download_rgba` was requested");

        let uncovered = (0..WIDE.height)
            .flat_map(|v| (0..WIDE.width).map(move |u| (u, v)))
            .filter(|(u, v)| {
                let p = image.pixel(*u, *v);
                p[0] > 250 && p[1] > 250 && p[2] > 250
            })
            .count();
        assert_eq!(
            uncovered, 0,
            "focal length {focal_length}: {uncovered} pixels of the view are not covered by the \
             intermediate the scene was rendered into"
        );
    }
}

/// A wide camera renders through five frusta, and each face measures depth along *its own*
/// optical axis. The inverse distance image is a distance along the ray instead, which is the same
/// thing only on the optical axis - so a point off to the side used to be read back, and picked
/// as the pivot of an interaction, at the wrong distance.
#[test]
fn depth_is_measured_along_the_ray_in_every_frustum() {
    const WIDE: ImageSize = ImageSize {
        width: 639,
        height: 479,
    };
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // (0, 0, 5) sits on the axis, in the forward face; the others are far enough off it to fall
    // into the side faces, at a metric z which is *not* their distance from the camera. The last
    // is 70 degrees off the axis, 2 m away and so nearer than the near plane in z - a face holds
    // it, since a face clips against its own axis, and along the ray it is a perfectly ordinary
    // 2 m: exactly the geometry a depth in z could show but never let you pick.
    let points = [
        [0.0f64, 0.0, 5.0],
        [4.0, 0.0, 2.0],
        [0.0, -3.5, 1.5],
        [1.33, 1.33, 0.68],
    ];

    // either side of the switch from the single plane to the frusta
    for focal_length in [560.0f64, 295.0, 254.0] {
        let camera = DynCameraF64::new_enhanced_unified(
            VecF64::from_array([focal_length, focal_length, 320.0, 240.0, 0.629, 1.02]),
            WIDE,
        );
        let properties = RenderCameraProperties::from_intrinsics(&camera);
        let clipping_planes = properties.clipping_planes;
        let mut renderer = OffscreenRenderer::new(&context, &properties);
        renderer.update_scene(
            points
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    make_point3_at(
                        format!("p{i}"),
                        &[[p[0] as f32, p[1] as f32, p[2] as f32]],
                        &Color::red(),
                        5.0,
                        Isometry3F64::identity(),
                    )
                })
                .collect(),
        );

        let result = renderer
            .render_params(&WIDE, &Isometry3F64::identity())
            .render();
        let inverse_distance = pollster::block_on(download_depth(
            false,
            clipping_planes.cast(),
            context.clone(),
            &WIDE,
            &result,
        ))
        .inverse_distance_image;

        for point in points {
            let uv = camera.cam_proj(VecF64::<3>::from_array(point));
            const MARGIN: f64 = 6.0;
            if !(uv[0] > MARGIN
                && uv[1] > MARGIN
                && uv[0] < WIDE.width as f64 - MARGIN
                && uv[1] < WIDE.height as f64 - MARGIN)
            {
                // outside this field of view, or so close to the edge that the marker is clipped
                continue;
            }
            let (u, v) = (uv[0].round() as usize, uv[1].round() as usize);

            // exactly what an interaction does with the pixel under the pointer: read the
            // inverse distance there, and put the pivot that far along the ray
            let distance = inverse_distance.distance(u, v) as f64;
            let pivot = camera.cam_unproj_to_unit_vector(VecF64::<2>::new(u as f64, v as f64));
            let pivot = pivot.vector() * distance;
            let point = VecF64::<3>::from_array(point);
            assert!(
                (pivot - point).norm() < 0.05,
                "focal length {focal_length}: the point {:?} is drawn at ({u}, {v}), where \
                 clicking puts the pivot at {:?} instead - {distance:.3} m away rather than \
                 {:.3}",
                point.as_slice(),
                pivot.as_slice(),
                point.norm(),
            );
        }
    }
}

/// The atlas keeps one slot per face of the hemisphere, but only the faces in view are rendered
/// into it - so the remaining slots hold an earlier frame, or nothing at all. Sampling those, or
/// converting the far plane of a face as if it were a surface, puts geometry in front of a camera
/// looking at an empty scene.
#[test]
fn an_empty_scene_holds_no_inverse_depth_at_all() {
    const WIDE: ImageSize = ImageSize {
        width: 639,
        height: 479,
    };
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    for focal_length in [560.0f64, 295.0, 254.0] {
        let camera = DynCameraF64::new_enhanced_unified(
            VecF64::from_array([focal_length, focal_length, 320.0, 240.0, 0.629, 1.02]),
            WIDE,
        );
        let properties = RenderCameraProperties::from_intrinsics(&camera);
        let mut renderer = OffscreenRenderer::new(&context, &properties);

        // a first frame with something in it, so that a stale atlas has something to leak
        renderer.update_scene(vec![make_point3_at(
            "p",
            &[[0.0f32, 0.0, 3.0]],
            &Color::red(),
            5.0,
            Isometry3F64::identity(),
        )]);
        renderer
            .render_params(&WIDE, &Isometry3F64::identity())
            .render();

        renderer.clear_renderables();
        let result = renderer
            .render_params(&WIDE, &Isometry3F64::identity())
            .render();
        let inverse_distance = pollster::block_on(download_depth(
            false,
            properties.clipping_planes.cast(),
            context.clone(),
            &WIDE,
            &result,
        ))
        .inverse_distance_image;

        let occupied = (0..WIDE.height)
            .flat_map(|v| (0..WIDE.width).map(move |u| (u, v)))
            .filter(|(u, v)| inverse_distance.distance(*u, *v).is_finite())
            .count();
        assert_eq!(
            occupied, 0,
            "focal length {focal_length}: the scene is empty, yet {occupied} pixels of the \
             inverse distance image hold a surface"
        );
    }
}

/// Which frustum a ray belongs to is decided twice: on the cpu, to pick the faces to render, and
/// in the shader, to pick the face to read back. Nothing forces the two to agree - and where they
/// do not, the shader reads a slot of the atlas which holds an earlier frame, or nothing at all.
///
/// The debug tint makes the disagreement visible: a pixel whose face was never rendered is tinted
/// neutral grey rather than in one of the five face colours. On an empty scene over the white
/// background, that is the only way a grey pixel can arise.
#[test]
fn every_ray_is_read_back_from_a_frustum_which_was_rendered() {
    const WIDE: ImageSize = ImageSize {
        width: 639,
        height: 479,
    };
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    for focal_length in [254.0f64, 265.0, 280.0, 295.0, 310.0, 330.0] {
        let camera = DynCameraF64::new_enhanced_unified(
            VecF64::from_array([focal_length, focal_length, 320.0, 240.0, 0.629, 1.02]),
            WIDE,
        );
        let properties = RenderCameraProperties::from_intrinsics(&camera);
        let mut renderer = OffscreenRenderer::new(&context, &properties);

        let image = renderer
            .render_params(&WIDE, &Isometry3F64::identity())
            .debug_frustum_planes(true)
            .download_rgba(true)
            .render()
            .rgba_image
            .expect("`download_rgba` was requested");

        let uncovered: Vec<_> = (0..WIDE.height)
            .flat_map(|v| (0..WIDE.width).map(move |u| (u, v)))
            .filter(|(u, v)| {
                let p = image.pixel(*u, *v);
                p[0] == p[1] && p[1] == p[2]
            })
            .collect();
        assert!(
            uncovered.is_empty(),
            "focal length {focal_length}: {} pixels - {:?} among them - are read back from a \
             frustum which was never rendered",
            uncovered.len(),
            &uncovered[..uncovered.len().min(4)]
        );
    }
}

/// 2d renderables are anchored in pixels of the final, distorted image - the focus marker of an
/// interaction among them, which is drawn at the pixel the pointer is over. They are drawn after
/// the distortion pass, from the same uniforms the scene was rendered with, and the pinhole model
/// among those describes the *intermediate*: a 90 degree square face when the view is wide enough
/// to need the frusta. Taken as the size of the image, that puts every 2d renderable somewhere
/// else - `v * 479 / 639` for a 639x479 image, which is what clicking into a wide view showed.
#[test]
fn a_2d_renderable_is_anchored_to_the_image_however_the_scene_is_rendered() {
    const WIDE: ImageSize = ImageSize {
        width: 639,
        height: 479,
    };
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // the frusta render into a square face as wide as the longer side of the view port, so a
    // view port narrower than the image takes the u axis out of step as well as the v axis
    for view_port in [WIDE, ImageSize::new(400, 300)] {
        // either side of the switch from the single plane to the frusta
        for focal_length in [560.0f64, 254.0] {
            let camera = DynCameraF64::new_enhanced_unified(
                VecF64::from_array([focal_length, focal_length, 320.0, 240.0, 0.629, 1.02]),
                WIDE,
            );
            let properties = RenderCameraProperties::from_intrinsics(&camera);
            let mut renderer = OffscreenRenderer::new(&context, &properties);

            for anchor in [[100.0f32, 100.0], [500.0, 400.0]] {
                renderer.clear_renderables();
                renderer.update_pixels(vec![make_point2("m", &[anchor], &Color::red(), 9.0)]);

                let image = renderer
                    .render_params(&view_port, &Isometry3F64::identity())
                    .download_rgba(true)
                    .render()
                    .rgba_image
                    .expect("`download_rgba` was requested");

                let (drawn, _) = centroid(&image, is_red).unwrap_or_else(|| {
                    panic!(
                        "view port {}x{}, focal length {focal_length}: the marker anchored at \
                         {anchor:?} was not drawn at all",
                        view_port.width, view_port.height
                    )
                });
                // the anchor is in image pixels, the image is the size of the view port
                let expected = VecF64::<2>::new(
                    (anchor[0] as f64 + 0.5) * view_port.width as f64 / WIDE.width as f64 - 0.5,
                    (anchor[1] as f64 + 0.5) * view_port.height as f64 / WIDE.height as f64 - 0.5,
                );
                assert!(
                    (drawn - expected).norm() < 1.0,
                    "view port {}x{}, focal length {focal_length}: the marker anchored at \
                     {anchor:?} belongs at {:?} but is drawn at {:?}",
                    view_port.width,
                    view_port.height,
                    expected.as_slice(),
                    drawn.as_slice(),
                );
            }
        }
    }
}

/// Surfaces are lit by a light fixed to the camera, so how bright a triangle comes out says which
/// way it faces. Two triangles of one colour: one square on to the camera, one steeply inclined.
///
/// The light sits a little off the optical axis on purpose - exactly on it, the shading term
/// depends only on the angle between the normal and the view direction, which gives a sphere no
/// terminator at all and reads as a flat disc.
#[test]
fn a_surface_is_shaded_by_the_way_it_faces() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // a quad square on to the camera, and one turned steeply away from it
    let facing = [[-0.6f32, -0.6, 4.0], [0.6, -0.6, 4.0], [0.6, 0.6, 4.0]];
    let inclined = [[-0.6f32, -0.6, 4.0], [0.6, -0.6, 5.2], [0.6, 0.6, 5.2]];

    // a shaded surface is dimmer than its own colour, which is the whole point here - so the
    // triangle is picked out by its hue rather than by how bright it is
    let is_reddish = |c: [u8; 4]| c[0] > 100 && c[1] < 80 && c[2] < 80;

    let brightness = |trig: [[f32; 3]; 3], renderer: &mut OffscreenRenderer| {
        renderer.clear_renderables();
        renderer.update_scene(vec![make_mesh3_at(
            "trig",
            &[(trig, Color::red())],
            Isometry3F64::identity(),
        )]);
        let image = render(renderer, Isometry3F64::identity());
        let (centre, count) = centroid(&image, is_reddish).expect("the triangle is drawn");
        assert!(count > 100, "only {count} pixels of the triangle are drawn");
        // at the centroid the triangle certainly covers the whole pixel. As an i32, since these
        // are compared with a margin and u8 arithmetic wraps rather than panics in release.
        image.pixel(centre[0] as usize, centre[1] as usize)[0] as i32
    };

    let facing_red = brightness(facing, &mut renderer);
    let inclined_red = brightness(inclined, &mut renderer);

    assert!(
        facing_red > inclined_red + 10,
        "a triangle facing the camera ({facing_red}) must be brighter than one turned away \
         from it ({inclined_red})"
    );
    assert!(
        inclined_red > 40,
        "the ambient floor must keep a surface turned away from the light off black, got \
         {inclined_red}"
    );
}

/// The light is fixed to the camera - but the pose a shader holds is the one the *intermediate*
/// is rendered from, which for a frustum face is not the camera at all. A light expressed in that
/// frame would swing from face to face.
///
/// A flat surface has one normal, and the light has one direction, so it must come out at one
/// brightness - all the way across a view wide enough that the surface spans several frusta.
#[test]
fn a_flat_surface_is_shaded_evenly_across_the_frusta() {
    const WIDE: ImageSize = ImageSize {
        width: 639,
        height: 479,
    };
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // 180 degrees across the diagonal, so the view is rendered through five faces
    let camera = DynCameraF64::new_enhanced_unified(
        VecF64::from_array([254.0, 254.0, 320.0, 240.0, 0.629, 1.02]),
        WIDE,
    );
    let properties = RenderCameraProperties::from_intrinsics(&camera);
    let mut renderer = OffscreenRenderer::new(&context, &properties);

    // wide enough, and near enough, to reach 70 degrees off the optical axis - well into the
    // faces either side of the forward one
    let quad = [
        (
            [[-6.0f32, -1.0, 0.0], [6.0, -1.0, 0.0], [6.0, 1.0, 0.0]],
            Color::red(),
        ),
        (
            [[-6.0f32, -1.0, 0.0], [6.0, 1.0, 0.0], [-6.0, 1.0, 0.0]],
            Color::red(),
        ),
    ];
    renderer.update_scene(vec![make_mesh3_at(
        "quad",
        &quad,
        // a camera pose of its own, so that the world, the camera and the entity are three
        // different frames rather than all the identity
        Isometry3::from_rotation_and_translation(
            Rotation3::exp(VecF64::<3>::new(0.2, 0.1, 0.3)),
            VecF64::<3>::new(0.0, 0.0, 2.0),
        ),
    )]);

    let image = renderer
        .render_params(
            &WIDE,
            &Isometry3::from_rotation_and_translation(
                Rotation3::exp(VecF64::<3>::new(0.1, -0.25, 0.15)),
                VecF64::<3>::new(0.0, 0.0, 0.0),
            ),
        )
        .download_rgba(true)
        .render()
        .rgba_image
        .expect("`download_rgba` was requested");

    let shades: Vec<i32> = (0..WIDE.height)
        .flat_map(|v| (0..WIDE.width).map(move |u| (u, v)))
        .map(|(u, v)| image.pixel(u, v))
        .filter(|p| p[0] > 60 && p[1] < 60 && p[2] < 60)
        .map(|p| p[0] as i32)
        .collect();

    assert!(
        shades.len() > 5000,
        "only {} pixels of the quad are drawn - it must span several faces for this to mean \
         anything",
        shades.len()
    );
    let brightest = shades.iter().max().unwrap();
    let darkest = shades.iter().min().unwrap();
    assert!(
        brightest - darkest <= 2,
        "the quad is flat and the light is one direction, yet it is shaded from {darkest} to \
         {brightest} across the view - the light differs between the frusta"
    );
}

/// A traced sphere is intersected with the ray through each pixel, so it is exact under the real
/// camera model: its distance is the distance to its surface, and its outline subtends exactly
/// the angle a sphere of that size at that range does - `asin(r / |c|)` - however wide the view
/// and whatever the scene was rasterized into.
#[test]
fn a_traced_sphere_is_exact() {
    const WIDE: ImageSize = ImageSize {
        width: 639,
        height: 479,
    };
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let center = VecF64::<3>::new(0.0, 0.0, 5.0);
    let radius = 1.0f64;

    // either side of the switch from the single plane to the frusta
    for focal_length in [560.0f64, 254.0] {
        let camera = DynCameraF64::new_enhanced_unified(
            VecF64::from_array([focal_length, focal_length, 320.0, 240.0, 0.629, 1.02]),
            WIDE,
        );
        let properties = RenderCameraProperties::from_intrinsics(&camera);
        let mut renderer = OffscreenRenderer::new(&context, &properties);
        renderer.update_scene(vec![make_sphere3_at(
            "sphere",
            &[(
                [center[0] as f32, center[1] as f32, center[2] as f32],
                radius as f32,
            )],
            &Color::red(),
            Isometry3F64::identity(),
        )]);

        let result = renderer
            .render_params(&WIDE, &Isometry3F64::identity())
            .download_rgba(true)
            .render();
        let image = result
            .rgba_image
            .clone()
            .expect("`download_rgba` was requested");
        let inverse_distance = pollster::block_on(download_depth(
            false,
            properties.clipping_planes.cast(),
            context.clone(),
            &WIDE,
            &result,
        ))
        .inverse_distance_image;

        // the nearest point of the sphere, straight down the optical axis
        let uv = camera.cam_proj(center);
        let (u, v) = (uv[0].round() as usize, uv[1].round() as usize);
        let distance = inverse_distance.distance(u, v) as f64;
        assert!(
            (distance - (center.norm() - radius)).abs() < 0.02,
            "focal length {focal_length}: the sphere's surface is {:.3} away, but the distance \
             image reads {distance:.3}",
            center.norm() - radius
        );

        // A small sphere far away, where the distance has to come out of numbers which nearly
        // cancel: `|c|^2 - r^2` against `along^2` is 90000 against 90000 to reach 0.0004, and in
        // f32 that is nothing at all.
        renderer.clear_renderables();
        renderer.update_scene(vec![make_sphere3_at(
            "far",
            &[([0.0f32, 0.0, 300.0], 0.02f32)],
            &Color::red(),
            Isometry3F64::identity(),
        )]);
        let far_result = renderer
            .render_params(&WIDE, &Isometry3F64::identity())
            .render();
        let far_distance = pollster::block_on(download_depth(
            false,
            properties.clipping_planes.cast(),
            context.clone(),
            &WIDE,
            &far_result,
        ))
        .inverse_distance_image
        .distance(320, 240) as f64;
        assert!(
            (far_distance - 299.98).abs() < 0.01,
            "focal length {focal_length}: a sphere of radius 0.02 at 300 m has its surface at \
             299.98, but the distance image reads {far_distance:.4}"
        );

        renderer.clear_renderables();
        renderer.update_scene(vec![make_sphere3_at(
            "sphere",
            &[(
                [center[0] as f32, center[1] as f32, center[2] as f32],
                radius as f32,
            )],
            &Color::red(),
            Isometry3F64::identity(),
        )]);

        // and its outline covers the solid angle a sphere subtends
        let is_reddish = |c: [u8; 4]| c[0] > 60 && c[1] < 60 && c[2] < 60;
        let (_, drawn) = centroid(&image, is_reddish).expect("the sphere is drawn");
        let half_angle = (radius / center.norm()).asin();
        let expected = (0..WIDE.height)
            .flat_map(|v| (0..WIDE.width).map(move |u| (u, v)))
            .filter(|(u, v)| {
                let ray = camera.cam_unproj_to_unit_vector(VecF64::<2>::new(*u as f64, *v as f64));
                ray.vector()[2].acos() < half_angle
            })
            .count();
        let off_by = (drawn as f64 - expected as f64).abs() / expected as f64;
        assert!(
            off_by < 0.02,
            "focal length {focal_length}: the sphere covers {drawn} pixels, but a sphere of \
             radius {radius} at {:.3} subtends {expected}",
            center.norm()
        );
    }
}

/// The traced primitives and the rasterized scene are composited by distance along the ray, which
/// is what both of them now measure - so a point in front of a sphere hides it, and one behind it
/// is hidden.
#[test]
fn a_traced_sphere_and_rasterized_geometry_take_turns_in_front() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let sphere = make_sphere3_at(
        "sphere",
        &[([0.0f32, 0.0, 5.0], 1.0f32)],
        &Color::red(),
        Isometry3F64::identity(),
    );

    for (name, z, sphere_wins) in [("in front of", 3.0f32, false), ("behind", 8.0, true)] {
        renderer.clear_renderables();
        renderer.update_scene(vec![
            sphere.clone(),
            make_point3_at(
                "point",
                &[[0.0f32, 0.0, z]],
                &Color::blue(),
                40.0,
                Isometry3F64::identity(),
            ),
        ]);
        let image = render(&mut renderer, Isometry3F64::identity());
        let middle = image.pixel(W / 2, H / 2);

        match sphere_wins {
            true => assert!(
                middle[0] > middle[2],
                "the point is {name} the sphere, so the sphere must be seen at the middle - \
                 found {:?}",
                [middle[0], middle[1], middle[2]]
            ),
            false => assert!(
                middle[2] > middle[0],
                "the point is {name} the sphere, so the point must be seen at the middle - \
                 found {:?}",
                [middle[0], middle[1], middle[2]]
            ),
        }
    }
}

/// An ellipsoid is traced the same way a sphere is - the sphere being the one whose shape is a
/// scaled identity - so the silhouette must match, pixel for pixel, what the intersection says on
/// the cpu. Anisotropic and rotated, so that every entry of the shape matters.
#[test]
fn a_traced_ellipsoid_matches_the_intersection_on_the_cpu() {
    const WIDE: ImageSize = ImageSize {
        width: 639,
        height: 479,
    };
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let center = VecF64::<3>::new(0.3, -0.2, 4.0);
    let rotation = Rotation3::<f64, 1, 0, 0>::exp(VecF64::<3>::new(0.4, 0.7, -0.3));
    let shape: MatF64<3, 3> =
        rotation.matrix() * MatF64::<3, 3>::from_diagonal(&VecF64::<3>::new(1.2, 0.5, 0.8));
    let to_unit_sphere = shape.try_inverse().expect("the shape is invertible");

    for focal_length in [560.0f64, 254.0] {
        let camera = DynCameraF64::new_enhanced_unified(
            VecF64::from_array([focal_length, focal_length, 320.0, 240.0, 0.629, 1.02]),
            WIDE,
        );
        let properties = RenderCameraProperties::from_intrinsics(&camera);
        let mut renderer = OffscreenRenderer::new(&context, &properties);
        renderer.update_scene(vec![make_ellipsoid3_at(
            "ellipsoid",
            vec![Ellipsoid3 {
                center: SVec::<f32, 3>::new(center[0] as f32, center[1] as f32, center[2] as f32),
                shape,
                color: Color::red(),
            }],
            Isometry3F64::identity(),
        )]);

        let image = renderer
            .render_params(&WIDE, &Isometry3F64::identity())
            .download_rgba(true)
            .render()
            .rgba_image
            .expect("`download_rgba` was requested");

        // the same intersection, in f64, against the ray of every pixel
        let mut drawn = 0usize;
        let mut expected = 0usize;
        let mut disagree = 0usize;
        for v in 0..WIDE.height {
            for u in 0..WIDE.width {
                let ray = camera
                    .cam_unproj_to_unit_vector(VecF64::<2>::new(u as f64, v as f64))
                    .vector();
                let o = to_unit_sphere * (-center);
                let d = to_unit_sphere * ray;
                let hits = d.dot(&d) - o.cross(&d).norm_squared() > 0.0 && center.dot(&ray) > 0.0;

                let p = image.pixel(u, v);
                let is_drawn = p[0] > 60 && p[1] < 60 && p[2] < 60;
                drawn += is_drawn as usize;
                expected += hits as usize;
                disagree += (is_drawn != hits) as usize;
            }
        }

        assert!(
            expected > 2000,
            "the ellipsoid barely covers the view: {expected} pixels"
        );
        // the disagreement should be a rim one pixel wide, where coverage is partial
        let perimeter = 4.0 * (expected as f64).sqrt();
        assert!(
            (disagree as f64) < 2.0 * perimeter,
            "focal length {focal_length}: the cpu says {expected} pixels are inside the \
             ellipsoid and the render draws {drawn}, disagreeing on {disagree} - more than the \
             rim of {perimeter:.0}"
        );
    }
}

/// The same conditioning the sphere needs, on a shape which is not a scaled identity: the
/// discriminant has to be taken as `|d|^2 - |o x d|^2` in the space where the ellipsoid is the
/// unit sphere, or a small far one is missed entirely.
#[test]
fn a_small_distant_ellipsoid_survives_its_own_arithmetic() {
    // The principal point sits on a pixel centre, so that the ray of the pixel sampled below runs
    // exactly through the middle of the view. An ellipsoid this small is far narrower than one
    // pixel, and half a pixel of aim would miss it entirely.
    const SIZE: ImageSize = ImageSize {
        width: 641,
        height: 481,
    };
    let Some((context, mut renderer)) = renderer(DynCameraF64::new_pinhole(
        VecF64::from_array([500.0, 500.0, 320.0, 240.0]),
        SIZE,
    )) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // semi-axes of a few centimetres, 300 m away: `|c|^2` is 90000 and `r^2` is 0.0009
    renderer.update_scene(vec![make_ellipsoid3_at(
        "far",
        vec![Ellipsoid3 {
            center: SVec::<f32, 3>::new(0.0, 0.0, 300.0),
            shape: MatF64::<3, 3>::from_diagonal(&VecF64::<3>::new(0.02, 0.05, 0.03)),
            color: Color::red(),
        }],
        Isometry3F64::identity(),
    )]);

    let result = renderer
        .render_params(&SIZE, &Isometry3F64::identity())
        .render();
    let distance = pollster::block_on(download_depth(
        false,
        renderer.camera_properties().clipping_planes.cast(),
        context,
        &SIZE,
        &result,
    ))
    .inverse_distance_image
    .distance(320, 240) as f64;

    assert!(
        (distance - 299.97).abs() < 0.01,
        "the near surface of the ellipsoid is 299.97 away - its semi-axis along the ray is 0.03 - \
         but the distance image reads {distance:.4}"
    );
}

/// A 2d point is round. It used to be square - the coverage was taken along whichever axis was
/// closest to an edge, which is a square's distance field rather than a disc's - so a point
/// marker did not look like the thing it marks, and drew 4/pi times too much of the image.
#[test]
fn a_2d_point_is_round() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    const RADIUS: f32 = 30.0;
    renderer.update_pixels(vec![make_point2(
        "point",
        &[[127.0f32, 127.0]],
        &Color::red(),
        2.0 * RADIUS,
    )]);
    let image = render(&mut renderer, Isometry3F64::identity());

    let (_, drawn) = centroid(&image, is_red).expect("the point is drawn");
    let area = core::f64::consts::PI * (RADIUS as f64).powi(2);
    assert!(
        (drawn as f64 - area).abs() < 0.1 * area,
        "a point of radius {RADIUS} covers {area:.0} pixels, but {drawn} are drawn - a square of \
         the same width would cover {:.0}",
        4.0 * (RADIUS as f64).powi(2)
    );

    // and the corners of its bounding box are outside it
    let corner = image.pixel(
        (127.0 - 0.8 * RADIUS as f64) as usize,
        (127.0 - 0.8 * RADIUS as f64) as usize,
    );
    assert!(
        !is_red([corner[0], corner[1], corner[2], corner[3]]),
        "the corner of the point's bounding box is drawn: {:?} - the marker is square",
        [corner[0], corner[1], corner[2]]
    );
}

/// A 2d ellipse covers the area its shape says - `pi a b` - and the circle is the one whose shape
/// is a scaled identity, so it agrees with the round point marker of the same radius.
#[test]
fn a_2d_ellipse_covers_the_area_of_its_shape() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let shape: MatF64<2, 2> = Rotation2::<f64, 1, 0, 0>::exp(VecF64::<1>::new(0.6)).matrix()
        * MatF64::<2, 2>::from_diagonal(&VecF64::<2>::new(40.0, 15.0));
    renderer.update_pixels(vec![named_ellipse2(
        "ellipse",
        vec![Ellipse2 {
            center: SVec::<f32, 2>::new(127.0, 127.0),
            shape,
            line_width: 0.0,
            color: Color::red(),
        }],
    )]);
    let image = render(&mut renderer, Isometry3F64::identity());

    let (centre, drawn) = centroid(&image, is_red).expect("the ellipse is drawn");
    let area = core::f64::consts::PI * 40.0 * 15.0;
    assert!(
        (drawn as f64 - area).abs() < 0.05 * area,
        "an ellipse with semi-axes 40 and 15 covers {area:.0} pixels, but {drawn} are drawn"
    );
    assert!(
        (centre - VecF64::<2>::new(127.0, 127.0)).norm() < 1.0,
        "the ellipse is centred at {:?} rather than at its anchor",
        centre.as_slice()
    );

    // rotated by the same angle, the extent along the major axis reaches 40 pixels from the centre
    let major =
        Rotation2::<f64, 1, 0, 0>::exp(VecF64::<1>::new(0.6)).matrix() * VecF64::<2>::new(1.0, 0.0);
    for (name, at, inside) in [("just inside", 38.0, true), ("just outside", 42.0, false)] {
        let probe = VecF64::<2>::new(127.0, 127.0) + major * at;
        let p = image.pixel(probe[0].round() as usize, probe[1].round() as usize);
        assert_eq!(
            is_red([p[0], p[1], p[2], p[3]]),
            inside,
            "{name} the major axis, at {:?}, the ellipse reads {:?}",
            probe.as_slice(),
            [p[0], p[1], p[2]]
        );
    }
}

/// An ellipse is a region *of the image* - a reprojection covariance, say - so it grows with the
/// 2d zoom, unlike a point marker, whose size is a view-port quantity and must not.
#[test]
fn a_2d_ellipse_grows_with_the_zoom() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    renderer.update_pixels(vec![
        named_ellipse2(
            "ellipse",
            vec![Ellipse2::circle(
                SVec::<f32, 2>::new(127.0, 127.0),
                20.0,
                0.0,
                Color::red(),
            )],
        ),
        // clear of the ellipse at both zooms, so neither hides the other
        make_point2("point", &[[160.0f32, 160.0]], &Color::blue(), 20.0),
    ]);

    let mut area = |zoom: TranslationAndScaling, pick: fn([u8; 4]) -> bool| {
        let image = renderer
            .render_params(&ImageSize::new(W, H), &Isometry3F64::identity())
            .zoom(zoom)
            .download_rgba(true)
            .render()
            .rgba_image
            .expect("`download_rgba` was requested");
        centroid(&image, pick).map(|(_, count)| count).unwrap_or(0)
    };

    let unzoomed = area(TranslationAndScaling::identity(), is_red);
    let zoom = TranslationAndScaling {
        translation: VecF64::<2>::new(-127.0, -127.0),
        scaling: VecF64::<2>::new(2.0, 2.0),
    };
    let zoomed = area(zoom, is_red);
    assert!(
        (zoomed as f64 - 4.0 * unzoomed as f64).abs() < 0.1 * 4.0 * unzoomed as f64,
        "at twice the zoom the ellipse should cover four times the {unzoomed} pixels it did, but \
         it covers {zoomed}"
    );

    // the point marker, drawn at the same place, keeps its size
    let point_unzoomed = area(TranslationAndScaling::identity(), is_blue);
    let point_zoomed = area(zoom, is_blue);
    assert!(
        (point_zoomed as f64 - point_unzoomed as f64).abs() < 0.1 * point_unzoomed as f64,
        "the point marker covered {point_unzoomed} pixels and now covers {point_zoomed} - its \
         size is a view-port quantity and must not follow the zoom"
    );
}

/// A disk, an ellipse, a rectangle and the whole plane are one surface with different bounds, so
/// the intersection is shared and only the predicate differs. Each is checked by the thing that
/// distinguishes it: the area of a disk, the corner a rectangle has and a disk has not, and the
/// hollow middle of an outlined one.
#[test]
fn the_planar_bounds_are_what_tells_them_apart() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    // the test camera: 200 pixels of focal length, so a disk of radius r at z covers
    // `pi (200 r / z)^2` pixels
    const FOCAL: f64 = 200.0;
    let axes = [
        SVec::<f32, 3>::new(0.5, 0.0, 0.0),
        SVec::<f32, 3>::new(0.0, 0.5, 0.0),
    ];
    let facing = SVec::<f32, 3>::new(0.0, 0.0, -1.0);
    let center = SVec::<f32, 3>::new(0.0, 0.0, 4.0);

    let draw = |renderer: &mut OffscreenRenderer, planar: Planar3| {
        renderer.clear_renderables();
        renderer.update_scene(vec![make_planar3("planar", vec![planar])]);
        render(renderer, Isometry3F64::identity())
    };

    // a disk covers the area its radius says
    let image = draw(
        &mut renderer,
        Planar3::disk(center, facing, 0.5, Color::red()),
    );
    let (_, drawn) = centroid(&image, is_red).expect("the disk is drawn");
    let expected = core::f64::consts::PI * (FOCAL * 0.5 / 4.0).powi(2);
    assert!(
        (drawn as f64 - expected).abs() < 0.05 * expected,
        "a disk of radius 0.5 at 4 m covers {expected:.0} pixels, but {drawn} are drawn"
    );

    // a rectangle of the same axes reaches its corners, which the disk does not
    let rectangle = draw(
        &mut renderer,
        Planar3::rectangle(center, axes, Color::red()),
    );
    let (_, rectangle_drawn) = centroid(&rectangle, is_red).expect("the rectangle is drawn");
    assert!(
        (rectangle_drawn as f64 - 4.0 / core::f64::consts::PI * drawn as f64).abs()
            < 0.05 * rectangle_drawn as f64,
        "the rectangle covers {rectangle_drawn} pixels against the disk's {drawn} - it should be \
         4/pi of it"
    );
    // the corner itself, three quarters of the way out along both axes
    let corner = (0.75 * FOCAL * 0.5 / 4.0) as usize;
    let at_corner = rectangle.pixel(W / 2 + corner, H / 2 + corner);
    assert!(
        is_red([at_corner[0], at_corner[1], at_corner[2], at_corner[3]]),
        "the rectangle's corner is not drawn: {:?}",
        [at_corner[0], at_corner[1], at_corner[2]]
    );

    // an outlined ellipse is a ring: drawn at the rim, not in the middle
    let ring = draw(
        &mut renderer,
        Planar3::ellipse(center, axes, Color::red()).outlined(4.0),
    );
    let middle = ring.pixel(W / 2, H / 2);
    assert!(
        !is_red([middle[0], middle[1], middle[2], middle[3]]),
        "the middle of a ring is filled in: {:?}",
        [middle[0], middle[1], middle[2]]
    );
    let (_, ring_drawn) = centroid(&ring, is_red).expect("the ring is drawn");
    assert!(
        ring_drawn < drawn / 2,
        "the ring covers {ring_drawn} pixels against the filled disk's {drawn} - it is not hollow"
    );
}

/// An unbounded plane cannot be rasterized without choosing a size, and its horizon is where the
/// distance along the ray runs to infinity - which is exactly what the inverse distance buffer
/// means by "nothing here". So the horizon converges on the background with no far plane to clamp
/// to.
#[test]
fn an_infinite_plane_reaches_its_horizon() {
    let Some((context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // a ground plane five metres below the camera, which looks along +z with +y down. Five, so
    // that the ground near the horizon is further away than the far plane - a distance a
    // rasterized quad could not have reached.
    renderer.update_scene(vec![make_planar3(
        "ground",
        vec![Planar3::plane(
            SVec::<f32, 3>::new(0.0, 5.0, 0.0),
            SVec::<f32, 3>::new(0.0, -1.0, 0.0),
            Color::red(),
        )],
    )]);

    let result = renderer
        .render_params(&ImageSize::new(W, H), &Isometry3F64::identity())
        .download_rgba(true)
        .render();
    let image = result
        .rgba_image
        .clone()
        .expect("`download_rgba` was requested");
    let inverse_distance = pollster::block_on(download_depth(
        false,
        renderer.camera_properties().clipping_planes.cast(),
        context,
        &ImageSize::new(W, H),
        &result,
    ))
    .inverse_distance_image;

    // a lit surface is dimmer than its own colour, so the ground is picked out by hue
    let is_reddish = |c: [u8; 4]| c[0] > 60 && c[1] < 60 && c[2] < 60;

    // the horizon is the row of the principal point: below it the ground, above it nothing
    let horizon = 127;
    for (name, v, expect_ground) in [
        ("below", horizon + 40, true),
        ("above", horizon - 40, false),
    ] {
        let p = image.pixel(W / 2, v);
        assert_eq!(
            is_reddish([p[0], p[1], p[2], p[3]]),
            expect_ground,
            "{name} the horizon the plane should {}be drawn, but the pixel is {:?}",
            match expect_ground {
                true => "",
                false => "not ",
            },
            [p[0], p[1], p[2]]
        );
    }

    // and it recedes: further down the image is nearer, and the horizon itself is unreachable
    let near = inverse_distance.distance(W / 2, horizon + 80) as f64;
    let far = inverse_distance.distance(W / 2, horizon + 5) as f64;
    assert!(
        near < far,
        "the ground should recede towards the horizon, but it reads {near:.2} low in the image \
         and {far:.2} near the horizon"
    );
    assert!(
        inverse_distance.inverse_distance(W / 2, horizon) <= 0.0,
        "the horizon itself is infinitely far, so it must read as background - it reads {}",
        inverse_distance.inverse_distance(W / 2, horizon)
    );

    // Nothing is clamped: one row below the horizon the ground is 2000 m away, twice the far
    // plane, and it says so. A quad would have had to stop somewhere.
    let just_below = inverse_distance.distance(W / 2, horizon + 1) as f64;
    let far = renderer.camera_properties().clipping_planes.far;
    assert!(
        just_below > 1.5 * far,
        "one row below the horizon the ground is about 2000 m away, but it reads \
         {just_below:.1} - the far plane is {far}"
    );
}

/// A grid on a plane is a pattern, filtered analytically over the footprint of each pixel. That
/// filtering is the point: near the horizon one pixel spans many grid periods, and a point-sampled
/// grid there alternates between landing on a line and missing it, which is noise. Filtered, it
/// converges on the average of the line over the period.
#[test]
fn a_grid_is_filtered_rather_than_sampled() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    const SPACING: f32 = 0.25;
    const LINE_WIDTH: f32 = 1.5;
    renderer.update_scene(vec![make_planar3(
        "ground",
        vec![
            Planar3::plane(
                SVec::<f32, 3>::new(0.0, 1.0, 0.0),
                SVec::<f32, 3>::new(0.0, -1.0, 0.0),
                Color::red(),
            )
            .with_grid(SPACING, LINE_WIDTH),
        ],
    )]);
    let image = render(&mut renderer, Isometry3F64::identity());

    // Close to the camera, low in the image, the lines are separate: the row alternates between
    // drawn and not, so it holds both.
    // The plane is red over a white background, so the green channel falls from one to zero as
    // the grid covers the pixel - it is the coverage, read off the image.
    let coverage_of = |v: usize| {
        (0..W)
            .map(|u| 1.0 - image.pixel(u, v)[1] as f64 / 255.0)
            .collect::<Vec<_>>()
    };
    // Any row down there will do, so long as it is not itself lying along one of the lines
    // running across the view - which would read as covered from end to end however well the grid
    // is drawn.
    let near = (240..256)
        .map(coverage_of)
        .max_by(|a, b| {
            let range = |c: &Vec<f64>| {
                c.iter().cloned().fold(f64::MIN, f64::max)
                    - c.iter().cloned().fold(f64::MAX, f64::min)
            };
            range(a).partial_cmp(&range(b)).expect("no NaN in an image")
        })
        .expect("the range is not empty");
    let near_min = near.iter().cloned().fold(f64::MAX, f64::min);
    let near_max = near.iter().cloned().fold(f64::MIN, f64::max);
    assert!(
        near_min < 0.3 && near_max > 0.7,
        "close to the camera the grid should be lines against gaps, but the row runs from \
         {near_min:.2} to {near_max:.2}"
    );
    // and the lines have edges: a filtered line covers its end pixels in part, where a sampled
    // one is on or off and nothing between
    let edges = near.iter().filter(|c| **c > 0.2 && **c < 0.8).count();
    assert!(
        edges >= 4,
        "only {edges} pixels of the row are partly covered - the lines have no antialiased edge, \
         so the grid is being sampled rather than filtered"
    );

    // Near the horizon a pixel spans several periods. Point sampled, the row would flicker
    // between landing on a line and missing it; filtered, the lines merge into a sheet, which
    // says nothing about scale - so the ruling gives way where it can no longer be resolved, and
    // the row is quiet either way. What is left is a faint even tint rather than nothing: the
    // plane goes on to its horizon, whether or not its scale can still be read there.
    let far = coverage_of(129);
    let mean = far.iter().sum::<f64>() / far.len() as f64;
    let spread = far
        .iter()
        .map(|c| (c - mean).abs())
        .fold(f64::MIN, f64::max);
    assert!(
        mean > 0.05 && mean < 0.4,
        "by the horizon the ruling should have given way to a faint tint, but the row averages \
         {mean:.2}"
    );
    assert!(
        spread < 0.35,
        "near the horizon the grid varies by {spread:.2} across the row - it is being sampled \
         rather than filtered"
    );

    // The band between is where a footprint taken along one axis alone shows up. Stepping *down*
    // the image moves a ground plane far further than stepping across it - near the horizon, many
    // grid periods against a fraction of one - so the lines running across the view have to be
    // ruled, and filtered, on their own terms. Measured down a column, off the centre line: they
    // come out as a handful of separate lines with clear ground between them. With one footprint
    // for both axes they stay at the spacing the near field asked for, which by here is finer
    // than the pixel, and the column crosses one at practically every row.
    let column: Vec<bool> = (129..150)
        .map(|v| 1.0 - image.pixel(100, v)[1] as f64 / 255.0 > 0.25)
        .collect();
    let lines = column.windows(2).filter(|w| w[1] && !w[0]).count() + usize::from(column[0]);
    let gaps = column.iter().filter(|covered| !**covered).count();
    assert!(
        lines <= 5 && gaps >= 8,
        "down twenty-one rows near the horizon the column crosses {lines} lines with {gaps} rows \
         of ground between them - the lines running across the view are being drawn at a spacing \
         this far off the plane cannot resolve"
    );
}

/// A capsule is every point within a radius of a segment, so it is round from any angle and its
/// hemispherical ends leave no seam. The arrow built from one - a capsule for the shaft, a cone
/// for the head, a disk to close it - is the shape a scene of poses and residuals is made of.
#[test]
fn a_capsule_and_a_cone_make_an_arrow() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    const FOCAL: f64 = 200.0;

    // a capsule across the view, seen broadside: a rectangle of `2 r` by its length, with a
    // half-disk at each end - which is `2 r L + pi r^2` in the plane it lies in
    let (from, to, radius) = (
        SVec::<f32, 3>::new(-0.6, 0.0, 4.0),
        SVec::<f32, 3>::new(0.6, 0.0, 4.0),
        0.15f32,
    );
    renderer.update_scene(vec![make_capsule3(
        "capsule",
        vec![Capsule3 {
            from,
            to,
            radius,
            color: Color::red(),
            flat_ends: false,
        }],
    )]);
    let image = render(&mut renderer, Isometry3F64::identity());
    let is_reddish = |c: [u8; 4]| c[0] > 60 && c[1] < 60 && c[2] < 60;
    let (centre, drawn) = centroid(&image, is_reddish).expect("the capsule is drawn");

    let scale = FOCAL / 4.0;
    let expected = (2.0 * radius as f64 * 1.2 + core::f64::consts::PI * (radius as f64).powi(2))
        * scale
        * scale;
    assert!(
        (drawn as f64 - expected).abs() < 0.08 * expected,
        "a capsule of radius {radius} and length 1.2 at 4 m covers {expected:.0} pixels, but \
         {drawn} are drawn"
    );
    assert!(
        (centre - VecF64::<2>::new(127.5, 127.5)).norm() < 2.0,
        "the capsule is centred at {:?} rather than between its ends",
        centre.as_slice()
    );

    // an arrow: the head is wider than the shaft, so the far end covers more of a column than
    // the near one
    renderer.clear_renderables();
    renderer.update_scene(make_arrow3(
        "arrow",
        &[(
            SVec::<f32, 3>::new(-0.8, 0.0, 4.0),
            SVec::<f32, 3>::new(0.8, 0.0, 4.0),
        )],
        0.05,
        &Color::red(),
    ));
    let arrow = render(&mut renderer, Isometry3F64::identity());
    let column = |u: usize| {
        (0..H)
            .filter(|v| {
                let p = arrow.pixel(u, *v);
                is_reddish([p[0], p[1], p[2], p[3]])
            })
            .count()
    };
    // the arrow runs from x = -0.8 to x = 0.8 at 4 m, which is u = 87 to u = 168; its head is
    // the last 0.3 of that, from u = 152
    // the shaft anywhere along its length, and the head at its widest, just ahead of the base
    let at_tail = column(95);
    let at_head = column(154);
    assert!(
        at_tail > 0 && at_head >= at_tail * 2,
        "the arrow's head should be wider than its shaft, but the shaft covers {at_tail} pixels \
         of its column and the head {at_head}"
    );
}

/// A checkerboard is half covered whatever the scale, so filtering it settles onto an even tone
/// rather than crowding into a sheet the way lines of a fixed screen width do. That means it needs
/// no fade - and so the plane keeps its surface all the way to the horizon.
///
/// A grid, which must fade, leaves the plane invisible before its horizon: with the example's
/// distorted camera the fade eats inwards from the edges of each row, so the lines end in a stub
/// around the vanishing point - 19 pixels of a 639 wide row, widening downwards into a wedge.
#[test]
fn a_checkered_plane_keeps_its_surface_to_the_horizon() {
    const WIDE: ImageSize = ImageSize {
        width: 639,
        height: 479,
    };
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let camera = DynCameraF64::new_enhanced_unified(
        VecF64::from_array([500.0, 500.0, 320.0, 240.0, 0.629, 1.02]),
        WIDE,
    );
    let properties = RenderCameraProperties::from_intrinsics(&camera);
    let mut renderer = OffscreenRenderer::new(&context, &properties);
    renderer.update_scene(vec![make_planar3(
        "ground",
        vec![
            Planar3::plane(
                SVec::<f32, 3>::new(0.0, 1.5, 0.0),
                SVec::<f32, 3>::new(0.0, -1.0, 0.0),
                Color::red(),
            )
            .with_checker(
                0.5,
                Color {
                    r: 0.3,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
            ),
        ],
    )]);
    let image = renderer
        .render_params(&WIDE, &Isometry3::trans_z(-5.0))
        .download_rgba(true)
        .render()
        .rgba_image
        .expect("`download_rgba` was requested");

    let tones = |v: usize| {
        (0..WIDE.width)
            .map(|u| image.pixel(u, v)[0] as f64 / 255.0)
            .collect::<Vec<_>>()
    };
    let spread = |ts: &[f64]| {
        ts.iter().cloned().fold(f64::MIN, f64::max) - ts.iter().cloned().fold(f64::MAX, f64::min)
    };

    // above the horizon there is nothing
    assert!(
        tones(238).iter().all(|t| *t > 0.99),
        "the plane is drawn above its own horizon"
    );

    // just below it, the whole row - not a stub around the vanishing point
    let near_horizon = tones(245);
    let drawn = near_horizon.iter().filter(|t| **t < 0.99).count();
    assert_eq!(
        drawn, WIDE.width,
        "just below the horizon only {drawn} of {} pixels are drawn - the plane is fading out \
         before its horizon rather than settling onto a tone",
        WIDE.width
    );
    assert!(
        spread(&near_horizon) < 0.03,
        "near the horizon the checker should have settled onto one tone, but the row spreads \
         over {:.2}",
        spread(&near_horizon)
    );

    // and near the camera the squares are there to see
    assert!(
        spread(&tones(430)) > 0.05,
        "close to the camera the checker should show its squares, but the row spreads over only \
         {:.2}",
        spread(&tones(430))
    );
}

/// Coordinate axes: three arrows from the origin, their shafts one colour and their tips red,
/// green and blue - so that the colour says which axis without three coloured arrows fighting
/// each other.
#[test]
fn coordinate_axes_have_black_shafts_and_rgb_tips() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    renderer.update_scene(make_axes_arrows3("axes", 1.0, Isometry3F64::identity()));
    // looking at the origin from in front, so +x runs right and +y runs down the image
    let image = render(&mut renderer, Isometry3::trans_z(-3.0));

    let is_dark = |c: [u8; 4]| c[0] < 90 && c[1] < 90 && c[2] < 90;
    let (shafts, shaft_pixels) = centroid(&image, is_dark).expect("the shafts are drawn");
    assert!(
        shaft_pixels > 200,
        "only {shaft_pixels} pixels of shaft are drawn"
    );
    // three shafts from the origin, two of which run right and down, so their centroid is below
    // and to the right of it
    assert!(
        shafts[0] > 127.0 && shafts[1] > 127.0,
        "the shafts should run right and down from the origin, but their centroid is {:?}",
        shafts.as_slice()
    );

    // each tip is where its axis points: +x to the right, +y down, +z towards the viewer and so
    // near the middle
    let (red, red_pixels) = centroid(&image, is_red).expect("the x tip is drawn");
    let (green, green_pixels) = centroid(&image, is_green).expect("the y tip is drawn");
    let (blue, blue_pixels) = centroid(&image, is_blue).expect("the z tip is drawn");
    // The tips are small - a head radius of 0.04 at 3 m is under three pixels across - and the z
    // one points straight away from this camera, so it is seen end on and smaller still.
    for (name, count) in [("x", red_pixels), ("y", green_pixels)] {
        assert!(count > 10, "the {name} tip covers only {count} pixels");
    }
    assert!(
        blue_pixels > 3,
        "the z tip covers only {blue_pixels} pixels"
    );
    assert!(
        red[0] > 160.0 && (red[1] - 127.5).abs() < 20.0,
        "the x tip should be out to the right, but it is at {:?}",
        red.as_slice()
    );
    assert!(
        green[1] > 160.0 && (green[0] - 127.5).abs() < 20.0,
        "the y tip should be below, but it is at {:?}",
        green.as_slice()
    );
    assert!(
        (blue - VecF64::<2>::new(127.5, 127.5)).norm() < 25.0,
        "the z tip points at the viewer, so it should be near the middle, but it is at {:?}",
        blue.as_slice()
    );
}

/// A camera's own frame is x right, y down, z forward. A bird's eye view of a z-up scene is that
/// frame turned to look along the scene's -z, with the camera's y along the scene's -y - which
/// puts x to the right of the image and y up it, the way a map is drawn.
///
/// This pins the convention the whole viewer rests on: flip a sign in it and everything still
/// renders, mirrored.
#[test]
fn a_birds_eye_camera_puts_x_right_and_y_up() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    renderer.update_scene(vec![
        make_sphere3("x", &[([1.0f32, 0.0, 0.0], 0.15f32)], &Color::red()),
        make_sphere3("y", &[([0.0f32, 1.0, 0.0], 0.15f32)], &Color::green()),
    ]);

    let looking_down = Isometry3::from_rotation_and_translation(
        Rotation3::try_from_mat(MatF64::<3, 3>::from_columns(&[
            VecF64::<3>::new(1.0, 0.0, 0.0),
            VecF64::<3>::new(0.0, -1.0, 0.0),
            VecF64::<3>::new(0.0, 0.0, -1.0),
        ]))
        .expect("the three axes are orthonormal"),
        VecF64::<3>::new(0.0, 0.0, 4.0),
    );
    let image = render(&mut renderer, looking_down);

    let (x_marker, _) = centroid(&image, is_red).expect("the +x marker is drawn");
    let (y_marker, _) = centroid(&image, is_green).expect("the +y marker is drawn");
    let middle = VecF64::<2>::new(127.5, 127.5);

    assert!(
        x_marker[0] > middle[0] + 20.0 && (x_marker[1] - middle[1]).abs() < 10.0,
        "+x of the scene should lie to the right of the image, but its marker is at {:?}",
        x_marker.as_slice()
    );
    assert!(
        y_marker[1] < middle[1] - 20.0 && (y_marker[0] - middle[0]).abs() < 10.0,
        "+y of the scene should lie up the image, but its marker is at {:?}",
        y_marker.as_slice()
    );
}

/// A ray running parallel to a disk's plane meets it infinitely far away, and its neighbour lands
/// somewhere else entirely - so the margin the coverage is measured from, and the step it is
/// measured against, are both enormous, and their ratio is not.
///
/// Without a guard that puts the coverage to nothing unless the pixel or one beside it is inside
/// the bound, that ratio draws the disk faintly along every direction parallel to its plane - a
/// great circle, which crosses the view as a hairline. It showed up in the example as a crack
/// across the sky, from a disk no wider than an arrow head.
#[test]
fn a_disk_is_not_drawn_along_the_directions_parallel_to_it() {
    const WIDE: ImageSize = ImageSize {
        width: 639,
        height: 479,
    };
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let camera = DynCameraF64::new_enhanced_unified(
        VecF64::from_array([500.0, 500.0, 320.0, 240.0, 0.629, 1.02]),
        WIDE,
    );
    let properties = RenderCameraProperties::from_intrinsics(&camera);
    let mut renderer = OffscreenRenderer::new(&context, &properties);

    // the bases of the three arrow heads of a set of axes, at the scale the example draws them
    renderer.update_scene(vec![make_planar3(
        "disks",
        (0..3)
            .map(|axis| {
                let mut normal = SVec::<f32, 3>::zeros();
                normal[axis] = 1.0;
                Planar3::disk(normal * 0.672, normal, 0.032, Color::red())
            })
            .collect(),
    )]);

    // the view it was reported from, which looks across the disk's plane
    let image = renderer
        .render_params(
            &WIDE,
            &Isometry3::from_rotation_and_translation(
                Rotation3::exp(VecF64::<3>::new(
                    -1.742_301_700_765_572_6,
                    -0.197_794_499_417_220_7,
                    0.482_162_372_833_677_4,
                )),
                VecF64::<3>::new(
                    1.708_855_340_557_972_3,
                    -3.620_886_271_044_731,
                    1.058_381_311_174_86,
                ),
            ),
        )
        .download_rgba(true)
        .render()
        .rgba_image
        .expect("`download_rgba` was requested");

    // The disk is a few pixels across at that range. A hairline along the directions parallel to
    // its plane crosses the whole view, so it is told apart by how many pixels it touches and how
    // far apart they lie.
    // Any pixel tinted towards the disk's colour, however faintly - the ghost is a wash rather
    // than a solid, so looking only for saturated red would miss it entirely.
    let tinted = (0..WIDE.height)
        .flat_map(|v| (0..WIDE.width).map(move |u| (u, v)))
        .filter(|(u, v)| {
            let p = image.pixel(*u, *v);
            p[0] as i32 > p[1] as i32 + 20 && p[0] as i32 > p[2] as i32 + 20
        })
        .count();
    assert!(
        tinted < 120,
        "{tinted} pixels are tinted towards the disks, which are a few pixels across between \
         them - they are being painted along the directions parallel to their planes"
    );
}

/// A field of solid covariance ellipsoids is an opaque mass which hides the scene and each other.
/// Rings - the outlines of where an ellipsoid meets its own three principal planes - show the same
/// shape and leave what is behind them visible.
#[test]
fn ellipsoid_rings_are_hollow() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let ellipsoid = Ellipsoid3 {
        center: SVec::<f32, 3>::new(0.0, 0.0, 4.0),
        shape: MatF64::<3, 3>::from_diagonal(&VecF64::<3>::new(0.5, 0.5, 0.5)),
        color: Color::red(),
    };

    // solid first, for the area to compare against
    renderer.update_scene(vec![make_ellipsoid3("solid", vec![ellipsoid.clone()])]);
    let solid = render(&mut renderer, Isometry3F64::identity());
    let is_reddish = |c: [u8; 4]| c[0] > 60 && c[1] < 60 && c[2] < 60;
    let (_, solid_pixels) = centroid(&solid, is_reddish).expect("the ellipsoid is drawn");

    renderer.clear_renderables();
    renderer.update_scene(vec![make_ellipsoid_rings3("rings", vec![ellipsoid], 2.0)]);
    let rings = render(&mut renderer, Isometry3F64::identity());
    let (_, ring_pixels) = centroid(&rings, is_reddish).expect("the rings are drawn");

    // the middle is open
    let middle = rings.pixel(W / 2, H / 2);
    assert!(
        !is_reddish([middle[0], middle[1], middle[2], middle[3]]),
        "the middle of the rings is filled in: {:?}",
        [middle[0], middle[1], middle[2]]
    );
    // and they take a small part of what the solid one did - three outlines against a disk
    assert!(
        ring_pixels > 100 && ring_pixels < solid_pixels / 3,
        "the rings cover {ring_pixels} pixels against the solid ellipsoid's {solid_pixels}"
    );
}

/// A texture on a surface raking away from the camera has many texels to a pixel. Point sampled,
/// one of them is picked arbitrarily and the surface seethes as the camera moves; filtered through
/// a chain of smaller copies, the far half settles onto the average of the texture.
#[test]
fn a_texture_is_filtered_where_it_recedes() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // a fine checkerboard, on a quad running away from the camera
    const TEXELS: usize = 512;
    let mut texture = MutImage4U8::from_image_size(ImageSize::new(TEXELS, TEXELS));
    for v in 0..TEXELS {
        for u in 0..TEXELS {
            let black = (u + v) % 2 == 0;
            *texture.mut_pixel(u, v) = match black {
                true => SVec::<u8, 4>::new(0, 0, 0, 255),
                false => SVec::<u8, 4>::new(255, 255, 255, 255),
            };
        }
    }
    let quad = [
        [
            ([-1.0f32, 0.3, 1.0], [0.0f32, 0.0]),
            ([1.0, 0.3, 1.0], [1.0, 0.0]),
            ([1.0, 0.3, 40.0], [1.0, 1.0]),
        ],
        [
            ([-1.0f32, 0.3, 1.0], [0.0f32, 0.0]),
            ([1.0, 0.3, 40.0], [1.0, 1.0]),
            ([-1.0, 0.3, 40.0], [0.0, 1.0]),
        ],
    ];
    renderer.update_scene(vec![make_textured_mesh3(
        "quad",
        &quad,
        texture.to_shared(),
    )]);
    let image = render(&mut renderer, Isometry3F64::identity());

    // Far up the quad, where a pixel covers many texels, a filtered texture is an even grey. Point
    // sampled it is a scatter of black and white, so the pixels either side of mid grey tell the
    // two apart.
    // The quad runs from row 187, where it is a metre away, to row 129 at forty metres. A pixel
    // covers many texels beyond about six metres, which is rows 130 to 137 - and the quad is only
    // some twenty pixels wide up there.
    //
    // What tells a filtered texture from a sampled one there is how much the pixels vary: with a
    // chain of smaller copies to read from, the far half settles towards the average of the
    // texture, and without one it keeps the full swing between black and white. Note that the
    // *chain* is what does this, not the filter mode - `Nearest` with a chain measures the same
    // as `Linear` with one, and `Linear` without one measures as though nothing were filtered at
    // all.
    let values: Vec<f64> = (130..137)
        .flat_map(|v| (118..138).map(move |u| (u, v)))
        .map(|(u, v)| image.pixel(u, v)[0] as f64)
        .collect();
    assert!(
        values.len() > 80,
        "only {} pixels of the quad were examined",
        values.len()
    );
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let spread =
        (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
    assert!(
        spread < 26.0,
        "far up the quad the texture varies by {spread:.1} of 255 - it is being sampled rather \
         than filtered, which is 31 against the 22 of a filtered one"
    );
}

/// Wireframe draws edges rather than surfaces, and what an edge is depends on how the thing was
/// drawn: a rasterized mesh has the edges of its triangles, a traced primitive has the silhouette
/// of the shape itself - not of any tessellation of it - and a 2d marker has its rim.
#[test]
fn wireframe_draws_edges_rather_than_surfaces() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    renderer.update_scene(vec![
        make_mesh3(
            "triangle",
            &[(
                [[-1.0f32, -1.0, 4.0], [1.0, -1.0, 4.0], [0.0, 1.0, 4.0]],
                Color::blue(),
            )],
        ),
        // Beside the triangle rather than behind it, so the whole of its face is there to be
        // compared against its outline. A ring of `WIREFRAME_WIDTH` around a disc of radius `r`
        // is `2 w / r` of it, so what is measured has to be most of a disc for the comparison to
        // say anything.
        make_sphere3("ball", &[([2.2f32, 0.0, 9.0], 1.2f32)], &Color::red()),
    ]);
    renderer.update_pixels(vec![make_point2(
        "marker",
        &[[200.0f32, 200.0]],
        &Color::green(),
        30.0,
    )]);

    let solid = renderer
        .render_params(&ImageSize::new(W, H), &Isometry3F64::identity())
        .download_rgba(true)
        .render()
        .rgba_image
        .expect("`download_rgba` was requested");
    let wire = renderer
        .render_params(&ImageSize::new(W, H), &Isometry3F64::identity())
        .wireframe(true)
        .download_rgba(true)
        .render()
        .rgba_image
        .expect("`download_rgba` was requested");

    // Shading dims a surface and an edge is drawn as it is given, so both have to be picked out
    // by hue rather than by brightness - otherwise the lit face of a thing counts for less than
    // its outline, and the comparison below measures the light instead of the wireframe.
    let is_reddish = |c: [u8; 4]| c[0] > 60 && c[1] < 60 && c[2] < 60;
    let is_blueish = |c: [u8; 4]| c[2] > 120 && c[0] < 90 && c[1] < 90;
    for (name, pick) in [
        ("the traced sphere", &is_reddish as &dyn Fn([u8; 4]) -> bool),
        ("the rasterized triangle", &is_blueish),
        ("the 2d marker", &is_green),
    ] {
        let (_, solid_pixels) = centroid(&solid, pick).unwrap_or_else(|| {
            panic!("{name} is not drawn at all");
        });
        let (_, wire_pixels) = centroid(&wire, pick).unwrap_or_else(|| {
            panic!("{name} vanishes in wireframe");
        });
        assert!(
            wire_pixels * 3 < solid_pixels,
            "{name} covers {wire_pixels} pixels as a wireframe against {solid_pixels} solid - it \
             is still being drawn as a surface"
        );
    }

    // and the middles are open: the surface is gone, only its edge is left
    for (name, u, v) in [
        ("the traced sphere", W / 2, H / 2),
        ("the 2d marker", 200, 200),
    ] {
        let p = wire.pixel(u, v);
        assert!(
            p[0] > 200 && p[1] > 200 && p[2] > 200,
            "the middle of {name} is filled in as a wireframe: {:?}",
            [p[0], p[1], p[2]]
        );
    }
}

/// Wireframe is per entity as well as per view: one thing can be opened up while the rest of the
/// scene stays solid, and the view's own switch turns on everything regardless.
#[test]
fn an_entity_can_be_a_wireframe_on_its_own() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // a traced sphere and a rasterized triangle, side by side
    let scene = |wire_ball: bool, wire_mesh: bool| {
        vec![
            make_sphere3("ball", &[([-0.8f32, 0.0, 5.0], 0.6f32)], &Color::red())
                .wireframe(wire_ball),
            make_mesh3(
                "triangle",
                &[(
                    [[0.4f32, -0.6, 5.0], [1.6, -0.6, 5.0], [1.0, 0.6, 5.0]],
                    Color::blue(),
                )],
            )
            .wireframe(wire_mesh),
        ]
    };
    let is_reddish = |c: [u8; 4]| c[0] > 60 && c[1] < 60 && c[2] < 60;
    let area = |renderer: &mut OffscreenRenderer, scene: Vec<SceneRenderable>, view_wire: bool| {
        renderer.clear_renderables();
        renderer.update_scene(scene);
        let image = renderer
            .render_params(&ImageSize::new(W, H), &Isometry3F64::identity())
            .wireframe(view_wire)
            .download_rgba(true)
            .render()
            .rgba_image
            .expect("`download_rgba` was requested");
        (
            centroid(&image, is_reddish).map(|(_, n)| n).unwrap_or(0),
            centroid(&image, is_blue).map(|(_, n)| n).unwrap_or(0),
        )
    };

    let (solid_ball, solid_mesh) = area(&mut renderer, scene(false, false), false);
    let (wire_ball, still_solid_mesh) = area(&mut renderer, scene(true, false), false);
    let (still_solid_ball, wire_mesh) = area(&mut renderer, scene(false, true), false);
    let (all_ball, all_mesh) = area(&mut renderer, scene(false, false), true);

    assert!(
        wire_ball * 3 < solid_ball,
        "the sphere alone was asked for as a wireframe, but it covers {wire_ball} pixels against \
         {solid_ball} solid"
    );
    assert_eq!(
        still_solid_mesh, solid_mesh,
        "the sphere was asked for as a wireframe, and the triangle changed with it"
    );
    assert!(
        wire_mesh * 3 < solid_mesh,
        "the triangle alone was asked for as a wireframe, but it covers {wire_mesh} pixels \
         against {solid_mesh} solid"
    );
    assert_eq!(
        still_solid_ball, solid_ball,
        "the triangle was asked for as a wireframe, and the sphere changed with it"
    );

    // and the view's own switch takes both, without either entity asking
    assert!(
        all_ball * 3 < solid_ball && all_mesh * 3 < solid_mesh,
        "the view was set to wireframe, but the sphere covers {all_ball} of {solid_ball} and the \
         triangle {all_mesh} of {solid_mesh}"
    );
}

/// A plane drawn as a wireframe is the same plane. Ruling it says where its surface is and how
/// big the squares on it are; it does not move the surface, and in particular the horizon - the
/// direction in which the plane runs away to infinity - is a property of the plane and not of how
/// it is shaded.
#[test]
fn a_ruled_plane_reaches_the_same_horizon_as_a_solid_one() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    renderer.update_scene(vec![make_planar3(
        "ground",
        vec![
            Planar3::plane(
                SVec::<f32, 3>::new(0.0, 5.0, 0.0),
                SVec::<f32, 3>::new(0.0, -1.0, 0.0),
                Color::red(),
            )
            .with_checker(1.0, Color::blue()),
        ],
    )]);

    // the topmost row at which the ground is drawn at all, which is the horizon it reaches
    let mut top_of_the_ground = |wireframe: bool| {
        let image = renderer
            .render_params(&ImageSize::new(W, H), &Isometry3F64::identity())
            .wireframe(wireframe)
            .download_rgba(true)
            .render()
            .rgba_image
            .expect("`download_rgba` was requested");
        for v in 0..H {
            // the background is white; anything the ground puts there is not
            if image.pixel(W / 2, v)[1] < 245 {
                return v;
            }
        }
        H
    };

    let solid = top_of_the_ground(false);
    let ruled = top_of_the_ground(true);
    assert!(
        ruled <= solid + 2,
        "the ground reaches row {solid} when solid but only row {ruled} when ruled, so the \
         wireframe has a horizon of its own"
    );
}

/// A ruling can only be read over about two decades of zoom: closer, and one line fills the
/// screen; further, and the lines merge into a sheet. So the grid steps the spacing it was given
/// asked for, keyed on how much of the plane each pixel covers - which makes it readable at any
/// distance, and puts a heavier line every fifth one to count by.
#[test]
fn a_grid_is_ruled_at_a_scale_the_pixel_can_read() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // The ground, ruled every metre, seen from a hundredth of a metre up and from fifty. The
    // first is far too fine to draw and the second far too coarse, so neither is what is drawn.
    let mut ruling_at = |height: f32| {
        renderer.update_scene(vec![make_planar3(
            "ground",
            vec![
                Planar3::plane(
                    SVec::<f32, 3>::new(0.0, height, 0.0),
                    SVec::<f32, 3>::new(0.0, -1.0, 0.0),
                    Color::red(),
                )
                .with_checker(1.0, Color::blue()),
            ],
        )]);
        let image = renderer
            .render_params(&ImageSize::new(W, H), &Isometry3F64::identity())
            .wireframe(true)
            .download_rgba(true)
            .render()
            .rgba_image
            .expect("`download_rgba` was requested");
        // the lines crossing one row, as runs of covered pixels: their number and their widths
        let row: Vec<bool> = (0..W).map(|u| image.pixel(u, 220)[1] < 200).collect();
        let mut runs = vec![];
        let mut run = 0;
        for covered in row {
            match covered {
                true => run += 1,
                false if run > 0 => {
                    runs.push(run);
                    run = 0;
                }
                false => {}
            }
        }
        runs
    };

    for height in [0.01f32, 50.0] {
        let runs = ruling_at(height);
        assert!(
            runs.len() >= 3 && runs.len() <= 40,
            "from {height} m up the ruling puts {} lines across the row: it is drawing the \
             spacing it was given rather than one this pixel can resolve",
            runs.len()
        );
        // and the decades are told apart by weight: not every line is the same width
        let thin = runs.iter().copied().min().expect("at least three lines");
        let thick = runs.iter().copied().max().expect("at least three lines");
        assert!(
            thick > thin,
            "every line of the ruling is {thin} px wide from {height} m up - there is no heavier \
             line to count by"
        );
    }
}

/// An ellipse is a region of the image and grows with the zoom, but its outline is a width in
/// view-port pixels and must not: a hairline round a magnified region stays a hairline.
///
/// The two are easy to conflate, because the margin which decides the outline is worked out in
/// image pixels while what it is compared against - the feather, the width - is in the pixels the
/// view port shows one for one. At a zoom of one those are the same thing, so getting the
/// conversion backwards costs nothing until somebody zooms in.
#[test]
fn an_ellipse_outline_holds_its_width_under_zoom() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let centre = 128.0f32;
    renderer.update_pixels(vec![named_ellipse2(
        "ellipse",
        vec![Ellipse2 {
            center: SVec::<f32, 2>::new(centre, centre),
            // anisotropic, so that a width which follows the shape rather than the screen shows
            // up as a difference between the two axes
            shape: MatF64::<2, 2>::new(25.0, 0.0, 0.0, 10.0),
            line_width: 3.0,
            color: Color::red(),
        }],
    )]);

    // the opaque core of the band, crossed along the row and down the column through the centre
    let mut band = |zoom: f64| {
        let held = centre as f64 - centre as f64 * zoom;
        let image = render_with(
            &mut renderer,
            Isometry3F64::identity(),
            false,
            TranslationAndScaling {
                translation: VecF64::<2>::new(held, held),
                scaling: VecF64::<2>::new(zoom, zoom),
            },
        );
        let opaque = |u: usize, v: usize| {
            let p = image.pixel(u, v);
            p[0] > 150 && p[1] < 120
        };
        let across = (centre as usize..W)
            .filter(|u| opaque(*u, centre as usize))
            .count();
        let down = (0..centre as usize)
            .filter(|v| opaque(centre as usize, *v))
            .count();
        (across, down)
    };

    let (across, down) = band(1.0);
    let (zoomed_across, zoomed_down) = band(3.0);
    assert!(
        across > 0 && down > 0,
        "the outline is not drawn at all: {across} px across, {down} down"
    );
    assert_eq!(
        (across, down),
        (zoomed_across, zoomed_down),
        "the outline is {across} by {down} px unzoomed and {zoomed_across} by {zoomed_down} at a \
         zoom of three - it is being drawn in the pixels of the image rather than of the screen"
    );
}
