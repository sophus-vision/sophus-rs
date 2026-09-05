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
        Color,
        Ellipse2,
        SceneRenderable,
        make_line2,
        make_mesh3,
        make_mesh3_at,
        make_point2,
        make_point3,
        make_point3_at,
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

/// Drawn as a wireframe, a surface shows its edges and nothing between them - so it covers far
/// fewer pixels than the same thing solid, and what is behind it shows through the middle.
#[test]
fn wireframe_draws_edges_rather_than_surfaces() {
    let Some((_context, mut renderer)) = renderer(pinhole()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    renderer.update_scene(vec![make_mesh3(
        "triangle",
        &[(
            [[-1.0f32, -1.0, 4.0], [1.0, -1.0, 4.0], [0.0, 1.0, 4.0]],
            Color::blue(),
        )],
    )]);
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

    // Shading dims a surface and an edge is drawn as it is given, so both are picked out by hue
    // rather than by brightness - otherwise the lit face counts for less than its own outline.
    let is_blueish = |c: [u8; 4]| c[2] > 120 && c[0] < 90 && c[1] < 90;
    for (name, pick) in [
        (
            "the rasterized triangle",
            &is_blueish as &dyn Fn([u8; 4]) -> bool,
        ),
        ("the 2d marker", &is_green),
    ] {
        let (_, solid_pixels) =
            centroid(&solid, pick).unwrap_or_else(|| panic!("{name} is not drawn at all"));
        let (_, wire_pixels) =
            centroid(&wire, pick).unwrap_or_else(|| panic!("{name} vanishes in wireframe"));
        assert!(
            wire_pixels * 3 < solid_pixels,
            "{name} covers {wire_pixels} pixels as a wireframe against {solid_pixels} solid - it \
             is still being drawn as a surface"
        );
    }

    // and the middle of the marker is open: the surface is gone, only its edge is left
    let middle = wire.pixel(200, 200);
    assert!(
        middle[0] > 200 && middle[1] > 200 && middle[2] > 200,
        "the middle of the 2d marker is filled in as a wireframe: {:?}",
        [middle[0], middle[1], middle[2]]
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
        make_point3("x", &[[1.0f32, 0.0, 0.0]], &Color::red(), 20.0),
        make_point3("y", &[[0.0f32, 1.0, 0.0]], &Color::green(), 20.0),
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
