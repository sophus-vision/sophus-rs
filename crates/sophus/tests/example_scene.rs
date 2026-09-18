//! The example scene has to survive back-face culling, which is on by default in the viewer.
//!
//! A triangle wound the wrong way round is not drawn at all, and says nothing about why - so the
//! billboard, the one part of that scene made of triangles, is checked here against the camera
//! the demo opens with.

use sophus::examples::viewer_example::make_park_scene;
use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};
use sophus_image::{
    ImageSize,
    prelude::*,
};
use sophus_lie::{
    Isometry3F64,
    Rotation3,
};
use sophus_renderer::{
    OffscreenRenderer,
    RenderContext,
    camera::{
        ClippingPlanes,
        RenderCameraProperties,
    },
};
use sophus_sensor::DynCameraF64;

#[test]
fn the_billboard_is_drawn_with_back_face_culling_on() {
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let image_size = ImageSize::new(639, 479);
    let renderer = &mut OffscreenRenderer::new(
        &context,
        &RenderCameraProperties::new(
            DynCameraF64::new_pinhole(VecF64::from_array([500.0, 500.0, 320.0, 240.0]), image_size),
            ClippingPlanes::default(),
        ),
    );
    renderer.update_scene(make_park_scene());

    // where the demo's distorted view stands, looking across the park
    let (eye, target) = (
        VecF64::<3>::new(-0.6, -7.6, 2.3),
        VecF64::<3>::new(0.0, 1.2, 1.1),
    );
    let forward = (target - eye).normalize();
    let up = VecF64::<3>::new(0.0, 0.0, 1.0);
    let down = (-up + up.dot(&forward) * forward).normalize();
    let scene_from_camera = Isometry3F64::from_rotation_and_translation(
        Rotation3::try_from_mat(MatF64::<3, 3>::from_columns(&[
            down.cross(&forward),
            down,
            forward,
        ]))
        .expect("the three axes are orthonormal"),
        eye,
    );

    let image = renderer
        .render_params(&image_size, &scene_from_camera)
        .backface_culling(true)
        .download_rgba(true)
        .render()
        .rgba_image
        .expect("`download_rgba` was requested");

    // The poster's header band, which nothing else up there is that colour - the trajectory is
    // the other orange in the scene, and it runs along the ground well below this.
    let band = (0..image_size.height / 2)
        .flat_map(|v| (0..image_size.width).map(move |u| (u, v)))
        .filter(|(u, v)| {
            let p = image.pixel(*u, *v);
            p[0] > 140 && (60..=150).contains(&p[1]) && p[2] < 110
        })
        .count();
    assert!(
        band > 50,
        "the billboard's poster covers {band} pixels of the upper half of the view - a triangle \
         is drawn from the side its normal points towards, so wound the other way round it is \
         culled and there is nothing there at all"
    );
}
