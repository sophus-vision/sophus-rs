//! Offscreen rendering benchmark.
//!
//! Times the pieces a viewer pays for every frame, so that changes to the render path can be
//! judged by a number rather than by inspection. Run with `just render-bench`.

use std::time::Instant;

use sophus_autodiff::linalg::{
    IsVector,
    SVec,
    VecF64,
};
use sophus_image::{
    ImageSize,
    MutImage4U8,
    prelude::IsMutImageView,
};
use sophus_lie::{
    Isometry3,
    Isometry3F64,
};
use sophus_renderer::{
    OffscreenRenderer,
    RenderContext,
    camera::RenderCameraProperties,
    renderables::{
        Color,
        SceneRenderable,
        make_line3,
        make_mesh3_at,
        make_point3,
    },
    textures::download_depth,
};
use sophus_sensor::DynCameraF64;

const VIEW_PORT: ImageSize = ImageSize {
    width: 640,
    height: 480,
};

fn scene() -> Vec<SceneRenderable> {
    let mut points = vec![];
    let mut segments = vec![];
    let mut triangles = vec![];
    for i in 0..2000 {
        let t = i as f32 * 0.01;
        points.push([t.sin(), t.cos(), 2.0 + 0.001 * i as f32]);
        segments.push([
            [t.sin(), t.cos(), 2.0 + 0.001 * i as f32],
            [t.cos(), t.sin(), 2.1 + 0.001 * i as f32],
        ]);
        triangles.push((
            [
                [t.sin(), t.cos(), 3.0],
                [t.cos(), t.sin(), 3.0],
                [t.sin(), t.sin(), 3.1],
            ],
            Color::blue(),
        ));
    }
    vec![
        make_point3("points", &points, &Color::red(), 3.0),
        make_line3("lines", &segments, &Color::green(), 2.0),
        make_mesh3_at("mesh", &triangles, Isometry3::trans_z(0.5)),
    ]
}

fn image(size: ImageSize, seed: u8) -> sophus_image::ArcImage4U8 {
    let mut img = MutImage4U8::from_image_size_and_val(size, SVec::<u8, 4>::new(0, 0, 0, 255));
    for v in 0..size.height {
        for u in 0..size.width {
            *img.mut_pixel(u, v) = SVec::<u8, 4>::new(u as u8, v as u8, seed, 255);
        }
    }
    img.to_shared()
}

fn bench(name: &str, iterations: usize, mut run: impl FnMut()) {
    run(); // warm up
    let start = Instant::now();
    for _ in 0..iterations {
        run();
    }
    let elapsed = start.elapsed().as_secs_f64() * 1e3;
    println!(
        "  {name:<44} {:>8.2} ms/iter   ({iterations} iterations)",
        elapsed / iterations as f64
    );
}

fn main() {
    let Some(context) = pollster::block_on(RenderContext::try_new()) else {
        eprintln!("no GPU available - nothing to measure");
        return;
    };

    let camera =
        DynCameraF64::new_pinhole(VecF64::from_array([500.0, 500.0, 320.0, 240.0]), VIEW_PORT);
    let properties = RenderCameraProperties::from_intrinsics(&camera);
    let clipping_planes = properties.clipping_planes.cast();

    println!("scene view ({} x {}):", VIEW_PORT.width, VIEW_PORT.height);
    {
        let mut renderer = OffscreenRenderer::new(&context, &properties);
        renderer.update_scene(scene());

        let renderables = scene();
        bench("update_scene (6000 renderables uploaded)", 100, || {
            renderer.update_scene(renderables.clone());
        });

        bench("render", 200, || {
            renderer
                .render_params(&VIEW_PORT, &Isometry3F64::identity())
                .render();
        });

        bench(
            "render + depth readback (only while interacting)",
            200,
            || {
                let result = renderer
                    .render_params(&VIEW_PORT, &Isometry3F64::identity())
                    .render();
                pollster::block_on(download_depth(
                    false,
                    clipping_planes,
                    context.clone(),
                    &VIEW_PORT,
                    &result,
                ));
            },
        );
    }

    println!("image view ({} x {}):", VIEW_PORT.width, VIEW_PORT.height);
    {
        let frame = image(VIEW_PORT, 7);
        let mut renderer = OffscreenRenderer::new(&context, &properties);
        renderer.reset_2d_frame(&properties.intrinsics, Some(&frame));

        bench("render", 200, || {
            renderer
                .render_params(&VIEW_PORT, &Isometry3F64::identity())
                .render();
        });

        let mut seed = 0u8;
        bench("reset_2d_frame (new background each frame)", 200, || {
            seed = seed.wrapping_add(1);
            renderer.reset_2d_frame(&properties.intrinsics, Some(&image(VIEW_PORT, seed)));
        });

        bench("OffscreenRenderer::new (full rebuild)", 50, || {
            let _ = OffscreenRenderer::new(&context, &properties);
        });

        let mut seed = 0u8;
        bench(
            "clear_renderables + reset_2d_frame (reuse path)",
            200,
            || {
                seed = seed.wrapping_add(1);
                renderer.clear_renderables();
                renderer.reset_2d_frame(&properties.intrinsics, Some(&image(VIEW_PORT, seed)));
            },
        );
    }
}
