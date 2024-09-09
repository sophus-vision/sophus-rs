use sophus::prelude::IsImageView;
use sophus_image::png::save_as_png;
use sophus_image::ImageSize;
use sophus_lie::Isometry3;
use sophus_sensor::DynCamera;
use sophus_viewer::renderables::color::Color;
use sophus_viewer::renderables::renderable3d::make_line3;
use sophus_viewer::renderables::renderable3d::make_mesh3_at;
use sophus_viewer::renderables::renderable3d::make_point3;
use sophus_viewer::renderer::types::TranslationAndScaling;
use sophus_viewer::renderer::Renderer;
use sophus_viewer::RenderContext;

struct OffscreenExample {
    renderer: Renderer,
}

impl OffscreenExample {
    pub fn new(render_state: &RenderContext) -> OffscreenExample {
        OffscreenExample {
            renderer: Renderer::new(
                render_state,
                &DynCamera::default_pinhole(ImageSize::new(639, 477)),
            ),
        }
    }
}

pub async fn run_offscreen() {
    let render_state = RenderContext::new().await;

    let mut offscreen = OffscreenExample::new(&render_state);

    let mut renderables3d = vec![];
    let trig_points = [[0.0, 0.0, -0.1], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    renderables3d.push(make_point3("points3", &trig_points, &Color::red(), 5.0));
    renderables3d.push(make_line3(
        "lines3",
        &[
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        &Color::green(),
        5.0,
    ));
    let blue = Color::blue();
    renderables3d.push(make_mesh3_at(
        "mesh",
        &[(trig_points, blue)],
        Isometry3::trans_z(3.0),
    ));

    offscreen.renderer.update_3d_renderables(renderables3d);

    let view_port_size = offscreen.renderer.intrinsics().image_size();

    let result = offscreen.renderer.render(
        &view_port_size,
        TranslationAndScaling::identity(),
        Isometry3::trans_z(-5.0),
        false,
        false,
        true,
    );

    let rgba_img_u8 = result.image_4u8.unwrap();

    save_as_png(
        &rgba_img_u8.image_view(),
        std::path::Path::new("11rgba.png"),
    );
}

fn main() {
    env_logger::init();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            run_offscreen().await;
        })
}
