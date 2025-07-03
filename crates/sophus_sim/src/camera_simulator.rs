use alloc::vec::Vec;

use sophus_image::ArcImage4U8;
use sophus_lie::Isometry3F64;
use sophus_renderer::{
    OffscreenRenderer,
    RenderContext,
    camera::RenderCameraProperties,
    renderables::SceneRenderable,
    textures::{
        DepthImage,
        download_depth,
    },
};

extern crate alloc;

/// camera simulator
pub struct CameraSimulator {
    renderer: OffscreenRenderer,
}

/// Simulated image
pub struct SimulatedImage {
    /// rgba
    pub rgba_image: ArcImage4U8,
    /// depth
    pub depth_image: DepthImage,
}

impl CameraSimulator {
    /// new simulator from context and camera intrinsics
    pub fn new(render_state: &RenderContext, camera_properties: &RenderCameraProperties) -> Self {
        CameraSimulator {
            renderer: OffscreenRenderer::new(render_state, camera_properties),
        }
    }

    /// update scene renderables
    pub fn update_3d_renderables(&mut self, renderables: Vec<SceneRenderable>) {
        self.renderer.update_scene(renderables);
    }

    /// render the simulated scene and returns corresponding simulated camera image.
    pub async fn render(&mut self, scene_from_camera: Isometry3F64) -> SimulatedImage {
        let view_port_size = self.renderer.intrinsics().image_size();

        let render_result = self
            .renderer
            .render_params(&view_port_size, &scene_from_camera)
            .download_rgba(true)
            .render();

        let clipping_planes = self.renderer.camera_properties.clipping_planes.cast();

        let render_result = download_depth(
            true,
            clipping_planes,
            self.renderer.render_context.clone(),
            &view_port_size,
            &render_result,
        )
        .await;
        SimulatedImage {
            rgba_image: render_result.rgba_image.unwrap(),
            depth_image: render_result.depth_image,
        }
    }
}
