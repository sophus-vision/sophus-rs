use crate::renderables::renderable3d::Renderable3d;
use crate::renderer::camera::properties::RenderCameraProperties;
use crate::renderer::textures::depth_image::DepthImage;
use crate::renderer::OffscreenRenderer;
use crate::RenderContext;
use sophus_image::arc_image::ArcImage4U8;
use sophus_lie::Isometry3F64;

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
    pub fn update_3d_renderables(&mut self, renderables: Vec<Renderable3d>) {
        self.renderer.update_3d_renderables(renderables);
    }

    /// render
    pub fn render(&mut self, scene_from_camera: Isometry3F64) -> SimulatedImage {
        let view_port_size = self.renderer.intrinsics().image_size();

        let result = self
            .renderer
            .render_params(&view_port_size, &scene_from_camera)
            .download_rgba(true)
            .render();

        SimulatedImage {
            rgba_image: result.rgba_image.unwrap(),
            depth_image: result.depth_image,
        }
    }
}
