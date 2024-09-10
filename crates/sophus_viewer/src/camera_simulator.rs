use sophus_image::arc_image::ArcImage4U8;
use sophus_lie::Isometry3F64;
use sophus_sensor::dyn_camera::DynCameraF64;

use crate::renderables::renderable3d::Renderable3d;
use crate::renderer::types::ClippingPlanesF64;
use crate::renderer::types::DepthImage;
use crate::renderer::types::TranslationAndScaling;
use crate::renderer::OffscreenRenderer;
use crate::RenderContext;

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
    pub fn new(
        render_state: &RenderContext,
        intrinsics: DynCameraF64,
        clipping_planes: ClippingPlanesF64,
    ) -> Self {
        CameraSimulator {
            renderer: OffscreenRenderer::new(render_state, intrinsics, clipping_planes),
        }
    }

    /// camera intrinsics
    pub fn intrinsics(&self) -> DynCameraF64 {
        self.renderer.intrinsics()
    }

    /// update scene renderables
    pub fn update_3d_renderables(&mut self, renderables: Vec<Renderable3d>) {
        self.renderer.update_3d_renderables(renderables);
    }

    /// render
    pub fn render(&mut self, scene_from_camera: Isometry3F64) -> SimulatedImage {
        let view_port_size = self.renderer.intrinsics().image_size();

        let compute_depth_texture = false;
        let backface_culling = false;
        let download_rgba = true;

        let result = self.renderer.render(
            &view_port_size,
            TranslationAndScaling::identity(),
            scene_from_camera,
            compute_depth_texture,
            backface_culling,
            download_rgba,
        );

        SimulatedImage {
            rgba_image: result.rgba_image.unwrap(),
            depth_image: result.depth_image,
        }
    }
}
