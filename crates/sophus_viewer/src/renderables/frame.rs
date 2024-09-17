use sophus_image::arc_image::ArcImage4U8;
use sophus_image::image_view::IsImageView;
use sophus_image::ImageSize;
use sophus_sensor::DynCamera;

use crate::renderer::camera::clipping_planes::ClippingPlanes;
use crate::renderer::camera::intrinsics::RenderIntrinsics;
use crate::renderer::camera::properties::RenderCameraProperties;

/// Frame to hold content
///
/// Invariants:
///   - The image, if available, must have the same size as the intrinsics.
///   - The image size must be non-zero.
#[derive(Clone, Debug)]
pub struct Frame {
    // Image, if available must have the same size as the intrinsics
    image: Option<ArcImage4U8>,
    /// Intrinsics
    camera_properties: RenderCameraProperties,
}

impl Frame {
    /// Try to create a new frame from an image and camera properties d
    ///
    /// Returns None if the image size is zero or the image size does not match the intrinsics.
    pub fn try_from(
        image: &ArcImage4U8,
        camera_properties: &RenderCameraProperties,
    ) -> Option<Frame> {
        if camera_properties.intrinsics.image_size().area() == 0 {
            return None;
        }
        if image.image_size() != camera_properties.intrinsics.image_size() {
            return None;
        }
        Some(Frame {
            image: Some(image.clone()),
            camera_properties: camera_properties.clone(),
        })
    }

    /// Create a new frame from an image
    ///
    /// Precondition: The image size must be non-zero.
    pub fn from_image(image: &ArcImage4U8) -> Frame {
        assert!(image.image_size().area() > 0);

        let camera_properties = RenderCameraProperties::default_from(image.image_size());

        Frame {
            image: Some(image.clone()),
            camera_properties,
        }
    }

    /// Create a new frame from intrinsics
    ///
    /// Precondition: The image size must be non-zero.
    pub fn from_intrinsics(intrinsics: &DynCamera<f64, 1>) -> Frame {
        assert!(intrinsics.image_size().area() > 0);
        Frame {
            image: None,
            camera_properties: RenderCameraProperties {
                intrinsics: RenderIntrinsics::new(intrinsics),
                clipping_planes: ClippingPlanes::default(),
            },
        }
    }

    /// Create a new frame from camera properties
    pub fn from_camera_properties(camera_properties: &RenderCameraProperties) -> Frame {
        Frame {
            image: None,
            camera_properties: camera_properties.clone(),
        }
    }

    /// Create a new frame from image size
    ///
    /// Precondition: The image size must be non-zero.
    pub fn from_size(view_size: &ImageSize) -> Frame {
        assert!(view_size.area() > 0);
        Frame {
            image: None,
            camera_properties: RenderCameraProperties::default_from(*view_size),
        }
    }

    /// Camera properties
    pub fn camera_properties(&self) -> &RenderCameraProperties {
        &self.camera_properties
    }

    /// Image, if available
    pub fn maybe_image(&self) -> Option<&ArcImage4U8> {
        self.image.as_ref()
    }
}
