use crate::camera::clipping_planes::ClippingPlanes;
use crate::camera::intrinsics::RenderIntrinsics;
use crate::camera::properties::RenderCameraProperties;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::image_view::IsImageView;
use sophus_image::ImageSize;
use sophus_sensor::dyn_camera::DynCameraF64;

/// Frame to hold content
///
/// Invariants:
///   - The image, if available, must have the same size as the intrinsics.
///   - The image size must be non-zero.
#[derive(Clone, Debug)]
pub struct ImageFrame {
    /// Image, if available must have the same size as the intrinsics
    pub image: Option<ArcImage4U8>,
    /// Intrinsics
    pub camera_properties: RenderCameraProperties,
}

impl ImageFrame {
    /// Try to create a new frame from an image and camera properties d
    ///
    /// Returns None if the image size is zero or the image size does not match the intrinsics.
    pub fn try_from(
        image: &ArcImage4U8,
        camera_properties: &RenderCameraProperties,
    ) -> Option<ImageFrame> {
        if camera_properties.intrinsics.image_size().area() == 0 {
            return None;
        }
        if image.image_size() != camera_properties.intrinsics.image_size() {
            return None;
        }
        Some(ImageFrame {
            image: Some(image.clone()),
            camera_properties: camera_properties.clone(),
        })
    }

    /// Create a new frame from an image
    ///
    /// Precondition: The image size must be non-zero.
    pub fn from_image(image: &ArcImage4U8) -> ImageFrame {
        assert!(image.image_size().area() > 0);

        let camera_properties = RenderCameraProperties::default_from(image.image_size());

        ImageFrame {
            image: Some(image.clone()),
            camera_properties,
        }
    }

    /// Create a new frame from intrinsics
    ///
    /// Precondition: The image size must be non-zero.
    pub fn from_intrinsics(intrinsics: &DynCameraF64) -> ImageFrame {
        assert!(intrinsics.image_size().area() > 0);
        ImageFrame {
            image: None,
            camera_properties: RenderCameraProperties {
                intrinsics: RenderIntrinsics::new(intrinsics),
                clipping_planes: ClippingPlanes::default(),
            },
        }
    }

    /// Create a new frame from camera properties
    pub fn from_camera_properties(camera_properties: &RenderCameraProperties) -> ImageFrame {
        ImageFrame {
            image: None,
            camera_properties: camera_properties.clone(),
        }
    }

    /// Create a new frame from image size
    ///
    /// Precondition: The image size must be non-zero.
    pub fn from_size(view_size: &ImageSize) -> ImageFrame {
        assert!(view_size.area() > 0);
        ImageFrame {
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
