use sophus_core::linalg::VecF64;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::image_view::IsImageView;
use sophus_image::ImageSize;
use sophus_sensor::DynCamera;

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
    intrinsics: DynCamera<f64, 1>,
}

impl Frame {
    /// Try to create a new frame from an image and intrinsics
    ///
    /// Returns None if the image size is zero or the image size does not match the intrinsics.
    pub fn try_from(image: &ArcImage4U8, intrinsics: &DynCamera<f64, 1>) -> Option<Frame> {
        if intrinsics.image_size().area() == 0 {
            return None;
        }
        if image.image_size() != intrinsics.image_size() {
            return None;
        }
        Some(Frame {
            image: Some(image.clone()),
            intrinsics: intrinsics.clone(),
        })
    }

    /// Create a new frame from an image
    ///
    /// Precondition: The image size must be non-zero.
    pub fn from_image(image: &ArcImage4U8) -> Frame {
        assert!(image.image_size().area() > 0);

        let intrinsics = DynCamera::new_pinhole(
            &VecF64::<4>::new(
                600.0,
                600.0,
                image.image_size().width as f64 * 0.5 - 0.5,
                image.image_size().height as f64 * 0.5 - 0.5,
            ),
            image.image_size(),
        );

        println!("!!!!!!!!intrinsics: {:?}", intrinsics);

        Frame {
            image: Some(image.clone()),
            intrinsics,
        }
    }

    /// Create a new frame from intrinsics
    ///
    /// Precondition: The image size must be non-zero.
    pub fn from_intrinsics(intrinsics: &DynCamera<f64, 1>) -> Frame {
        assert!(intrinsics.image_size().area() > 0);
        Frame {
            image: None,
            intrinsics: intrinsics.clone(),
        }
    }

    /// Create a new frame from image size
    ///
    /// Precondition: The image size must be non-zero.
    pub fn from_size(view_size: &ImageSize) -> Frame {
        assert!(view_size.area() > 0);
        Frame {
            image: None,
            intrinsics: DynCamera::new_pinhole(
                &VecF64::<4>::new(
                    600.0,
                    600.0,
                    view_size.width as f64 * 0.5 - 0.5,
                    view_size.height as f64 * 0.5 - 0.5,
                ),
                *view_size,
            ),
        }
    }

    /// Intrinsics
    pub fn intrinsics(&self) -> &DynCamera<f64, 1> {
        &self.intrinsics
    }

    /// Image, if available
    pub fn maybe_image(&self) -> Option<&ArcImage4U8> {
        self.image.as_ref()
    }
}
