use eframe::egui::mutex::Mutex;
use sophus_autodiff::linalg::SVec;
use sophus_image::{
    color_map::BlueWhiteRedBlackColorMap,
    prelude::*,
    ArcImage4U8,
    ArcImageF32,
    MutImage4U8,
    MutImageF32,
};

use crate::camera::ClippingPlanesF32;

/// depth image
pub struct DepthImage {
    /// ndc depth values
    pub ndc_z_image: ArcImageF32,
    /// clipping planes
    pub clipping_planes: ClippingPlanesF32,
    /// color mapped depth image cache
    color_mapped_cache: Mutex<Option<ArcImage4U8>>,
    /// metric depth image cache
    metric_depth_cache: Mutex<Option<ArcImageF32>>,
}

/// ndc z to color
pub fn ndc_z_to_color(ndc_z: f32) -> SVec<u8, 4> {
    if ndc_z == 1.0 {
        // map background to pitch black
        return SVec::<u8, 4>::new(0, 0, 0, 255);
    }
    // scale to [0.0 - 0.9] range, so that far away [dark red] differs
    // from background [pitch black].
    let z = 0.9 * ndc_z;
    // z is squared to get higher dynamic range for far objects.
    let rgb = BlueWhiteRedBlackColorMap::f32_to_rgb(z * z);
    SVec::<u8, 4>::new(rgb[0], rgb[1], rgb[2], 255)
}

impl DepthImage {
    /// new depth image
    pub fn new(ndc_z_image: ArcImageF32, clipping_planes: ClippingPlanesF32) -> Self {
        DepthImage {
            ndc_z_image,
            clipping_planes,
            color_mapped_cache: Mutex::new(None),
            metric_depth_cache: Mutex::new(None),
        }
    }

    /// return color mapped depth
    pub fn color_mapped(&self) -> ArcImage4U8 {
        let mut cached_image = self.color_mapped_cache.lock();

        match cached_image.as_mut() {
            Some(cached_image) => cached_image.clone(),
            None => {
                let mut image_rgba = MutImage4U8::from_image_size(self.ndc_z_image.image_size());

                for v in 0..image_rgba.image_size().height {
                    for u in 0..image_rgba.image_size().width {
                        let z = self.ndc_z_image.pixel(u, v);
                        *image_rgba.mut_pixel(u, v) = ndc_z_to_color(z);
                    }
                }
                let shared = ArcImage4U8::from(image_rgba);
                *cached_image = Some(shared.clone());
                shared
            }
        }
    }

    /// return metric depth image
    pub fn metric_depth(&self) -> ArcImageF32 {
        let mut cached_image = self.metric_depth_cache.lock();

        match cached_image.as_mut() {
            Some(cached_image) => cached_image.clone(),
            None => {
                let mut depth = MutImageF32::from_image_size(self.ndc_z_image.image_size());
                for v in 0..depth.image_size().height {
                    for u in 0..depth.image_size().width {
                        let z = self.ndc_z_image.pixel(u, v);

                        *depth.mut_pixel(u, v) = self.clipping_planes.metric_z_from_ndc_z(z);
                    }
                }
                let shared = ArcImageF32::from(depth);
                *cached_image = Some(shared.clone());
                shared
            }
        }
    }
}
