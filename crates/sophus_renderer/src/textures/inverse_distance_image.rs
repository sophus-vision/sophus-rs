use eframe::egui;
use sophus_autodiff::linalg::SVec;
use sophus_image::{
    ArcImage4U8,
    ArcImageF32,
    MutImage4U8,
    MutImageF32,
    color_map::BlueWhiteRedBlackColorMap,
    prelude::*,
};

use crate::camera::{
    ClippingPlanesF32,
    RenderIntrinsics,
};

/// Inverse distance image.
///
/// One inverse distance per pixel - one over the metric distance to the surface seen there,
/// measured along the *ray* through that pixel rather than along the optical axis. Distance, not
/// depth: depth is the component along the optical axis, which is what a rasterizer's own buffer
/// and an rgb-d camera hold, and the two part company as soon as the surface is off axis.
///
/// It is the range of the inverse depth parameterisation - the name that carries in the
/// literature - which pairs it with a unit bearing, and
/// [RenderIntrinsics::cam_unproj_to_unit_vector] gives that bearing for the pixel.
///
/// A camera which sees more than a right angle has rays for which a distance along the axis is a
/// poor measure and, at exactly a right angle, no measure at all - it is zero whatever the
/// surface, so it says nothing about where the surface is. The distance along the ray is well
/// behaved wherever the camera can see.
///
/// Zero is a pixel holding nothing, which is the same thing as a surface infinitely far away - so
/// unlike a depth, the value needs no separate marker for the background. Note that a non-zero
/// inverse distance is not necessarily inside the clipping planes: what is visible was decided
/// when the scene was rasterized, and a surface within the far plane along the optical axis can
/// be further than that away from the camera.
#[derive(Clone)]
pub struct InverseDistanceImage {
    /// one over the metric distance along the ray through each pixel, in 1/m
    pub image: ArcImageF32,
    /// clipping planes
    pub clipping_planes: ClippingPlanesF32,
    /// color mapped image cache
    color_mapped_cache: egui::mutex::Mutex<Option<ArcImage4U8>>,
}

/// Whether this pixel holds no surface at all.
pub fn is_background(inverse_distance: f32) -> bool {
    inverse_distance <= 0.0 || inverse_distance.is_nan()
}

/// Metric distance along the ray, from an inverse distance - [f32::INFINITY] for the background.
pub fn metric_distance(inverse_distance: f32) -> f32 {
    match is_background(inverse_distance) {
        true => f32::INFINITY,
        false => 1.0 / inverse_distance,
    }
}

/// Inverse distance to color.
pub fn inverse_distance_to_color(
    inverse_distance: f32,
    clipping_planes: ClippingPlanesF32,
) -> SVec<u8, 4> {
    match is_background(inverse_distance) {
        true => normalized_to_color(f32::NAN),
        // the same curve the rasterizer's depth follows, which in inverse distance is a straight
        // line: 0 at the near plane, 1 at the far one
        false => normalized_to_color(
            clipping_planes.far / (clipping_planes.far - clipping_planes.near)
                * (1.0 - clipping_planes.near * inverse_distance),
        ),
    }
}

/// Color of a depth normalized to the clipping planes - 0 at the near plane, 1 at the far one,
/// and anything not finite for a pixel holding no surface.
pub fn normalized_to_color(normalized: f32) -> SVec<u8, 4> {
    if !normalized.is_finite() {
        // map background to pitch black
        return SVec::<u8, 4>::new(0, 0, 0, 255);
    }
    // scale to [0.0 - 0.9] range, so that far away [dark red] differs
    // from background [pitch black].
    let z = 0.9 * normalized.clamp(0.0, 1.0);
    // z is squared to get higher dynamic range for far objects.
    let rgb = BlueWhiteRedBlackColorMap::f32_to_rgb(z * z);
    SVec::<u8, 4>::new(rgb[0], rgb[1], rgb[2], 255)
}

impl InverseDistanceImage {
    /// new inverse distance image
    pub fn new(image: ArcImageF32, clipping_planes: ClippingPlanesF32) -> Self {
        InverseDistanceImage {
            image,
            clipping_planes,
            color_mapped_cache: egui::mutex::Mutex::new(None),
        }
    }

    /// The inverse distance at a pixel, zero where there is no surface.
    pub fn inverse_distance(&self, u: usize, v: usize) -> f32 {
        self.image.pixel(u, v)
    }

    /// The metric distance along the ray through a pixel, [f32::INFINITY] where there is nothing.
    pub fn distance(&self, u: usize, v: usize) -> f32 {
        metric_distance(self.inverse_distance(u, v))
    }

    /// return color mapped inverse distance
    pub fn color_mapped(&self) -> ArcImage4U8 {
        let mut cached_image = self.color_mapped_cache.lock();

        match cached_image.as_mut() {
            Some(cached_image) => cached_image.clone(),
            None => {
                let mut image_rgba = MutImage4U8::from_image_size(self.image.image_size());

                for v in 0..image_rgba.image_size().height {
                    for u in 0..image_rgba.image_size().width {
                        *image_rgba.mut_pixel(u, v) =
                            inverse_distance_to_color(self.image.pixel(u, v), self.clipping_planes);
                    }
                }
                let shared = ArcImage4U8::from(image_rgba);
                *cached_image = Some(shared.clone());
                shared
            }
        }
    }

    /// The distance along the *optical axis* - depth in the rgb-d sense - for each pixel, in m.
    ///
    /// This is the conventional form for a depth camera, and the one to hand to anything which
    /// expects `z`. It is derived rather than stored, since it is the parameterisation which
    /// degenerates: at a right angle off the axis it is zero for every surface, and beyond that
    /// there is no z to speak of.
    pub fn metric_z(&self, intrinsics: &RenderIntrinsics) -> ArcImageF32 {
        let mut z = MutImageF32::from_image_size(self.image.image_size());
        for v in 0..z.image_size().height {
            for u in 0..z.image_size().width {
                let ray = intrinsics.cam_unproj_to_unit_vector(
                    &sophus_autodiff::linalg::VecF64::<2>::new(u as f64, v as f64),
                );
                *z.mut_pixel(u, v) = self.distance(u, v) * ray[2] as f32;
            }
        }
        ArcImageF32::from(z)
    }
}
