use sophus_autodiff::linalg::{
    VecF32,
    VecF64,
};
use sophus_geo::{
    LineF64,
    Ray,
    Ray2,
    UnitVector,
};
use sophus_renderer::renderables::{
    named_line3,
    Color,
    LineSegment3,
    SceneRenderable,
};

use crate::examples::optics_sim::{
    aperture_stop::ApertureStop,
    convex_lens::BiConvexLens2,
    detector::Detector,
    element::{
        gray_color,
        Element,
    },
};

/// Unproject a 2D vector to 3D by setting the z-coordinate to 0.
pub fn unproj2(vec2: VecF32<2>) -> VecF32<3> {
    VecF32::<3>::new(vec2[0], vec2[1], 0.0)
}

/// LightPath struct represents a light path in an optical system.
pub struct LightPath {
    name: String,
    scene_point_ray: Ray2<f64, 1, 0, 0>,
    rays: Vec<Ray2<f64, 1, 0, 0>>,
    detector: Detector,
    aperture: ApertureStop,
    chief_ray: bool,
    color: Color,
    /// The image point on the detector plane.
    pub image_point: Option<f64>,
}

impl LightPath {
    /// Creates a new LightPath instance.
    pub fn from(
        name: impl ToString,
        point: VecF64<2>,
        angle_rad: f64,
        detector: Detector,
        aperture: ApertureStop,
        chief_ray: bool,
        color: Color,
    ) -> LightPath {
        LightPath {
            name: name.to_string(),
            scene_point_ray: Ray {
                origin: point,
                dir: UnitVector::from_vector_and_normalize(&VecF64::<2>::new(
                    angle_rad.cos(),
                    angle_rad.sin(),
                )),
            },
            detector,
            aperture,
            rays: Vec::new(),
            chief_ray,
            color,
            image_point: None,
        }
    }

    /// Propagates the light path through the lens system.
    pub fn propagate(&mut self, lens: &BiConvexLens2<f64, 0, 0>) {
        self.rays.clear();

        let result = lens.refract(self.scene_point_ray.clone());
        if let Some(rays) = result {
            self.rays.extend_from_slice(&rays);

            let aperature_point = self
                .aperture
                .aperture_plane()
                .rays_intersect(&self.rays[1])
                .unwrap();

            if aperature_point[1].abs() <= self.aperture.radius + 0.0001 {
                let p = self
                    .detector
                    .image_plane()
                    .rays_intersect(&self.rays[1])
                    .unwrap();

                self.image_point = Some(p[1]);
            }
        }
    }
}

impl Element for LightPath {
    fn to_renderable3(&self) -> SceneRenderable {
        let mut prev_ray = self.scene_point_ray.clone();
        let mut line_segments = vec![];

        if self.chief_ray {
            if let Some(point) =
                LineF64::from_point_pair(VecF64::<2>::new(1.0, 0.0), VecF64::<2>::new(-1.0, 0.0))
                    .rays_intersect(&prev_ray)
            {
                line_segments.push(LineSegment3 {
                    p0: unproj2(prev_ray.origin.cast()),
                    p1: unproj2(point.cast()),
                    color: gray_color(),
                    line_width: 0.5,
                });
            }
        }

        for ray in self.rays.iter() {
            line_segments.push(LineSegment3 {
                p0: unproj2(prev_ray.origin.cast()),
                p1: unproj2(ray.origin.cast()),
                color: self.color,
                line_width: 1.5,
            });
            prev_ray = ray.clone();
        }

        let aperature_point = self
            .aperture
            .aperture_plane()
            .rays_intersect(&prev_ray)
            .unwrap();

        if aperature_point[1].abs() <= self.aperture.radius + 0.0001 {
            line_segments.push(LineSegment3 {
                p0: unproj2(prev_ray.origin.cast()),
                p1: unproj2(
                    self.detector
                        .image_plane()
                        .rays_intersect(&prev_ray)
                        .unwrap()
                        .cast(),
                ),
                color: self.color,
                line_width: 1.5,
            });
        } else {
            line_segments.push(LineSegment3 {
                p0: unproj2(prev_ray.origin.cast()),
                p1: unproj2(aperature_point.cast()),
                color: self.color,
                line_width: 1.5,
            });
        }

        named_line3(self.name.clone(), line_segments)
    }
}
