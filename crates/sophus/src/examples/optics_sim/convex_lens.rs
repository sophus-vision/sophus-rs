use sophus_autodiff::linalg::{
    IsVector,
    VecF32,
    VecF64,
};
use sophus_geo::{
    Circle,
    Ray,
    Ray2,
    UnitVector,
};
use sophus_lie::{
    prelude::IsSingleScalar,
    Rotation2,
};
use sophus_renderer::renderables::{
    named_line3,
    Color,
    LineSegment3,
    SceneRenderable,
};

use crate::examples::optics_sim::{
    element::{
        gray_color,
        Element,
    },
    light_path::unproj2,
};

/// A struct representing a circular segment of a lens.
pub struct Circularsection {
    /// The starting angle of the segment.
    pub start: f64,
    /// The range of the segment.
    pub range: f64,
}

/// A struct representing a bi-convex lens with two circular surfaces.
pub struct BiConvexLens2<S: IsSingleScalar<DM, DN> + PartialOrd, const DM: usize, const DN: usize> {
    /// The front surface of the lens.
    pub front_surface: Circle<S, 1, DM, DN>,
    /// The back surface of the lens.
    pub back_surface: Circle<S, 1, DM, DN>,
    /// The front section of the lens.
    pub front_section: Circularsection,
    /// The back section of the lens.
    pub back_section: Circularsection,
    /// The material index of the lens.
    pub material_index: S,
}

/// lens
pub type BiConvexLens2F64 = BiConvexLens2<f64, 0, 0>;

impl<S: IsSingleScalar<DM, DN> + PartialOrd, const DM: usize, const DN: usize>
    BiConvexLens2<S, DM, DN>
{
    /// Creates a new BiConvexLens2 instance.
    pub fn new(
        front_surface: Circle<S, 1, DM, DN>,
        back_surface: Circle<S, 1, DM, DN>,
        material_index: S,
    ) -> Self {
        let intersection_points = front_surface.intersect_circle(&back_surface).unwrap();

        let mut front_angles = vec![];
        let mut back_angles = vec![];

        for intersection_point in intersection_points {
            let front_intersection = intersection_point - front_surface.center;
            let back_intersection = intersection_point - back_surface.center;

            front_angles.push(
                front_intersection
                    .elem(1)
                    .atan2(front_intersection.elem(0))
                    .single_real_scalar(),
            );

            back_angles.push(
                back_intersection
                    .elem(1)
                    .atan2(back_intersection.elem(0))
                    .single_real_scalar(),
            );
        }

        BiConvexLens2 {
            front_surface,
            front_section: Circularsection {
                start: front_angles[0],
                range: Rotation2::exp(VecF64::<1>::new(front_angles[0]))
                    .inverse()
                    .group_mul(&Rotation2::<f64, 1, 0, 0>::exp(VecF64::<1>::new(
                        front_angles[1],
                    )))
                    .log()[0],
            },
            back_surface,
            back_section: Circularsection {
                start: back_angles[1],
                range: back_angles[0] - back_angles[1],
            },
            material_index,
        }
    }

    /// Refract the ray through the lens.
    pub fn refract(&self, incident_ray: Ray2<S, 1, DM, DN>) -> Option<[Ray2<S, 1, DM, DN>; 2]> {
        let front_surface_point = self.front_surface.ray_intersect(&incident_ray)?;
        let front_normal = front_surface_point - self.front_surface.center;

        let dir1 = incident_ray.dir.refract(
            UnitVector::from_vector_and_normalize(&front_normal),
            S::from_f64(1.0) / self.material_index,
        )?;

        let ray1 = Ray {
            origin: front_surface_point,
            dir: dir1,
        };

        let back_surface_point = self.back_surface.ray_intersect(&ray1)?;
        let back_normal = back_surface_point - self.back_surface.center;

        let dir2 = ray1.dir.refract(
            UnitVector::from_vector_and_normalize(&back_normal),
            self.material_index,
        )?;

        Some([
            ray1,
            Ray {
                origin: back_surface_point,
                dir: dir2,
            },
        ])
    }
}

fn create_circle_segment(
    center: VecF32<2>,
    radius: f32,
    start_angle: f32,
    angle_range: f32,
    num_segments: usize,
    color: Color,
) -> Vec<LineSegment3> {
    let mut segments: Vec<LineSegment3> = vec![];

    let angle_increment = angle_range / num_segments as f32;

    let mut previous_point = VecF32::<3>::new(
        center.x + radius * start_angle.cos(),
        center.y + radius * start_angle.sin(),
        0.0,
    );

    for i in 0..=num_segments {
        let angle = i as f32 * angle_increment + start_angle;
        let next_point = VecF32::<3>::new(
            center.x + radius * angle.cos(),
            center.y + radius * angle.sin(),
            0.0,
        );

        segments.push(LineSegment3 {
            p0: previous_point,
            p1: next_point,
            color,
            line_width: 2.0,
        });

        previous_point = next_point;
    }

    segments
}

impl Element for BiConvexLens2F64 {
    fn to_renderable3(&self) -> SceneRenderable {
        let mut segments: Vec<LineSegment3> = vec![];

        segments.extend(create_circle_segment(
            self.front_surface.center.cast(),
            self.front_surface.radius as f32,
            self.front_section.start as f32,
            self.front_section.range as f32,
            20,
            gray_color(),
        ));
        segments.extend(create_circle_segment(
            self.back_surface.center.cast(),
            self.back_surface.radius as f32,
            self.back_section.start as f32,
            self.back_section.range as f32,
            20,
            Color {
                r: 0.3,
                g: 0.3,
                b: 0.3,
                a: 1.0,
            },
        ));

        if let Some(points) = self.front_surface.intersect_circle(&self.back_surface) {
            segments.push(LineSegment3 {
                p0: unproj2(points[0].cast()),
                p1: unproj2(points[1].cast()),
                color: Color {
                    r: 0.3,
                    g: 0.3,
                    b: 0.3,
                    a: 1.0,
                },
                line_width: 1.0,
            });
        }

        named_line3("lens", segments)
    }
}
