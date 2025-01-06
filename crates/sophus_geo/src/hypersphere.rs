use sophus_autodiff::linalg::EPS_F64;
use sophus_lie::prelude::*;

use crate::ray::Ray;

/// N-Sphere
pub struct HyperSphere<
    S: IsScalar<BATCH, DM, DN>,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    /// center
    pub center: S::Vector<DIM>,
    /// radius
    pub radius: S,
}

/// Circle
pub type Circle<S, const B: usize, const DM: usize, const DN: usize> = HyperSphere<S, 2, B, DM, DN>;

/// Circle
pub type CircleF64 = Circle<f64, 1, 0, 0>;

impl<S: IsSingleScalar<DM, DN> + PartialOrd, const DM: usize, const DN: usize>
    Circle<S, 1, DM, DN>
{
    /// circle-circle intersection
    pub fn intersect_circle(&self, other: &Circle<S, 1, DM, DN>) -> Option<[S::Vector<2>; 2]> {
        let dir = other.center - self.center;
        let d = dir.norm();

        let r1 = self.radius;
        let r2 = other.radius;

        // No intersection
        if d > r1 + r2 || d < (r1 - r2).abs() {
            return None;
        }

        // Distance from the center of the first circle to the midpoint of the intersection line
        let a = (r1 * r1 - r2 * r2 + d * d) / (S::from_f64(2.0) * d);

        // Height from the midpoint to the intersection points
        let h = (r1 * r1 - a * a).sqrt();

        // Midpoint coordinates
        let m = self.center + dir.scaled(a / d);

        let ortho_dir = S::Vector::from_array([dir.get_elem(1), dir.get_elem(0)]);

        let up = ortho_dir.scaled(h / d);

        Some([m + up, m - up])
    }
}

/// Ray-hyperphere intersection
pub struct HypersphereRayIntersection {}

/// Line-hypersphere intersection
pub enum LineHypersphereIntersection<
    S: IsSingleScalar<DM, DN>,
    const DIM: usize,
    const DM: usize,
    const DN: usize,
> {
    /// point pair
    Points([S::Vector<DIM>; 2]),
    /// tangent point
    TangentPoint(S::Vector<DIM>),
}

/// 2d line circle intersection
pub type LineCircleIntersection<S, const DM: usize, const DN: usize> =
    LineHypersphereIntersection<S, 2, DM, DN>;
/// 3d line circle intersection
pub type LineSphereIntersection<S, const DM: usize, const DN: usize> =
    LineHypersphereIntersection<S, 3, DM, DN>;

impl<
        S: IsSingleScalar<DM, DN> + PartialOrd,
        const DIM: usize,
        const DM: usize,
        const DN: usize,
    > HyperSphere<S, DIM, 1, DM, DN>
{
    /// Function to calculate the intersection between a line: l: o + t*d,
    /// and a hypersphere.
    ///
    /// Returns a parameter tuple (t1, t2), or None if the line does not
    /// intersect the sphere.
    pub fn line_intersect_parameters<const DOF: usize>(
        &self,
        ray: &Ray<S, DOF, DIM, 1, DM, DN>,
    ) -> Option<[S; 2]> {
        let oc = ray.origin - self.center;
        let a = ray.dir.vector().dot(ray.dir.vector());
        let b = S::from_f64(2.0) * oc.clone().dot(ray.dir.vector());
        let c = oc.clone().dot(oc) - self.radius * self.radius;

        // Compute the discriminant
        let discriminant = b * b - S::from_f64(4.0) * a * c;

        if discriminant < S::from_f64(0.0) {
            return None; // No intersection
        }

        let sqrt_discriminant = discriminant.sqrt();

        // Compute both possible solutions (t1 and t2)
        let t1 = (-b - sqrt_discriminant) / (S::from_f64(2.0) * a);
        let t2 = (-b + sqrt_discriminant) / (S::from_f64(2.0) * a);

        Some([t1, t2])
    }

    /// Returns closest intersection point in front of ray, and None if the ray does not intersect
    /// the sphere.
    ///
    /// Intersection points for t<eps are ignored. If all intersection points are needed,
    /// use Self::line_intersect_parameters directly.
    pub fn ray_intersect_with_eps<const DOF: usize>(
        &self,
        ray: &Ray<S, DOF, DIM, 1, DM, DN>,
        eps: f64,
    ) -> Option<S::Vector<DIM>> {
        let t = self.line_intersect_parameters(ray)?;
        let t0 = &t[0];
        let t1 = &t[1];

        let t0_pos = t0.single_real_scalar() >= eps;
        let t1_pos = t1.single_real_scalar() >= eps;

        let t: Option<S> = match (t0_pos, t1_pos) {
            (true, true) => {
                if t0 < t1 {
                    Some(*t0)
                } else {
                    Some(*t1)
                }
            }
            (true, false) => Some(*t0),
            (false, true) => Some(*t1),
            (false, false) => None,
        };

        Some(ray.at(t?))
    }

    /// Self::ray_intersect_with_eps with default eps.
    pub fn ray_intersect<const DOF: usize>(
        &self,
        ray: &Ray<S, DOF, DIM, 1, DM, DN>,
    ) -> Option<S::Vector<DIM>> {
        self.ray_intersect_with_eps(ray, EPS_F64)
    }

    /// calculates line intersection
    pub fn line_intersect<const DOF: usize>(
        &self,
        ray: &Ray<S, DOF, DIM, 1, DM, DN>,
    ) -> Option<LineHypersphereIntersection<S, DIM, DM, DN>> {
        self.line_intersect_with_eps(ray, EPS_F64)
    }

    /// calculates line intersection
    pub fn line_intersect_with_eps<const DOF: usize>(
        &self,
        ray: &Ray<S, DOF, DIM, 1, DM, DN>,
        eps: f64,
    ) -> Option<LineHypersphereIntersection<S, DIM, DM, DN>> {
        let t = self.line_intersect_parameters(ray)?;
        let t0 = t[0];
        let t1 = t[1];
        if (t0 - t1).abs().single_real_scalar() < eps {
            return Some(LineHypersphereIntersection::TangentPoint(
                ray.clone().at(t0),
            ));
        }
        Some(LineHypersphereIntersection::Points([
            ray.clone().at(t0),
            ray.at(t1),
        ]))
    }
}
