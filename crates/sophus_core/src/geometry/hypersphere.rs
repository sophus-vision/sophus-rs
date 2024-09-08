use crate::geometry::ray::Ray;
use crate::linalg::EPS_F64;
use crate::prelude::IsScalar;
use crate::prelude::IsSingleScalar;
use crate::prelude::IsVector;

/// N-Sphere
pub struct HyperSphere<S: IsScalar<BATCH_SIZE>, const DIM: usize, const BATCH_SIZE: usize> {
    /// center
    pub center: S::Vector<DIM>,
    /// radius
    pub radius: S,
}

/// Circle
pub type Circle<S, const B: usize> = HyperSphere<S, 2, B>;

/// Circle
pub type CircleF64 = Circle<f64, 1>;

impl<S: IsSingleScalar + PartialOrd> Circle<S, 1> {
    /// circle-circle intersection
    pub fn intersect_circle(&self, other: &Circle<S, 1>) -> Option<[S::Vector<2>; 2]> {
        let dir = other.center.clone() - self.center.clone();
        let d = dir.norm();

        let r1 = self.radius.clone();
        let r2 = other.radius.clone();

        // No intersection
        if d > r1.clone() + r2.clone() || d < (r1.clone() - r2.clone()).abs() {
            return None;
        }

        // Distance from the center of the first circle to the midpoint of the intersection line
        let a = (r1.clone() * r1.clone() - r2.clone() * r2.clone() + d.clone() * d.clone())
            / (S::from_f64(2.0) * d.clone());

        // Height from the midpoint to the intersection points
        let h = (r1.clone() * r1 - a.clone() * a.clone()).sqrt();

        // Midpoint coordinates
        let m = self.center.clone() + dir.scaled(a.clone() / d.clone());

        let ortho_dir = S::Vector::from_array([dir.get_elem(1), dir.get_elem(0)]);

        let up = ortho_dir.scaled(h / d);

        Some([m.clone() + up.clone(), m.clone() - up])
    }
}

/// Ray-hyperphere intersection
pub struct HypersphereRayIntersection {}

/// Line-hypersphere intersection
pub enum LineHypersphereIntersection<S: IsSingleScalar, const DIM: usize> {
    /// point pair
    Points([S::Vector<DIM>; 2]),
    /// tangent point
    TangentPoint(S::Vector<DIM>),
}

/// 2d line circle intersection
pub type LineCircleIntersection<S> = LineHypersphereIntersection<S, 2>;
/// 3d line circle intersection
pub type LineSphereIntersection<S> = LineHypersphereIntersection<S, 3>;

impl<S: IsSingleScalar + PartialOrd, const DIM: usize> HyperSphere<S, DIM, 1> {
    /// Function to calculate the intersection between a line: l: o + t*d,
    /// and a hypersphere.
    ///
    /// Returns a parameter tuple (t1, t2), or None if the line does not
    /// intersect the sphere.
    pub fn line_intersect_parameters<const DOF: usize>(
        &self,
        ray: &Ray<S, DOF, DIM, 1>,
    ) -> Option<[S; 2]> {
        let oc = ray.origin.clone() - self.center.clone();
        let a = ray.dir.vector().dot(ray.dir.vector());
        let b = S::from_f64(2.0) * oc.clone().dot(ray.dir.vector());
        let c = oc.clone().dot(oc) - self.radius.clone() * self.radius.clone();

        // Compute the discriminant
        let discriminant = b.clone() * b.clone() - S::from_f64(4.0) * a.clone() * c;

        if discriminant < S::from_f64(0.0) {
            return None; // No intersection
        }

        let sqrt_discriminant = discriminant.sqrt();

        // Compute both possible solutions (t1 and t2)
        let t1 = (-b.clone() - sqrt_discriminant.clone()) / (S::from_f64(2.0) * a.clone());
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
        ray: &Ray<S, DOF, DIM, 1>,
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
                    Some(t0.clone())
                } else {
                    Some(t1.clone())
                }
            }
            (true, false) => Some(t0.clone()),
            (false, true) => Some(t1.clone()),
            (false, false) => None,
        };

        Some(ray.at(t?))
    }

    /// Self::ray_intersect_with_eps with default eps.
    pub fn ray_intersect<const DOF: usize>(
        &self,
        ray: &Ray<S, DOF, DIM, 1>,
    ) -> Option<S::Vector<DIM>> {
        self.ray_intersect_with_eps(ray, EPS_F64)
    }

    /// calculates line intersection
    pub fn line_intersect<const DOF: usize>(
        &self,
        ray: &Ray<S, DOF, DIM, 1>,
    ) -> Option<LineHypersphereIntersection<S, DIM>> {
        self.line_intersect_with_eps(ray, EPS_F64)
    }

    /// calculates line intersection
    pub fn line_intersect_with_eps<const DOF: usize>(
        &self,
        ray: &Ray<S, DOF, DIM, 1>,
        eps: f64,
    ) -> Option<LineHypersphereIntersection<S, DIM>> {
        let t = self.line_intersect_parameters(ray)?;
        let t0 = t[0].clone();
        let t1 = t[1].clone();
        if (t0.clone() - t1.clone()).abs().single_real_scalar() < eps {
            return Some(LineHypersphereIntersection::TangentPoint(
                ray.clone().at(t0.clone()),
            ));
        }
        Some(LineHypersphereIntersection::Points([
            ray.clone().at(t0),
            ray.at(t1),
        ]))
    }
}
