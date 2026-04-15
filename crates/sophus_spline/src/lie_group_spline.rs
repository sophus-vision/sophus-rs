//! Cubic B-spline on Lie groups.
//!
//! Uses the cumulative basis function approach (Lovegrove et al., BMVC 2013):
//!
//! ```text
//! T(u) = T₀ · exp(b₀(u) · Ω₀) · exp(b₁(u) · Ω₁) · exp(b₂(u) · Ω₂)
//! ```
//!
//! where `Ωᵢ = log(Tᵢ⁻¹ · Tᵢ₊₁)` are the relative tangent vectors between
//! consecutive control points, and `b₀, b₁, b₂` are the cumulative cubic
//! B-spline basis functions (same as the Euclidean case).
//!
//! Velocity is computed via the product rule on the manifold using adjoint
//! transport.

use sophus_autodiff::prelude::*;
use sophus_lie::{
    IsLieGroupImpl,
    LieGroup,
};

use crate::spline_segment::{
    CubicBasisFunction,
    SegmentCase,
};

extern crate alloc;

/// Type alias to reduce verbosity.
type Group<
    S,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const DM: usize,
    const DN: usize,
    G,
> = LieGroup<S, DOF, PARAMS, POINT, AMBIENT, 1, DM, DN, G>;

/// Cubic B-spline segment on a Lie group.
///
/// Given four control points `[T_prev, T₀, T₁, T₂]`, evaluates the spline
/// at parameter `u ∈ [0, 1)` using the product-of-exponentials formula.
pub struct LieGroupBSplineSegment<
    S: IsSingleScalar<DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const DM: usize,
    const DN: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, 1, DM, DN>,
> {
    /// Segment boundary case (affects basis function clamping).
    pub case: SegmentCase,
    /// The four control points for this segment: [prev, 0, 1, 2].
    pub control_points: [Group<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G>; 4],
}

impl<
    S: IsSingleScalar<DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const DM: usize,
    const DN: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, 1, DM, DN>,
> LieGroupBSplineSegment<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G>
{
    /// Compute the three tangent vectors Ωᵢ = log(Tᵢ⁻¹ · Tᵢ₊₁) for this segment.
    ///
    /// Boundary handling:
    /// - First segment: Ω₀ = 0 (T_prev is clamped to T₀)
    /// - Last segment:  Ω₂ = 0 (T₂ is clamped to T₁)
    /// - Normal:        all three are computed from consecutive pairs
    fn omegas(&self) -> [S::Vector<DOF>; 3] {
        let zero = S::Vector::<DOF>::zeros();
        match self.case {
            SegmentCase::First => {
                let omega1 = (self.control_points[1].inverse() * &self.control_points[2]).log();
                let omega2 = (self.control_points[2].inverse() * &self.control_points[3]).log();
                [zero, omega1, omega2]
            }
            SegmentCase::Normal => {
                let omega0 = (self.control_points[0].inverse() * &self.control_points[1]).log();
                let omega1 = (self.control_points[1].inverse() * &self.control_points[2]).log();
                let omega2 = (self.control_points[2].inverse() * &self.control_points[3]).log();
                [omega0, omega1, omega2]
            }
            SegmentCase::Last => {
                let omega0 = (self.control_points[0].inverse() * &self.control_points[1]).log();
                let omega1 = (self.control_points[1].inverse() * &self.control_points[2]).log();
                [omega0, omega1, zero]
            }
        }
    }

    /// Base control point for this segment.
    fn base_point(&self) -> &Group<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G> {
        match self.case {
            SegmentCase::First => &self.control_points[1],
            SegmentCase::Normal | SegmentCase::Last => &self.control_points[0],
        }
    }

    /// Interpolate at parameter `u ∈ [0, 1)`.
    ///
    /// Returns `T₀ · exp(b₀·Ω₀) · exp(b₁·Ω₁) · exp(b₂·Ω₂)`.
    pub fn interpolate(&self, u: S) -> Group<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G> {
        let omegas = self.omegas();
        let b = CubicBasisFunction::<S, DM, DN>::b(u);

        let a0 =
            Group::<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G>::exp(omegas[0].scaled(b.elem(0)));
        let a1 =
            Group::<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G>::exp(omegas[1].scaled(b.elem(1)));
        let a2 =
            Group::<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G>::exp(omegas[2].scaled(b.elem(2)));

        self.base_point() * &(a0 * &(a1 * &a2))
    }

    /// Body-frame velocity at parameter `u`.
    ///
    /// Returns the tangent vector `ω(u) ∈ 𝔤` such that `dT/dt = T(u) · hat(ω(u))`.
    ///
    /// Derived from the product rule on `A₀ · A₁ · A₂`:
    /// `ω = Adj(A₁A₂)⁻¹ · ḃ₀·Ω₀ + Adj(A₂)⁻¹ · ḃ₁·Ω₁ + ḃ₂·Ω₂`
    pub fn velocity(&self, u: S, delta_t: S) -> S::Vector<DOF> {
        let omegas = self.omegas();
        let db = CubicBasisFunction::<S, DM, DN>::du_b(u, delta_t);
        let b = CubicBasisFunction::<S, DM, DN>::b(u);

        let a1 =
            Group::<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G>::exp(omegas[1].scaled(b.elem(1)));
        let a2 =
            Group::<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G>::exp(omegas[2].scaled(b.elem(2)));

        let a1a2 = &a1 * &a2;
        let adj_a1a2_inv = a1a2.inverse().adj();
        let adj_a2_inv = a2.inverse().adj();

        adj_a1a2_inv * omegas[0].scaled(db.elem(0))
            + adj_a2_inv * omegas[1].scaled(db.elem(1))
            + omegas[2].scaled(db.elem(2))
    }
}

/// Cubic B-spline on a Lie group.
///
/// Control points are Lie group elements. Interpolation uses the cumulative
/// basis function approach to produce smooth (C²) curves on the group manifold.
pub struct LieGroupCubicBSpline<
    S: IsSingleScalar<DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const DM: usize,
    const DN: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, 1, DM, DN>,
> {
    /// Control points (Lie group elements).
    pub control_points: alloc::vec::Vec<Group<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G>>,
    /// Time interval between consecutive control points.
    pub delta_t: S,
    /// Start time.
    pub t0: S,
}

impl<
    S: IsSingleScalar<DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const DM: usize,
    const DN: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, 1, DM, DN>,
> LieGroupCubicBSpline<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G>
{
    /// Number of spline segments.
    pub fn num_segments(&self) -> usize {
        assert!(self.control_points.len() >= 2);
        self.control_points.len() - 1
    }

    /// Maximum valid time.
    pub fn t_max(&self) -> S {
        self.t0 + S::from_f64(self.num_segments() as f64) * self.delta_t
    }

    /// Convert time to segment index and local parameter `u ∈ [0, 1)`.
    pub fn index_and_u(&self, t: S) -> (usize, S) {
        assert!(t.greater_equal(&self.t0).all());
        assert!(t.less_equal(&self.t_max()).all());

        let normalized = (t - self.t0) / self.delta_t;
        let mut idx = normalized.i64_floor() as usize;
        let mut u = normalized.fract();

        if idx >= self.num_segments() {
            idx = self.num_segments() - 1;
            u = S::from_f64(1.0);
        }
        (idx, u)
    }

    /// Build a segment for the given index.
    fn segment(
        &self,
        segment_idx: usize,
    ) -> LieGroupBSplineSegment<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G> {
        let n = self.num_segments();
        assert!(segment_idx < n);

        let case = if segment_idx == 0 {
            SegmentCase::First
        } else if segment_idx == n - 1 {
            SegmentCase::Last
        } else {
            SegmentCase::Normal
        };

        let idx_prev = if segment_idx == 0 { 0 } else { segment_idx - 1 };
        let idx_0 = segment_idx;
        let idx_1 = segment_idx + 1;
        let idx_2 = (segment_idx + 2).min(self.control_points.len() - 1);

        LieGroupBSplineSegment {
            case,
            control_points: [
                self.control_points[idx_prev],
                self.control_points[idx_0],
                self.control_points[idx_1],
                self.control_points[idx_2],
            ],
        }
    }

    /// Interpolate the spline at time `t`.
    pub fn interpolate(&self, t: S) -> Group<S, DOF, PARAMS, POINT, AMBIENT, DM, DN, G> {
        let (idx, u) = self.index_and_u(t);
        self.segment(idx).interpolate(u)
    }

    /// Body-frame velocity at time `t`.
    pub fn velocity(&self, t: S) -> S::Vector<DOF> {
        let (idx, u) = self.index_and_u(t);
        self.segment(idx).velocity(u, self.delta_t)
    }

    /// Indices of control points that influence the spline at time `t`.
    pub fn idx_involved(&self, t: S) -> alloc::vec::Vec<usize> {
        let (segment_idx, _) = self.index_and_u(t);
        let idx_prev = if segment_idx == 0 { 0 } else { segment_idx - 1 };
        let idx_2 = (segment_idx + 2).min(self.control_points.len() - 1);
        alloc::vec![idx_prev, segment_idx, segment_idx + 1, idx_2]
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use sophus_autodiff::linalg::{
        IsVector,
        VecF64,
    };
    use sophus_lie::{
        Rotation2F64,
        Rotation3F64,
    };

    use super::*;

    #[test]
    fn lie_group_spline_so2() {
        let angles = [0.0, 0.5, 1.2, 0.8, -0.3, 0.1];
        let control_points: alloc::vec::Vec<Rotation2F64> = angles
            .iter()
            .map(|&a| Rotation2F64::exp(VecF64::<1>::new(a)))
            .collect();

        let spline = LieGroupCubicBSpline {
            control_points,
            delta_t: 1.0,
            t0: 0.0,
        };

        // Continuity at segment boundaries.
        for i in 1..spline.num_segments() {
            let t = i as f64;
            let from_left = spline.interpolate(t - 1e-8);
            let from_right = spline.interpolate(t + 1e-8);
            let diff = (from_left.inverse() * &from_right).log();
            assert_abs_diff_eq!(diff[0], 0.0, epsilon = 1e-5);
        }

        // Velocity vs numerical differentiation.
        let dt = 1e-6;
        for t in [0.5, 1.0, 1.5, 2.5, 3.5, 4.0] {
            if t + dt > spline.t_max() {
                continue;
            }
            let v_analytic = spline.velocity(t);
            let t_curr = spline.interpolate(t);
            let t_next = spline.interpolate(t + dt);
            let v_numeric = IsVector::scaled(&(t_curr.inverse() * &t_next).log(), 1.0 / dt);
            assert_abs_diff_eq!(v_analytic[0], v_numeric[0], epsilon = 1e-3);
        }
    }

    #[test]
    fn lie_group_spline_so3() {
        let tangents = [
            VecF64::<3>::new(0.0, 0.0, 0.0),
            VecF64::<3>::new(0.3, 0.1, -0.2),
            VecF64::<3>::new(0.5, -0.4, 0.1),
            VecF64::<3>::new(-0.2, 0.6, 0.3),
            VecF64::<3>::new(0.1, -0.1, -0.5),
        ];
        let control_points: alloc::vec::Vec<Rotation3F64> =
            tangents.iter().map(|t| Rotation3F64::exp(*t)).collect();

        let spline = LieGroupCubicBSpline {
            control_points,
            delta_t: 1.0,
            t0: 0.0,
        };

        // Continuity at segment boundaries.
        for i in 1..spline.num_segments() {
            let t = i as f64;
            let from_left = spline.interpolate(t - 1e-8);
            let from_right = spline.interpolate(t + 1e-8);
            let diff = (from_left.inverse() * &from_right).log();
            assert_abs_diff_eq!(diff.norm(), 0.0, epsilon = 1e-4);
        }

        // Velocity vs numerical differentiation.
        let dt = 1e-6;
        for t in [0.5, 1.5, 2.5, 3.0] {
            if t + dt > spline.t_max() {
                continue;
            }
            let v_analytic = spline.velocity(t);
            let t_curr = spline.interpolate(t);
            let t_next = spline.interpolate(t + dt);
            let v_numeric = IsVector::scaled(&(t_curr.inverse() * &t_next).log(), 1.0 / dt);
            for d in 0..3 {
                assert_abs_diff_eq!(v_analytic[d], v_numeric[d], epsilon = 1e-3);
            }
        }
    }
}
