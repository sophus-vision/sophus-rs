use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};

/// 3×6 translation Jacobian for SE(3) left-perturbation: `[−hat(t), I]`.
///
/// Maps a 6-DOF se(3) perturbation `(ω, v)` to the change in translation:
///   `dt = −hat(t) · ω + v`
pub(crate) fn translation_jac(t: VecF64<3>) -> MatF64<3, 6> {
    let mut j = MatF64::<3, 6>::zeros();
    // -hat(t) block (columns 0-2)
    j[(0, 1)] = t[2];
    j[(0, 2)] = -t[1];
    j[(1, 0)] = -t[2];
    j[(1, 2)] = t[0];
    j[(2, 0)] = t[1];
    j[(2, 1)] = -t[0];
    // I block (columns 3-5)
    j[(0, 3)] = 1.0;
    j[(1, 4)] = 1.0;
    j[(2, 5)] = 1.0;
    j
}

/// Half-plane inequality constraint for SE(3) poses.
pub mod half_plane;
/// Circle obstacle inequality constraint for 2D scalar points.
pub mod scalar_circle;
/// Ellipse obstacle inequality constraint for 2D scalar points.
pub mod scalar_ellipse;
/// Half-plane inequality constraint for 2D scalar points.
pub mod scalar_half_plane;
/// Circle obstacle inequality constraint for SE(2) poses.
pub mod se2_circle;
/// Circle obstacle constraint evaluated at an SE(2) Lie group B-spline sample point.
pub mod se2_spline_circle;
/// Lateral velocity bound for an SE(2) Lie group B-spline.
pub mod se2_spline_lateral_vel;
pub use half_plane::HalfPlaneConstraint;
pub use scalar_circle::ScalarCircleConstraint;
pub use scalar_ellipse::ScalarEllipseConstraint;
pub use scalar_half_plane::ScalarHalfPlaneConstraint;
pub use se2_circle::SE2CircleConstraint;
pub use se2_spline_circle::SE2SplineCircleConstraint;
pub use se2_spline_lateral_vel::SE2SplineLateralVelConstraint;
