/// Bundle adjustment example problem
pub mod ba_problem;
/// Bundle adjustment with scale constraint (Schur + equality constraint example)
pub mod ba_scale_constraint;
/// Camera calibration example problem
pub mod cam_calib;
/// Circle + obstacle: combined equality and inequality constraint example
pub mod circle_obstacle;
/// 2D constrained point — simplest inequality constraint example
pub mod constrained_point2d;
/// Corridor navigation via half-plane inequality constraints
pub mod corridor_navigation;
/// Inverse depth point estimation example problem
pub mod inverse_depth_estimation;
/// linear equality constraint toy example
pub mod linear_eq_toy_problem;
/// non-linear equality constraint toy example
pub mod non_linear_eq_toy_problem;
/// Pose graph example problem
pub mod pose_circle;
/// Simple prior example problem
pub mod simple_prior;
/// Solver validation tests (death tests + canonical cases)
pub mod solver_validation;
/// Spline-based 2D trajectory optimization with obstacle avoidance
pub mod spline_trajectory;
