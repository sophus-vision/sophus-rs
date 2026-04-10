/// Bundle adjustment demo (standard + scale constraint toggle).
pub mod bundle_adjustment;
/// Offscreen camera simulator example.
#[cfg(feature = "std")]
pub mod camera_sim;
/// Circle + obstacle demo (equality + inequality constraints).
pub mod circle_obstacle_demo;
/// Corridor navigation demo.
pub mod corridor_demo;
/// Demo application with optics simulation, bundle adjustment, and 3D viewer.
pub mod demo_app;
/// Inverse depth estimation demo with covariance ellipsoid visualization.
pub mod inverse_depth;
/// Shared optimization widget helpers.
pub mod opt_widget;
/// Optics simulation components.
pub mod optics_sim;
/// 2D inequality-constrained point demo (point pulled toward target, outside obstacles).
pub mod point2d_demo;
/// Spline trajectory optimization demo.
pub mod spline_traj_demo;
/// Viewer example utilities.
pub mod viewer_example;
