/// Bundle adjustment demo (standard + scale constraint toggle).
pub mod bundle_adjustment;
/// Offscreen camera simulator example.
#[cfg(feature = "std")]
pub mod camera_sim;
/// Demo application with optics simulation, bundle adjustment, and 3D viewer.
pub mod demo_app;
/// Inverse depth estimation demo with covariance ellipsoid visualization.
pub mod inverse_depth;
/// Shared optimization widget helpers.
pub mod opt_widget;
/// Optics simulation components.
pub mod optics_sim;
/// Viewer example utilities.
pub mod viewer_example;
