/// hyper-plane: line in 2d, plane in 3d, ...
pub mod hyperplane;
/// n-Sphere: circle, sphere, ...
pub mod hypersphere;
/// ray
pub mod ray;
/// region
pub mod region;
/// unit vector
pub mod unit_vector;

/// sophus_geo prelude
pub mod prelude {
    pub use sophus_autodiff::prelude::*;
    pub use sophus_lie::prelude::*;

    pub use crate::region::IsRegion;
}
