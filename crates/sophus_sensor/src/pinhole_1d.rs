use core::borrow::Borrow;

use sophus_autodiff::linalg::{
    IsMatrix,
    IsScalar,
    IsVector,
};

/// 1D pinhole camera model for 2D point-to-1D pixel projection.
///
/// Projects a 2D point `(x, z)` in the camera frame to a 1D pixel coordinate:
///   `u = fx * x/z + cx`
///
/// Parameters: `(fx, cx)`.
///
/// This is the 2D analog of the standard 3D pinhole camera. It is useful for
/// planar (SE(2)) bundle adjustment problems where the camera observes only
/// a horizontal coordinate.
#[derive(Debug, Clone, Copy)]
pub struct Pinhole1dCamera<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    params: S::Vector<2>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Pinhole1dCamera<S, BATCH, DM, DN>
{
    /// Create from focal length and principal point.
    pub fn new(fx: S, cx: S) -> Self {
        Self {
            params: S::Vector::<2>::from_array([fx, cx]),
        }
    }

    /// Create from parameter vector `[fx, cx]`.
    pub fn from_params(params: S::Vector<2>) -> Self {
        Self { params }
    }

    /// Focal length `fx`.
    pub fn fx(&self) -> S {
        self.params.elem(0)
    }

    /// Principal point `cx`.
    pub fn cx(&self) -> S {
        self.params.elem(1)
    }

    /// Project a 2D camera-frame point `(x, z)` to a 1D pixel coordinate.
    ///
    /// `u = fx * x/z + cx`
    pub fn proj<P: Borrow<S::Vector<2>>>(&self, point_in_camera: P) -> S {
        let p = point_in_camera.borrow();
        let fx = self.params.elem(0);
        let cx = self.params.elem(1);
        fx * p.elem(0) / p.elem(1) + cx
    }

    /// Jacobian of `proj` w.r.t. the 2D camera-frame point `(x, z)`.
    ///
    /// Returns a 1x2 matrix: `[fx/z, -fx*x/z²]`.
    pub fn dx_proj_x<P: Borrow<S::Vector<2>>>(&self, point_in_camera: P) -> S::Matrix<1, 2> {
        let p = point_in_camera.borrow();
        let fx = self.params.elem(0);
        let z = p.elem(1);
        let z2 = z * z;
        S::Matrix::<1, 2>::from_array2([[fx / z, -fx * p.elem(0) / z2]])
    }
}

/// Convenience alias for f64 scalar.
pub type Pinhole1dCameraF64 = Pinhole1dCamera<f64, 1, 0, 0>;

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use sophus_autodiff::{
        dual::{
            DualScalar,
            DualVector,
        },
        linalg::VecF64,
        prelude::*,
    };

    use super::*;

    #[test]
    fn proj_basic() {
        let cam = Pinhole1dCameraF64::new(300.0, 320.0);
        let p = VecF64::<2>::new(1.0, 5.0);
        let u = cam.proj(p);
        assert_abs_diff_eq!(u, 300.0 * 0.2 + 320.0, epsilon = 1e-12);
    }

    #[test]
    fn dx_proj_x_matches_autodiff() {
        let fx = 300.0;
        let cx = 320.0;
        let point = VecF64::<2>::new(2.0, 8.0);

        // Analytic Jacobian (1x2)
        let cam = Pinhole1dCameraF64::new(fx, cx);
        let analytic = cam.dx_proj_x(point);

        // Autodiff: f: R^2 -> R, so DualScalar<f64, 2, 1> stores a 2x1 derivative.
        type DS = DualScalar<f64, 2, 1>;
        let dual_cam = Pinhole1dCamera::<DS, 1, 2, 1>::new(DS::from_f64(fx), DS::from_f64(cx));
        let dual_point = DualVector::<f64, 2, 2, 1>::var(point);
        let dual_u: DS = dual_cam.proj(dual_point);
        // derivative() returns SMatrix<f64, 2, 1> = gradient as column vector
        let grad = dual_u.derivative();

        for c in 0..2 {
            assert_abs_diff_eq!(
                analytic.elem([0, c]).real_part(),
                grad[(c, 0)],
                epsilon = 1e-10
            );
        }
    }
}
