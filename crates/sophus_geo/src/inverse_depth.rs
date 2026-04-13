use sophus_lie::prelude::{
    IsMatrix,
    IsScalar,
    IsSingleScalar,
    IsVector,
};

/// Point in inverse-depth parameterization.
///
/// Stores `(a₁, ..., aₙ₋₁, ψ)` where:
/// - `aᵢ = xᵢ / z` (bearing ratios)
/// - `ψ = 1 / z` (inverse depth)
///
/// Cartesian coordinates are `(x₁, ..., xₙ₋₁, z) = (a₁/ψ, ..., aₙ₋₁/ψ, 1/ψ)`.
///
/// This parameterization is useful for optimization because:
/// - Uncertainty in `(a, ψ)` is approximately Gaussian even for distant points.
/// - Depth uncertainty in Cartesian space is nonlinearly distorted; in inverse-depth space it
///   becomes a well-conditioned ellipse.
/// - The scaled transform `R · (a₁, ..., aₙ₋₁, 1)ᵀ + ψ · t` avoids division by ψ.
///
/// ## References
///
/// Civera, Davison, Montiel: "Inverse Depth Parametrization for Monocular SLAM" (2008).
/// Eade: "Monocular SLAM with Inverse-Depth Landmarks" (2008).
#[derive(Clone, Debug)]
pub struct InverseDepthPoint<
    S: IsScalar<BATCH, DM, DN>,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    /// The inverse-depth parameters `(a₁, ..., aₙ₋₁, ψ)`.
    pub params: S::Vector<DIM>,
}

/// 2D inverse-depth point in the xz-plane: `(a, ψ) = (x/z, 1/z)`.
pub type InverseDepthPoint2<S, const B: usize, const DM: usize, const DN: usize> =
    InverseDepthPoint<S, 2, B, DM, DN>;
/// 3D inverse-depth point: `(a, b, ψ) = (x/z, y/z, 1/z)`.
pub type InverseDepthPoint3<S, const B: usize, const DM: usize, const DN: usize> =
    InverseDepthPoint<S, 3, B, DM, DN>;

/// 2D inverse-depth point with f64 scalar.
pub type InverseDepthPoint2F64 = InverseDepthPoint2<f64, 1, 0, 0>;
/// 3D inverse-depth point with f64 scalar.
pub type InverseDepthPoint3F64 = InverseDepthPoint3<f64, 1, 0, 0>;

// ── 2D specialization ───────────────────────────────────────────────────

impl<S: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize> InverseDepthPoint2<S, 1, DM, DN> {
    /// Create from inverse-depth parameters `(a, ψ)`.
    pub fn from_params(params: S::Vector<2>) -> Self {
        Self { params }
    }

    /// Create from Cartesian `(x, z)`. Returns `None` if `z ≈ 0`.
    pub fn from_cartesian(cartesian: S::Vector<2>) -> Option<Self> {
        let z = cartesian.elem(1);
        if z.abs().real_part() < S::RealScalar::from_f64(1e-10) {
            return None;
        }
        let z_inv = S::from_f64(1.0) / z;
        Some(Self {
            params: S::Vector::<2>::from_array([cartesian.elem(0) * z_inv, z_inv]),
        })
    }

    /// Convert to Cartesian `(x, z)`. Returns `None` if `ψ ≈ 0` (point at infinity).
    pub fn to_cartesian(&self) -> Option<S::Vector<2>> {
        let psi = self.params.elem(1);
        if psi.abs().real_part() < S::RealScalar::from_f64(1e-10) {
            return None;
        }
        let psi_inv = S::from_f64(1.0) / psi;
        Some(S::Vector::<2>::from_array([
            self.params.elem(0) * psi_inv,
            psi_inv,
        ]))
    }

    /// The inverse depth `ψ = 1/z`.
    pub fn psi(&self) -> S {
        self.params.elem(1)
    }

    /// The depth `z = 1/ψ`. Returns `None` if `ψ ≈ 0`.
    pub fn depth(&self) -> Option<S> {
        let psi = self.psi();
        if psi.abs().real_part() < S::RealScalar::from_f64(1e-10) {
            return None;
        }
        Some(S::from_f64(1.0) / psi)
    }

    /// Scaled point in a camera frame: `R * (a, 1)^T + ψ * t`.
    ///
    /// This avoids dividing by ψ when transforming inverse-depth points.
    /// The result is related to the Cartesian camera-frame point by
    /// `scaled_point = ψ * p_cam`.
    pub fn scaled_point(&self, rot: S::Matrix<2, 2>, t: S::Vector<2>) -> S::Vector<2> {
        let dir = S::Vector::<2>::from_array([self.params.elem(0), S::from_f64(1.0)]);
        rot * dir + t.scaled(self.psi())
    }

    /// Jacobian of [`scaled_point`](Self::scaled_point) w.r.t. the parameters `(a, ψ)`.
    ///
    /// Returns a 2x2 matrix `[R * e₀ | t]` where `e₀ = (1, 0)^T`.
    pub fn dx_scaled_point_d_params(
        &self,
        rot: S::Matrix<2, 2>,
        t: S::Vector<2>,
    ) -> S::Matrix<2, 2> {
        let r0 = rot.get_col_vec(0);
        let mut m = S::Matrix::<2, 2>::zeros();
        m.set_col_vec(0, r0);
        m.set_col_vec(1, t);
        m
    }
}

// ── 3D specialization ───────────────────────────────────────────────────

impl<S: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize> InverseDepthPoint3<S, 1, DM, DN> {
    /// Create from inverse-depth parameters `(a, b, ψ)`.
    pub fn from_params(params: S::Vector<3>) -> Self {
        Self { params }
    }

    /// Create from Cartesian `(x, y, z)`. Returns `None` if `z ≈ 0`.
    pub fn from_cartesian(cartesian: S::Vector<3>) -> Option<Self> {
        let z = cartesian.elem(2);
        if z.abs().real_part() < S::RealScalar::from_f64(1e-10) {
            return None;
        }
        let z_inv = S::from_f64(1.0) / z;
        Some(Self {
            params: S::Vector::<3>::from_array([
                cartesian.elem(0) * z_inv,
                cartesian.elem(1) * z_inv,
                z_inv,
            ]),
        })
    }

    /// Convert to Cartesian `(x, y, z)`. Returns `None` if `ψ ≈ 0` (point at infinity).
    pub fn to_cartesian(&self) -> Option<S::Vector<3>> {
        let psi = self.params.elem(2);
        if psi.abs().real_part() < S::RealScalar::from_f64(1e-10) {
            return None;
        }
        let psi_inv = S::from_f64(1.0) / psi;
        Some(S::Vector::<3>::from_array([
            self.params.elem(0) * psi_inv,
            self.params.elem(1) * psi_inv,
            psi_inv,
        ]))
    }

    /// The inverse depth `ψ = 1/z`.
    pub fn psi(&self) -> S {
        self.params.elem(2)
    }

    /// The depth `z = 1/ψ`. Returns `None` if `ψ ≈ 0`.
    pub fn depth(&self) -> Option<S> {
        let psi = self.psi();
        if psi.abs().real_part() < S::RealScalar::from_f64(1e-10) {
            return None;
        }
        Some(S::from_f64(1.0) / psi)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use sophus_autodiff::linalg::VecF64;

    use super::*;

    #[test]
    fn roundtrip_2d() {
        let cart = VecF64::<2>::new(3.0, 10.0);
        let inv = InverseDepthPoint2F64::from_cartesian(cart).unwrap();

        assert_abs_diff_eq!(inv.params[0], 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(inv.params[1], 0.1, epsilon = 1e-12);

        let back = inv.to_cartesian().unwrap();
        assert_abs_diff_eq!(back[0], cart[0], epsilon = 1e-12);
        assert_abs_diff_eq!(back[1], cart[1], epsilon = 1e-12);
    }

    #[test]
    fn roundtrip_3d() {
        let cart = VecF64::<3>::new(2.0, -1.0, 5.0);
        let inv = InverseDepthPoint3F64::from_cartesian(cart).unwrap();

        assert_abs_diff_eq!(inv.params[0], 0.4, epsilon = 1e-12);
        assert_abs_diff_eq!(inv.params[1], -0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(inv.params[2], 0.2, epsilon = 1e-12);

        let back = inv.to_cartesian().unwrap();
        assert_abs_diff_eq!(back[0], cart[0], epsilon = 1e-12);
        assert_abs_diff_eq!(back[1], cart[1], epsilon = 1e-12);
        assert_abs_diff_eq!(back[2], cart[2], epsilon = 1e-12);
    }

    #[test]
    fn depth_and_psi() {
        let inv = InverseDepthPoint2F64::from_params(VecF64::<2>::new(0.5, 0.25));
        assert_abs_diff_eq!(inv.psi(), 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(inv.depth().unwrap(), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn distant_point_small_psi() {
        let cart = VecF64::<2>::new(0.0, 1000.0);
        let inv = InverseDepthPoint2F64::from_cartesian(cart).unwrap();
        assert_abs_diff_eq!(inv.psi(), 0.001, epsilon = 1e-12);
        assert_abs_diff_eq!(inv.depth().unwrap(), 1000.0, epsilon = 1e-6);
    }

    #[test]
    fn zero_depth_returns_none() {
        let cart = VecF64::<2>::new(1.0, 0.0);
        assert!(InverseDepthPoint2F64::from_cartesian(cart).is_none());

        let cart3 = VecF64::<3>::new(1.0, 2.0, 0.0);
        assert!(InverseDepthPoint3F64::from_cartesian(cart3).is_none());
    }

    #[test]
    fn zero_psi_returns_none() {
        let inv = InverseDepthPoint2F64::from_params(VecF64::<2>::new(0.5, 0.0));
        assert!(inv.to_cartesian().is_none());
        assert!(inv.depth().is_none());
    }

    #[test]
    fn scaled_point_basic() {
        use sophus_autodiff::linalg::MatF64;

        let inv = InverseDepthPoint2F64::from_params(VecF64::<2>::new(0.3, 0.1));
        let rot = MatF64::<2, 2>::identity();
        let t = VecF64::<2>::new(1.0, 2.0);

        // scaled_p = R * (0.3, 1.0) + 0.1 * (1.0, 2.0) = (0.4, 1.2)
        let sp = inv.scaled_point(rot, t);
        assert_abs_diff_eq!(sp[0], 0.4, epsilon = 1e-12);
        assert_abs_diff_eq!(sp[1], 1.2, epsilon = 1e-12);
    }

    #[test]
    fn dx_scaled_point_d_params_matches_autodiff() {
        use sophus_autodiff::{
            dual::{
                DualScalar,
                DualVector,
            },
            prelude::*,
        };
        use sophus_lie::Rotation2;

        let a_psi = VecF64::<2>::new(0.3, 0.1);
        let angle = 0.4;
        let rot_f64 = Rotation2::<f64, 1, 0, 0>::exp(VecF64::<1>::new(angle)).matrix();
        let t_f64 = VecF64::<2>::new(1.5, -0.7);

        // Analytic Jacobian
        let inv = InverseDepthPoint2F64::from_params(a_psi);
        let analytic = inv.dx_scaled_point_d_params(rot_f64, t_f64);

        // Autodiff: scaled_point maps R^2 -> R^2, so we want a 2x2 Jacobian.
        type DS = DualScalar<f64, 2, 1>;

        let rot_dual = <DS as IsScalar<1, 2, 1>>::Matrix::<2, 2>::from_real_matrix(rot_f64);
        let t_dual = <DS as IsScalar<1, 2, 1>>::Vector::<2>::from_real_vector(t_f64);
        let dual_params = DualVector::<f64, 2, 2, 1>::var(a_psi);
        let dual_inv = InverseDepthPoint2::<DS, 1, 2, 1>::from_params(dual_params);
        let dual_sp = dual_inv.scaled_point(rot_dual, t_dual);

        // dual_sp is a DualVector<f64, 2, 2, 1>; jacobian() -> SMatrix<f64, 2, 2>
        let autodiff_jac = dual_sp.jacobian();

        for r in 0..2 {
            for c in 0..2 {
                assert_abs_diff_eq!(
                    analytic.elem([r, c]).real_part(),
                    autodiff_jac[(r, c)],
                    epsilon = 1e-10
                );
            }
        }
    }
}
