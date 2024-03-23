use crate::dual::dual_matrix::DualM;
use crate::dual::dual_scalar::Dual;
use crate::dual::dual_vector::DualV;
use crate::types::MatF64;
use crate::types::VecF64;

use sophus_tensor::mut_tensor::MutTensorDD;
use sophus_tensor::view::IsTensorLike;

/// Scalar-valued map on a vector space.
///
/// This is a function which takes a vector and returns a scalar:
///
///   f: ℝᵐ -> ℝ
///
/// These functions are also called a scalar fields (on vector spaces).
///
pub struct ScalarValuedMapFromVector;

impl ScalarValuedMapFromVector {
    /// Finite difference quotient of the scalar-valued map.
    ///
    /// The derivative is a vector or rank-1 tensor of shape (Rᵢ).
    ///
    /// Since all operations are batched, it returns a (B x Rᵢ) rank-2 tensor.
    pub fn sym_diff_quotient<TFn, const INROWS: usize>(
        scalar_valued: TFn,
        a: VecF64<INROWS>,
        eps: f64,
    ) -> VecF64<INROWS>
    where
        TFn: Fn(VecF64<INROWS>) -> f64,
    {
        let mut out = VecF64::<INROWS>::zeros();

        for r in 0..INROWS {
            let mut a_plus = a;
            a_plus[r] += eps;

            let mut a_minus = a;
            a_minus[r] -= eps;

            out[r] = (scalar_valued(a_plus) - scalar_valued(a_minus)) / (2.0 * eps);
        }
        out
    }

    /// Auto differentiation of the scalar-valued map.
    pub fn fw_autodiff<TFn, const INROWS: usize>(
        scalar_valued: TFn,
        a: VecF64<INROWS>,
    ) -> VecF64<INROWS>
    where
        TFn: Fn(DualV<INROWS>) -> Dual,
    {
        let jacobian: MutTensorDD<f64> = scalar_valued(DualV::v(a)).dij_val.unwrap();
        assert_eq!(jacobian.dims(), [INROWS, 1]);
        let mut out = VecF64::zeros();

        for r in 0..jacobian.dims()[0] {
            out[r] = jacobian.get([r, 0]);
        }
        out
    }
}

/// Scalar-valued map on a product space (= space of matrices).
///
/// This is a function which takes a matrix and returns a scalar:
///
///   f: ℝᵐ x ℝⁿ -> ℝ
///
/// These functions are also called a scalar fields (on product spaces).
///
pub struct ScalarValuedMapFromMatrix;

impl ScalarValuedMapFromMatrix {
    /// Finite difference quotient of the scalar-valued map.
    ///
    /// The derivative is a matrix or rank-2 tensor of shape (Rᵢ x Cⱼ).
    pub fn sym_diff_quotient<TFn, const INROWS: usize, const INCOLS: usize>(
        scalar_valued: TFn,
        a: MatF64<INROWS, INCOLS>,
        eps: f64,
    ) -> MatF64<INROWS, INCOLS>
    where
        TFn: Fn(MatF64<INROWS, INCOLS>) -> f64,
    {
        let mut out = MatF64::<INROWS, INCOLS>::zeros();

        for r in 0..INROWS {
            for c in 0..INCOLS {
                let mut a_plus = a;
                a_plus[(r, c)] += eps;
                let mut a_minus = a;
                a_minus[(r, c)] -= eps;

                out[(r, c)] = (scalar_valued(a_plus) - scalar_valued(a_minus)) / (2.0 * eps);
            }
        }
        out
    }

    /// Auto differentiation of the scalar-valued map.
    pub fn fw_autodiff<TFn, const INROWS: usize, const INCOLS: usize>(
        scalar_valued: TFn,
        a: MatF64<INROWS, INCOLS>,
    ) -> MatF64<INROWS, INCOLS>
    where
        TFn: Fn(DualM<INROWS, INCOLS>) -> Dual,
    {
        let jacobian: MutTensorDD<f64> = scalar_valued(DualM::v(a)).dij_val.unwrap();
        assert_eq!(jacobian.dims(), [INROWS, INCOLS]);
        let mut out = MatF64::zeros();

        for r in 0..jacobian.dims()[0] {
            for c in 0..jacobian.dims()[1] {
                out[(r, c)] = jacobian.get([r, c]);
            }
        }
        out
    }
}

mod test {
    #[cfg(test)]
    use crate::maps::scalar_valued_maps::ScalarValuedMapFromMatrix;
    #[cfg(test)]
    use crate::maps::scalar_valued_maps::ScalarValuedMapFromVector;
    #[cfg(test)]
    use crate::types::matrix::IsMatrix;
    #[cfg(test)]
    use crate::types::scalar::IsScalar;
    #[cfg(test)]
    use crate::types::vector::IsVector;
    #[cfg(test)]
    use crate::types::MatF64;
    #[cfg(test)]
    use crate::types::VecF64;

    #[test]
    fn test_scalar_valued_map_from_vector() {
        let a = VecF64::<2>::new(0.1, 0.4);

        fn f<S: IsScalar<1>, V2: IsVector<S, 2, 1>>(x: V2) -> S {
            x.norm()
        }

        let finite_diff = ScalarValuedMapFromVector::sym_diff_quotient(f, a, 1e-6);
        let auto_grad = ScalarValuedMapFromVector::fw_autodiff(f, a);
        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
    }

    #[test]
    fn test_batched_scalar_valued_map_from_matrix() {
        //      [[ a,   b ]]
        //  det [[ c,   d ]] = ad - bc
        //      [[ e,   f ]]

        fn determinant_fn<S: IsScalar<1>, M32: IsMatrix<S, 3, 2, 1>>(mat: M32) -> S {
            let a = mat.get((0, 0));
            let b = mat.get((0, 1));

            let c = mat.get((1, 0));
            let d = mat.get((1, 1));

            (a * d) - (b * c)
        }

        let mut mat = MatF64::<3, 2>::zeros();
        mat[(0, 0)] = 4.6;
        mat[(1, 0)] = 1.6;
        mat[(1, 1)] = 0.6;

        let finite_diff = ScalarValuedMapFromMatrix::sym_diff_quotient(determinant_fn, mat, 1e-6);
        let auto_grad = ScalarValuedMapFromMatrix::fw_autodiff(determinant_fn, mat);
        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
    }
}
