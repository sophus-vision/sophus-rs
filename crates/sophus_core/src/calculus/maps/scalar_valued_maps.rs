use crate::prelude::*;
use crate::tensor::mut_tensor::MutTensorDD;

/// Scalar-valued map on a vector space.
///
/// This is a function which takes a vector and returns a scalar:
///
///   f: ℝᵐ -> ℝ
///
/// These functions are also called a scalar fields (on vector spaces).
///
pub struct ScalarValuedMapFromVector<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: std::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH>, const BATCH: usize> ScalarValuedMapFromVector<S, BATCH> {
    /// Finite difference quotient of the scalar-valued map.
    ///
    /// The derivative is a vector or rank-1 tensor of shape (Rᵢ).
    pub fn sym_diff_quotient<TFn, const INROWS: usize>(
        scalar_valued: TFn,
        a: S::RealVector<INROWS>,
        eps: f64,
    ) -> S::RealVector<INROWS>
    where
        TFn: Fn(S::RealVector<INROWS>) -> S::RealScalar,
    {
        let mut out = S::RealVector::<INROWS>::zeros();

        for r in 0..INROWS {
            let mut a_plus = a;
            a_plus[r] += S::RealScalar::from_f64(eps);

            let mut a_minus = a;
            a_minus[r] -= S::RealScalar::from_f64(eps);

            out.set_elem(
                r,
                (scalar_valued(a_plus) - scalar_valued(a_minus))
                    / S::RealScalar::from_f64(2.0 * eps),
            );
        }
        out
    }
}

impl<D: IsDualScalar<BATCH>, const BATCH: usize> ScalarValuedMapFromVector<D, BATCH> {
    /// Auto differentiation of the scalar-valued map.
    pub fn fw_autodiff<TFn, const INROWS: usize>(
        scalar_valued: TFn,
        a: D::RealVector<INROWS>,
    ) -> D::RealVector<INROWS>
    where
        TFn: Fn(D::DualVector<INROWS>) -> D,
    {
        let jacobian: MutTensorDD<D::RealScalar> = scalar_valued(D::vector_with_dij(a))
            .dij_val()
            .unwrap()
            .clone();
        assert_eq!(jacobian.dims(), [INROWS, 1]);
        let mut out = D::RealVector::<INROWS>::zeros();

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
pub struct ScalarValuedMapFromMatrix<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: std::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH>, const BATCH: usize> ScalarValuedMapFromMatrix<S, BATCH> {
    /// Finite difference quotient of the scalar-valued map.
    ///
    /// The derivative is a matrix or rank-2 tensor of shape (Rᵢ x Cⱼ).
    pub fn sym_diff_quotient<TFn, const INROWS: usize, const INCOLS: usize>(
        scalar_valued: TFn,
        a: S::RealMatrix<INROWS, INCOLS>,
        eps: f64,
    ) -> S::RealMatrix<INROWS, INCOLS>
    where
        TFn: Fn(S::RealMatrix<INROWS, INCOLS>) -> S::RealScalar,
    {
        let mut out = S::RealMatrix::<INROWS, INCOLS>::zeros();

        for r in 0..INROWS {
            for c in 0..INCOLS {
                let mut a_plus = a;
                a_plus[(r, c)] += S::RealScalar::from_f64(eps);
                let mut a_minus = a;
                a_minus[(r, c)] -= S::RealScalar::from_f64(eps);

                out[(r, c)] = (scalar_valued(a_plus) - scalar_valued(a_minus))
                    / S::RealScalar::from_f64(2.0 * eps);
            }
        }
        out
    }
}

impl<D: IsDualScalar<BATCH>, const BATCH: usize> ScalarValuedMapFromMatrix<D, BATCH> {
    /// Auto differentiation of the scalar-valued map.
    pub fn fw_autodiff<TFn, const INROWS: usize, const INCOLS: usize>(
        scalar_valued: TFn,
        a: D::RealMatrix<INROWS, INCOLS>,
    ) -> D::RealMatrix<INROWS, INCOLS>
    where
        TFn: Fn(D::DualMatrix<INROWS, INCOLS>) -> D,
    {
        let jacobian: MutTensorDD<D::RealScalar> = scalar_valued(D::matrix_with_dij(a))
            .dij_val()
            .unwrap()
            .clone();
        assert_eq!(jacobian.dims(), [INROWS, INCOLS]);
        let mut out = D::RealMatrix::<INROWS, INCOLS>::zeros();

        for r in 0..jacobian.dims()[0] {
            for c in 0..jacobian.dims()[1] {
                out[(r, c)] = jacobian.get([r, c]);
            }
        }
        out
    }
}

#[test]
fn scalar_valued_map_tests() {
    use crate::calculus::dual::dual_scalar::DualBatchScalar;
    use crate::calculus::dual::dual_scalar::DualScalar;
    use crate::linalg::BatchScalarF64;

    #[cfg(test)]
    trait Test {
        fn run();
    }

    macro_rules! def_scalar_valued_map_test_template {
        ($batch:literal, $scalar: ty, $dual_scalar: ty
    ) => {
            #[cfg(test)]
            impl Test for $scalar {
                fn run() {
                    use crate::linalg::vector::IsVector;

                    let a = <$scalar as IsScalar<$batch>>::Vector::<2>::new(
                        <$scalar>::from_f64(0.1),
                        <$scalar>::from_f64(0.4),
                    );

                    fn f<S: IsScalar<BATCH>, const BATCH: usize>(x: S::Vector<2>) -> S {
                        x.norm()
                    }

                    let finite_diff =
                        ScalarValuedMapFromVector::<$scalar, $batch>::sym_diff_quotient(f, a, 1e-6);
                    let auto_grad =
                        ScalarValuedMapFromVector::<$dual_scalar, $batch>::fw_autodiff(f, a);
                    approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                    //      [[ a,   b ]]
                    //  det [[ c,   d ]] = ad - bc
                    //      [[ e,   f ]]

                    fn determinant_fn<S: IsScalar<BATCH>, const BATCH: usize>(
                        mat: S::Matrix<3, 2>,
                    ) -> S {
                        let a = mat.get_elem([0, 0]);
                        let b = mat.get_elem([0, 1]);

                        let c = mat.get_elem([1, 0]);
                        let d = mat.get_elem([1, 1]);

                        (a * d) - (b * c)
                    }

                    let mut mat = <$scalar as IsScalar<$batch>>::Matrix::<3, 2>::zeros();
                    mat[(0, 0)] = <$scalar>::from_f64(4.6);
                    mat[(1, 0)] = <$scalar>::from_f64(1.6);
                    mat[(1, 1)] = <$scalar>::from_f64(0.6);

                    let finite_diff =
                        ScalarValuedMapFromMatrix::<$scalar, $batch>::sym_diff_quotient(
                            determinant_fn,
                            mat,
                            1e-6,
                        );
                    let auto_grad = ScalarValuedMapFromMatrix::<$dual_scalar, $batch>::fw_autodiff(
                        determinant_fn,
                        mat,
                    );
                    approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                }
            }
        };
    }

    def_scalar_valued_map_test_template!(1, f64, DualScalar);
    def_scalar_valued_map_test_template!(2, BatchScalarF64<2>, DualBatchScalar<2>);
    def_scalar_valued_map_test_template!(4, BatchScalarF64<4>, DualBatchScalar<4>);
    def_scalar_valued_map_test_template!(8, BatchScalarF64<8>, DualBatchScalar<8>);
    def_scalar_valued_map_test_template!(16, BatchScalarF64<16>, DualBatchScalar<16>);
    def_scalar_valued_map_test_template!(32, BatchScalarF64<32>, DualBatchScalar<32>);
    def_scalar_valued_map_test_template!(64, BatchScalarF64<64>, DualBatchScalar<64>);

    f64::run();
    BatchScalarF64::<2>::run();
    BatchScalarF64::<4>::run();
    BatchScalarF64::<8>::run();
    BatchScalarF64::<16>::run();
    BatchScalarF64::<32>::run();
    BatchScalarF64::<64>::run();
}
