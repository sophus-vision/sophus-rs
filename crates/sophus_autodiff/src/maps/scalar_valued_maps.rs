use crate::prelude::*;

/// Scalar-valued map on a vector space.
///
/// This is a function which takes a vector and returns a scalar:
///
///   f: ℝᵐ -> ℝ
///
/// These functions are also called a scalar fields (on vector spaces).
pub struct ScalarValuedVectorMap<S: IsScalar<BATCH, 0, 0>, const BATCH: usize> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH>, const BATCH: usize> ScalarValuedVectorMap<S, BATCH> {
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

            *out.elem_mut(r) = (scalar_valued(a_plus) - scalar_valued(a_minus))
                / S::RealScalar::from_f64(2.0 * eps);
        }
        out
    }
}

/// Scalar-valued map on a product space (= space of matrices).
///
/// This is a function which takes a matrix and returns a scalar:
///
///   f: ℝᵐ x ℝⁿ -> ℝ
pub struct ScalarValuedMatrixMap<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH>, const BATCH: usize> ScalarValuedMatrixMap<S, BATCH, 0, 0> {
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

#[test]
fn scalar_valued_map_tests() {
    #[cfg(feature = "simd")]
    use crate::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;
    use crate::{
        dual::dual_scalar::DualScalar,
        linalg::EPS_F64,
    };

    #[cfg(test)]
    trait Test {
        fn run();
    }

    macro_rules! def_scalar_valued_map_test_template {
        ($batch:literal, $scalar: ty, $dual_scalar: ty, $dual_scalar2: ty
    ) => {
            #[cfg(test)]
            impl Test for $scalar {
                fn run() {
                    use crate::linalg::vector::IsVector;

                    let a = <$scalar as IsScalar<$batch, 0, 0>>::Vector::<2>::new(
                        <$scalar>::from_f64(0.1),
                        <$scalar>::from_f64(0.4),
                    );

                    fn f<
                        S: IsScalar<BATCH, DM, DN>,
                        const BATCH: usize,
                        const DM: usize,
                        const DN: usize,
                    >(
                        x: S::Vector<2>,
                    ) -> S {
                        x.norm()
                    }

                    let finite_diff =
                        ScalarValuedVectorMap::<$scalar, $batch>::sym_diff_quotient(f, a, EPS_F64);
                    let auto_grad =
                        f::<$dual_scalar, $batch, 2, 1>(<$dual_scalar>::vector_var(a)).derivative();

                    approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                    //      [[ a,   b ]]
                    //  det [[ c,   d ]] = ad - bc
                    //      [[ e,   f ]]

                    fn determinant_fn<
                        S: IsScalar<BATCH, DM, DN>,
                        const BATCH: usize,
                        const DM: usize,
                        const DN: usize,
                    >(
                        mat: S::Matrix<3, 2>,
                    ) -> S {
                        let a = mat.elem([0, 0]);
                        let b = mat.elem([0, 1]);

                        let c = mat.elem([1, 0]);
                        let d = mat.elem([1, 1]);

                        (a * d) - (b * c)
                    }

                    let mut mat = <$scalar as IsScalar<$batch, 0, 0>>::Matrix::<3, 2>::zeros();
                    mat[(0, 0)] = <$scalar>::from_f64(4.6);
                    mat[(1, 0)] = <$scalar>::from_f64(1.6);
                    mat[(1, 1)] = <$scalar>::from_f64(0.6);

                    let finite_diff =
                        ScalarValuedMatrixMap::<$scalar, $batch, 0, 0>::sym_diff_quotient(
                            determinant_fn,
                            mat,
                            EPS_F64,
                        );
                    let auto_grad = determinant_fn::<$dual_scalar2, $batch, 3, 2>(
                        <$dual_scalar2>::matrix_var(mat),
                    )
                    .derivative();
                    approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                }
            }
        };
    }

    def_scalar_valued_map_test_template!(1, f64, DualScalar<2,1>, DualScalar<3,2>);
    #[cfg(feature = "simd")]
    def_scalar_valued_map_test_template!(2, BatchScalarF64<2>, DualBatchScalar<2,2,1>,DualBatchScalar<2,3,2>);
    #[cfg(feature = "simd")]
    def_scalar_valued_map_test_template!(4, BatchScalarF64<4>, DualBatchScalar<4,2,1>,DualBatchScalar<4,3,2>);
    #[cfg(feature = "simd")]
    def_scalar_valued_map_test_template!(8, BatchScalarF64<8>, DualBatchScalar<8,2,1>,DualBatchScalar<8,3,2>);
    #[cfg(feature = "simd")]
    def_scalar_valued_map_test_template!(16, BatchScalarF64<16>, DualBatchScalar<16,2,1>,DualBatchScalar<16,3,2>);
    #[cfg(feature = "simd")]
    def_scalar_valued_map_test_template!(32, BatchScalarF64<32>, DualBatchScalar<32,2,1>,DualBatchScalar<32,3,2>);
    #[cfg(feature = "simd")]
    def_scalar_valued_map_test_template!(64, BatchScalarF64<64>, DualBatchScalar<64,2,1>,DualBatchScalar<64,3,2>);

    f64::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<2>::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<4>::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<8>::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<16>::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<32>::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<64>::run();
}
