use crate::prelude::IsScalar;

/// Trait for scalar dual numbers
pub trait IsDualScalar<const BATCH: usize, const DM: usize, const DN: usize>:
    IsScalar<BATCH, DM, DN>
{
    /// Create a new dual scalar from real scalar for auto-differentiation with respect to self
    ///
    /// Typically this is not called directly, but through using a curve auto-differentiation call:
    ///
    ///  - ScalarValuedCurve::fw_autodiff(...);
    ///  - VectorValuedCurve::fw_autodiff(...);
    ///  - MatrixValuedCurve::fw_autodiff(...);
    fn var(val: Self::RealScalar) -> Self;

    /// Create a new dual vector from a real vector for auto-differentiation with respect to self
    ///
    /// Typically this is not called directly, but through using a map auto-differentiation call:
    ///
    ///  - ScalarValuedVectorMap::fw_autodiff(...);
    ///  - VectorValuedVectorMap::fw_autodiff(...);
    ///  - MatrixValuedVectorMap::fw_autodiff(...);
    fn vector_var<const ROWS: usize>(val: Self::RealVector<ROWS>)
        -> Self::DualVector<ROWS, DM, DN>;

    /// Create a new dual matrix from a real matrix for auto-differentiation with respect to self
    ///
    /// Typically this is not called directly, but through using a map auto-differentiation call:
    ///
    ///  - ScalarValueMatrixMap::fw_autodiff(...);
    ///  - VectorValuedMatrixMap::fw_autodiff(...);
    ///  - MatrixValuedMatrixMap::fw_autodiff(...);
    fn matrix_var<const ROWS: usize, const COLS: usize>(
        val: Self::RealMatrix<ROWS, COLS>,
    ) -> Self::DualMatrix<ROWS, COLS, DM, DN>;

    /// Get the derivative
    fn derivative(&self) -> Self::RealMatrix<DM, DN>;
}

#[test]
fn dual_scalar_tests() {
    #[cfg(feature = "simd")]
    use crate::calculus::dual::DualBatchScalar;
    use crate::calculus::dual::DualScalar;
    use crate::calculus::maps::curves::ScalarValuedCurve;
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;
    use crate::linalg::EPS_F64;

    trait DualScalarTest {
        fn run_dual_scalar_test();
    }
    macro_rules! def_dual_scalar_test_template {
        ($batch:literal, $scalar: ty, $dual_scalar: ty) => {
            impl DualScalarTest for $dual_scalar {
                fn run_dual_scalar_test() {
                    let b = <$scalar>::from_f64(12.0);
                    for i in 1..10 {
                        let a: $scalar = <$scalar>::from_f64(0.1 * (i as f64));

                        fn abs_fn(x: $scalar) -> $scalar {
                            x.abs()
                        }
                        fn dual_abs_fn(x: $dual_scalar) -> $dual_scalar {
                            x.abs()
                        }
                        let abs_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(abs_fn, a, EPS_F64);
                        let abs_auto_grad = ScalarValuedCurve::fw_autodiff(dual_abs_fn, a);
                        approx::assert_abs_diff_eq!(
                            abs_finite_diff,
                            abs_auto_grad,
                            epsilon = 0.0001
                        );

                        fn acos_fn(x: $scalar) -> $scalar {
                            x.acos()
                        }
                        fn dual_acos_fn(x: $dual_scalar) -> $dual_scalar {
                            x.acos()
                        }
                        let acos_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(acos_fn, a, EPS_F64);
                        let acos_auto_grad = ScalarValuedCurve::fw_autodiff(dual_acos_fn, a);
                        approx::assert_abs_diff_eq!(
                            acos_finite_diff,
                            acos_auto_grad,
                            epsilon = 0.0001
                        );

                        fn asin_fn(x: $scalar) -> $scalar {
                            x.asin()
                        }
                        fn dual_asin_fn(x: $dual_scalar) -> $dual_scalar {
                            x.asin()
                        }
                        let asin_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(asin_fn, a, EPS_F64);
                        let asin_auto_grad = ScalarValuedCurve::fw_autodiff(dual_asin_fn, a);
                        approx::assert_abs_diff_eq!(
                            asin_finite_diff,
                            asin_auto_grad,
                            epsilon = 0.0001
                        );

                        fn atan_fn(x: $scalar) -> $scalar {
                            x.atan()
                        }
                        fn dual_atan_fn(x: $dual_scalar) -> $dual_scalar {
                            x.atan()
                        }
                        let atan_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(atan_fn, a, EPS_F64);
                        let atan_auto_grad = ScalarValuedCurve::fw_autodiff(dual_atan_fn, a);
                        approx::assert_abs_diff_eq!(
                            atan_finite_diff,
                            atan_auto_grad,
                            epsilon = 0.0001
                        );

                        fn cos_fn(x: $scalar) -> $scalar {
                            x.cos()
                        }
                        fn dual_cos_fn(x: $dual_scalar) -> $dual_scalar {
                            x.cos()
                        }
                        let cos_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(cos_fn, a, EPS_F64);
                        let cos_auto_grad = ScalarValuedCurve::fw_autodiff(dual_cos_fn, a);
                        approx::assert_abs_diff_eq!(
                            cos_finite_diff,
                            cos_auto_grad,
                            epsilon = 0.0001
                        );

                        fn signum_fn(x: $scalar) -> $scalar {
                            x.signum()
                        }
                        fn dual_signum_fn(x: $dual_scalar) -> $dual_scalar {
                            x.signum()
                        }
                        let signum_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(signum_fn, a, EPS_F64);
                        let signum_auto_grad = ScalarValuedCurve::fw_autodiff(dual_signum_fn, a);
                        approx::assert_abs_diff_eq!(
                            signum_finite_diff,
                            signum_auto_grad,
                            epsilon = 0.0001
                        );

                        fn sin_fn(x: $scalar) -> $scalar {
                            x.sin()
                        }
                        fn dual_sin_fn(x: $dual_scalar) -> $dual_scalar {
                            x.sin()
                        }
                        let sin_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(sin_fn, a, EPS_F64);
                        let sin_auto_grad = ScalarValuedCurve::fw_autodiff(dual_sin_fn, a);
                        approx::assert_abs_diff_eq!(
                            sin_finite_diff,
                            sin_auto_grad,
                            epsilon = 0.0001
                        );

                        fn sqrt_fn(x: $scalar) -> $scalar {
                            x.sqrt()
                        }
                        fn dual_sqrt_fn(x: $dual_scalar) -> $dual_scalar {
                            x.sqrt()
                        }
                        let sqrt_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(sqrt_fn, a, EPS_F64);
                        let sqrt_auto_grad = ScalarValuedCurve::fw_autodiff(dual_sqrt_fn, a);
                        approx::assert_abs_diff_eq!(
                            sqrt_finite_diff,
                            sqrt_auto_grad,
                            epsilon = 0.0001
                        );

                        fn tan_fn(x: $scalar) -> $scalar {
                            x.tan()
                        }
                        fn dual_tan_fn(x: $dual_scalar) -> $dual_scalar {
                            x.tan()
                        }
                        let tan_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(tan_fn, a, EPS_F64);
                        let tan_auto_grad = ScalarValuedCurve::fw_autodiff(dual_tan_fn, a);
                        approx::assert_abs_diff_eq!(
                            tan_finite_diff,
                            tan_auto_grad,
                            epsilon = 0.0001
                        );

                        // f(x) = x^2
                        fn square_fn(x: $scalar) -> $scalar {
                            x.clone() * x
                        }
                        fn dual_square_fn(x: $dual_scalar) -> $dual_scalar {
                            x.clone() * x
                        }
                        let finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(square_fn, a, EPS_F64);
                        let auto_grad = ScalarValuedCurve::fw_autodiff(dual_square_fn, a);
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                        {
                            fn add_fn(x: $scalar, y: $scalar) -> $scalar {
                                x + y
                            }
                            fn dual_add_fn(x: $dual_scalar, y: $dual_scalar) -> $dual_scalar {
                                x + y
                            }

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| add_fn(x, b), a, EPS_F64);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_add_fn(x, <$dual_scalar>::from_real_scalar(b)),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| add_fn(b, x), a, EPS_F64);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_add_fn(<$dual_scalar>::from_real_scalar(b), x),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                        }

                        {
                            fn sub_fn(x: $scalar, y: $scalar) -> $scalar {
                                x - y
                            }
                            fn dual_sub_fn(x: $dual_scalar, y: $dual_scalar) -> $dual_scalar {
                                x - y
                            }
                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| sub_fn(x, b), a, EPS_F64);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_sub_fn(x, <$dual_scalar>::from_real_scalar(b)),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| sub_fn(b, x), a, EPS_F64);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_sub_fn(<$dual_scalar>::from_real_scalar(b), x),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                        }

                        {
                            fn mul_fn(x: $scalar, y: $scalar) -> $scalar {
                                x * y
                            }
                            fn dual_mul_fn(x: $dual_scalar, y: $dual_scalar) -> $dual_scalar {
                                x * y
                            }
                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| mul_fn(x, b), a, EPS_F64);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_mul_fn(x, <$dual_scalar>::from_real_scalar(b)),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| mul_fn(x, b), a, EPS_F64);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_mul_fn(x, <$dual_scalar>::from_real_scalar(b)),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                        }

                        fn div_fn(x: $scalar, y: $scalar) -> $scalar {
                            x / y
                        }
                        fn dual_div_fn(x: $dual_scalar, y: $dual_scalar) -> $dual_scalar {
                            x / y
                        }
                        let finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| div_fn(x, b), a, EPS_F64);
                        let auto_grad = ScalarValuedCurve::fw_autodiff(
                            |x| dual_div_fn(x, <$dual_scalar>::from_real_scalar(b)),
                            a,
                        );
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                        let finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| div_fn(x, b), a, EPS_F64);
                        let auto_grad = ScalarValuedCurve::fw_autodiff(
                            |x| dual_div_fn(x, <$dual_scalar>::from_real_scalar(b)),
                            a,
                        );
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                        let finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| div_fn(b, x), a, EPS_F64);
                        let auto_grad = ScalarValuedCurve::fw_autodiff(
                            |x| dual_div_fn(<$dual_scalar>::from_real_scalar(b), x),
                            a,
                        );
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                        let finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| div_fn(x, b), a, EPS_F64);
                        let auto_grad = ScalarValuedCurve::fw_autodiff(
                            |x| dual_div_fn(x, <$dual_scalar>::from_real_scalar(b)),
                            a,
                        );
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                    }
                }
            }
        };
    }

    def_dual_scalar_test_template!(1, f64, DualScalar<1,1>);
    #[cfg(feature = "simd")]
    def_dual_scalar_test_template!(2, BatchScalarF64<2>, DualBatchScalar<2,1,1>);
    #[cfg(feature = "simd")]
    def_dual_scalar_test_template!(4, BatchScalarF64<4>, DualBatchScalar<4,1,1>);
    #[cfg(feature = "simd")]
    def_dual_scalar_test_template!(8, BatchScalarF64<8>, DualBatchScalar<8,1,1>);

    DualScalar::run_dual_scalar_test();
    #[cfg(feature = "simd")]
    DualBatchScalar::<2, 1, 1>::run_dual_scalar_test();
    #[cfg(feature = "simd")]
    DualBatchScalar::<4, 1, 1>::run_dual_scalar_test();
    #[cfg(feature = "simd")]
    DualBatchScalar::<8, 1, 1>::run_dual_scalar_test();
}
