use crate::prelude::IsScalar;

/// A trait for scalar types that store both a real part and an infinitesimal part,
/// enabling *forward-mode automatic differentiation* (AD) of scalar functions.
///
/// Typical implementations include [`DualScalar<DM, DN>`][crate::dual::DualScalar]
/// for single-lane usage (`BATCH=1`), or [`DualBatchScalar<BATCH, DM,
/// DN>`][crate::dual::DualBatchScalar] when using SIMD-based batch derivatives.
///
/// # Const Parameters
/// - `BATCH`: Number of lanes (1 if non-SIMD).
/// - `DM`, `DN`: Dimensions of the derivative. For example, `DM=3, DN=1` indicates storing partial
///   derivatives w.r.t. a 3D input, resulting in a 3×1 Jacobian for each scalar.
pub trait IsDualScalar<const BATCH: usize, const DM: usize, const DN: usize>:
    IsScalar<BATCH, DM, DN>
{
    /// Creates a dual scalar from a “real” scalar, marking it as a variable for AD.
    ///
    /// The newly created dual scalar will have derivative = identity w.r.t. this variable,
    /// meaning the real part is `val` and the infinitesimal part is a unit derivative.
    ///
    /// # Example
    /// ```rust
    /// use sophus_autodiff::{
    ///     dual::DualScalar,
    ///     prelude::*,
    /// };
    ///
    /// // Suppose we want to differentiate f(x) = x² at x = 0.1
    /// // First, create the variable as a dual:
    /// let x_dual = DualScalar::<1, 1>::var(0.1);
    /// // Then define f in terms of the dual variable:
    /// let f_dual = x_dual * x_dual;
    /// // We can now get the derivative with respect to x:
    /// let dfdx = f_dual.curve_derivative(); // Should be 2 * 0.1 = 0.2
    /// ```
    fn var(val: Self::RealScalar) -> Self;

    /// Creates a dual vector of dimension `ROWS` from a purely real vector,
    /// establishing each component as a variable for AD.
    fn vector_var<const ROWS: usize>(val: Self::RealVector<ROWS>)
        -> Self::DualVector<ROWS, DM, DN>;

    /// Creates a dual matrix of size `[ROWS, COLS]` from a purely real matrix,
    /// establishing each element as a variable for AD.
    fn matrix_var<const ROWS: usize, const COLS: usize>(
        val: Self::RealMatrix<ROWS, COLS>,
    ) -> Self::DualMatrix<ROWS, COLS, DM, DN>;

    /// Returns the infinitesimal part (derivative) of this scalar, as a real matrix
    /// of shape `[DM, DN]`.
    ///
    /// # Interpretation
    /// - If `DM=1, DN=1`, this corresponds to a scalar derivative in `ℝ`.
    /// - If `DM>1, DN=1`, it corresponds to a vector derivative in `ℝ^DM`.
    /// - For more complex shapes, it can represent Jacobians or Hessians as needed.
    fn derivative(&self) -> Self::RealMatrix<DM, DN>;
}

/// A trait extension indicating the scalar is the result of a curve `f: ℝ -> ℝ`,
/// enabling a simpler interface to retrieve the 1D derivative.
///
/// # Const Parameter
/// - `BATCH`: Number of lanes (e.g., 1 for single-lane or >1 for SIMD).
///
/// Implementors typically return a real scalar via `curve_derivative()`.
pub trait IsDualScalarFromCurve<S: IsDualScalar<BATCH, 1, 1>, const BATCH: usize>:
    IsDualScalar<BATCH, 1, 1>
{
    /// Returns the derivative as a single real number, since `f: ℝ -> ℝ`.
    ///
    /// # Example
    /// ```rust
    /// use sophus_autodiff::{
    ///     dual::DualScalar,
    ///     prelude::*,
    /// };
    /// // Suppose we want to differentiate f(x)=x³ at x=2:
    /// // First, we wrap x=2 as a dual variable:
    /// let x_dual = DualScalar::<1, 1>::var(2.0);
    /// // Then compute f(x):
    /// let f_dual = x_dual * x_dual * x_dual;
    /// // The derivative is 3 * x² = 12:
    /// let derivative = f_dual.curve_derivative();
    /// assert_eq!(derivative, 12.0);
    /// ```
    fn curve_derivative(&self) -> S::RealScalar;
}

#[test]
fn dual_scalar_tests() {
    #[cfg(feature = "simd")]
    use crate::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;
    use crate::{
        dual::DualScalar,
        linalg::EPS_F64,
        maps::ScalarValuedCurve,
    };

    trait DualScalarTest {
        fn run_dual_scalar_test();
    }

    // This macro defines a template that implements test logic for each pair of
    // (regular scalar, dual scalar) and for a given BATCH lane size.
    macro_rules! def_dual_scalar_test_template {
        ($batch:literal, $scalar: ty, $dual_scalar: ty) => {
            impl DualScalarTest for $dual_scalar {
                fn run_dual_scalar_test() {
                    // We'll test various math functions with a finite-difference approach
                    // and compare it to the dual-based derivative.
                    let b = <$scalar>::from_f64(12.0);
                    for i in 1..10 {
                        let a: $scalar = <$scalar>::from_f64(0.1 * (i as f64));

                        let abs_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| x.abs(), a, EPS_F64);
                        let abs_auto_grad = <$dual_scalar>::var(a).abs().curve_derivative();
                        approx::assert_abs_diff_eq!(
                            abs_finite_diff,
                            abs_auto_grad,
                            epsilon = 0.0001
                        );

                        let acos_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| x.acos(), a, EPS_F64);
                        let acos_auto_grad = <$dual_scalar>::var(a).acos().curve_derivative();
                        approx::assert_abs_diff_eq!(
                            acos_finite_diff,
                            acos_auto_grad,
                            epsilon = 0.0001
                        );

                        let asin_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| x.asin(), a, EPS_F64);
                        let asin_auto_grad = <$dual_scalar>::var(a).asin().curve_derivative();
                        approx::assert_abs_diff_eq!(
                            asin_finite_diff,
                            asin_auto_grad,
                            epsilon = 0.0001
                        );

                        let atan_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| x.atan(), a, EPS_F64);
                        let atan_auto_grad = <$dual_scalar>::var(a).atan().curve_derivative();
                        approx::assert_abs_diff_eq!(
                            atan_finite_diff,
                            atan_auto_grad,
                            epsilon = 0.0001
                        );

                        let cos_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| x.cos(), a, EPS_F64);
                        let cos_auto_grad = <$dual_scalar>::var(a).cos().curve_derivative();
                        approx::assert_abs_diff_eq!(
                            cos_finite_diff,
                            cos_auto_grad,
                            epsilon = 0.0001
                        );

                        let signum_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| x.signum(), a, EPS_F64);
                        let signum_auto_grad = <$dual_scalar>::var(a).signum().curve_derivative();
                        approx::assert_abs_diff_eq!(
                            signum_finite_diff,
                            signum_auto_grad,
                            epsilon = 0.0001
                        );

                        let exp_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| x.exp(), a, EPS_F64);
                        let exp_auto_grad = <$dual_scalar>::var(a).exp().curve_derivative();
                        approx::assert_abs_diff_eq!(
                            exp_finite_diff,
                            exp_auto_grad,
                            epsilon = 0.0001
                        );

                        let ln_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| x.ln(), a, EPS_F64);
                        let ln_auto_grad = <$dual_scalar>::var(a).ln().curve_derivative();
                        approx::assert_abs_diff_eq!(ln_finite_diff, ln_auto_grad, epsilon = 0.0001);

                        let sin_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| x.sin(), a, EPS_F64);
                        let sin_auto_grad = <$dual_scalar>::var(a).sin().curve_derivative();
                        approx::assert_abs_diff_eq!(
                            sin_finite_diff,
                            sin_auto_grad,
                            epsilon = 0.0001
                        );

                        let sqrt_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| x.sqrt(), a, EPS_F64);
                        let sqrt_auto_grad = <$dual_scalar>::var(a).sqrt().curve_derivative();
                        approx::assert_abs_diff_eq!(
                            sqrt_finite_diff,
                            sqrt_auto_grad,
                            epsilon = 0.0001
                        );

                        let tan_finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| x.tan(), a, EPS_F64);
                        let tan_auto_grad = <$dual_scalar>::var(a).tan().curve_derivative();
                        approx::assert_abs_diff_eq!(
                            tan_finite_diff,
                            tan_auto_grad,
                            epsilon = 0.0001
                        );

                        // f(x) = x^2
                        fn square_fn(x: $scalar) -> $scalar {
                            x * x
                        }
                        fn dual_square_fn(x: $dual_scalar) -> $dual_scalar {
                            x * x
                        }
                        let finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(square_fn, a, EPS_F64);
                        let auto_grad = dual_square_fn(<$dual_scalar>::var(a)).curve_derivative();
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                        // Additional tests for arithmetic ops (add, sub, mul, div)
                        {
                            fn add_fn(x: $scalar, y: $scalar) -> $scalar {
                                x + y
                            }
                            fn dual_add_fn(x: $dual_scalar, y: $dual_scalar) -> $dual_scalar {
                                x + y
                            }

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| add_fn(x, b), a, EPS_F64);
                            let auto_grad = dual_add_fn(
                                <$dual_scalar>::var(a),
                                <$dual_scalar>::from_real_scalar(b),
                            )
                            .curve_derivative();

                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| add_fn(b, x), a, EPS_F64);
                            let auto_grad = dual_add_fn(
                                <$dual_scalar>::from_real_scalar(b),
                                <$dual_scalar>::var(a),
                            )
                            .curve_derivative();
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
                            let auto_grad = dual_sub_fn(
                                <$dual_scalar>::var(a),
                                <$dual_scalar>::from_real_scalar(b),
                            )
                            .curve_derivative();
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| sub_fn(b, x), a, EPS_F64);
                            let auto_grad = dual_sub_fn(
                                <$dual_scalar>::from_real_scalar(b),
                                <$dual_scalar>::var(a),
                            )
                            .curve_derivative();
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
                            let auto_grad = dual_mul_fn(
                                <$dual_scalar>::var(a),
                                <$dual_scalar>::from_real_scalar(b),
                            )
                            .curve_derivative();
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| mul_fn(b, x), a, EPS_F64);
                            let auto_grad = dual_mul_fn(
                                <$dual_scalar>::from_real_scalar(b),
                                <$dual_scalar>::var(a),
                            )
                            .curve_derivative();
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                        }

                        {
                            fn div_fn(x: $scalar, y: $scalar) -> $scalar {
                                x / y
                            }
                            fn dual_div_fn(x: $dual_scalar, y: $dual_scalar) -> $dual_scalar {
                                x / y
                            }
                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| div_fn(x, b), a, EPS_F64);
                            let auto_grad = dual_div_fn(
                                <$dual_scalar>::var(a),
                                <$dual_scalar>::from_real_scalar(b),
                            )
                            .curve_derivative();
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| div_fn(b, x), a, EPS_F64);
                            let auto_grad = dual_div_fn(
                                <$dual_scalar>::from_real_scalar(b),
                                <$dual_scalar>::var(a),
                            )
                            .curve_derivative();
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| div_fn(b, x), a, EPS_F64);
                            let auto_grad = dual_div_fn(
                                <$dual_scalar>::from_real_scalar(b),
                                <$dual_scalar>::var(a),
                            )
                            .curve_derivative();
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| div_fn(x, b), a, EPS_F64);
                            let auto_grad = dual_div_fn(
                                <$dual_scalar>::var(a),
                                <$dual_scalar>::from_real_scalar(b),
                            )
                            .curve_derivative();
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                        }
                    }
                }
            }
        };
    }

    // Expand the test template for each combination of (batch, real scalar, dual scalar).
    def_dual_scalar_test_template!(1, f64, DualScalar<1,1>);
    #[cfg(feature = "simd")]
    def_dual_scalar_test_template!(2, BatchScalarF64<2>, DualBatchScalar<2,1,1>);
    #[cfg(feature = "simd")]
    def_dual_scalar_test_template!(4, BatchScalarF64<4>, DualBatchScalar<4,1,1>);
    #[cfg(feature = "simd")]
    def_dual_scalar_test_template!(8, BatchScalarF64<8>, DualBatchScalar<8,1,1>);

    // Now run the test suites:
    DualScalar::run_dual_scalar_test();
    #[cfg(feature = "simd")]
    DualBatchScalar::<2, 1, 1>::run_dual_scalar_test();
    #[cfg(feature = "simd")]
    DualBatchScalar::<4, 1, 1>::run_dual_scalar_test();
    #[cfg(feature = "simd")]
    DualBatchScalar::<8, 1, 1>::run_dual_scalar_test();
}
