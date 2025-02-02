use crate::prelude::*;

extern crate alloc;

/// A smooth curve in ℝ.
///
/// This is a function which takes a scalar and returns a scalar:
///
///  f: ℝ -> ℝ
pub struct ScalarValuedCurve<S, const BATCH: usize> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsScalar<BATCH, 0, 0>, const BATCH: usize> ScalarValuedCurve<S, BATCH> {
    /// Finite difference quotient of the scalar-valued curve.
    ///
    /// The derivative is also a scalar.
    pub fn sym_diff_quotient<TFn>(curve: TFn, a: S, h: f64) -> S
    where
        TFn: Fn(S) -> S,
    {
        let hh = S::from_f64(h);
        (curve(a + hh) - curve(a - hh)) / S::from_f64(2.0 * h)
    }
}

/// A smooth curve in ℝʳ.
///
/// This is a function which takes a scalar and returns a vector:
///
///   f: ℝ -> ℝʳ
pub struct VectorValuedCurve<S, const BATCH: usize> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsScalar<BATCH, 0, 0>, const BATCH: usize> VectorValuedCurve<S, BATCH> {
    /// Finite difference quotient of the vector-valued curve.
    ///
    /// The derivative is also a vector.
    pub fn sym_diff_quotient<TFn, const ROWS: usize>(curve: TFn, a: S, h: f64) -> S::Vector<ROWS>
    where
        TFn: Fn(S) -> S::Vector<ROWS>,
    {
        let hh = S::from_f64(h);
        (curve(a + hh) - curve(a - hh)).scaled(S::from_f64(1.0 / (2.0 * h)))
    }
}

/// A smooth curve in ℝʳ x ℝᶜ.
///
/// This is a function which takes a scalar and returns a matrix:
///   f: ℝ -> ℝʳ x ℝᶜ
pub struct MatrixValuedCurve<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsScalar<BATCH, 0, 0>, const BATCH: usize> MatrixValuedCurve<S, BATCH, 0, 0> {
    /// Finite difference quotient of the matrix-valued curve.
    ///
    /// The derivative is also a matrix.
    pub fn sym_diff_quotient<TFn, const ROWS: usize, const COLS: usize>(
        curve: TFn,
        a: S,
        h: f64,
    ) -> S::Matrix<ROWS, COLS>
    where
        TFn: Fn(S) -> S::Matrix<ROWS, COLS>,
    {
        let hh = S::from_f64(h);
        (curve(a + hh) - curve(a - hh)).scaled(S::from_f64(1.0 / (2.0 * h)))
    }
}

#[test]
fn curve_test() {
    #[cfg(feature = "simd")]
    use crate::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;
    use crate::{
        dual::DualScalar,
        linalg::{
            scalar::IsScalar,
            EPS_F64,
        },
    };

    trait CurveTest {
        fn run_curve_test();
    }

    macro_rules! def_curve_test_template {
        ($batch:literal, $scalar: ty, $dual_scalar: ty
    ) => {
            impl CurveTest for $dual_scalar {
                fn run_curve_test() {
                    use crate::linalg::vector::IsVector;

                    for i in 0..10 {
                        let a = <$scalar>::from_f64(0.1 * (i as f64));

                        // f(x) = x^2
                        fn square_fn<
                            S: IsScalar<BATCH, DM, DN>,
                            const BATCH: usize,
                            const DM: usize,
                            const DN: usize,
                        >(
                            x: S,
                        ) -> S {
                            x.clone() * x
                        }
                        let finite_diff = ScalarValuedCurve::<$scalar, $batch>::sym_diff_quotient(
                            square_fn,
                            a.clone(),
                            EPS_F64,
                        );
                        let auto_grad = square_fn(<$dual_scalar>::var(a)).derivative()[0];
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0002);
                    }

                    for i in 0..10 {
                        let a = <$scalar>::from_f64(0.1 * (i as f64));

                        // f(x) = [cos(x), sin(x)]
                        fn trig_fn<
                            S: IsScalar<BATCH, DM, DN>,
                            const BATCH: usize,
                            const DM: usize,
                            const DN: usize,
                        >(
                            x: S,
                        ) -> S::Vector<2> {
                            S::Vector::<2>::from_array([x.clone().cos(), x.sin()])
                        }

                        let finite_diff = VectorValuedCurve::<$scalar, $batch>::sym_diff_quotient(
                            trig_fn,
                            a.clone(),
                            EPS_F64,
                        );
                        let auto_grad = trig_fn(<$dual_scalar>::var(a)).jacobian();

                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0003);
                    }

                    for i in 0..10 {
                        let a = <$scalar>::from_f64(0.1 * (i as f64));

                        // f(x) = [[ cos(x), sin(x), 0],
                        //         [-sin(x), cos(x), 0]]
                        fn fn_x<
                            S: IsScalar<BATCH, DM, DN>,
                            const BATCH: usize,
                            const DM: usize,
                            const DN: usize,
                        >(
                            x: S,
                        ) -> S::Matrix<2, 3> {
                            let sin = x.clone().sin();
                            let cos = x.clone().cos();

                            S::Matrix::from_array2([
                                [cos.clone(), sin.clone(), S::from_f64(0.0)],
                                [-sin, cos, S::from_f64(0.0)],
                            ])
                        }

                        let finite_diff =
                            MatrixValuedCurve::<$scalar, $batch, 0, 0>::sym_diff_quotient(
                                fn_x,
                                a.clone(),
                                EPS_F64,
                            );
                        let auto_grad = fn_x(<$dual_scalar>::var(a)).scalarfield_derivative();
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                    }
                }
            }
        };
    }

    def_curve_test_template!(1, f64, DualScalar<1,1>);
    #[cfg(feature = "simd")]
    def_curve_test_template!(2, BatchScalarF64<2>, DualBatchScalar<2,1,1>);
    #[cfg(feature = "simd")]
    def_curve_test_template!(4, BatchScalarF64<4>, DualBatchScalar<4, 1, 1>);
    #[cfg(feature = "simd")]
    def_curve_test_template!(8, BatchScalarF64<8>, DualBatchScalar<8,1,1>);

    DualScalar::run_curve_test();
    #[cfg(feature = "simd")]
    DualBatchScalar::<2, 1, 1>::run_curve_test();
    #[cfg(feature = "simd")]
    DualBatchScalar::<4, 1, 1>::run_curve_test();
    #[cfg(feature = "simd")]
    DualBatchScalar::<8, 1, 1>::run_curve_test();
}
