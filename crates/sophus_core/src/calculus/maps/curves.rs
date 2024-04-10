use crate::linalg::SMat;
use crate::prelude::*;
use nalgebra::SVector;

/// A smooth curve in ℝ.
///
/// This is a function which takes a scalar and returns a scalar:
///
///  f: ℝ -> ℝ
pub struct ScalarValuedCurve<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: std::marker::PhantomData<S>,
}

impl<S: IsScalar<BATCH>, const BATCH: usize> ScalarValuedCurve<S, BATCH> {
    /// Finite difference quotient of the scalar-valued curve.
    ///
    /// The derivative is also a scalar.
    pub fn sym_diff_quotient<TFn>(curve: TFn, a: S, h: f64) -> S
    where
        TFn: Fn(S) -> S,
    {
        let hh = S::from_f64(h);
        (curve(a.clone() + hh.clone()) - curve(a - hh)) / S::from_f64(2.0 * h)
    }
}

impl<D: IsDualScalar<BATCH>, const BATCH: usize> ScalarValuedCurve<D, BATCH> {
    /// Auto differentiation of the scalar-valued curve.
    pub fn fw_autodiff<TFn>(curve: TFn, a: D::RealScalar) -> D::RealScalar
    where
        TFn: Fn(D) -> D,
    {
        curve(D::new_with_dij(a))
            .dij_val()
            .clone()
            .unwrap()
            .get([0, 0])
    }
}

/// A smooth curve in ℝʳ.
///
/// This is a function which takes a scalar and returns a vector:
///
///   f: ℝ -> ℝʳ
pub struct VectorValuedCurve<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: std::marker::PhantomData<S>,
}

impl<S: IsScalar<BATCH>, const BATCH: usize> VectorValuedCurve<S, BATCH> {
    /// Finite difference quotient of the vector-valued curve.
    ///
    /// The derivative is also a vector.
    pub fn sym_diff_quotient<TFn, const ROWS: usize>(curve: TFn, a: S, h: f64) -> S::Vector<ROWS>
    where
        TFn: Fn(S) -> S::Vector<ROWS>,
    {
        let hh = S::from_f64(h);
        (curve(a.clone() + hh.clone()) - curve(a - hh)).scaled(S::from_f64(1.0 / (2.0 * h)))
    }
}

impl<D: IsDualScalar<BATCH>, const BATCH: usize> VectorValuedCurve<D, BATCH> {
    /// Auto differentiation of the vector-valued curve.
    pub fn fw_autodiff<TFn, const ROWS: usize>(
        curve: TFn,
        a: D::RealScalar,
    ) -> SVector<D::RealScalar, ROWS>
    where
        TFn: Fn(D) -> D::Vector<ROWS>,
        D::Vector<ROWS>: IsDualVector<D, ROWS, BATCH>,
    {
        curve(D::new_with_dij(a)).dij_val().unwrap().get([0, 0])
    }
}

/// A smooth curve in ℝʳ x ℝᶜ.
///
/// This is a function which takes a scalar and returns a matrix:
///   f: ℝ -> ℝʳ x ℝᶜ
pub struct MatrixValuedCurve<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: std::marker::PhantomData<S>,
}

impl<S: IsScalar<BATCH>, const BATCH: usize> MatrixValuedCurve<S, BATCH> {
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
        (curve(a.clone() + hh.clone()) - curve(a - hh)).scaled(S::from_f64(1.0 / (2.0 * h)))
    }
}

impl<D: IsDualScalar<BATCH>, const BATCH: usize> MatrixValuedCurve<D, BATCH> {
    /// Auto differentiation of the matrix-valued curve.
    pub fn fw_autodiff<TFn, const ROWS: usize, const COLS: usize>(
        curve: TFn,
        a: D::RealScalar,
    ) -> SMat<<D as IsScalar<BATCH>>::RealScalar, ROWS, COLS>
    where
        TFn: Fn(D) -> D::Matrix<ROWS, COLS>,
        D::Matrix<ROWS, COLS>: IsDualMatrix<D, ROWS, COLS, BATCH>,
    {
        curve(D::new_with_dij(a)).dij_val().unwrap().get([0, 0])
    }
}

#[test]
fn curve_test() {
    use crate::calculus::dual::dual_scalar::DualBatchScalar;
    use crate::calculus::dual::dual_scalar::DualScalar;
    use crate::linalg::scalar::IsScalar;
    use crate::linalg::BatchScalarF64;

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
                        fn square_fn<S: IsScalar<BATCH>, const BATCH: usize>(x: S) -> S {
                            x.clone() * x
                        }
                        let finite_diff = ScalarValuedCurve::<$scalar, $batch>::sym_diff_quotient(
                            square_fn,
                            a.clone(),
                            1e-6,
                        );
                        let auto_grad =
                            ScalarValuedCurve::<$dual_scalar, $batch>::fw_autodiff(square_fn, a);
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                    }

                    for i in 0..10 {
                        let a = <$scalar>::from_f64(0.1 * (i as f64));

                        // f(x) = [cos(x), sin(x)]
                        fn trig_fn<S: IsScalar<BATCH>, const BATCH: usize>(x: S) -> S::Vector<2> {
                            S::Vector::<2>::from_array([x.clone().cos(), x.sin()])
                        }

                        let finite_diff = VectorValuedCurve::<$scalar, $batch>::sym_diff_quotient(
                            trig_fn,
                            a.clone(),
                            1e-6,
                        );
                        let auto_grad =
                            VectorValuedCurve::<$dual_scalar, $batch>::fw_autodiff(trig_fn, a);
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                    }

                    for i in 0..10 {
                        let a = <$scalar>::from_f64(0.1 * (i as f64));

                        // f(x) = [[ cos(x), sin(x), 0],
                        //         [-sin(x), cos(x), 0]]
                        fn fn_x<S: IsScalar<BATCH>, const BATCH: usize>(x: S) -> S::Matrix<2, 3> {
                            let sin = x.clone().sin();
                            let cos = x.clone().cos();

                            S::Matrix::from_array2([
                                [cos.clone(), sin.clone(), S::from_f64(0.0)],
                                [-sin, cos, S::from_f64(0.0)],
                            ])
                        }

                        let finite_diff = MatrixValuedCurve::<$scalar, $batch>::sym_diff_quotient(
                            fn_x,
                            a.clone(),
                            1e-6,
                        );
                        let auto_grad =
                            MatrixValuedCurve::<$dual_scalar, $batch>::fw_autodiff(fn_x, a);
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                    }
                }
            }
        };
    }

    def_curve_test_template!(1, f64, DualScalar);
    def_curve_test_template!(2, BatchScalarF64<2>, DualBatchScalar<2>);
    def_curve_test_template!(4, BatchScalarF64<4>, DualBatchScalar<4>);
    def_curve_test_template!(8, BatchScalarF64<8>, DualBatchScalar<8>);

    DualScalar::run_curve_test();
    DualBatchScalar::<2>::run_curve_test();
    DualBatchScalar::<4>::run_curve_test();
    DualBatchScalar::<8>::run_curve_test();
}
