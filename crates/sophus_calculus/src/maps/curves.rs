use crate::dual::dual_matrix::DualM;
use crate::dual::dual_scalar::Dual;
use crate::dual::dual_vector::DualV;
use crate::types::MatF64;
use crate::types::VecF64;

use sophus_tensor::view::IsTensorLike;

/// A smooth curve in ℝ.
///
/// This is a function which takes a scalar and returns a scalar:
///
///  f: ℝ -> ℝ
pub struct ScalarValuedCurve;

impl ScalarValuedCurve {
    /// Finite difference quotient of the scalar-valued curve.
    ///
    /// The derivative is also a scalar.
    ///
    /// Since all operations are batched, the function returns a vector of scalars, i.e. a rank-1
    /// tensor.
    pub fn sym_diff_quotient<TFn>(curve: TFn, a: f64, h: f64) -> f64
    where
        TFn: Fn(f64) -> f64,
    {
        (curve(a + h) - curve(a - h)) / (2.0 * h)
    }

    /// Auto differentiation of the scalar-valued curve.
    pub fn fw_autodiff<TFn>(curve: TFn, a: f64) -> f64
    where
        TFn: Fn(Dual) -> Dual,
    {
        curve(Dual::v(a)).dij_val.unwrap().get([0, 0])
    }
}

/// A smooth curve in ℝʳ.
///
/// This is a function which takes a scalar and returns a vector:
///
///   f: ℝ -> ℝʳ
pub struct VectorValuedCurve;

impl VectorValuedCurve {
    /// Finite difference quotient of the vector-valued curve.
    ///
    /// The derivative is also a vector.
    ///
    /// Since all operations are batched, the function returns a vector of vector, i.e. a rank-2
    /// tensor.
    pub fn sym_diff_quotient<TFn, const ROWS: usize>(curve: TFn, a: f64, h: f64) -> VecF64<ROWS>
    where
        TFn: Fn(f64) -> VecF64<ROWS>,
    {
        (curve(a + h) - curve(a - h)) / (2.0 * h)
    }

    /// Auto differentiation of the vector-valued curve.
    pub fn fw_autodiff<TFn, const ROWS: usize>(curve: TFn, a: f64) -> VecF64<ROWS>
    where
        TFn: Fn(Dual) -> DualV<ROWS>,
    {
        curve(Dual::v(a)).dij_val.unwrap().get([0, 0])
    }
}

/// A smooth curve in ℝʳ x ℝᶜ.
///
/// This is a function which takes a scalar and returns a matrix:
///   f: ℝ -> ℝʳ x ℝᶜ
pub struct MatrixValuedCurve;

impl MatrixValuedCurve {
    /// Finite difference quotient of the matrix-valued curve.
    ///
    /// The derivative is also a matrix.
    ///
    /// Since all operations are batched, the function returns a vector of matrices, i.e. a rank-3
    /// tensor.
    pub fn sym_diff_quotient<TFn, const ROWS: usize, const COLS: usize>(
        curve: TFn,
        a: f64,
        h: f64,
    ) -> MatF64<ROWS, COLS>
    where
        TFn: Fn(f64) -> MatF64<ROWS, COLS>,
    {
        (curve(a + h) - curve(a - h)) / (2.0 * h)
    }

    /// Auto differentiation of the matrix-valued curve.
    pub fn fw_autodiff<TFn, const ROWS: usize, const COLS: usize>(
        curve: TFn,
        a: f64,
    ) -> MatF64<ROWS, COLS>
    where
        TFn: Fn(Dual) -> DualM<ROWS, COLS>,
    {
        curve(Dual::v(a)).dij_val.unwrap().get([0, 0])
    }
}

mod test {
    #[cfg(test)]
    use crate::types::matrix::IsMatrix;
    #[cfg(test)]
    use crate::types::scalar::IsScalar;
    #[cfg(test)]
    use crate::types::vector::IsVector;

    #[test]
    fn scalar_valued() {
        use super::ScalarValuedCurve;

        for i in 0..10 {
            let a = 0.1 * (i as f64);

            // f(x) = x^2
            fn square_fn<S: IsScalar<1>>(x: S) -> S {
                x.clone() * x
            }
            let finite_diff = ScalarValuedCurve::sym_diff_quotient(square_fn, a, 1e-6);
            let auto_grad = ScalarValuedCurve::fw_autodiff(square_fn, a);
            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
        }
    }

    #[test]
    fn vector_valued() {
        use super::VectorValuedCurve;

        for i in 0..10 {
            let a = 0.1 * (i as f64);

            // f(x) = [cos(x), sin(x)]
            fn trig_fn<S: IsScalar<1>, V2: IsVector<S, 2, 1>>(x: S) -> V2 {
                V2::from_array([x.clone().cos(), x.sin()])
            }

            let finite_diff = VectorValuedCurve::sym_diff_quotient(trig_fn, a, 1e-6);
            let auto_grad = VectorValuedCurve::fw_autodiff(trig_fn, a);
            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
        }
    }

    #[test]
    fn matrix_valued() {
        use super::MatrixValuedCurve;

        for i in 0..10 {
            let a = 0.1 * (i as f64);

            // f(x) = [[ cos(x), sin(x), 0],
            //         [-sin(x), cos(x), 0]]
            fn fn_x<S: IsScalar<1>, M23: IsMatrix<S, 2, 3, 1>>(x: S) -> M23 {
                let sin = x.clone().sin();
                let cos = x.clone().cos();

                M23::from_array2([
                    [cos.clone(), sin.clone(), S::c(0.0)],
                    [-sin, cos, S::c(0.0)],
                ])
            }

            let finite_diff = MatrixValuedCurve::sym_diff_quotient(fn_x, a, 1e-6);
            let auto_grad = MatrixValuedCurve::fw_autodiff(fn_x, a);
            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
        }
    }
}
