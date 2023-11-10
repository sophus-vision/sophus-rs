use super::batch_types::*;
use dfdx_core::prelude::*;
// A smooth curve in ℝ.
//
// This is a function which takes a scalar and returns a scalar:
//
//  f: ℝ -> ℝ
pub struct ScalarValuedCurve;

impl ScalarValuedCurve {
    // Finite difference quotient of the scalar-valued curve.
    //
    // The derivative is also a scalar.
    //
    // Since all operations are batched, the function returns a vector of scalars, i.e. a rank-1
    // tensor.
    pub fn sym_diff_quotient<TFn, const BATCH: usize>(curve: TFn, a: S<BATCH>, h: f64) -> S<BATCH>
    where
        TFn: Fn(S<BATCH>) -> S<BATCH>,
    {
        (curve(a.clone() + h) - curve(a - h)) / (2.0 * h)
    }

    pub fn sym_diff_quotient_from_taped<TFn, const BATCH: usize>(
        curve: TFn,
        a: S<BATCH>,
        h: f64,
    ) -> S<BATCH>
    where
        TFn: Fn(TapedS<BATCH>) -> TapedS<BATCH>,
    {
        Self::sym_diff_quotient(|x| curve(x.retaped()).split_tape().0, a, h)
    }

    pub fn auto_grad<TFn, const BATCH: usize>(curve: TFn, a: S<BATCH>) -> S<BATCH>
    where
        TFn: Fn(TapedS<BATCH>) -> TapedS<BATCH>,
    {
        let mut v: S<BATCH> = a.device().zeros();

        for b in 0..BATCH {
            let (a_prime, tape) = a.leaky_trace().split_tape();
            let a_prime = a_prime.put_tape(std::sync::Arc::new(std::sync::Mutex::new(tape)));
            let out = curve(a_prime);
            let o_b = out.select(Cpu::default().tensor(b));
            let g = o_b.backward();
            v[[b]] = g.get(&a)[[b]];
        }
        v
    }
}

// A smooth curve in ℝʳ.
//
// This is a function which takes a scalar and returns a vector:
//
//   f: ℝ -> ℝʳ
pub struct VectorValuedCurve;

impl VectorValuedCurve {
    // Finite difference quotient of the vector-valued curve.
    //
    // The derivative is also a vector.
    //
    // Since all operations are batched, the function returns a vector of vector, i.e. a rank-2
    // tensor.
    pub fn sym_diff_quotient<TFn, const BATCH: usize, const ROWS: usize>(
        curve: TFn,
        a: S<BATCH>,
        h: f64,
    ) -> V<BATCH, ROWS>
    where
        TFn: Fn(S<BATCH>) -> V<BATCH, ROWS>,
    {
        (curve(a.clone() + h) - curve(a - h)) / (2.0 * h)
    }

    pub fn sym_diff_quotient_from_taped<TFn, const BATCH: usize, const ROWS: usize>(
        curve: TFn,
        a: S<BATCH>,
        h: f64,
    ) -> V<BATCH, ROWS>
    where
        TFn: Fn(TapedS<BATCH>) -> TapedV<BATCH, ROWS>,
    {
        Self::sym_diff_quotient(|x| curve(x.retaped()).split_tape().0, a, h)
    }

    pub fn auto_grad<TFn, const BATCH: usize, const INROWS: usize>(
        curve: TFn,
        a: S<BATCH>,
    ) -> V<BATCH, INROWS>
    where
        TFn: Fn(TapedS<BATCH>) -> TapedV<BATCH, INROWS>,
    {
        let v: V<BATCH, 0> = a.device().zeros();
        let mut v: Tensor<(Const<BATCH>, usize), f64, _> = v.realize();

        for r in 0..INROWS {
            let batch: S<BATCH> = ScalarValuedCurve::auto_grad(
                |s| {
                    let v = curve(s);
                    let v_r: TapedV<BATCH, 1> = v.slice((.., r..r + 1)).realize();
                    v_r.reshape()
                },
                a.clone(),
            );
            let outrow: V<BATCH, 1> = batch.reshape();
            v = (v, outrow).concat_along(Axis::<1>);
        }
        v.realize()
    }
}

// A smooth curve in ℝʳ x ℝᶜ.
//
// This is a function which takes a scalar and returns a matrix:
//   f: ℝ -> ℝʳ x ℝᶜ
pub struct MatrixValuedCurve;

impl MatrixValuedCurve {
    // Finite difference quotient of the matrix-valued curve.
    //
    // The derivative is also a matrix.
    //
    // Since all operations are batched, the function returns a vector of matrices, i.e. a rank-3
    // tensor.
    pub fn sym_diff_quotient<TFn, const BATCH: usize, const ROWS: usize, const COLS: usize>(
        curve: TFn,
        a: S<BATCH>,
        h: f64,
    ) -> M<BATCH, ROWS, COLS>
    where
        TFn: Fn(S<BATCH>) -> M<BATCH, ROWS, COLS>,
    {
        (curve(a.clone() + h) - curve(a - h)) / (2.0 * h)
    }

    pub fn sym_diff_quotient_from_taped<
        TFn,
        const BATCH: usize,
        const ROWS: usize,
        const COLS: usize,
    >(
        curve: TFn,
        a: S<BATCH>,
        h: f64,
    ) -> M<BATCH, ROWS, COLS>
    where
        TFn: Fn(TapedS<BATCH>) -> TapedM<BATCH, ROWS, COLS>,
    {
        Self::sym_diff_quotient(|x| curve(x.retaped()).split_tape().0, a, h)
    }

    pub fn auto_grad<TFn, const BATCH: usize, const INROWS: usize, const INCOLS: usize>(
        curve: TFn,
        a: S<BATCH>,
    ) -> M<BATCH, INROWS, INCOLS>
    where
        TFn: Fn(TapedS<BATCH>) -> TapedM<BATCH, INROWS, INCOLS>,
    {
        let m: M<BATCH, INROWS, 0> = a.device().zeros();
        let mut m: Tensor<(Const<BATCH>, Const<INROWS>, usize), f64, _> = m.realize();

        for c in 0..INCOLS {
            let batch: V<BATCH, INROWS> = VectorValuedCurve::auto_grad(
                |v| {
                    let m = curve(v);
                    let m_c: TapedM<BATCH, INROWS, 1> = m.slice((.., .., c..c + 1)).realize();
                    m_c.reshape()
                },
                a.clone(),
            );
            let outcols: M<BATCH, INROWS, 1> = batch.reshape();
            m = (m, outcols).concat_along(Axis::<2>);
        }
        m.realize()
    }
}

mod test {

    #[test]
    fn scalar_valued() {
        use super::ScalarValuedCurve;
        use crate::{assert_tensors_relative_eq_rank1, calculus::curves::*};

        let dev = dfdx_core::tensor::Cpu::default();

        const BATCH: usize = 4;

        for i in 0..10 {
            let mut a: S<BATCH> = dev.ones() * (i as f64);

            for b in 0..BATCH {
                a[[b]] += (b as f64) * 0.1;
            }

            // f(x) = x^2
            let fn_x = |x: TapedS<BATCH>| -> TapedS<BATCH> { x.square() };

            let auto_grad = ScalarValuedCurve::auto_grad(fn_x, a.clone());
            let finite_diff =
                ScalarValuedCurve::sym_diff_quotient_from_taped(fn_x, a.clone(), 1e-6);

            assert_tensors_relative_eq_rank1!(auto_grad, finite_diff.clone(), 0.0001);
        }
    }

    #[test]
    fn vector_valued() {
        use super::VectorValuedCurve;
        use crate::calculus::make::*;
        use crate::{assert_tensors_relative_eq_rank2, calculus::curves::*};

        let dev = dfdx_core::tensor::Cpu::default();

        const BATCH: usize = 4;

        for i in 0..10 {
            let mut a: S<BATCH> = dev.ones() * (i as f64);

            for b in 0..BATCH {
                a[[b]] += (b as f64) * 0.1;
            }

            // f(x) = [cos(x), sin(x)]
            let fn_x =
                |x: TapedS<BATCH>| -> TapedV<BATCH, 2> { make_vec2(x.clone().cos(), x.sin()) };

            let auto_grad = VectorValuedCurve::auto_grad(fn_x, a.clone());
            let finite_diff =
                VectorValuedCurve::sym_diff_quotient_from_taped(fn_x, a.clone(), 1e-6);

            assert_tensors_relative_eq_rank2!(auto_grad, finite_diff, 0.0001);
        }
    }

    #[test]
    fn matrix_valued() {
        use super::MatrixValuedCurve;
        use crate::calculus::make::*;
        use crate::{assert_tensors_relative_eq_rank3, calculus::curves::*};

        let dev = dfdx_core::tensor::Cpu::default();

        const BATCH: usize = 4;

        for i in 0..10 {
            let mut a: S<BATCH> = dev.ones() * (i as f64);

            for b in 0..BATCH {
                a[[b]] += (b as f64) * 0.1;
            }

            // f(x) = [[ cos(x), sin(x), 0],
            //         [-sin(x), cos(x), 0]]
            let fn_x = |x: TapedS<BATCH>| -> TapedM<BATCH, 2, 3> {
                let zero: S<BATCH> = x.device().zeros();
                let sin: TapedS<BATCH> = x.clone().sin();
                let cos: TapedS<BATCH> = x.clone().cos();

                make_2rowblock_mat(
                    make_3col_mat(cos.clone(), sin.clone(), zero.retaped()),
                    make_3col_mat(-sin, cos, zero.retaped()),
                )
            };

            let auto_grad = MatrixValuedCurve::auto_grad(fn_x, a.clone());
            let finite_diff =
                MatrixValuedCurve::sym_diff_quotient_from_taped(fn_x, a.clone(), 1e-6);

            // You need to update the macro name and define one for rank 3 tensors
            assert_tensors_relative_eq_rank3!(auto_grad, finite_diff, 0.0001);
        }
    }
}
