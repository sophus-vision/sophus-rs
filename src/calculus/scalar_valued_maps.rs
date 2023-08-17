use super::batch_types::*;
use dfdx::{shapes::*, tensor::*, tensor_ops::*};

// Scalar-valued map on a vector space.
//
// This is a function which takes a vector and returns a scalar:
//
//   f: ℝᵐ -> ℝ
//
// These functions are also called a scalar fields (on vector spaces).
//
pub struct ScalarValuedMapFromVector;

impl ScalarValuedMapFromVector {
    // Finite difference quotient of the scalar-valued map.
    //
    // The derivative is a vector or rank-1 tensor of shape (Rᵢ).
    //
    // Since all operations are batched, it returns a (B x Rᵢ) rank-2 tensor.
    pub fn sym_diff_quotient<TFn, const BATCH: usize, const INROWS: usize>(
        scalar_valued: TFn,
        a: V<BATCH, INROWS>,
        eps: f64,
    ) -> V<BATCH, INROWS>
    where
        TFn: Fn(V<BATCH, INROWS>) -> S<BATCH>,
    {
        let dev = a.device();
        let row0: Tensor<(Const<BATCH>, Const<0>), f64, _> = dev.zeros();
        let mut rows: Tensor<(Const<BATCH>, usize), f64, _> = row0.realize();

        for r in 0..INROWS {
            let mut a_plus = a.clone();
            for b in 0..BATCH {
                a_plus[[b, r]] += eps;
            }
            let mut a_minus = a.clone();
            for b in 0..BATCH {
                a_minus[[b, r]] -= eps;
            }
            let row: Tensor<(Const<BATCH>, Const<1>), f64, _> =
                ((scalar_valued(a_plus) - scalar_valued(a_minus)) / (2.0 * eps)).reshape();
            rows = (rows, row).concat_along(Axis::<1>);
        }
        rows.realize()
    }

    pub fn sym_diff_quotient_from_taped<TFn, const BATCH: usize, const INROWS: usize>(
        scalar_valued: TFn,
        a: V<BATCH, INROWS>,
        eps: f64,
    ) -> V<BATCH, INROWS>
    where
        TFn: Fn(TapedV<BATCH, INROWS>) -> TapedS<BATCH>,
    {
        Self::sym_diff_quotient(|x| scalar_valued(x.retaped()).split_tape().0, a, eps)
    }

    pub fn auto_grad<TFn, const BATCH: usize, const INROWS: usize>(
        scalar_valued: TFn,
        a: V<BATCH, INROWS>,
    ) -> V<BATCH, INROWS>
    where
        TFn: Fn(TapedV<BATCH, INROWS>) -> TapedS<BATCH>,
    {
        let batch0: V<0, INROWS> = a.device().zeros();
        let mut batches: Tensor<(usize, Const<INROWS>), f64, _> = batch0.realize();

        for b in 0..BATCH {
            let (a_prime, tape) = a.leaky_trace().split_tape();
            let a_prime = a_prime.put_tape(std::sync::Arc::new(std::sync::Mutex::new(tape)));
            let out = scalar_valued(a_prime);
            let o_b = out.select(Cpu::default().tensor(b));
            let g = o_b.backward();
            let gv = g.get(&a);
            let batch: V<1, INROWS> = gv.select(Cpu::default().tensor(b)).reshape();

            batches = (batches, batch).concat_along(Axis::<0>);
        }
        batches.realize()
    }
}

// Scalar-valued map on a product space (= space of matrices).
//
// This is a function which takes a matrix and returns a scalar:
//
//   f: ℝᵐ x ℝⁿ -> ℝ
//
// These functions are also called a scalar fields (on product spaces).
//
pub struct ScalarValuedMapFromMatrix;

impl ScalarValuedMapFromMatrix {
    // Finite difference quotient of the scalar-valued map.
    //
    // The derivative is a matrix or rank-2 tensor of shape (Rᵢ x Cⱼ).
    //
    // Since all operations are batched, this function returns a (B x Rᵢ x Cᵢ) rank-3 tensor.
    pub fn sym_diff_quotient<TFn, const BATCH: usize, const INROWS: usize, const INCOLS: usize>(
        scalar_valued: TFn,
        a: M<BATCH, INROWS, INCOLS>,
        eps: f64,
    ) -> M<BATCH, INROWS, INCOLS>
    where
        TFn: Fn(M<BATCH, INROWS, INCOLS>) -> S<BATCH>,
    {
        let dev = a.device();
        let row0: Tensor<(Const<BATCH>, Const<0>, Const<INCOLS>), f64, _> = dev.zeros();
        let mut rows: Tensor<(Const<BATCH>, usize, Const<INCOLS>), f64, _> = row0.realize();

        for r in 0..INROWS {
            let col0: Tensor<(Const<BATCH>, Const<1>, Const<0>), f64, _> = dev.zeros();
            let mut cols: Tensor<(Const<BATCH>, Const<1>, usize), f64, _> = col0.realize();
            for c in 0..INCOLS {
                let mut a_plus = a.clone();
                for b in 0..BATCH {
                    a_plus[[b, r, c]] += eps;
                }
                let mut a_minus = a.clone();
                for b in 0..BATCH {
                    a_minus[[b, r, c]] -= eps;
                }
                let col: Tensor<(Const<BATCH>, Const<1>, Const<1>), f64, _> =
                    ((scalar_valued(a_plus) - scalar_valued(a_minus)) / (2.0 * eps)).reshape();
                cols = (cols, col).concat_along(Axis::<2>);
            }
            let cols: Tensor<(Const<BATCH>, Const<1>, Const<INCOLS>), f64, _> = cols.realize();
            rows = (rows, cols).concat_along(Axis::<1>);
        }
        rows.realize()
    }

    pub fn sym_diff_quotient_from_taped<
        TFn,
        const BATCH: usize,
        const INROWS: usize,
        const INCOLS: usize,
    >(
        scalar_valued: TFn,
        a: M<BATCH, INROWS, INCOLS>,
        eps: f64,
    ) -> M<BATCH, INROWS, INCOLS>
    where
        TFn: Fn(TapedM<BATCH, INROWS, INCOLS>) -> TapedS<BATCH>,
    {
        Self::sym_diff_quotient(|x| scalar_valued(x.retaped()).split_tape().0, a, eps)
    }

    pub fn auto_grad<TFn, const BATCH: usize, const INROWS: usize, const INCOLS: usize>(
        scalar_valued: TFn,
        a: M<BATCH, INROWS, INCOLS>,
    ) -> M<BATCH, INROWS, INCOLS>
    where
        TFn: Fn(TapedM<BATCH, INROWS, INCOLS>) -> TapedS<BATCH>,
    {
        let batch0: M<0, INROWS, INCOLS> = a.device().zeros();
        let mut batches: Tensor<(usize, Const<INROWS>, Const<INCOLS>), f64, _> = batch0.realize();

        for b in 0..BATCH {
            let (a_prime, tape) = a.leaky_trace().split_tape();
            let a_prime = a_prime.put_tape(std::sync::Arc::new(std::sync::Mutex::new(tape)));
            let out = scalar_valued(a_prime);
            let o_b = out.select(Cpu::default().tensor(b));
            let g = o_b.backward();
            let gv = g.get(&a);
            let batch: M<1, INROWS, INCOLS> = gv.select(Cpu::default().tensor(b)).reshape();

            batches = (batches, batch).concat_along(Axis::<0>);
        }
        batches.realize()
    }
}

mod test {
    use crate::*;
    use crate::calculus::make::*;

    use super::*;
    use dfdx::tensor::Cpu;

    #[allow(dead_code)]
    fn test_batched_scalar_valued_map_from_vector<const BATCH: usize>() {
        let dev = Cpu::default();
        let mut a = dev.ones();

        for b in 0..BATCH {
            a[[b, 0]] += b as f64;
        }
        let eps = 1e-6;
        let f = |x: TapedV<BATCH, 2>| -> TapedS<BATCH> { x.square().sum().sqrt() };
        let df = ScalarValuedMapFromVector::sym_diff_quotient_from_taped(f, a.clone(), eps);

        let auto_grad = ScalarValuedMapFromVector::auto_grad(f, a.clone());

        let nrm = f(a.retaped()).split_tape().0;

        let analytic: V<BATCH, 2> = a / nrm.broadcast();
        assert_tensors_relative_eq_rank2!(df, analytic, 1e-6);
        assert_tensors_relative_eq_rank2!(df, auto_grad, 1e-6);
    }

    #[test]
    fn test_scalar_valued_map_from_vector() {
        test_batched_scalar_valued_map_from_vector::<1>();
        test_batched_scalar_valued_map_from_vector::<2>();
        test_batched_scalar_valued_map_from_vector::<4>();
    }

    #[allow(dead_code)]
    fn test_batched_scalar_valued_map_from_matrix<const BATCH: usize>() {
        let dev = Cpu::default();

        let eps = 1e-6;

        //      [[ a,   b ]]
        //  det [[ c,   d ]] = ad - bc
        //      [[ e,   f ]]
        let determinant_fn = |mat: TapedM<BATCH, 3, 2>| -> TapedS<BATCH> {
            // let (m, tape) = mat.split_tape();

            let a = mat.clone().slice((.., 0..1, 0..1));
            let b = mat.clone().slice((.., 0..1, 1..2));

            let c = mat.clone().slice((.., 1..2, 0..1));
            let d = mat.clone().slice((.., 1..2, 1..2));

            let r: TapedM<BATCH, 1, 1> = (a * d - b * c).realize();
            r.reshape()
        };

        let mat = dev.sample_uniform();

        let df = ScalarValuedMapFromMatrix::sym_diff_quotient_from_taped(
            determinant_fn,
            mat.clone(),
            eps,
        );
        let auto_grad = ScalarValuedMapFromMatrix::auto_grad(determinant_fn, mat.clone());

        //            [[ a   b ]]   [[  d  -c ]]
        //  adjunct_t [[ c   d ]] = [[ -b   a ]]
        //            [[ e   f ]]   [[  0   0 ]]
        let adjunct_t = |mat: M<BATCH, 3, 2>| -> M<BATCH, 3, 2> {
            let a: M<BATCH, 1, 1> = mat.clone().slice((.., 0..1, 0..1)).realize();
            let b: M<BATCH, 1, 1> = mat.clone().slice((.., 0..1, 1..2)).realize();

            let c: M<BATCH, 1, 1> = mat.clone().slice((.., 1..2, 0..1)).realize();
            let d: M<BATCH, 1, 1> = mat.clone().slice((.., 1..2, 1..2)).realize();

            let zero: M<BATCH, 1, 2> = a.device().zeros();

            make_3rowblock_mat(
                make_2blockcol_mat(d, -c), //
                make_2blockcol_mat(-b, a),
                zero,
            )
        };
        let analytic = adjunct_t(mat);

        assert_tensors_relative_eq_rank3!(df, analytic, 1e-6);
        assert_tensors_relative_eq_rank3!(df, auto_grad, 1e-6);
    }

    #[test]
    fn test_scalar_valued_map_from_matrix() {
        test_batched_scalar_valued_map_from_matrix::<1>();
        test_batched_scalar_valued_map_from_matrix::<2>();
        test_batched_scalar_valued_map_from_matrix::<4>();
    }
}
