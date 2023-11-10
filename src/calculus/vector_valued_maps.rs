use super::{
    scalar_valued_maps::*,
    batch_types::*,
};
use dfdx_core::{shapes::*, tensor::*, tensor_ops::*};

// Vector-valued map on a vector space.
//
// This is a function which takes a vector and returns a vector:
//
//  f: ℝᵐ -> ℝʳ
//
// These functions are also called vector fields (on vector spaces).
//
pub struct VectorValuedMapFromVector;

impl VectorValuedMapFromVector {
    // Finite difference quotient of the vector-valued map.
    //
    // The derivative is a matrix or rank-2 tensor with shape (Rᵢ x Rₒ).
    //
    // Since all operations are batched, the function returns a (B x Rₒ x Rᵢ) rank-3 tensor.
    pub fn sym_diff_quotient<TFn, const BATCH: usize, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: V<BATCH, INROWS>,
        eps: f64,
    ) -> VFromV<BATCH, OUTROWS, INROWS>
    where
        TFn: Fn(V<BATCH, INROWS>) -> V<BATCH, OUTROWS>,
    {
        let dev = a.device();
        let row0: Tensor<(Const<BATCH>, Const<OUTROWS>, Const<0>), f64, _> = dev.zeros();
        let mut rows: Tensor<(Const<BATCH>, Const<OUTROWS>, usize), f64, _> = row0.realize();

        for r in 0..INROWS {
            let mut a_plus = a.clone();
            for b in 0..BATCH {
                a_plus[[b, r]] += eps;
            }
            let mut a_minus = a.clone();
            for b in 0..BATCH {
                a_minus[[b, r]] -= eps;
            }
            let row: Tensor<(Const<BATCH>, Const<OUTROWS>, Const<1>), f64, _> =
                ((vector_valued(a_plus) - vector_valued(a_minus)) / (2.0 * eps)).reshape();
            rows = (rows, row).concat_along(Axis::<2>);
        }
        rows.realize()
    }

    pub fn sym_diff_quotient_from_taped<
        TFn,
        const BATCH: usize,
        const OUTROWS: usize,
        const INROWS: usize,
    >(
        vector_valued: TFn,
        a: V<BATCH, INROWS>,
        eps: f64,
    ) -> VFromV<BATCH, OUTROWS, INROWS>
    where
        TFn: Fn(TapedV<BATCH, INROWS>) -> TapedV<BATCH, OUTROWS>,
    {
        Self::sym_diff_quotient(|x| vector_valued(x.retaped()).split_tape().0, a, eps)
    }

    pub fn auto_grad<TFn, const BATCH: usize, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: V<BATCH, INROWS>,
    ) -> VFromV<BATCH, OUTROWS, INROWS>
    where
        TFn: Fn(TapedV<BATCH, INROWS>) -> TapedV<BATCH, OUTROWS>,
    {
        let outrow0: M<BATCH, 0, INROWS> = a.device().zeros();
        let mut outrows: Tensor<(Const<BATCH>, usize, Const<INROWS>), f64, _> = outrow0.realize();

        for r in 0..OUTROWS {
            let batch: V<BATCH, INROWS> = ScalarValuedMapFromVector::auto_grad(
                |x| {
                    let v = vector_valued(x);
                    let v_b: TapedV<BATCH, 1> = v.slice((.., r..r + 1)).realize();
                    v_b.reshape()
                },
                a.clone(),
            );
            let outrow: M<BATCH, 1, INROWS> = batch.reshape();

            outrows = (outrows, outrow).concat_along(Axis::<1>);
        }
        outrows.realize()
    }
}

// Vector-valued map on a product space (= space of matrices).
//
// This is a function which takes a matrix and returns a vector:
//
//  f: ℝᵐ x ℝⁿ -> ℝʳ
//
// This type of function is also called a vector field (on product spaces).
//
pub struct VectorValuedMapFromMatrix;

impl VectorValuedMapFromMatrix {
    // Finite difference quotient of the vector-valued map.
    //
    // The derivative is a matrix or rank-3 tensor with shape (Rₒ x Rᵢ x Cⱼ).
    //
    // Since all operations are batched, the function returns a (B x Rₒ x Rᵢ x Cᵢ) rank-4 tensor.
    pub fn sym_diff_quotient<
        TFn,
        const BATCH: usize,
        const OUTROWS: usize,
        const INROWS: usize,
        const INCOLS: usize,
    >(
        vector_valued: TFn,
        a: M<BATCH, INROWS, INCOLS>,
        eps: f64,
    ) -> VFromM<BATCH, OUTROWS, INROWS, INCOLS>
    where
        TFn: Fn(M<BATCH, INROWS, INCOLS>) -> V<BATCH, OUTROWS>,
    {
        let dev = a.device();
        let row0: Tensor<(Const<BATCH>, Const<OUTROWS>, Const<0>, Const<INCOLS>), f64, _> =
            dev.zeros();
        let mut rows: Tensor<(Const<BATCH>, Const<OUTROWS>, usize, Const<INCOLS>), f64, _> =
            row0.realize();

        for r in 0..INROWS {
            let col0: Tensor<(Const<BATCH>, Const<OUTROWS>, Const<1>, Const<0>), f64, _> =
                dev.zeros();
            let mut cols: Tensor<(Const<BATCH>, Const<OUTROWS>, Const<1>, usize), f64, _> =
                col0.realize();
            for c in 0..INCOLS {
                let mut a_plus = a.clone();
                for b in 0..BATCH {
                    a_plus[[b, r, c]] += eps;
                }
                let mut a_minus = a.clone();
                for b in 0..BATCH {
                    a_minus[[b, r, c]] -= eps;
                }
                let col: Tensor<(Const<BATCH>, Const<OUTROWS>, Const<1>, Const<1>), f64, _> =
                    ((vector_valued(a_plus) - vector_valued(a_minus)) / (2.0 * eps)).reshape();
                cols = (cols, col).concat_along(Axis::<3>);
            }
            let cols: Tensor<(Const<BATCH>, Const<OUTROWS>, Const<1>, Const<INCOLS>), f64, _> =
                cols.realize();
            rows = (rows, cols).concat_along(Axis::<2>);
        }
        rows.realize()
    }

    pub fn sym_diff_quotient_from_taped<
        TFn,
        const BATCH: usize,
        const OUTROWS: usize,
        const INROWS: usize,
        const INCOLS: usize,
    >(
        vector_valued: TFn,
        a: M<BATCH, INROWS, INCOLS>,
        eps: f64,
    ) -> VFromM<BATCH, OUTROWS, INROWS, INCOLS>
    where
        TFn: Fn(
            TapedM<BATCH, INROWS, INCOLS>,
        ) -> TapedV<BATCH, OUTROWS>,
    {
        Self::sym_diff_quotient(|x| vector_valued(x.retaped()).split_tape().0, a, eps)
    }

    pub fn auto_grad<
        TFn,
        const BATCH: usize,
        const OUTROWS: usize,
        const INROWS: usize,
        const INCOLS: usize,
    >(
        vector_valued: TFn,
        a: M<BATCH, INROWS, INCOLS>,
    ) -> VFromM<BATCH, OUTROWS, INROWS, INCOLS>
    where
        TFn: Fn(TapedM<BATCH, INROWS, INCOLS>) -> TapedV<BATCH, OUTROWS>,
    {
        let outrow0: VFromM<BATCH, 0, INROWS, INCOLS> = a.device().zeros();
        let mut outrows: Tensor<(Const<BATCH>, usize, Const<INROWS>, Const<INCOLS>), f64, _> =
            outrow0.realize();

        for r in 0..OUTROWS {
            let batch: M<BATCH, INROWS, INCOLS> = ScalarValuedMapFromMatrix::auto_grad(
                |x| {
                    let v = vector_valued(x);
                    let v_b: TapedV<BATCH, 1> = v.slice((.., r..r + 1)).realize();
                    v_b.reshape()
                },
                a.clone(),
            );
            let outrow: VFromM<BATCH, 1, INROWS, INCOLS> = batch.reshape();

            outrows = (outrows, outrow).concat_along(Axis::<1>);
        }
        outrows.realize()
    }
}

mod test {

    use crate::*;
    use crate::calculus::batch_types::*;
    use super::*;
    use dfdx_core::tensor::Cpu;

    #[allow(dead_code)]
    fn test_batched_vector_valued_map_from_vector<const BATCH: usize>() {
        let dev = Cpu::default();
        let a = dev.sample_uniform();

        let eps = 1e-6;

        //       [[ x ]]   [[ x / z ]]
        //  proj [[ y ]] = [[       ]]
        //       [[ z ]]   [[ y / z ]]
        let proj_fn = |v: TapedV<BATCH, 3>| -> TapedV<BATCH, 2> {
            let x = v.clone().slice((.., 0..1));
            let y = v.clone().slice((.., 1..2));
            let z = v.clone().slice((.., 2..3));

            (x / z.clone(), y / z).concat_along(Axis::<1>).realize()
        };

        let df = VectorValuedMapFromVector::sym_diff_quotient_from_taped(proj_fn, a.clone(), eps);

        let auto_grad = VectorValuedMapFromVector::auto_grad(proj_fn, a.clone());

        assert_tensors_relative_eq_rank3!(df, auto_grad, 1e-3);
    }

    #[test]
    fn test_vector_valued_map_from_vector() {
        test_batched_vector_valued_map_from_vector::<1>();
        test_batched_vector_valued_map_from_vector::<2>();
        test_batched_vector_valued_map_from_vector::<4>();
    }

    #[allow(dead_code)]
    fn test_batched_vector_valued_map_from_matrix<const BATCH: usize>() {
        let dev = Cpu::default();

        let eps = 1e-6;
        //       [[ a  b ]]   [[ a + b ]]
        //  proj [[ c  d ]] = [[ c + d ]]
        //       [[ e  f ]]   [[ e + f ]]
        //                    [[   1   ]]
        let f = |x: TapedM<BATCH, 3, 2>| -> TapedV<BATCH, 4> {
            let summed: TapedV<BATCH, 3> = x.sum::<_, Axis<2>>();
            let one: V<BATCH, 1> = summed.device().ones();
            (summed, one).concat_along(Axis::<1>).realize()
        };

        let a = dev.sample_uniform();
        let df = VectorValuedMapFromMatrix::sym_diff_quotient_from_taped(f, a.clone(), eps);
        let auto_grad = VectorValuedMapFromMatrix::auto_grad(f, a.clone());

        assert_tensors_relative_eq_rank4!(df, auto_grad, 1e-6);
    }

    #[test]
    fn test_vector_valued_map_from_matrix() {
        test_batched_vector_valued_map_from_matrix::<1>();
        test_batched_vector_valued_map_from_matrix::<2>();
        test_batched_vector_valued_map_from_matrix::<4>();
    }
}
