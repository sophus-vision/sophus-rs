use super::{
    make::*,
    vector_valued_maps::*, batch_types::*,
};
use dfdx_core::{shapes::*, tensor::*, tensor_ops::*};

// Matrix-valued map on a vector space.
//
// This is a function which takes a vector and returns a matrix:
//
//  f: ℝᵐ -> ℝʳ x ℝᶜ
//
pub struct MatrixValuedMapFromVector;

impl MatrixValuedMapFromVector {
    // Finite difference quotient of the matrix-valued map.
    //
    // The derivative is a rank-3 tensor with shape (Rᵢ x Rₒ x Cₒ).
    //
    // Since all operations are batched, the function returns a (B x Rᵢ x Rₒ x Cₒ) rank-4 tensor.
    pub fn sym_diff_quotient<
        TFn,
        const BATCH: usize,
        const OUTROWS: usize,
        const OUTCOLS: usize,
        const INROWS: usize,
    >(
        vector_field: TFn,
        a: V<BATCH, INROWS>,
        eps: f64,
    ) -> MFromV<BATCH, OUTROWS, OUTCOLS, INROWS>
    where
        TFn: Fn(V<BATCH, INROWS>) -> M<BATCH, OUTROWS, OUTCOLS>,
    {
        let dev = a.device();
        let row0: Tensor<(Const<BATCH>, Const<OUTROWS>, Const<OUTCOLS>, Const<0>), f64, _> =
            dev.zeros();
        let mut rows: Tensor<(Const<BATCH>, Const<OUTROWS>, Const<OUTCOLS>, usize), f64, _> =
            row0.realize();

        for i1 in 0..INROWS {
            let mut a_plus = a.clone();
            for b in 0..BATCH {
                a_plus[[b, i1]] += eps;
            }
            let mut a_minus = a.clone();
            for b in 0..BATCH {
                a_minus[[b, i1]] -= eps;
            }
            let row: Tensor<(Const<BATCH>, Const<OUTROWS>, Const<OUTCOLS>, Const<1>), f64, _> =
                ((vector_field(a_plus) - vector_field(a_minus)) / (2.0 * eps)).reshape();
            rows = (rows, row).concat_along(Axis::<3>);
        }
        rows.realize::<Rank4<BATCH, OUTROWS, OUTCOLS, INROWS>>()
    }

    pub fn sym_diff_quotient_from_taped<
        TFn,
        const BATCH: usize,
        const OUTROWS: usize,
        const OUTCOLS: usize,
        const INROWS: usize,
    >(
        matrix_valued: TFn,
        a: V<BATCH, INROWS>,
        eps: f64,
    ) -> MFromV<BATCH, OUTROWS, OUTCOLS, INROWS>
    where
        TFn: Fn(TapedV<BATCH, INROWS>) -> TapedM<BATCH, OUTROWS, OUTCOLS>,
    {
        Self::sym_diff_quotient(|x| matrix_valued(x.retaped()).split_tape().0, a, eps)
    }

    pub fn auto_grad<
        TFn,
        const BATCH: usize,
        const OUTROWS: usize,
        const OUTCOLS: usize,
        const INROWS: usize,
    >(
        matrix_valued: TFn,
        a: V<BATCH, INROWS>,
    ) -> MFromV<BATCH, OUTROWS, OUTCOLS, INROWS>
    where
        TFn: Fn(TapedV<BATCH, INROWS>) -> TapedM<BATCH, OUTROWS, OUTCOLS>,
    {
        let outcol0: MFromV<BATCH, OUTROWS, 0, INROWS> = a.device().zeros();
        let mut outcols: Tensor<(Const<BATCH>, Const<OUTROWS>, usize, Const<INROWS>), f64, _> =
            outcol0.realize();

        for c in 0..OUTCOLS {
            let batch: VFromV<BATCH, OUTROWS, INROWS> = VectorValuedMapFromVector::auto_grad(
                |x| {
                    let m = matrix_valued(x);
                    let m_c: TapedM<BATCH, OUTROWS, 1> = m.slice((.., .., c..c + 1)).realize();
                    m_c.reshape()
                },
                a.clone(),
            );
            let outcol: MFromV<BATCH, OUTROWS, 1, INROWS> = batch.reshape();

            outcols = (outcols, outcol).concat_along(Axis::<2>);
        }
        outcols.realize()
    }
}

// Matrix-valued map on a product space (=matrices).
//
// This is a function which takes a matrix and returns a matrix:
//
//  f: ℝᵐ x ℝⁿ -> ℝʳ x ℝᶜ
//
pub struct MatrixValuedMapFromMatrix;

impl MatrixValuedMapFromMatrix {
    // Finite difference quotient of the matrix-valued map.
    //
    // The derivative is a rank-4 tensor with shape (Rᵢ x Rₒ x Rₒ x Cₒ).
    //
    // Since all operations are batched, the function returns a (B x Rᵢ x Rₒ x Rₒ x Cₒ) rank-5 tensor.
    pub fn sym_diff_quotient<
        TFn,
        const BATCH: usize,
        const OUTROWS: usize,
        const OUTCOLS: usize,
        const INROWS: usize,
        const INCOLS: usize,
    >(
        vector_field: TFn,
        a: M<BATCH, INROWS, INCOLS>,
        eps: f64,
    ) -> MFromM<BATCH, OUTROWS, OUTCOLS, INROWS, INCOLS>
    where
        TFn: Fn(M<BATCH, INROWS, INCOLS>) -> M<BATCH, OUTROWS, OUTCOLS>,
    {
        let dev = a.device();
        let row0: Tensor<
            (
                Const<BATCH>,
                Const<OUTROWS>,
                Const<OUTCOLS>,
                Const<0>,
                Const<INCOLS>,
            ),
            f64,
            _,
        > = dev.zeros();
        let mut rows: Tensor<
            (
                Const<BATCH>,
                Const<OUTROWS>,
                Const<OUTCOLS>,
                usize,
                Const<INCOLS>,
            ),
            f64,
            _,
        > = row0.realize();

        for i1 in 0..INROWS {
            let col0: Tensor<
                (
                    Const<BATCH>,
                    Const<OUTROWS>,
                    Const<OUTCOLS>,
                    Const<1>,
                    Const<0>,
                ),
                f64,
                _,
            > = dev.zeros();
            let mut cols: Tensor<
                (
                    Const<BATCH>,
                    Const<OUTROWS>,
                    Const<OUTCOLS>,
                    Const<1>,
                    usize,
                ),
                f64,
                _,
            > = col0.realize();
            for i0 in 0..INCOLS {
                let mut a_plus = a.clone();
                for b in 0..BATCH {
                    a_plus[[b, i1, i0]] += eps;
                }
                let mut a_minus = a.clone();
                for b in 0..BATCH {
                    a_minus[[b, i1, i0]] -= eps;
                }
                let col: Tensor<
                    (
                        Const<BATCH>,
                        Const<OUTROWS>,
                        Const<OUTCOLS>,
                        Const<1>,
                        Const<1>,
                    ),
                    f64,
                    _,
                > = ((vector_field(a_plus) - vector_field(a_minus)) / (2.0 * eps)).reshape();
                cols = (cols, col).concat_along(Axis::<4>);
            }
            let cols: Tensor<
                (
                    Const<BATCH>,
                    Const<OUTROWS>,
                    Const<OUTCOLS>,
                    Const<1>,
                    Const<INCOLS>,
                ),
                f64,
                _,
            > = cols.realize();
            rows = (rows, cols).concat_along(Axis::<3>);
        }
        rows.realize()
    }

    pub fn sym_diff_quotient_from_taped<
        TFn,
        const BATCH: usize,
        const OUTROWS: usize,
        const OUTCOLS: usize,
        const INROWS: usize,
        const INCOLS: usize,
    >(
        matrix_valued: TFn,
        a: M<BATCH, INROWS, INCOLS>,
        eps: f64,
    ) -> MFromM<BATCH, OUTROWS, OUTCOLS, INROWS, INCOLS>
    where
        TFn: Fn(TapedM<BATCH, INROWS, INCOLS>) -> TapedM<BATCH, OUTROWS, OUTCOLS>,
    {
        Self::sym_diff_quotient(|x| matrix_valued(x.retaped()).split_tape().0, a, eps)
    }

    pub fn auto_grad<
        TFn,
        const BATCH: usize,
        const OUTROWS: usize,
        const OUTCOLS: usize,
        const INROWS: usize,
        const INCOLS: usize,
    >(
        matrix_valued: TFn,
        a: M<BATCH, INROWS, INCOLS>,
    ) -> MFromM<BATCH, OUTROWS, OUTCOLS, INROWS, INCOLS>
    where
        TFn: Fn(TapedM<BATCH, INROWS, INCOLS>) -> TapedM<BATCH, OUTROWS, OUTCOLS>,
    {
        let outcol0: MFromM<BATCH, OUTROWS, 0, INROWS, INCOLS> = a.device().zeros();
        let mut outcols: Tensor<
            (
                Const<BATCH>,
                Const<OUTROWS>,
                usize,
                Const<INROWS>,
                Const<INROWS>,
            ),
            f64,
            _,
        > = outcol0.realize();

        for c in 0..OUTCOLS {
            let batch: VFromM<BATCH, OUTROWS, INROWS, INCOLS> =
                VectorValuedMapFromMatrix::auto_grad(
                    |x| {
                        let m = matrix_valued(x);
                        let m_c: TapedM<BATCH, OUTROWS, 1> = m.slice((.., .., c..c + 1)).realize();
                        m_c.reshape()
                    },
                    a.clone(),
                );
            let outcol: MFromM<BATCH, OUTROWS, 1, INROWS, INROWS> = batch.reshape();

            outcols = (outcols, outcol).concat_along(Axis::<2>);
        }
        outcols.realize()
    }
}

mod test {

    use crate::assert_tensors_relative_eq_rank4;
    use crate::assert_tensors_relative_eq_rank5;

    use super::*;
    use dfdx_core::tensor::Cpu;

    #[allow(dead_code)]
    fn test_batched_matrix_valued_map_from_vector<const BATCH: usize>() {
        let dev = Cpu::default();
        let a = dev.sample_uniform();

        let eps = 1e-6;
        //      [[ i ]]
        //      [[   ]]
        //      [[ j ]]      [[                      ]]
        //      [[   ]]      [[   0   -k    j    x   ]]
        //      [[ k ]]      [[                      ]]
        //  hat [[   ]]   =  [[   k    0   -i    y   ]]
        //      [[ x ]]      [[                      ]]
        //      [[   ]]      [[  -j    i    0    z   ]]
        //      [[ y ]]      [[                      ]]
        //      [[   ]]
        //      [[ z ]]
        let hat_fn = |v: TapedV<BATCH, 6>| -> TapedM<BATCH, 3, 4> {
            let v: TapedM<BATCH, 6, 1> = v.reshape();
            let i: TapedM<BATCH, 1, 1> = v.clone().slice((.., 0..1, ..)).realize();
            let j: TapedM<BATCH, 1, 1> = v.clone().slice((.., 1..2, ..)).realize();
            let k: TapedM<BATCH, 1, 1> = v.clone().slice((.., 2..3, ..)).realize();
            let x: TapedM<BATCH, 1, 1> = v.clone().slice((.., 3..4, ..)).realize();
            let y: TapedM<BATCH, 1, 1> = v.clone().slice((.., 4..5, ..)).realize();
            let z: TapedM<BATCH, 1, 1> = v.clone().slice((.., 5..6, ..)).realize();
            let zero: M<BATCH, 1, 1> = z.device().zeros();

            let row0 = make_4blockcol_mat(zero.retaped(), k.clone().negate(), j.clone(), x);
            let row1 = make_4blockcol_mat(k, zero.retaped(), -i.clone(), y);
            let make_2col_mat = make_4blockcol_mat(-j, i, zero.retaped(), z);

            make_3rowblock_mat(row0, row1, make_2col_mat)
        };

        let df = MatrixValuedMapFromVector::sym_diff_quotient_from_taped(hat_fn, a.clone(), eps);
        let auto_grad = MatrixValuedMapFromVector::auto_grad(hat_fn, a.clone());

        assert_tensors_relative_eq_rank4!(df, auto_grad, 1e-6);
    }

    #[test]
    fn test_matrix_valued_map_from_vector() {
        test_batched_matrix_valued_map_from_vector::<1>();
        test_batched_matrix_valued_map_from_vector::<2>();
        test_batched_matrix_valued_map_from_vector::<4>();
    }

    #[allow(dead_code)]
    fn test_batched_matrix_valued_map_from_matrix<const BATCH: usize>() {
        let dev = Cpu::default();
        let a = dev.sample_uniform();

        let eps = 1e-6;
        //      [[ a   b ]]       1    [[  d  -b ]]
        //  inv [[       ]] =  ------- [[        ]]
        //      [[ c   d ]]    ad - bc [[ -c   a ]]
        let f = |m: TapedM<BATCH, 2, 2>| -> TapedM<BATCH, 2, 2> {
            let a: TapedM<BATCH, 1, 1> = m.clone().slice((.., 0..1, 0..1)).realize();
            let b: TapedM<BATCH, 1, 1> = m.clone().slice((.., 0..1, 1..2)).realize();

            let c: TapedM<BATCH, 1, 1> = m.clone().slice((.., 1..2, 0..1)).realize();
            let d: TapedM<BATCH, 1, 1> = m.clone().slice((.., 1..2, 1..2)).realize();

            let det : TapedS<BATCH> = (a.clone() * d.clone() - b.clone() * c.clone()).reshape();

            make_2rowblock_mat(make_2blockcol_mat(b, -d), make_2blockcol_mat(-c, a)) * det.broadcast()
        };

        let df = MatrixValuedMapFromMatrix::sym_diff_quotient_from_taped(f, a.clone(), eps);
        let auto_grad = MatrixValuedMapFromMatrix::auto_grad(f, a.clone());

        assert_tensors_relative_eq_rank5!(df, auto_grad, 1e-6);
    }

    #[test]
    fn test_matrix_valued_map_from_matrix() {
        test_batched_matrix_valued_map_from_matrix::<1>();
        test_batched_matrix_valued_map_from_matrix::<2>();
        test_batched_matrix_valued_map_from_matrix::<4>();
    }
}
