use nalgebra::SMatrix;

use crate::{
    dual::MatrixValuedDerivative,
    prelude::*,
};

/// Matrix-valued map on a vector space.
///
/// This is a function which takes a vector and returns a matrix:
///
///  `f: ℝᵐ -> ℝʳˣᶜ`
pub struct MatrixValuedVectorMap<S, const BATCH: usize> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH, RealScalar = S>, const BATCH: usize> MatrixValuedVectorMap<S, BATCH> {
    /// Finite difference quotient of the matrix-valued map.
    pub fn sym_diff_quotient<TFn, const OUTROWS: usize, const OUTCOLS: usize, const INROWS: usize>(
        matrix_valued: TFn,
        a: S::RealVector<INROWS>,
        eps: f64,
    ) -> MatrixValuedDerivative<S, OUTROWS, OUTCOLS, BATCH, INROWS, 1>
    where
        TFn: Fn(S::RealVector<INROWS>) -> SMatrix<S, OUTROWS, OUTCOLS>,
        SMatrix<S, OUTROWS, OUTCOLS>: IsRealMatrix<S, OUTROWS, OUTCOLS, BATCH>,
    {
        let mut out = MatrixValuedDerivative::<S, OUTROWS, OUTCOLS, BATCH, INROWS, 1>::zeros();
        let eps_v = S::RealScalar::from_f64(eps);

        for r in 0..INROWS {
            let mut a_plus = a;

            a_plus[r] += eps_v;

            let mut a_minus = a;
            a_minus[r] -= eps_v;

            let val = (matrix_valued(a_plus) - matrix_valued(a_minus))
                .scaled(S::from_f64(1.0 / (2.0 * eps)));

            for i in 0..OUTROWS {
                for j in 0..OUTCOLS {
                    out.out_mat[(i, j)][r] = val[(i, j)];
                }
            }
        }
        out
    }
}

/// Matrix-valued map on a product space (=matrices).
///
/// This is a function which takes a matrix and returns a matrix:
///
///  `f: ℝᵐˣⁿ -> ℝʳˣᶜ`
pub struct MatrixValuedMatrixMap<S: IsScalar<BATCH, 0, 0>, const BATCH: usize> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH, RealScalar = S>, const BATCH: usize> MatrixValuedMatrixMap<S, BATCH> {
    /// Finite difference quotient of the matrix-valued map.
    pub fn sym_diff_quotient<
        TFn,
        const OUTROWS: usize,
        const OUTCOLS: usize,
        const INROWS: usize,
        const INCOLS: usize,
    >(
        vector_field: TFn,
        a: S::RealMatrix<INROWS, INCOLS>,
        eps: f64,
    ) -> MatrixValuedDerivative<S, OUTROWS, OUTCOLS, BATCH, INROWS, INCOLS>
    where
        TFn: Fn(S::RealMatrix<INROWS, INCOLS>) -> SMatrix<S, OUTROWS, OUTCOLS>,
        SMatrix<S, OUTROWS, OUTCOLS>: IsRealMatrix<S, OUTROWS, OUTCOLS, BATCH>,
    {
        let mut out = MatrixValuedDerivative::<S, OUTROWS, OUTCOLS, BATCH, INROWS, INCOLS>::zeros();
        let eps_v = S::RealScalar::from_f64(eps);
        for i1 in 0..INROWS {
            for i0 in 0..INCOLS {
                let mut a_plus = a;

                a_plus[(i1, i0)] += eps_v;

                let mut a_minus = a;
                a_minus[(i1, i0)] -= eps_v;

                let val = (vector_field(a_plus) - vector_field(a_minus))
                    .scaled(S::from_f64(1.0 / (2.0 * eps)));

                for r in 0..OUTROWS {
                    for c in 0..OUTCOLS {
                        out.out_mat[(r, c)][(i1, i0)] = val[(r, c)];
                    }
                }
            }
        }
        out
    }
}

#[test]
fn matrix_valued_map_from_vector_tests() {
    #[cfg(feature = "simd")]
    use crate::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;
    use crate::{
        dual::DualScalar,
        linalg::{
            IsScalar,
            IsVector,
            EPS_F64,
        },
        maps::MatrixValuedVectorMap,
    };

    #[cfg(test)]
    trait Test {
        fn run();
    }

    macro_rules! def_test_template {
        ( $scalar:ty, $dual_scalar_6: ty,$dual_scalar_2_2: ty, $batch:literal
    ) => {
            #[cfg(test)]
            impl Test for $scalar {
                fn run() {
                    {
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
                        fn hat_fn<
                            S: IsScalar<BATCH, DM, DN>,
                            const BATCH: usize,
                            const DM: usize,
                            const DN: usize,
                        >(
                            v: S::Vector<6>,
                        ) -> S::Matrix<3, 4> {
                            let i = v.elem(0);
                            let j = v.elem(1);
                            let k = v.elem(2);
                            let ni = -i;
                            let nj = -j;
                            let nk = -k;
                            let x = v.elem(3);
                            let y = v.elem(4);
                            let z = v.elem(5);

                            S::Matrix::<3, 4>::from_array2([
                                [S::from_f64(0.0), nk, j, x],
                                [k, S::from_f64(0.0), ni, y],
                                [nj, i, S::from_f64(0.0), z],
                            ])
                        }

                        let a = <$scalar as IsScalar<$batch, 0, 0>>::Vector::<6>::new(
                            <$scalar>::from_f64(0.1),
                            <$scalar>::from_f64(0.2),
                            <$scalar>::from_f64(0.4),
                            <$scalar>::from_f64(0.7),
                            <$scalar>::from_f64(0.8),
                            <$scalar>::from_f64(0.9),
                        );

                        let finite_diff =
                            MatrixValuedVectorMap::<$scalar, $batch>::sym_diff_quotient(
                                hat_fn::<$scalar, $batch, 0, 0>,
                                a,
                                EPS_F64,
                            );
                        let auto_grad =
                            hat_fn::<$dual_scalar_6, $batch, 6, 1>(<$dual_scalar_6>::vector_var(a))
                                .derivative();
                        for r in 0..6 {
                            approx::assert_abs_diff_eq!(
                                finite_diff.out_mat[r],
                                auto_grad.out_mat[r],
                                epsilon = 0.0001
                            );
                        }
                    }

                    //      [[ a   b ]]       1    [[  d  -b ]]
                    //  inv [[       ]] =  ------- [[        ]]
                    //      [[ c   d ]]    ad - bc [[ -c   a ]]

                    fn f<
                        S: IsScalar<BATCH, DM, DN>,
                        const BATCH: usize,
                        const DM: usize,
                        const DN: usize,
                    >(
                        m: S::Matrix<2, 2>,
                    ) -> S::Matrix<2, 2> {
                        let a = m.elem([0, 0]);
                        let b = m.elem([0, 1]);

                        let c = m.elem([1, 0]);
                        let d = m.elem([1, 1]);

                        let det = S::from_f64(1.0) / (a * d - (b * c));

                        S::Matrix::from_array2([[det * d, -det * b], [-det * c, det * a]])
                    }
                    let a = <$scalar as IsScalar<$batch, 0, 0>>::Matrix::<2, 2>::new(
                        <$scalar>::from_f64(0.1),
                        <$scalar>::from_f64(0.2),
                        <$scalar>::from_f64(0.4),
                        <$scalar>::from_f64(0.7),
                    );

                    let finite_diff = MatrixValuedMatrixMap::<$scalar, $batch>::sym_diff_quotient(
                        f::<$scalar, $batch, 0, 0>,
                        a,
                        EPS_F64,
                    );
                    let auto_grad =
                        f::<$dual_scalar_2_2, $batch, 2, 2>(<$dual_scalar_2_2>::matrix_var(a))
                            .derivative();

                    for r in 0..2 {
                        for c in 0..2 {
                            approx::assert_abs_diff_eq!(
                                finite_diff.out_mat[(r, c)],
                                auto_grad.out_mat[(r, c)],
                                epsilon = 2.0
                            );
                        }
                    }
                }
            }
        };
    }

    def_test_template!(
        f64,
        DualScalar<6, 1>,
        DualScalar<2, 2>,
        1);
    #[cfg(feature = "simd")]
    def_test_template!(
        BatchScalarF64<2>,
        DualBatchScalar<2, 6, 1>,
        DualBatchScalar<2, 2, 2>,
        2);
    #[cfg(feature = "simd")]
    def_test_template!(
        BatchScalarF64<4>,
        DualBatchScalar<4, 6, 1>,
        DualBatchScalar<4, 2, 2>,
        4);

    f64::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<2>::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<4>::run();
}
