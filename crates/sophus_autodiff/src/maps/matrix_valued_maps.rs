use crate::dual::matrix::MatrixValuedDerivative;
use crate::prelude::*;
use nalgebra::SMatrix;

/// Matrix-valued map on a vector space.
///
/// This is a function which takes a vector and returns a matrix:
///
///  f: ℝᵐ -> ℝʳ x ℝᶜ
///
pub struct MatrixValuedVectorMap<S, const BATCH: usize, const DM: usize, const DN: usize> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH, RealScalar = S>, const BATCH: usize>
    MatrixValuedVectorMap<S, BATCH, 0, 0>
{
    /// Finite difference quotient of the matrix-valued map.
    ///
    /// The derivative is a rank-3 tensor with shape (Rₒ x Cₒ x Rᵢ).
    ///
    /// For efficiency reasons, we return Rᵢ x [Rₒ x Cₒ]
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

impl<
        D: IsDualScalar<BATCH, INROWS, 1, DualScalar<INROWS, 1> = D>,
        const BATCH: usize,
        const INROWS: usize,
    > MatrixValuedVectorMap<D, BATCH, INROWS, 1>
{
    /// Auto differentiation of the matrix-valued map.
    pub fn fw_autodiff<TFn, const OUTROWS: usize, const OUTCOLS: usize>(
        matrix_valued: TFn,
        a: D::RealVector<INROWS>,
    ) -> MatrixValuedDerivative<D::RealScalar, OUTROWS, OUTCOLS, BATCH, INROWS, 1>
    where
        TFn: Fn(D::DualVector<INROWS, INROWS, 1>) -> D::DualMatrix<OUTROWS, OUTCOLS, INROWS, 1>,
    {
        matrix_valued(D::vector_var(a)).derivative()
    }
}

/// Matrix-valued map on a product space (=matrices).
///
/// This is a function which takes a matrix and returns a matrix:
///
///  f: ℝᵐ x ℝⁿ -> ℝʳ x ℝᶜ
///
pub struct MatrixValuedMatrixMap<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH, RealScalar = S>, const BATCH: usize>
    MatrixValuedMatrixMap<S, BATCH, 0, 0>
{
    /// Finite difference quotient of the matrix-valued map.
    ///
    /// The derivative is a rank-4 tensor with shape (Rₒ x Cₒ x Rᵢ x Cᵢ).
    ///
    /// For efficiency reasons, we return Rᵢ x Cᵢ x [Rₒ x Cₒ]
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

impl<
        D: IsDualScalar<BATCH, INROWS, INCOLS, DualScalar<INROWS, INCOLS> = D>,
        const BATCH: usize,
        const INROWS: usize,
        const INCOLS: usize,
    > MatrixValuedMatrixMap<D, BATCH, INROWS, INCOLS>
{
    /// Auto differentiation of the matrix-valued map.
    pub fn fw_autodiff<TFn, const OUTROWS: usize, const OUTCOLS: usize>(
        matrix_valued: TFn,
        a: D::RealMatrix<INROWS, INCOLS>,
    ) -> MatrixValuedDerivative<D::RealScalar, OUTROWS, OUTCOLS, BATCH, INROWS, INCOLS>
    where
        TFn: Fn(
            D::DualMatrix<INROWS, INCOLS, INROWS, INCOLS>,
        ) -> D::DualMatrix<OUTROWS, OUTCOLS, INROWS, INCOLS>,
    {
        matrix_valued(D::matrix_var(a)).derivative()
    }
}

#[test]
fn matrix_valued_map_from_vector_tests() {
    use crate::dual::dual_scalar::DualScalar;
    #[cfg(feature = "simd")]
    use crate::dual::DualBatchScalar;
    use crate::linalg::scalar::IsScalar;
    use crate::linalg::vector::IsVector;
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;
    use crate::linalg::EPS_F64;
    use crate::maps::matrix_valued_maps::MatrixValuedVectorMap;

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
                            let i = v.get_elem(0);
                            let j = v.get_elem(1);
                            let k = v.get_elem(2);
                            let ni = -i.clone();
                            let nj = -j.clone();
                            let nk = -k.clone();
                            let x = v.get_elem(3);
                            let y = v.get_elem(4);
                            let z = v.get_elem(5);

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
                            MatrixValuedVectorMap::<$scalar, $batch, 0, 0>::sym_diff_quotient(
                                hat_fn::<$scalar, $batch, 0, 0>,
                                a,
                                EPS_F64,
                            );
                        let auto_grad =
                            MatrixValuedVectorMap::<$dual_scalar_6, $batch, 6, 1>::fw_autodiff(
                                hat_fn::<$dual_scalar_6, $batch, 6, 1>,
                                a,
                            );
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
                        let a = m.get_elem([0, 0]);
                        let b = m.get_elem([0, 1]);

                        let c = m.get_elem([1, 0]);
                        let d = m.get_elem([1, 1]);

                        let det =
                            S::from_f64(1.0) / (a.clone() * d.clone() - (b.clone() * c.clone()));

                        S::Matrix::from_array2([
                            [det.clone() * d, -det.clone() * b],
                            [-det.clone() * c, det * a],
                        ])
                    }
                    let a = <$scalar as IsScalar<$batch, 0, 0>>::Matrix::<2, 2>::new(
                        <$scalar>::from_f64(0.1),
                        <$scalar>::from_f64(0.2),
                        <$scalar>::from_f64(0.4),
                        <$scalar>::from_f64(0.7),
                    );

                    let finite_diff =
                        MatrixValuedMatrixMap::<$scalar, $batch, 0, 0>::sym_diff_quotient(
                            f::<$scalar, $batch, 0, 0>,
                            a,
                            EPS_F64,
                        );
                    let auto_grad =
                        MatrixValuedMatrixMap::<$dual_scalar_2_2, $batch, 2, 2>::fw_autodiff(
                            f::<$dual_scalar_2_2, $batch, 2, 2>,
                            a,
                        );

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
