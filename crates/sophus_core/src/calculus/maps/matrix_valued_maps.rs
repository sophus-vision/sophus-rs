use crate::linalg::SMat;
use crate::prelude::*;
use crate::tensor::mut_tensor::MutTensorDDRC;
use crate::tensor::mut_tensor::MutTensorDRC;
use nalgebra::SMatrix;
use std::marker::PhantomData;

/// Matrix-valued map on a vector space.
///
/// This is a function which takes a vector and returns a matrix:
///
///  f: ℝᵐ -> ℝʳ x ℝᶜ
///
pub struct MatrixValuedMapFromVector<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: std::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH, RealScalar = S>, const BATCH: usize>
    MatrixValuedMapFromVector<S, BATCH>
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
    ) -> MutTensorDRC<S, OUTROWS, OUTCOLS>
    where
        TFn: Fn(S::RealVector<INROWS>) -> SMatrix<S, OUTROWS, OUTCOLS>,
        SMatrix<S, OUTROWS, OUTCOLS>: IsRealMatrix<S, OUTROWS, OUTCOLS, BATCH>,
    {
        let mut out = MutTensorDRC::<S, OUTROWS, OUTCOLS>::from_shape([INROWS]);
        let eps_v = S::RealScalar::from_f64(eps);

        for i1 in 0..INROWS {
            let mut a_plus = a;

            a_plus[i1] += eps_v;

            let mut a_minus = a;
            a_minus[i1] -= eps_v;

            let val = (matrix_valued(a_plus) - matrix_valued(a_minus))
                .scaled(S::from_f64(1.0 / (2.0 * eps)));

            *out.mut_view().get_mut([i1]) = val;
        }
        out
    }
}

impl<D: IsDualScalar<BATCH, DualScalar = D>, const BATCH: usize>
    MatrixValuedMapFromVector<D, BATCH>
{
    /// Auto differentiation of the matrix-valued map.
    pub fn fw_autodiff<TFn, const OUTROWS: usize, const OUTCOLS: usize, const INROWS: usize>(
        matrix_valued: TFn,
        a: D::RealVector<INROWS>,
    ) -> MutTensorDRC<D::RealScalar, OUTROWS, OUTCOLS>
    where
        TFn: Fn(D::DualVector<INROWS>) -> D::DualMatrix<OUTROWS, OUTCOLS>,
    {
        MutTensorDRC {
            mut_array: matrix_valued(D::vector_with_dij(a))
                .dij_val()
                .unwrap()
                .mut_array
                .into_shape([INROWS])
                .unwrap(),
            phantom: PhantomData,
        }
    }
}

/// Matrix-valued map on a product space (=matrices).
///
/// This is a function which takes a matrix and returns a matrix:
///
///  f: ℝᵐ x ℝⁿ -> ℝʳ x ℝᶜ
///
pub struct MatrixValuedMapFromMatrix<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: std::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH, RealScalar = S>, const BATCH: usize>
    MatrixValuedMapFromMatrix<S, BATCH>
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
    ) -> MutTensorDDRC<S, OUTROWS, OUTCOLS>
    where
        TFn: Fn(S::RealMatrix<INROWS, INCOLS>) -> SMatrix<S, OUTROWS, OUTCOLS>,
        SMatrix<S, OUTROWS, OUTCOLS>: IsRealMatrix<S, OUTROWS, OUTCOLS, BATCH>,
    {
        let mut out = MutTensorDDRC::<S, OUTROWS, OUTCOLS>::from_shape_and_val(
            [INROWS, INCOLS],
            SMat::<S, OUTROWS, OUTCOLS>::zeros(),
        );
        let eps_v = S::RealScalar::from_f64(eps);
        for i1 in 0..INROWS {
            for i0 in 0..INCOLS {
                let mut a_plus = a;

                a_plus[(i1, i0)] += eps_v;

                let mut a_minus = a;
                a_minus[(i1, i0)] -= eps_v;

                let val = (vector_field(a_plus) - vector_field(a_minus))
                    .scaled(S::from_f64(1.0 / (2.0 * eps)));

                *out.mut_view().get_mut([i1, i0]) = val;
            }
        }
        out
    }
}

impl<D: IsDualScalar<BATCH, DualScalar = D>, const BATCH: usize>
    MatrixValuedMapFromMatrix<D, BATCH>
{
    /// Auto differentiation of the matrix-valued map.
    pub fn fw_autodiff<
        TFn,
        const OUTROWS: usize,
        const OUTCOLS: usize,
        const INROWS: usize,
        const INCOLS: usize,
    >(
        matrix_valued: TFn,
        a: D::RealMatrix<INROWS, INCOLS>,
    ) -> MutTensorDDRC<D::RealScalar, OUTROWS, OUTCOLS>
    where
        TFn: Fn(D::DualMatrix<INROWS, INCOLS>) -> D::DualMatrix<OUTROWS, OUTCOLS>,
    {
        matrix_valued(D::matrix_with_dij(a)).dij_val().unwrap()
    }
}

#[test]
fn matrix_valued_map_from_vector_tests() {
    use crate::calculus::dual::dual_scalar::DualBatchScalar;
    use crate::calculus::dual::dual_scalar::DualScalar;
    use crate::calculus::maps::matrix_valued_maps::MatrixValuedMapFromVector;

    use crate::linalg::scalar::IsScalar;
    use crate::linalg::vector::IsVector;
    use crate::linalg::BatchScalarF64;
    use crate::tensor::tensor_view::IsTensorLike;

    #[cfg(test)]
    trait Test {
        fn run();
    }

    macro_rules! def_test_template {
        ( $scalar:ty, $dual_scalar: ty, $batch:literal
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
                        fn hat_fn<S: IsScalar<BATCH>, const BATCH: usize>(
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

                        let a = <$scalar as IsScalar<$batch>>::Vector::<6>::new(
                            <$scalar>::from_f64(0.1),
                            <$scalar>::from_f64(0.2),
                            <$scalar>::from_f64(0.4),
                            <$scalar>::from_f64(0.7),
                            <$scalar>::from_f64(0.8),
                            <$scalar>::from_f64(0.9),
                        );

                        let finite_diff =
                            MatrixValuedMapFromVector::<$scalar, $batch>::sym_diff_quotient(
                                hat_fn::<$scalar, $batch>,
                                a,
                                1e-6,
                            );
                        let auto_grad =
                            MatrixValuedMapFromVector::<$dual_scalar, $batch>::fw_autodiff(
                                hat_fn::<$dual_scalar, $batch>,
                                a,
                            );
                        approx::assert_abs_diff_eq!(
                            finite_diff.view().elem_view(),
                            auto_grad.view().elem_view(),
                            epsilon = 0.0001
                        );
                    }

                    //      [[ a   b ]]       1    [[  d  -b ]]
                    //  inv [[       ]] =  ------- [[        ]]
                    //      [[ c   d ]]    ad - bc [[ -c   a ]]

                    fn f<S: IsScalar<BATCH>, const BATCH: usize>(
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
                    let a = <$scalar as IsScalar<$batch>>::Matrix::<2, 2>::new(
                        <$scalar>::from_f64(0.1),
                        <$scalar>::from_f64(0.2),
                        <$scalar>::from_f64(0.4),
                        <$scalar>::from_f64(0.7),
                    );

                    let finite_diff =
                        MatrixValuedMapFromMatrix::<$scalar, $batch>::sym_diff_quotient(
                            f::<$scalar, $batch>,
                            a,
                            1e-6,
                        );
                    let auto_grad = MatrixValuedMapFromMatrix::<$dual_scalar, $batch>::fw_autodiff(
                        f::<$dual_scalar, $batch>,
                        a,
                    );

                    approx::assert_abs_diff_eq!(
                        finite_diff.view().elem_view(),
                        auto_grad.view().elem_view(),
                        epsilon = 2.0
                    );
                }
            }
        };
    }

    def_test_template!(f64, DualScalar, 1);
    def_test_template!(BatchScalarF64<2>, DualBatchScalar<2>, 2);
    def_test_template!(BatchScalarF64<4>, DualBatchScalar<4>, 4);

    f64::run();
    BatchScalarF64::<2>::run();
    BatchScalarF64::<4>::run();
}
