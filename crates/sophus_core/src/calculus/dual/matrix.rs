use super::scalar::IsDualScalar;
use crate::linalg::SMat;
use crate::prelude::IsMatrix;
use crate::prelude::IsRealScalar;

/// Derivative of a matrix-valued map.
pub struct MatrixValuedDerivative<
    S,
    const OUTROWS: usize,
    const OUTCOLS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    /// output matrix of input matrices
    pub out_mat: SMat<SMat<S, DM, DN>, OUTROWS, OUTCOLS>,
}

impl<
        S: IsRealScalar<BATCH, RealScalar = S>,
        const OUTROWS: usize,
        const OUTCOLS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > MatrixValuedDerivative<S, OUTROWS, OUTCOLS, BATCH, DM, DN>
{
    /// zero
    pub fn zeros() -> Self {
        MatrixValuedDerivative {
            out_mat: SMat::zeros(),
        }
    }
}

/// Trait for scalar dual numbers
pub trait IsDualMatrix<
    S: IsDualScalar<BATCH, DM, DN>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: IsMatrix<S, ROWS, COLS, BATCH, DM, DN>
{
    /// Create a new dual matrix from a real matrix for auto-differentiation with respect to self
    ///
    /// Typically this is not called directly, but through using a map auto-differentiation call:
    ///
    ///  - ScalarValueMatrixMap::fw_autodiff(...);
    ///  - VectorValuedMatrixMap::fw_autodiff(...);
    ///  - MatrixValuedMatrixMap::fw_autodiff(...);
    fn var(val: S::RealMatrix<ROWS, COLS>) -> Self;

    /// Get the derivative
    fn derivative(self) -> MatrixValuedDerivative<S::RealScalar, ROWS, COLS, BATCH, DM, DN>;
}

#[test]
fn dual_matrix_tests() {
    #[cfg(feature = "simd")]
    use crate::calculus::dual::DualBatchScalar;
    use crate::calculus::dual::DualScalar;
    use crate::calculus::maps::matrix_valued_maps::MatrixValuedMatrixMap;
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;
    use crate::linalg::EPS_F64;
    use crate::prelude::IsScalar;

    #[cfg(test)]
    trait Test {
        fn run();
    }

    macro_rules! def_test_template {
        ( $scalar:ty, $dual_scalar_2_4: ty, $dual_scalar_4_1: ty, $dual_scalar_4_4: ty, $batch:literal
    ) => {
            #[cfg(test)]
            impl Test for $scalar {
                fn run() {
                    let m_2x4 = <$scalar as IsScalar<$batch,0,0>>::Matrix::<2, 4>::from_f64_array2([
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                    ]);
                    let m_4x1 = <$scalar as IsScalar<$batch,0,0>>::Matrix::<4, 1>::from_f64_array2([
                        [1.0],
                        [2.0],
                        [3.0],
                        [4.0],
                    ]);

                    fn mat_mul_fn<
                        S: IsScalar<BATCH, DM, DN>,
                        const BATCH: usize,
                        const DM: usize,
                        const DN: usize,
                    >(
                        x: S::Matrix<2, 4>,
                        y: S::Matrix<4, 1>,
                    ) -> S::Matrix<2, 1> {
                        x.mat_mul(y)
                    }
                    let finite_diff =
                        MatrixValuedMatrixMap::<$scalar, $batch,0,0>::sym_diff_quotient(
                            |x| mat_mul_fn::<$scalar, $batch,0,0>(x, m_4x1),
                            m_2x4,
                            EPS_F64,
                        );
                    let auto_grad = MatrixValuedMatrixMap::<$dual_scalar_2_4, $batch, 2, 4>::fw_autodiff(
                        |x| {
                            mat_mul_fn::<$dual_scalar_2_4, $batch, 2, 4>(
                                x,
                                <$dual_scalar_2_4 as IsScalar<$batch, 2, 4>>::Matrix::from_real_matrix(m_4x1),
                            )
                        },
                        m_2x4,
                    );

                    for i in 0..2 {
                        for j in 0..1 {
                            approx::assert_abs_diff_eq!(
                                finite_diff.out_mat[(i, j)],
                                auto_grad.out_mat[(i, j)],
                                epsilon = 0.0001
                            );
                        }
                    }

                    let finite_diff = MatrixValuedMatrixMap::sym_diff_quotient(
                        |x| mat_mul_fn::<$scalar, $batch,0,0>(m_2x4, x),
                        m_4x1,
                        EPS_F64,
                    );
                    let auto_grad = MatrixValuedMatrixMap::<$dual_scalar_4_1, $batch, 4, 1>::fw_autodiff(
                        |x| {
                            mat_mul_fn::<$dual_scalar_4_1, $batch, 4, 1>(
                                <$dual_scalar_4_1 as IsScalar<$batch, 4, 1>>::Matrix::from_real_matrix(m_2x4),
                                x,
                            )
                        },
                        m_4x1,
                    );

                    for i in 0..2 {
                        for j in 0..1 {
                            approx::assert_abs_diff_eq!(
                                finite_diff.out_mat[(i, j)],
                                auto_grad.out_mat[(i, j)],
                                epsilon = 0.0001
                            );
                        }
                    }

                    fn mat_mul2_fn<
                        S: IsScalar<BATCH, DM, DN>,
                        const BATCH: usize,
                        const DM: usize,
                        const DN: usize,
                    >(
                        x: S::Matrix<4, 4>,
                    ) -> S::Matrix<4, 4> {
                        x.mat_mul(&x)
                    }

                    let m_4x4 = <$scalar as IsScalar<$batch,0,0>>::Matrix::<4, 4>::from_f64_array2([
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                    ]);

                    let finite_diff =
                        MatrixValuedMatrixMap::<$scalar, $batch,0,0>::sym_diff_quotient(
                            mat_mul2_fn::<$scalar, $batch,0,0>,
                            m_4x4,
                            EPS_F64,
                        );
                    let auto_grad = MatrixValuedMatrixMap::<$dual_scalar_4_4, $batch,4, 4>::fw_autodiff(
                        mat_mul2_fn::<$dual_scalar_4_4, $batch, 4 ,4>,
                        m_4x4,
                    );

                    for i in 0..2 {
                        for j in 0..4 {
                            approx::assert_abs_diff_eq!(
                                finite_diff.out_mat[(i, j)],
                                auto_grad.out_mat[(i, j)],
                                epsilon = 0.0001
                            );
                        }
                    }

                }
            }
        };
    }

    def_test_template!(f64, DualScalar<2,4>, DualScalar<4,1>, DualScalar<4,4>, 1);
    #[cfg(feature = "simd")]
    def_test_template!(BatchScalarF64<2>, DualBatchScalar<2, 2, 4>, DualBatchScalar<2, 4, 1>,DualBatchScalar<2, 4, 4>, 2);
    #[cfg(feature = "simd")]
    def_test_template!(BatchScalarF64<4>, DualBatchScalar<4, 2, 4>,DualBatchScalar<4, 4, 1>, DualBatchScalar<4, 4, 4>, 4);

    f64::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<2>::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<4>::run();
}
