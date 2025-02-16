use super::scalar::IsDualScalar;
use crate::{
    linalg::SMat,
    prelude::{
        IsMatrix,
        IsRealScalar,
    },
};

/// A structure that holds the derivative of a matrix-valued map with respect to its inputs.
///
/// For a function f: X -> ℝʳ, or similarly a batch variant in SIMD,
/// this structure stores each output dimension’s derivative block.
///
/// - `out_vec[i]` contains a derivative matrix (for dimension `i` of the output).
pub struct MatrixValuedDerivative<
    S,
    const OUTROWS: usize,
    const OUTCOLS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    /// The OUTROWS x OUTCOLS output matrix, where each element is a
    /// matrix of shape `[DM × DN]` storing the derivative for that output lane.
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
    /// Creates a new instance set to all zeros.
    pub fn zeros() -> Self {
        MatrixValuedDerivative {
            out_mat: SMat::zeros(),
        }
    }
}

/// A trait for a *dual matrix*, supporting forward-mode AD on matrix quantities.
///
/// A dual matrix is a matrix whose entries are themselves dual scalars
/// (with partial derivatives). This trait extends [`IsMatrix`] with AD-specific methods:
///
/// - Constructing a "variable" matrix from a real matrix (marking it for differentiation).
/// - Retrieving the entire derivative as a [`MatrixValuedDerivative`].
///
/// # Generic Parameters
/// - `S`: A type implementing [`IsDualScalar`].
/// - `ROWS`, `COLS`: Matrix dimensions.
/// - `BATCH`, `DM`, `DN`: Additional parameters for batch usage and the derivative shape.
pub trait IsDualMatrix<
    S: IsDualScalar<BATCH, DM, DN>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: IsMatrix<S, ROWS, COLS, BATCH, DM, DN>
{
    /// Creates a new dual matrix from a purely *real* matrix, marking its entries as variables
    /// for auto-differentiation.is turned into a dual scalar with derivative
    /// identity relative to that component.
    fn var(val: S::RealMatrix<ROWS, COLS>) -> Self;

    /// Returns the derivative as a [`MatrixValuedDerivative`].
    ///
    /// If the matrix has an `ROWS` x `COLS` shape, each element’s derivative is a matrix of shape
    /// `[DM × DN]`, so the overall derivative is `ROWS` x `COLS` grid of those matrices.
    fn derivative(self) -> MatrixValuedDerivative<S::RealScalar, ROWS, COLS, BATCH, DM, DN>;
}

/// A helper trait marking that this dual matrix is the result of a *curve* f: ℝ -> ℝʳˣᶜ, letting
/// you retrieve a simpler derivative form for each dimension of the output.
pub trait IsDualMatrixFromCurve<
    S: IsDualScalar<BATCH, 1, 1>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH: usize,
>: IsDualMatrix<S, ROWS, COLS, BATCH, 1, 1>
{
    /// Returns the derivative of this matrix w.r.t. the single scalar input,
    /// as a real matrix of shape `ROWS` x `COLS`.
    ///
    /// For example, if `f: ℝ -> ℝ³ˣ²`, then `curve_derivative()` is a 3x2 real matrix.
    fn curve_derivative(&self) -> S::RealMatrix<ROWS, COLS>;
}

#[test]
fn dual_matrix_tests() {
    #[cfg(feature = "simd")]
    use crate::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;
    use crate::{
        dual::DualScalar,
        linalg::EPS_F64,
        maps::MatrixValuedMatrixMap,
        prelude::IsScalar,
    };

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
                    let m_2x4 =
                        <$scalar as IsScalar<$batch, 0, 0>>::Matrix::<2, 4>::from_f64_array2([
                            [1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                        ]);
                    let m_4x1 =
                        <$scalar as IsScalar<$batch, 0, 0>>::Matrix::<4, 1>::from_f64_array2([
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
                    let finite_diff = MatrixValuedMatrixMap::<$scalar, $batch>::sym_diff_quotient(
                        |x| mat_mul_fn::<$scalar, $batch, 0, 0>(x, m_4x1),
                        m_2x4,
                        EPS_F64,
                    );
                    let auto_grad = mat_mul_fn::<$dual_scalar_2_4, $batch, 2, 4>(
                        <$dual_scalar_2_4>::matrix_var(m_2x4),
                        <$dual_scalar_2_4 as IsScalar<$batch, 2, 4>>::Matrix::from_real_matrix(
                            m_4x1,
                        ),
                    )
                    .derivative();

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
                        |x| mat_mul_fn::<$scalar, $batch, 0, 0>(m_2x4, x),
                        m_4x1,
                        EPS_F64,
                    );
                    let auto_grad = mat_mul_fn::<$dual_scalar_4_1, $batch, 4, 1>(
                        <$dual_scalar_4_1 as IsScalar<$batch, 4, 1>>::Matrix::from_real_matrix(
                            m_2x4,
                        ),
                        <$dual_scalar_4_1>::matrix_var(m_4x1),
                    )
                    .derivative();

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

                    let m_4x4 =
                        <$scalar as IsScalar<$batch, 0, 0>>::Matrix::<4, 4>::from_f64_array2([
                            [1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                        ]);

                    let finite_diff = MatrixValuedMatrixMap::<$scalar, $batch>::sym_diff_quotient(
                        mat_mul2_fn::<$scalar, $batch, 0, 0>,
                        m_4x4,
                        EPS_F64,
                    );
                    let auto_grad = mat_mul2_fn::<$dual_scalar_4_4, $batch, 4, 4>(
                        <$dual_scalar_4_4>::matrix_var(m_4x4),
                    )
                    .derivative();

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
