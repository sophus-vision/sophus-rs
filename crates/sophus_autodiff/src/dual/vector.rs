use super::scalar::IsDualScalar;
use crate::{
    linalg::{
        SMat,
        SVec,
    },
    prelude::{
        IsRealScalar,
        IsVector,
    },
};

/// A structure that holds the derivative of a vector-valued map with respect to its inputs.
///
/// For a function `f: X -> ℝʳ`, or similarly a batch variant in SIMD,
/// this structure stores each output dimension’s derivative block.
///
/// - `out_vec[i]` contains a derivative matrix (for dimension `i` of the output).
pub struct VectorValuedDerivative<
    S,
    const OUTROWS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    /// The output vector of dimension `OUTROWS`, where each element is a
    /// matrix of shape `[DM × DN]` storing the derivative for that output lane.
    pub out_vec: SVec<SMat<S, DM, DN>, OUTROWS>,
}

impl<
        S: IsRealScalar<BATCH, RealScalar = S>,
        const OUTROWS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > VectorValuedDerivative<S, OUTROWS, BATCH, DM, DN>
{
    /// Creates a new instance set to all zeros.
    pub fn zeros() -> Self {
        VectorValuedDerivative {
            out_vec: SVec::zeros(),
        }
    }
}

/// A trait for a *dual vector*, supporting forward-mode AD on vector quantities.
///
/// A dual vector is a vector whose entries are themselves dual scalars
/// (with partial derivatives). This trait extends [`IsVector`] with AD-specific methods:
///
/// - Constructing a "variable" vector from a real vector (marking it for differentiation).
/// - Retrieving the entire derivative as a [`VectorValuedDerivative`].
///
/// # Generic Parameters
/// - `S`: A type implementing [`IsDualScalar`].
/// - `ROWS`: Number of entries in the vector.
/// - `BATCH`, `DM`, `DN`: Additional parameters for batch usage and the derivative shape.
pub trait IsDualVector<
    S: IsDualScalar<BATCH, DM, DN>,
    const ROWS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: IsVector<S, ROWS, BATCH, DM, DN>
{
    /// Creates a new dual vector from a purely *real* vector, marking its entries as variables
    /// for auto-differentiation.
    fn var(val: S::RealVector<ROWS>) -> Self;

    /// Returns the derivative as a [`VectorValuedDerivative`].
    ///
    /// If the vector has dimension `ROWS`, each element’s derivative is a matrix of shape `[DM ×
    /// DN]`, so the overall derivative is `[ROWS]` of those matrices.
    fn derivative(&self) -> VectorValuedDerivative<S::RealScalar, ROWS, BATCH, DM, DN>;
}

/// A helper trait marking that this dual vector is the result of a *curve* `f: ℝ -> ℝʳ`,
/// letting you retrieve a simpler derivative form for each dimension of the output.
pub trait IsDualVectorFromCurve<
    S: IsDualScalar<BATCH, 1, 1>, // scalar must represent "DM=1, DN=1" scenario
    const ROWS: usize,
    const BATCH: usize,
>: IsDualVector<S, ROWS, BATCH, 1, 1>
{
    /// Returns the derivative of this vector w.r.t. the single scalar input,
    /// as a real vector of dimension `ROWS`.
    ///
    /// For example, if `f: ℝ -> ℝ^3`, then `curve_derivative()` is a 3D real vector.
    fn curve_derivative(&self) -> S::RealVector<ROWS>;
}

/// A convenience trait for dual vectors that store partial derivatives with respect to a
/// multi-dimensional input, but only a *single* output dimension is used. More precisely, this
/// trait is for `S: IsDualScalar<BATCH, DM, 1>` vectors.
///
/// It exposes a `jacobian()` method returning a real matrix \([ROWS × DM]\).
///
/// # Example
///
/// If the vector dimension is `ROWS`, and each entry is a dual scalar with shape `[DM × 1]`,
/// the overall derivative can be collected in a `[ROWS × DM]` real matrix.
pub trait HasJacobian<
    S: IsDualScalar<BATCH, DM, 1>,
    const ROWS: usize,
    const BATCH: usize,
    const DM: usize,
>: IsVector<S, ROWS, BATCH, DM, 1>
{
    /// Returns the [ROWS × DM] matrix collecting each component's derivative row.
    fn jacobian(&self) -> S::RealMatrix<ROWS, DM>;
}

#[test]
fn dual_vector_tests() {
    #[cfg(feature = "simd")]
    use crate::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;
    use crate::{
        dual::dual_scalar::DualScalar,
        linalg::{
            IsVector,
            EPS_F64,
        },
        maps::{
            ScalarValuedVectorMap,
            VectorValuedVectorMap,
        },
        points::example_points,
        prelude::IsScalar,
    };

    // A trait for test runs:
    #[cfg(test)]
    trait Test {
        fn run();
    }

    // A macro generating test code for both single-lane and batch-lane types.
    macro_rules! def_test_template {
        ($scalar:ty, $dual_scalar4_1: ty, $batch:literal) => {
            #[cfg(test)]
            impl Test for $scalar {
                fn run() {
                    let points = example_points::<$scalar, 4, $batch,0,0>();

                    // We'll test a few scenarios with dot() and scaling, verifying
                    // that the derivative matches finite-difference approximations.
                    for p in points.clone() {
                        for p1 in points.clone() {
                            // 1) dot_fn: compute dot product => scalar
                            {
                                fn dot_fn<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>(
                                    x: S::Vector<4>,
                                    y: S::Vector<4>,
                                ) -> S {
                                    x.dot(y)
                                }

                                let finite_diff =
                                    ScalarValuedVectorMap::<$scalar, $batch>::sym_diff_quotient(
                                        |x| dot_fn(x, p1),
                                        p,
                                        EPS_F64,
                                    );
                                let auto_grad = dot_fn::<$dual_scalar4_1, $batch,4,1>(
                                    <$dual_scalar4_1>::vector_var(p),
                                    <$dual_scalar4_1 as IsScalar<$batch,4,1>>::Vector::<4>::from_real_vector(p1),
                                ).derivative();

                                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                            }

                            // 2) scale a vector: x -> x * 0.99 or x -> p1[0] * x
                            {
                                fn scale_fn<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>(
                                    x: S::Vector<4>,
                                    s: S
                                ) -> S::Vector<4> {
                                    x.scaled(s)
                                }

                                // scale the vector by a constant 0.99
                                let finite_diff = VectorValuedVectorMap::<$scalar, $batch>::sym_diff_quotient_jacobian(
                                    |x| scale_fn::<$scalar, $batch,0,0>(x, <$scalar>::from_f64(0.99)),
                                    p,
                                    EPS_F64,
                                );
                                let auto_grad = scale_fn::<$dual_scalar4_1, $batch,4,1>(
                                    <$dual_scalar4_1>::vector_var(p),
                                    <$dual_scalar4_1>::from_f64(0.99)
                                ).jacobian();

                                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                                // scale the vector by p1[0], which is a single scalar from p1
                                // so we only look at the partial w.r.t. x, not p1
                                let finite_diff = VectorValuedVectorMap::<$scalar, $batch>::sym_diff_quotient_jacobian(
                                    |x| scale_fn::<$scalar, $batch,0,0>(p1, x[0]),
                                    p,
                                    EPS_F64,
                                );
                                // convert p1 to a dual vector, then get the first element
                                let auto_grad = scale_fn::<$dual_scalar4_1, $batch,4,1>(
                                    <$dual_scalar4_1 as IsScalar<$batch,4,1>>::Vector::<4>::from_real_vector(p1),
                                    <$dual_scalar4_1>::vector_var(p).elem(0)
                                ).jacobian();

                                approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                            }
                        }
                    }
                }
            }
        };
    }

    // Generate tests for single-lane f64 and batch-lane batch scalars
    def_test_template!(f64, DualScalar<4,1>, 1);
    #[cfg(feature = "simd")]
    def_test_template!(BatchScalarF64<2>, DualBatchScalar<2, 4, 1>, 2);
    #[cfg(feature = "simd")]
    def_test_template!(BatchScalarF64<4>, DualBatchScalar<4, 4, 1>, 4);

    // Run the tests:
    f64::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<2>::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<4>::run();
}
