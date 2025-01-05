use super::scalar::IsDualScalar;
use crate::linalg::SMat;
use crate::linalg::SVec;
use crate::prelude::IsRealScalar;
use crate::prelude::IsVector;

/// Derivative of a vector-valued map.
pub struct VectorValuedDerivative<
    S,
    const OUTROWS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    /// output vector of input matrices
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
    /// zero
    pub fn zeros() -> Self {
        VectorValuedDerivative {
            out_vec: SVec::zeros(),
        }
    }
}

/// Trait for scalar dual numbers
pub trait IsDualVector<
    S: IsDualScalar<BATCH, DM, DN>,
    const ROWS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: IsVector<S, ROWS, BATCH, DM, DN>
{
    /// Create a new dual vector from a real vector for auto-differentiation with respect to self
    ///
    /// Typically this is not called directly, but through using a map auto-differentiation call:
    ///
    ///  - ScalarValuedVectorMap::fw_autodiff(...);
    ///  - VectorValuedVectorMap::fw_autodiff(...);
    ///  - MatrixValuedVectorMap::fw_autodiff(...);
    fn var(val: S::RealVector<ROWS>) -> Self;

    /// Get the derivative
    fn derivative(&self) -> VectorValuedDerivative<S::RealScalar, ROWS, BATCH, DM, DN>;
}

/// Trait for scalar dual numbers
pub trait HasJacobian<
    S: IsDualScalar<BATCH, DM, 1>,
    const ROWS: usize,
    const BATCH: usize,
    const DM: usize,
>: IsVector<S, ROWS, BATCH, DM, 1>
{
    /// Get the derivative
    fn jacobian(&self) -> S::RealMatrix<ROWS, DM>;
}

#[test]
fn dual_vector_tests() {
    use crate::dual::dual_scalar::DualScalar;
    #[cfg(feature = "simd")]
    use crate::dual::DualBatchScalar;
    use crate::linalg::vector::IsVector;
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;
    use crate::linalg::EPS_F64;
    use crate::maps::scalar_valued_maps::ScalarValuedVectorMap;
    use crate::maps::vector_valued_maps::VectorValuedVectorMap;
    use crate::points::example_points;
    use crate::prelude::IsScalar;

    #[cfg(test)]
    trait Test {
        fn run();
    }

    macro_rules! def_test_template {
        ( $scalar:ty, $dual_scalar4_1: ty, $batch:literal
    ) => {
            #[cfg(test)]
            impl Test for $scalar {
                fn run() {
                    let points = example_points::<$scalar, 4, $batch,0,0>();

                    for p in points.clone() {
                        for p1 in points.clone() {
                            {
                                fn dot_fn<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>(
                                    x: S::Vector<4>,
                                    y: S::Vector<4>,
                                ) -> S {
                                    x.dot(y)
                                }
                                let finite_diff =
                                    ScalarValuedVectorMap::<$scalar, $batch,0,0>::sym_diff_quotient(
                                        |x| dot_fn(x, p1),
                                        p,
                                        EPS_F64,
                                    );
                                let auto_grad =
                                    ScalarValuedVectorMap::<$dual_scalar4_1, $batch,4,1>::fw_autodiff(
                                        |x| {
                                            dot_fn(
                                                x,
                                                <$dual_scalar4_1 as IsScalar<$batch,4,1>>::Vector::<4>::from_real_vector(p1),
                                            )
                                        },
                                        p,
                                    );
                                approx::assert_abs_diff_eq!(
                                    finite_diff,
                                    auto_grad,
                                    epsilon = 0.0001
                                );
                            }

                            fn dot_fn<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>(x: S::Vector<4>, s: S) -> S::Vector<4> {
                                x.scaled(s)
                            }
                            let finite_diff = VectorValuedVectorMap::<$scalar, $batch,0,0>::sym_diff_quotient_jacobian(
                                |x| dot_fn::<$scalar, $batch,0,0>(x, <$scalar>::from_f64(0.99)),
                                p,
                                EPS_F64,
                            );
                            let auto_grad = VectorValuedVectorMap::<$dual_scalar4_1, $batch,4,1>::fw_autodiff_jacobian(
                                |x| dot_fn::<$dual_scalar4_1, $batch,4,1>(x, <$dual_scalar4_1>::from_f64(0.99)),
                                p,
                            );
                            approx::assert_abs_diff_eq!(
                                finite_diff,
                                auto_grad,
                                epsilon = 0.0001
                            );


                            let finite_diff = VectorValuedVectorMap::<$scalar, $batch,0,0>::sym_diff_quotient_jacobian(
                                |x| dot_fn::<$scalar, $batch,0,0>(p1, x[0]),
                                p,
                                EPS_F64,
                            );
                            let auto_grad = VectorValuedVectorMap::<$dual_scalar4_1, $batch,4,1>::fw_autodiff_jacobian(
                                |x| {
                                    dot_fn::<$dual_scalar4_1, $batch,4,1>(
                                        <$dual_scalar4_1 as IsScalar<$batch,4,1>>::Vector::from_real_vector(p1),
                                        x.get_elem(0),
                                    )
                                },
                                p,
                            );
                            approx::assert_abs_diff_eq!(
                                finite_diff,
                                auto_grad,
                                epsilon = 0.0001
                            );
                        }
                    }
                }
            }
        };
    }

    def_test_template!(f64, DualScalar<4,1>, 1);
    #[cfg(feature = "simd")]
    def_test_template!(BatchScalarF64<2>, DualBatchScalar<2, 4, 1>, 2);
    #[cfg(feature = "simd")]
    def_test_template!(BatchScalarF64<4>, DualBatchScalar<4, 4, 1>, 4);

    f64::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<2>::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<4>::run();
}
