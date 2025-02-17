use crate::{
    dual::VectorValuedDerivative,
    linalg::SVec,
    prelude::*,
};

/// Vector-valued map on a vector space.
///
/// This is a function which takes a vector and returns a vector:
///
///  `f: ℝᵐ -> ℝʳ`
///
/// These functions are also called vector fields (on vector space).
pub struct VectorValuedVectorMap<S: IsScalar<BATCH, 0, 0>, const BATCH: usize> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH, RealScalar = S>, const BATCH: usize> VectorValuedVectorMap<S, BATCH> {
    /// Finite difference quotient of the vector-valued map.
    pub fn sym_diff_quotient_jacobian<TFn, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: S::RealVector<INROWS>,
        eps: f64,
    ) -> S::RealMatrix<OUTROWS, INROWS>
    where
        TFn: Fn(S::RealVector<INROWS>) -> SVec<S, OUTROWS>,
        SVec<S, OUTROWS>: IsVector<S, OUTROWS, BATCH, 0, 0>,
    {
        let mut sjac = S::RealMatrix::<OUTROWS, INROWS>::zeros();

        let eps_v = S::RealScalar::from_f64(eps);

        for r in 0..INROWS {
            let mut a_plus = a;
            a_plus[r] += eps_v;

            let mut a_minus = a;
            a_minus[r] -= eps_v;
            let d = (vector_valued(a_plus) - vector_valued(a_minus))
                .scaled(S::from_f64(1.0 / (2.0 * eps)));

            for c in 0..OUTROWS {
                sjac[(c, r)] = d[(c, 0)];
            }
        }
        sjac
    }

    /// Finite difference quotient of the vector-valued map.
    ///
    /// The derivative is a matrix or rank-2 tensor with shape (Rₒ x Rᵢ).
    pub fn sym_diff_quotient<TFn, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: S::RealVector<INROWS>,
        eps: f64,
    ) -> VectorValuedDerivative<S, OUTROWS, BATCH, INROWS, 1>
    where
        TFn: Fn(S::RealVector<INROWS>) -> SVec<S, OUTROWS>,
        SVec<S, OUTROWS>: IsVector<S, OUTROWS, BATCH, 0, 0>,
    {
        let mut sjac = VectorValuedDerivative::<S, OUTROWS, BATCH, INROWS, 1>::zeros();

        let eps_v = S::RealScalar::from_f64(eps);

        for r in 0..INROWS {
            let mut a_plus = a;
            a_plus[r] += eps_v;

            let mut a_minus = a;
            a_minus[r] -= eps_v;
            let d = (vector_valued(a_plus) - vector_valued(a_minus))
                .scaled(S::from_f64(1.0 / (2.0 * eps)));

            for c in 0..OUTROWS {
                sjac.out_vec[c][r] = d[(c, 0)];
            }
        }
        sjac

        //  todo!()
    }
}

/// Vector-valued map on a product space (= space of matrices).
///
/// This is a function which takes a matrix and returns a vector:
///
///  `f: ℝᵐˣⁿ -> ℝʳ`
///
/// This type of function is also called a vector field (on product spaces).
pub struct VectorValuedMatrixMap<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: core::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH, RealScalar = S>, const BATCH: usize>
    VectorValuedMatrixMap<S, BATCH, 0, 0>
{
    /// Finite difference quotient of the vector-valued map.
    pub fn sym_diff_quotient<TFn, const OUTROWS: usize, const INROWS: usize, const INCOLS: usize>(
        vector_valued: TFn,
        a: S::RealMatrix<INROWS, INCOLS>,
        eps: f64,
    ) -> VectorValuedDerivative<S, OUTROWS, BATCH, INROWS, INCOLS>
    where
        TFn: Fn(S::RealMatrix<INROWS, INCOLS>) -> SVec<S, OUTROWS>,
        SVec<S, OUTROWS>: IsVector<S, OUTROWS, BATCH, 0, 0>,
    {
        let mut out = VectorValuedDerivative::zeros();
        let eps_b = S::RealScalar::from_f64(eps);

        for c in 0..INCOLS {
            for r in 0..INROWS {
                let mut a_plus = a;

                a_plus[(r, c)] += eps_b;

                let mut a_minus = a;

                a_minus[(r, c)] -= eps_b;

                let vv = (vector_valued(a_plus) - vector_valued(a_minus))
                    .scaled(S::from_f64(1.0 / (2.0 * eps)));
                for or in 0..OUTROWS {
                    out.out_vec[or][(r, c)] = vv[or];
                }
            }
        }
        out
    }
}

#[test]
fn vector_valued_map_from_vector_tests() {
    #[cfg(feature = "simd")]
    use crate::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;
    use crate::{
        dual::DualScalar,
        linalg::{
            IsVector,
            EPS_F64,
        },
        maps::vector_valued_maps::{
            VectorValuedMatrixMap,
            VectorValuedVectorMap,
        },
    };

    #[cfg(test)]
    trait Test {
        fn run();
    }

    macro_rules! def_test_template {
        ( $scalar:ty, $dual_scalar: ty,$dual_scalar2: ty, $batch:literal
    ) => {
            #[cfg(test)]
            impl Test for $scalar {
                fn run() {
                    {
                        let a = <$scalar as IsScalar<$batch, 0, 0>>::RealVector::<3>::new(
                            <$scalar>::from_f64(0.6),
                            <$scalar>::from_f64(2.2),
                            <$scalar>::from_f64(1.1),
                        );

                        //       [[ x ]]   [[ x / z ]]
                        //  proj [[ y ]] = [[       ]]
                        //       [[ z ]]   [[ y / z ]]
                        fn proj_fn<
                            S: IsScalar<BATCH, DM, DN>,
                            const BATCH: usize,
                            const DM: usize,
                            const DN: usize,
                        >(
                            v: S::Vector<3>,
                        ) -> S::Vector<2> {
                            let x = IsVector::elem(&v, 0);
                            let y = IsVector::elem(&v, 1);
                            let z = IsVector::elem(&v, 2);
                            S::Vector::<2>::from_array([x / z, y / z])
                        }

                        let sfinite_diff =
                            VectorValuedVectorMap::<$scalar, $batch>::sym_diff_quotient(
                                proj_fn::<$scalar, $batch, 0, 0>,
                                a,
                                EPS_F64,
                            );
                        let sauto_grad =
                            proj_fn::<$dual_scalar, $batch, 3, 1>(<$dual_scalar>::vector_var(a))
                                .derivative();
                        for i in 0..2 {
                            approx::assert_abs_diff_eq!(
                                sfinite_diff.out_vec[i],
                                sauto_grad.out_vec[i],
                                epsilon = 0.0001
                            );
                        }

                        let sfinite_diff =
                            VectorValuedVectorMap::<$scalar, $batch>::sym_diff_quotient_jacobian(
                                proj_fn::<$scalar, $batch, 0, 0>,
                                a,
                                EPS_F64,
                            );
                        let sauto_grad =
                            proj_fn::<$dual_scalar, $batch, 3, 1>(<$dual_scalar>::vector_var(a))
                                .jacobian();

                        approx::assert_abs_diff_eq!(sfinite_diff, sauto_grad, epsilon = 0.0001);
                    }

                    fn f<
                        S: IsScalar<BATCH, DM, DN>,
                        const BATCH: usize,
                        const DM: usize,
                        const DN: usize,
                    >(
                        x: S::Matrix<3, 2>,
                    ) -> S::Vector<4> {
                        let a = x.elem([0, 0]);
                        let b = x.elem([0, 1]);
                        let c = x.elem([1, 0]);
                        let d = x.elem([1, 1]);
                        let e = x.elem([2, 0]);
                        let f = x.elem([2, 1]);

                        S::Vector::<4>::from_array([a + b, c + d, e + f, S::from_f64(1.0)])
                    }

                    let mut mat = <$scalar as IsScalar<$batch, 0, 0>>::RealMatrix::<3, 2>::zeros();
                    mat[(0, 0)] = <$scalar>::from_f64(-4.6);
                    mat[(0, 1)] = <$scalar>::from_f64(-1.6);
                    mat[(1, 0)] = <$scalar>::from_f64(0.6);
                    mat[(1, 1)] = <$scalar>::from_f64(1.6);
                    mat[(2, 0)] = <$scalar>::from_f64(-1.6);
                    mat[(2, 1)] = <$scalar>::from_f64(0.2);

                    let finite_diff =
                        VectorValuedMatrixMap::<$scalar, $batch, 0, 0>::sym_diff_quotient(
                            f::<$scalar, $batch, 0, 0>,
                            mat,
                            EPS_F64,
                        );
                    let auto_grad =
                        f::<$dual_scalar2, $batch, 3, 2>(<$dual_scalar2>::matrix_var(mat))
                            .derivative();
                    for i in 0..3 {
                        approx::assert_abs_diff_eq!(
                            finite_diff.out_vec[(i, 0)],
                            auto_grad.out_vec[(i, 0)],
                            epsilon = 0.0001
                        );
                    }
                }
            }
        };
    }

    def_test_template!(
        f64,
        DualScalar<3, 1>,
        DualScalar<3, 2>,
        1);
    #[cfg(feature = "simd")]
    def_test_template!(
        BatchScalarF64<2>,
        DualBatchScalar<2, 3, 1>,
        DualBatchScalar<2, 3, 2>,
        2);
    #[cfg(feature = "simd")]
    def_test_template!(
        BatchScalarF64<4>,
        DualBatchScalar<4, 3, 1>,
        DualBatchScalar<4, 3, 2>,
        4);

    f64::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<2>::run();
    #[cfg(feature = "simd")]
    BatchScalarF64::<4>::run();
}
