use crate::linalg::SVec;
use crate::prelude::*;
use crate::tensor::mut_tensor::MutTensorDDR;
use crate::tensor::mut_tensor::MutTensorDR;
use std::marker::PhantomData;

/// Vector-valued map on a vector space.
///
/// This is a function which takes a vector and returns a vector:
///
///  f: ℝᵐ -> ℝʳ
///
/// These functions are also called vector fields (on vector space).
///
pub struct VectorValuedMapFromVector<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: std::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH, RealScalar = S>, const BATCH: usize>
    VectorValuedMapFromVector<S, BATCH>
{
    /// Finite difference quotient of the vector-valued map.
    ///
    /// The derivative is a matrix or rank-2 tensor with shape (Rₒ x Rᵢ).
    ///
    /// For efficiency reasons, we return the transpose Rᵢ x (Rₒ)
    ///
    pub fn sym_diff_quotient<TFn, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: S::RealVector<INROWS>,
        eps: f64,
    ) -> MutTensorDR<S, OUTROWS>
    where
        TFn: Fn(S::RealVector<INROWS>) -> SVec<S, OUTROWS>,
        SVec<S, OUTROWS>: IsVector<S, OUTROWS, BATCH>,
    {
        let mut out = MutTensorDR::<S, OUTROWS>::from_shape([INROWS]);
        let eps_v = S::RealScalar::from_f64(eps);

        for r in 0..INROWS {
            let mut a_plus = a;
            a_plus[r] += eps_v;

            let mut a_minus = a;
            a_minus[r] -= eps_v;
            let d = (vector_valued(a_plus) - vector_valued(a_minus))
                .scaled(S::from_f64(1.0 / (2.0 * eps)));

            out.get_mut([r]).copy_from(&d);
        }
        out
    }

    /// Finite difference quotient of the vector-valued map.
    ///
    /// The derivative is a matrix or rank-2 tensor with shape (Rₒ x Rᵢ).
    ///
    pub fn static_sym_diff_quotient<TFn, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: S::RealVector<INROWS>,
        eps: f64,
    ) -> S::RealMatrix<OUTROWS, INROWS>
    where
        TFn: Fn(S::RealVector<INROWS>) -> SVec<S, OUTROWS>,
        SVec<S, OUTROWS>: IsVector<S, OUTROWS, BATCH>,
    {
        let jac = Self::sym_diff_quotient(vector_valued, a, eps);
        let mut sjac = S::RealMatrix::<OUTROWS, INROWS>::zeros();

        for r in 0..INROWS {
            let v = jac.get([r]);
            for c in 0..OUTROWS {
                sjac[(c, r)] = v[c];
            }
        }

        sjac

        // todo!()
    }
}

impl<D: IsDualScalar<BATCH, DualScalar = D>, const BATCH: usize>
    VectorValuedMapFromVector<D, BATCH>
{
    /// Auto differentiation of the vector-valued map.
    pub fn fw_autodiff<TFn, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: D::RealVector<INROWS>,
    ) -> MutTensorDR<D::RealScalar, OUTROWS>
    where
        TFn: Fn(D::DualVector<INROWS>) -> D::DualVector<OUTROWS>,
    {
        let v = vector_valued(D::vector_with_dij(a));
        let d = v.dij_val();
        if d.is_none() {
            return MutTensorDR::from_shape([INROWS]);
        }

        MutTensorDR {
            mut_array: d.unwrap().mut_array.into_shape([INROWS]).unwrap(),
            phantom: PhantomData,
        }
    }

    /// Auto differentiation of the vector-valued map.
    pub fn static_fw_autodiff<TFn, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: D::RealVector<INROWS>,
    ) -> D::RealMatrix<OUTROWS, INROWS>
    where
        TFn: Fn(D::DualVector<INROWS>) -> D::DualVector<OUTROWS>,
    {
        let jac = Self::fw_autodiff(vector_valued, a);
        let mut sjac = D::RealMatrix::<OUTROWS, INROWS>::zeros();

        for r in 0..INROWS {
            let v = jac.get([r]);
            for c in 0..OUTROWS {
                sjac[(c, r)] = v[c];
            }
        }

        sjac
    }
}

/// Vector-valued map on a product space (= space of matrices).
///
/// This is a function which takes a matrix and returns a vector:
///
///  f: ℝᵐ x ℝⁿ -> ℝʳ
///
/// This type of function is also called a vector field (on product spaces).
///
pub struct VectorValuedMapFromMatrix<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: std::marker::PhantomData<S>,
}

impl<S: IsRealScalar<BATCH>, const BATCH: usize> VectorValuedMapFromMatrix<S, BATCH> {
    /// Finite difference quotient of the vector-valued map.
    ///
    /// The derivative is a matrix or rank-3 tensor with shape (Rₒ x Rᵢ x Cᵢ).
    ///
    /// For efficiency reasons, we return Rᵢ x Cᵢ x (Rₒ)
    ///
    pub fn sym_diff_quotient<TFn, const OUTROWS: usize, const INROWS: usize, const INCOLS: usize>(
        vector_valued: TFn,
        a: S::RealMatrix<INROWS, INCOLS>,
        eps: f64,
    ) -> MutTensorDDR<S, OUTROWS>
    where
        TFn: Fn(S::RealMatrix<INROWS, INCOLS>) -> SVec<S, OUTROWS>,
        SVec<S, OUTROWS>: IsVector<S, OUTROWS, BATCH>,
    {
        let mut out = MutTensorDDR::<S, OUTROWS>::from_shape([INROWS, INCOLS]);
        let eps_b = S::RealScalar::from_f64(eps);

        for c in 0..INCOLS {
            for r in 0..INROWS {
                let mut a_plus = a;

                a_plus[(r, c)] += eps_b;

                let mut a_minus = a;

                a_minus[(r, c)] -= eps_b;

                let vv = (vector_valued(a_plus) - vector_valued(a_minus))
                    .scaled(S::from_f64(1.0 / (2.0 * eps)));
                *out.mut_view().get_mut([r, c]) = vv;
            }
        }
        out
    }
}

impl<D: IsDualScalar<BATCH, DualScalar = D>, const BATCH: usize>
    VectorValuedMapFromMatrix<D, BATCH>
{
    /// Auto differentiation of the vector-valued map.
    pub fn fw_autodiff<TFn, const OUTROWS: usize, const INROWS: usize, const INCOLS: usize>(
        vector_valued: TFn,
        a: D::RealMatrix<INROWS, INCOLS>,
    ) -> MutTensorDDR<D::RealScalar, OUTROWS>
    where
        TFn: Fn(D::DualMatrix<INROWS, INCOLS>) -> D::DualVector<OUTROWS>,
    {
        vector_valued(D::matrix_with_dij(a)).dij_val().unwrap()
    }
}

#[test]
fn vector_valued_map_from_vector_tests() {
    use crate::calculus::dual::dual_scalar::DualBatchScalar;
    use crate::calculus::dual::dual_scalar::DualScalar;
    use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromMatrix;
    use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;

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
                        let a = <$scalar as IsScalar<$batch>>::RealVector::<3>::new(
                            <$scalar>::from_f64(0.6),
                            <$scalar>::from_f64(2.2),
                            <$scalar>::from_f64(1.1),
                        );

                        //       [[ x ]]   [[ x / z ]]
                        //  proj [[ y ]] = [[       ]]
                        //       [[ z ]]   [[ y / z ]]
                        fn proj_fn<S: IsScalar<BATCH>, const BATCH: usize>(
                            v: S::Vector<3>,
                        ) -> S::Vector<2> {
                            let x = IsVector::get_elem(&v, 0);
                            let y = IsVector::get_elem(&v, 1);
                            let z = IsVector::get_elem(&v, 2);
                            S::Vector::<2>::from_array([x / z.clone(), y / z])
                        }

                        let finite_diff =
                            VectorValuedMapFromVector::<$scalar, $batch>::sym_diff_quotient(
                                proj_fn::<$scalar, $batch>,
                                a,
                                1e-6,
                            );
                        let auto_grad =
                            VectorValuedMapFromVector::<$dual_scalar, $batch>::fw_autodiff(
                                proj_fn::<$dual_scalar, $batch>,
                                a,
                            );
                        for i in 0..2 {
                            approx::assert_abs_diff_eq!(
                                finite_diff.get([i]),
                                auto_grad.get([i]),
                                epsilon = 0.0001
                            );
                        }

                        let sfinite_diff =
                            VectorValuedMapFromVector::<$scalar, $batch>::static_sym_diff_quotient(
                                proj_fn::<$scalar, $batch>,
                                a,
                                1e-6,
                            );
                        let sauto_grad =
                            VectorValuedMapFromVector::<$dual_scalar, $batch>::static_fw_autodiff(
                                proj_fn::<$dual_scalar, $batch>,
                                a,
                            );
                        approx::assert_abs_diff_eq!(sfinite_diff, sauto_grad, epsilon = 0.0001);
                    }

                    fn f<S: IsScalar<BATCH>, const BATCH: usize>(
                        x: S::Matrix<3, 2>,
                    ) -> S::Vector<4> {
                        let a = x.get_elem([0, 0]);
                        let b = x.get_elem([0, 1]);
                        let c = x.get_elem([1, 0]);
                        let d = x.get_elem([1, 1]);
                        let e = x.get_elem([2, 0]);
                        let f = x.get_elem([2, 1]);

                        S::Vector::<4>::from_array([a + b, c + d, e + f, S::from_f64(1.0)])
                    }

                    let mut mat = <$scalar as IsScalar<$batch>>::RealMatrix::<3, 2>::zeros();
                    mat[(0, 0)] = <$scalar>::from_f64(-4.6);
                    mat[(0, 1)] = <$scalar>::from_f64(-1.6);
                    mat[(1, 0)] = <$scalar>::from_f64(0.6);
                    mat[(1, 1)] = <$scalar>::from_f64(1.6);
                    mat[(2, 0)] = <$scalar>::from_f64(-1.6);
                    mat[(2, 1)] = <$scalar>::from_f64(0.2);

                    let finite_diff =
                        VectorValuedMapFromMatrix::<$scalar, $batch>::sym_diff_quotient(
                            f::<$scalar, $batch>,
                            mat,
                            1e-6,
                        );
                    let auto_grad = VectorValuedMapFromMatrix::<$dual_scalar, $batch>::fw_autodiff(
                        f::<$dual_scalar, $batch>,
                        mat,
                    );
                    approx::assert_abs_diff_eq!(
                        finite_diff.elem_view(),
                        auto_grad.elem_view(),
                        epsilon = 0.0001
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
