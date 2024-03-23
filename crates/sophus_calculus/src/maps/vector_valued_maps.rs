use crate::dual::dual_matrix::DualM;
use crate::dual::dual_vector::DualV;
use crate::types::MatF64;
use crate::types::VecF64;

use sophus_tensor::mut_tensor::MutTensorDDR;
use sophus_tensor::mut_tensor::MutTensorDR;
use sophus_tensor::mut_view::IsMutTensorLike;
use sophus_tensor::view::IsTensorLike;

use std::marker::PhantomData;

/// Vector-valued map on a vector space.
///
/// This is a function which takes a vector and returns a vector:
///
///  f: ℝᵐ -> ℝʳ
///
/// These functions are also called vector fields (on vector space x s).
///
pub struct VectorValuedMapFromVector;

impl VectorValuedMapFromVector {
    /// Finite difference quotient of the vector-valued map.
    ///
    /// The derivative is a matrix or rank-2 tensor with shape (Rₒ x Rᵢ).
    ///
    /// For efficiency reasons, we return the transpose Rᵢ x (Rₒ)
    ///
    pub fn sym_diff_quotient<TFn, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: VecF64<INROWS>,
        eps: f64,
    ) -> MutTensorDR<f64, OUTROWS>
    where
        TFn: Fn(VecF64<INROWS>) -> VecF64<OUTROWS>,
    {
        let mut out = MutTensorDR::<f64, OUTROWS>::from_shape([INROWS]);

        for r in 0..INROWS {
            let mut a_plus = a;
            a_plus[r] += eps;

            let mut a_minus = a;
            a_minus[r] -= eps;

            out.get_mut([r])
                .copy_from(&((vector_valued(a_plus) - vector_valued(a_minus)) / (2.0 * eps)));
        }
        out
    }

    /// Auto differentiation of the vector-valued map.
    pub fn fw_autodiff<TFn, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: VecF64<INROWS>,
    ) -> MutTensorDR<f64, OUTROWS>
    where
        TFn: Fn(DualV<INROWS>) -> DualV<OUTROWS>,
    {
        let d = vector_valued(DualV::v(a)).dij_val;
        if d.is_none() {
            return MutTensorDR::from_shape([INROWS]);
        }

        MutTensorDR {
            mut_array: d.unwrap().mut_array.into_shape([INROWS]).unwrap(),
            phantom: PhantomData,
        }
    }

    /// Finite difference quotient of the vector-valued map.
    ///
    /// The derivative is a matrix or rank-2 tensor with shape (Rₒ x Rᵢ).
    ///
    pub fn static_sym_diff_quotient<TFn, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: VecF64<INROWS>,
        eps: f64,
    ) -> MatF64<OUTROWS, INROWS>
    where
        TFn: Fn(VecF64<INROWS>) -> VecF64<OUTROWS>,
    {
        let jac = Self::sym_diff_quotient(vector_valued, a, eps);
        let mut sjac = MatF64::<OUTROWS, INROWS>::zeros();

        for r in 0..INROWS {
            let v = jac.get([r]);
            sjac.fixed_view_mut::<OUTROWS, 1>(0, r).copy_from(&v);
        }

        sjac
    }

    /// Auto differentiation of the vector-valued map.
    pub fn static_fw_autodiff<TFn, const OUTROWS: usize, const INROWS: usize>(
        vector_valued: TFn,
        a: VecF64<INROWS>,
    ) -> MatF64<OUTROWS, INROWS>
    where
        TFn: Fn(DualV<INROWS>) -> DualV<OUTROWS>,
    {
        let jac = Self::fw_autodiff(vector_valued, a);
        let mut sjac = MatF64::<OUTROWS, INROWS>::zeros();

        for r in 0..INROWS {
            let v = jac.get([r]);
            sjac.fixed_view_mut::<OUTROWS, 1>(0, r).copy_from(&v);
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
pub struct VectorValuedMapFromMatrix;

impl VectorValuedMapFromMatrix {
    /// Finite difference quotient of the vector-valued map.
    ///
    /// The derivative is a matrix or rank-3 tensor with shape (Rₒ x Rᵢ x Cᵢ).
    ///
    /// For efficiency reasons, we return Rᵢ x Cᵢ x (Rₒ)
    ///
    pub fn sym_diff_quotient<TFn, const OUTROWS: usize, const INROWS: usize, const INCOLS: usize>(
        vector_valued: TFn,
        a: MatF64<INROWS, INCOLS>,
        eps: f64,
    ) -> MutTensorDDR<f64, OUTROWS>
    where
        TFn: Fn(MatF64<INROWS, INCOLS>) -> VecF64<OUTROWS>,
    {
        let mut out = MutTensorDDR::<f64, OUTROWS>::from_shape([INROWS, INCOLS]);

        for c in 0..INCOLS {
            for r in 0..INROWS {
                let mut a_plus = a;

                a_plus[(r, c)] += eps;

                let mut a_minus = a;

                a_minus[(r, c)] -= eps;

                let vv = (vector_valued(a_plus) - vector_valued(a_minus)) / (2.0 * eps);
                *out.mut_view().get_mut([r, c]) = vv;
            }
        }
        out
    }

    /// Auto differentiation of the vector-valued map.
    pub fn fw_autodiff<TFn, const OUTROWS: usize, const INROWS: usize, const INCOLS: usize>(
        vector_valued: TFn,
        a: MatF64<INROWS, INCOLS>,
    ) -> MutTensorDDR<f64, OUTROWS>
    where
        TFn: Fn(DualM<INROWS, INCOLS>) -> DualV<OUTROWS>,
    {
        vector_valued(DualM::v(a)).dij_val.unwrap()
    }
}

mod test {

    #[test]
    fn test_batched_vector_valued_map_from_vector() {
        use crate::maps::vector_valued_maps::VectorValuedMapFromVector;
        use crate::types::scalar::IsScalar;
        use crate::types::vector::IsVector;
        use crate::types::VecF64;
        use sophus_tensor::view::IsTensorLike;

        let a = VecF64::<3>::new(0.6, 2.2, 1.1);

        //       [[ x ]]   [[ x / z ]]
        //  proj [[ y ]] = [[       ]]
        //       [[ z ]]   [[ y / z ]]
        fn proj_fn<S: IsScalar<1>, V2: IsVector<S, 2, 1>, V3: IsVector<S, 3, 1>>(v: V3) -> V2 {
            let x = v.get(0);
            let y = v.get(1);
            let z = v.get(2);

            V2::from_array([x / z.clone(), y / z])
        }

        let finite_diff = VectorValuedMapFromVector::sym_diff_quotient(proj_fn, a, 1e-6);
        let auto_grad = VectorValuedMapFromVector::fw_autodiff(proj_fn, a);
        for i in 0..2 {
            approx::assert_abs_diff_eq!(finite_diff.get([i]), auto_grad.get([i]), epsilon = 0.0001);
        }

        let sfinite_diff = VectorValuedMapFromVector::static_sym_diff_quotient(proj_fn, a, 1e-6);
        let sauto_grad = VectorValuedMapFromVector::static_fw_autodiff(proj_fn, a);
        approx::assert_abs_diff_eq!(sfinite_diff, sauto_grad, epsilon = 0.0001);
    }

    #[test]
    fn test_batched_vector_valued_map_from_matrix() {
        use crate::maps::vector_valued_maps::VectorValuedMapFromMatrix;
        use crate::types::matrix::IsMatrix;
        use crate::types::scalar::IsScalar;
        use crate::types::vector::IsVector;
        use crate::types::MatF64;
        use sophus_tensor::view::IsTensorLike;

        fn f<S: IsScalar<1>, M32: IsMatrix<S, 3, 2, 1>, V4: IsVector<S, 4, 1>>(x: M32) -> V4 {
            let a = x.get((0, 0));
            let b = x.get((0, 1));
            let c = x.get((1, 0));
            let d = x.get((1, 1));
            let e = x.get((2, 0));
            let f = x.get((2, 1));

            V4::from_array([a + b, c + d, e + f, S::c(1.0)])
        }

        let mut mat = MatF64::<3, 2>::zeros();
        mat[(0, 0)] = -4.6;
        mat[(0, 1)] = -1.6;
        mat[(1, 0)] = 0.6;
        mat[(1, 1)] = 1.6;
        mat[(2, 0)] = -1.6;
        mat[(2, 1)] = 0.2;

        let finite_diff = VectorValuedMapFromMatrix::sym_diff_quotient(f, mat, 1e-6);
        let auto_grad = VectorValuedMapFromMatrix::fw_autodiff(f, mat);
        approx::assert_abs_diff_eq!(
            finite_diff.elem_view(),
            auto_grad.elem_view(),
            epsilon = 0.0001
        );
    }
}
