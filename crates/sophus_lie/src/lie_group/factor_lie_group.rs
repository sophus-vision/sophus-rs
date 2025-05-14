use core::borrow::Borrow;

use approx::assert_relative_eq;
#[cfg(feature = "simd")]
use sophus_autodiff::dual::DualBatchScalar;
#[cfg(feature = "simd")]
use sophus_autodiff::linalg::BatchScalarF64;
use sophus_autodiff::{
    dual::DualScalar,
    manifold::IsTangent,
    maps::MatrixValuedVectorMap,
};

use crate::{
    IsRealLieFactorGroupImpl,
    Rotation2,
    Rotation3,
    lie_group::LieGroup,
    prelude::*,
};

impl<
    S: IsRealScalar<BATCH, RealScalar = S>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const BATCH: usize,
    G: IsRealLieFactorGroupImpl<S, DOF, PARAMS, POINT, BATCH>,
> LieGroup<S, DOF, PARAMS, POINT, POINT, BATCH, 0, 0, G>
{
    /// V matrix - used in the exponential map
    pub fn mat_v<T>(tangent: T) -> S::Matrix<POINT, POINT>
    where
        T: Borrow<S::Vector<DOF>>,
    {
        G::mat_v(tangent.borrow())
    }

    /// V matrix inverse - used in the logarithmic map
    pub fn mat_v_inverse<T>(tangent: T) -> S::Matrix<POINT, POINT>
    where
        T: Borrow<S::Vector<DOF>>,
    {
        G::mat_v_inverse(tangent.borrow())
    }

    /// derivative of V matrix
    pub fn dx_mat_v<T>(tangent: T) -> [S::Matrix<POINT, POINT>; DOF]
    where
        T: Borrow<S::Vector<DOF>>,
    {
        G::dx_mat_v(tangent.borrow())
    }

    /// derivative of V matrix inverse
    pub fn dx_mat_v_inverse<T>(tangent: T) -> [S::Matrix<POINT, POINT>; DOF]
    where
        T: Borrow<S::Vector<DOF>>,
    {
        G::dx_mat_v_inverse(tangent.borrow())
    }

    /// derivative of V matrix times point
    pub fn dparams_matrix_times_point<PA, PO>(params: PA, point: PO) -> S::Matrix<POINT, PARAMS>
    where
        PA: Borrow<S::Vector<PARAMS>>,
        PO: Borrow<S::Vector<POINT>>,
    {
        G::dparams_matrix_times_point(params.borrow(), point.borrow())
    }
}

/// A trait for Lie groups.
pub trait RealFactorLieGroupTest {
    /// Run all tests.
    fn run_real_factor_tests() {
        Self::mat_v_test();
        Self::test_mat_v_jacobian();
    }

    /// Test mat_v and mat_v_inverse.
    fn mat_v_test();

    /// Test hat and vee operators.
    fn test_mat_v_jacobian();
}

macro_rules! def_real_group_test_template {
    ($scalar:ty, $dual_scalar:ty, $group: ty, $dual_group: ty, $batch:literal
) => {
        impl RealFactorLieGroupTest for $group {
            fn mat_v_test() {
                use crate::IsLieGroup;
                use sophus_autodiff::linalg::IsScalar;

                const POINT: usize = <$group>::POINT;

                for t in <$group>::tangent_examples() {
                    let mat_v = Self::mat_v(&t);
                    let mat_v_inverse = Self::mat_v_inverse(&t);

                    assert_relative_eq!(
                        mat_v.mat_mul(mat_v_inverse),
                        <$scalar as IsScalar<$batch, 0, 0>>::Matrix::<POINT, POINT>::identity(),
                        epsilon = 0.0001
                    );
                }
            }

            fn test_mat_v_jacobian() {
                use crate::IsLieGroup;
                use log::info;
                use sophus_autodiff::linalg::IsScalar;
                use sophus_autodiff::linalg::IsVector;
                use sophus_autodiff::params::HasParams;
                use sophus_autodiff::points::example_points;

                const DOF: usize = <$group>::DOF;
                const POINT: usize = <$group>::POINT;
                const PARAMS: usize = <$group>::PARAMS;

                for t in <$group>::tangent_examples() {
                    let mat_v_jacobian = Self::dx_mat_v(&t);

                    let mat_v_x = |t: <$scalar as IsScalar<$batch,0,0>>::Vector<DOF>|
                                -> <$scalar as IsScalar<$batch,0,0>>::Matrix<POINT, POINT>
                            {
                                Self::mat_v(&t)
                            };

                    let num_diff =
                        MatrixValuedVectorMap::<$scalar, $batch>::sym_diff_quotient(
                            mat_v_x, t, 0.0001,
                        );

                    for i in 0..DOF {
                        for r in 0..POINT {
                            for c in 0..POINT {
                                info!("i: {}", i);
                                assert_relative_eq!(
                                    mat_v_jacobian[i][(r, c)],
                                    num_diff.out_mat[(r, c)][i],
                                    epsilon = 0.001
                                );
                            }
                        }
                    }

                    let mat_v_inv_jacobian = Self::dx_mat_v_inverse(&t);

                    let mat_v_x_inv = |t: <$scalar as IsScalar<$batch,0,0>>::Vector<DOF>|
                       -> <$scalar as IsScalar<$batch,0,0>>::Matrix<POINT, POINT> { Self::mat_v_inverse(&t) };
                    let num_diff = MatrixValuedVectorMap::sym_diff_quotient(mat_v_x_inv, t, 0.0001);

                    for i in 0..DOF {
                        for r in 0..POINT {
                            for c in 0..POINT {
                                info!("i: {}", i);
                                assert_relative_eq!(mat_v_inv_jacobian[i][(r,c)], num_diff.out_mat[(r,c)][i], epsilon = 0.001);
                            }
                        }
                    }
                }
                for p in example_points::<$scalar, POINT, $batch,0,0>() {
                    for a in Self::element_examples() {
                        let dual_p =
                            <$dual_scalar as IsScalar<$batch,PARAMS,1>>::Vector::from_real_vector(p);

                        let dual_fn = |x: <$dual_scalar as IsScalar<$batch,PARAMS,1>>::Vector<PARAMS>|
                            -> <$dual_scalar as IsScalar<$batch,PARAMS,1>>::Vector<POINT>
                            {
                                <$dual_group>::from_params(&x).matrix() * dual_p
                            };

                        let auto_diff =
                            dual_fn(<$dual_scalar>::vector_var(*a.params())).jacobian();
                        let analytic_diff = Self::dparams_matrix_times_point(a.params(), &p);
                        assert_relative_eq!(analytic_diff, auto_diff, epsilon = 0.001);
                    }
                }
            }
        }
    };
}

def_real_group_test_template!(
    f64,
    DualScalar<2,1>,
    Rotation2<f64, 1,0,0>,
    Rotation2<DualScalar<2,1>, 1,2,1>,
    1);
#[cfg(feature = "simd")]
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8, 2, 1>,
    Rotation2<BatchScalarF64<8>, 8, 0, 0>,
    Rotation2<DualBatchScalar<8, 2, 1>, 8, 2, 1>,
    8
);

def_real_group_test_template!(
    f64,
    DualScalar<4, 1>,
    Rotation3<f64, 1, 0, 0>,
    Rotation3<DualScalar<4, 1>, 1, 4, 1>,
    1);
#[cfg(feature = "simd")]
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8, 4, 1>,
    Rotation3<BatchScalarF64<8>, 8, 0, 0>,
    Rotation3<DualBatchScalar<8, 4, 1>, 8, 4, 1>,
    8
);
