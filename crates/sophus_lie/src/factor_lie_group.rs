use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::traits::IsRealLieFactorGroupImpl;
use crate::Rotation2;
use crate::Rotation3;
use approx::assert_relative_eq;
use sophus_core::calculus::dual::DualBatchScalar;
use sophus_core::calculus::dual::DualScalar;
use sophus_core::calculus::maps::MatrixValuedMapFromVector;
use sophus_core::linalg::BatchScalarF64;
use sophus_core::manifold::traits::TangentImpl;

impl<
        S: IsRealScalar<BATCH_SIZE, RealScalar = S>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const BATCH_SIZE: usize,
        G: IsRealLieFactorGroupImpl<S, DOF, PARAMS, POINT, BATCH_SIZE>,
    > LieGroup<S, DOF, PARAMS, POINT, POINT, BATCH_SIZE, G>
{
    /// V matrix - used in the exponential map
    pub fn mat_v(tangent: &S::Vector<DOF>) -> S::Matrix<POINT, POINT> {
        G::mat_v(tangent)
    }

    /// V matrix inverse - used in the logarithmic map
    pub fn mat_v_inverse(tangent: &S::Vector<DOF>) -> S::Matrix<POINT, POINT> {
        G::mat_v_inverse(tangent)
    }

    /// derivative of V matrix
    pub fn dx_mat_v(tangent: &S::Vector<DOF>) -> [S::Matrix<POINT, POINT>; DOF] {
        G::dx_mat_v(tangent)
    }

    /// derivative of V matrix inverse
    pub fn dx_mat_v_inverse(tangent: &S::Vector<DOF>) -> [S::Matrix<POINT, POINT>; DOF] {
        G::dx_mat_v_inverse(tangent)
    }

    /// derivative of V matrix times point
    pub fn dparams_matrix_times_point(
        params: &S::Vector<PARAMS>,
        point: &S::Vector<POINT>,
    ) -> S::Matrix<POINT, PARAMS> {
        G::dparams_matrix_times_point(params, point)
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
                use crate::traits::IsLieGroup;
                use sophus_core::linalg::scalar::IsScalar;

                const POINT: usize = <$group>::POINT;

                for t in <$group>::tangent_examples() {
                    let mat_v = Self::mat_v(&t);
                    let mat_v_inverse = Self::mat_v_inverse(&t);

                    assert_relative_eq!(
                        mat_v.mat_mul(mat_v_inverse),
                        <$scalar as IsScalar<$batch>>::Matrix::<POINT, POINT>::identity(),
                        epsilon = 0.0001
                    );
                }
            }

            fn test_mat_v_jacobian() {
                use crate::traits::IsLieGroup;
                use sophus_core::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
                use sophus_core::linalg::scalar::IsScalar;
                use sophus_core::linalg::vector::IsVector;
                use sophus_core::params::HasParams;
                use sophus_core::points::example_points;

                const DOF: usize = <$group>::DOF;
                const POINT: usize = <$group>::POINT;
                const PARAMS: usize = <$group>::PARAMS;
                use sophus_core::tensor::tensor_view::IsTensorLike;

                for t in <$group>::tangent_examples() {
                    let mat_v_jacobian = Self::dx_mat_v(&t);

                    let mat_v_x = |t: <$scalar as IsScalar<$batch>>::Vector<DOF>|
                        -> <$scalar as IsScalar<$batch>>::Matrix<POINT, POINT>
                    {
                        Self::mat_v(&t)
                    };

                    let num_diff = MatrixValuedMapFromVector::<$scalar, $batch>::sym_diff_quotient(
                        mat_v_x, t, 0.0001,
                    );

                    for i in 0..DOF {
                        println!("i: {}", i);
                        assert_relative_eq!(mat_v_jacobian[i], num_diff.get([i]), epsilon = 0.001);
                    }

                    let mat_v_inv_jacobian = Self::dx_mat_v_inverse(&t);

                    let mat_v_x_inv = |t: <$scalar as IsScalar<$batch>>::Vector<DOF>|
                       -> <$scalar as IsScalar<$batch>>::Matrix<POINT, POINT> { Self::mat_v_inverse(&t) };
                    let num_diff = MatrixValuedMapFromVector::sym_diff_quotient(mat_v_x_inv, t, 0.0001);

                    for i in 0..DOF {
                        println!("i: {}", i);
                        assert_relative_eq!(mat_v_inv_jacobian[i], num_diff.get([i]), epsilon = 0.001);
                    }
                }
                for p in example_points::<$scalar, POINT, $batch>() {
                    for a in Self::element_examples() {
                        let dual_p =
                            <$dual_scalar as IsScalar<$batch>>::Vector::from_real_vector(p.clone());

                        let dual_fn = |x: <$dual_scalar as IsScalar<$batch>>::Vector<PARAMS>|
                            -> <$dual_scalar as IsScalar<$batch>>::Vector<POINT>
                            {
                                <$dual_group>::from_params(&x).matrix() * dual_p.clone()
                            };

                        let auto_diff =
                            VectorValuedMapFromVector::<$dual_scalar, $batch>::static_fw_autodiff
                            (
                                dual_fn,
                                *a.params(),
                            );
                        let analytic_diff = Self::dparams_matrix_times_point(a.params(), &p);
                        assert_relative_eq!(analytic_diff, auto_diff, epsilon = 0.001);
                    }
                }
            }
        }
    };
}

def_real_group_test_template!(f64, DualScalar, Rotation2<f64, 1>, Rotation2<DualScalar, 1>,  1);
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8>,
    Rotation2<BatchScalarF64<8>, 8>,
    Rotation2<DualBatchScalar<8>, 8>,
    8
);

def_real_group_test_template!(f64, DualScalar, Rotation3<f64, 1>, Rotation3<DualScalar, 1>,  1);
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8>,
    Rotation3<BatchScalarF64<8>, 8>,
    Rotation3<DualBatchScalar<8>, 8>,
    8
);
