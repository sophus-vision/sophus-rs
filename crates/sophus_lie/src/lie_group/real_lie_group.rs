use crate::groups::isometry2::Isometry2F64;
use crate::groups::isometry3::Isometry3F64;
use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::traits::IsLieGroupImpl;
use crate::traits::IsRealLieGroupImpl;
use crate::Isometry2;
use crate::Isometry3;
use crate::Rotation2;
use crate::Rotation3;
use approx::assert_relative_eq;
use nalgebra::SVector;
use sophus_core::calculus::dual::DualScalar;
use sophus_core::calculus::maps::MatrixValuedMapFromVector;
use sophus_core::calculus::maps::VectorValuedMapFromMatrix;
use sophus_core::calculus::maps::VectorValuedMapFromVector;

extern crate alloc;

#[cfg(feature = "simd")]
use sophus_core::calculus::dual::dual_batch_scalar::DualBatchScalar;
#[cfg(feature = "simd")]
use sophus_core::linalg::BatchScalarF64;

use core::borrow::Borrow;
use core::fmt::Display;
use core::fmt::Formatter;

impl<
        S: IsRealScalar<BATCH_SIZE, RealScalar = S>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsRealLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
    > LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
where
    SVector<S, DOF>: IsVector<S, DOF, BATCH_SIZE>,
{
    /// derivative of exponential map at the identity
    pub fn dx_exp_x_at_0() -> S::Matrix<PARAMS, DOF> {
        G::dx_exp_x_at_0()
    }

    /// derivative of exponential map times point at the identity
    pub fn dx_exp_x_times_point_at_0<P>(point: P) -> S::Matrix<POINT, DOF>
    where
        P: Borrow<S::Vector<POINT>>,
    {
        G::dx_exp_x_times_point_at_0(point.borrow())
    }

    /// are there multiple shortest paths to the identity?
    pub fn has_shortest_path_ambiguity(&self) -> S::Mask {
        G::has_shortest_path_ambiguity(&self.params)
    }

    /// derivative of exponential map
    pub fn dx_exp<T>(tangent: T) -> S::Matrix<PARAMS, DOF>
    where
        T: Borrow<S::Vector<DOF>>,
    {
        G::dx_exp(tangent.borrow())
    }

    /// derivative of logarithmic map
    pub fn dx_log_x<P>(params: P) -> S::Matrix<DOF, PARAMS>
    where
        P: Borrow<S::Vector<PARAMS>>,
    {
        G::dx_log_x(params.borrow())
    }

    /// dual representation of the group
    pub fn to_dual_c(
        self,
    ) -> LieGroup<S::DualScalar, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G::DualG> {
        LieGroup::from_params(self.params.to_dual())
    }

    /// derivative of log(exp(x)) at the identity
    pub fn dx_log_a_exp_x_b_at_0<A, B>(a: A, b: B) -> S::Matrix<DOF, DOF>
    where
        A: Borrow<Self>,
        B: Borrow<Self>,
    {
        let a = a.borrow();
        let b = b.borrow();
        let ab = a * b;
        Self::dx_log_x(ab.params())
            .mat_mul(Self::da_a_mul_b(Self::identity(), ab))
            .mat_mul(Self::dx_exp_x_at_0())
            .mat_mul(Self::adj(a))
    }

    /// derivative of group multiplication with respect to the first argument
    pub fn da_a_mul_b<A, B>(a: A, b: B) -> S::Matrix<PARAMS, PARAMS>
    where
        A: Borrow<Self>,
        B: Borrow<Self>,
    {
        let a = a.borrow();
        let b = b.borrow();
        G::da_a_mul_b(a.params(), b.params())
    }

    /// derivative of group multiplication with respect to the second argument
    pub fn db_a_mul_b<A, B>(a: A, b: B) -> S::Matrix<PARAMS, PARAMS>
    where
        A: Borrow<Self>,
        B: Borrow<Self>,
    {
        let a = a.borrow();
        let b = b.borrow();
        G::db_a_mul_b(a.params(), b.params())
    }

    /// derivative of matrix representation with respect to the internal parameters
    ///
    /// precondition: column index in [0, AMBIENT-1]
    pub fn dparams_matrix(&self, col_idx: usize) -> S::Matrix<POINT, PARAMS> {
        G::dparams_matrix(&self.params, col_idx)
    }

    /// derivative of matrix representation: (exp(x) * T).col(i) at x=0
    ///
    /// precondition: column index in [0, AMBIENT-1]
    pub fn dx_exp_x_time_matrix_at_0(&self, col_idx: usize) -> S::Matrix<POINT, DOF> {
        self.dparams_matrix(col_idx)
            .mat_mul(Self::da_a_mul_b(Self::identity(), self))
            .mat_mul(&Self::dx_exp_x_at_0())
    }
}

impl<
        S: IsScalar<BATCH_SIZE>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
    > Display for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.compact())
    }
}

/// A trait for Lie groups.
pub trait RealLieGroupTest {
    /// Run all tests.
    fn run_real_tests() {
        Self::adjoint_jacobian_tests();
        Self::exp_log_jacobians_tests();
        Self::hat_jacobians_tests();
        Self::mul_jacobians_tests();
        Self::matrix_jacobians_tests();
        Self::interpolation_test();
    }

    /// Test hat and vee operators.
    fn hat_jacobians_tests();

    /// Test group multiplication jacobians.
    fn mul_jacobians_tests();

    /// Test matrix jacobians.
    fn matrix_jacobians_tests();

    /// Test adjoint jacobian.
    fn adjoint_jacobian_tests();

    /// exp_log_jacobians_tests
    fn exp_log_jacobians_tests();

    /// interpolation_test
    fn interpolation_test();
}

macro_rules! def_real_group_test_template {
    ($scalar:ty, $dual_scalar:ty, $group: ty, $dual_group: ty, $batch:literal
) => {
        impl RealLieGroupTest for $group {

            fn adjoint_jacobian_tests() {
                use crate::traits::IsLieGroup;
                const DOF: usize = <$group>::DOF;
                use sophus_core::manifold::traits::TangentImpl;

                let tangent_examples: alloc::vec::Vec<<$scalar as IsScalar<$batch>>::Vector<DOF>>
                    = <$group>::tangent_examples();

                for a in &tangent_examples {
                    let ad_a: <$scalar as IsScalar<$batch>>::Matrix<DOF, DOF> = <$group>::ad(a);

                    for b in &tangent_examples {
                        if DOF > 0 {
                            let lambda = |x: <$scalar as IsScalar<$batch>>::Vector<DOF>| {
                                let lhs = <$group>::hat(a).mat_mul(<$group>::hat(&x));
                                let rhs = <$group>::hat(&x).mat_mul(<$group>::hat(a));
                                <$group>::vee(&(lhs - rhs))
                            };

                            let num_diff_ad_a =
                                VectorValuedMapFromVector::<$scalar, $batch>
                                    ::static_sym_diff_quotient
                                (
                                    |x| {
                                        lambda(x)
                                    },
                                    *b.real_vector(),
                                    0.0001,
                                );
                            approx::assert_relative_eq!(
                                ad_a.real_matrix(),
                                num_diff_ad_a.real_matrix(),
                                epsilon = 0.0001
                            );

                            let dual_a =
                                <$dual_scalar as IsScalar<$batch>>::Vector::from_real_vector
                                (
                                    a.clone()
                                );

                            let auto_diff_ad_a
                                = VectorValuedMapFromVector::<$dual_scalar, $batch>::fw_autodiff
                                (
                                    |x| {
                                        let hat_x = <$dual_group>::hat(&x);
                                        let hat_a = <$dual_group>::hat(&dual_a);
                                        let mul = hat_a.mat_mul(hat_x.clone())
                                            - hat_x.mat_mul(hat_a);
                                        <$dual_group>::vee(&mul)
                                    },
                                    *b,
                                );

                            for i in 0..DOF {
                                assert_relative_eq!(
                                    ad_a.get_col_vec(i),
                                    auto_diff_ad_a.get([i]),
                                    epsilon = 0.001
                                );
                            }

                        }
                    }
                }
            }

            fn exp_log_jacobians_tests(){
                use crate::traits::IsLieGroup;
                const DOF: usize = <$group>::DOF;
                const POINT: usize = <$group>::POINT;
                const PARAMS: usize = <$group>::PARAMS;

                use sophus_core::manifold::traits::TangentImpl;
                use sophus_core::points::example_points;

                for t in <$group>::tangent_examples() {
                    // x == log(exp(x))

                    let log_exp_t: <$scalar as IsScalar<$batch>>::Vector<DOF> =
                        Self::log(&Self::exp(&t));
                    assert_relative_eq!(t, log_exp_t, epsilon = 0.0001);

                    // dx exp(x).matrix
                    {
                        let exp_t = |t: <$scalar as IsScalar<$batch>>::Vector<DOF>|
                            -> <$scalar as IsScalar<$batch>>::Vector<PARAMS>
                            {
                                *Self::exp(&t).params()
                            };
                        let dual_exp_t = |vv: <$dual_scalar as IsScalar<$batch>>::Vector<DOF>|
                            -> <$dual_scalar as IsScalar<$batch>>::Vector<PARAMS>
                            {
                                <$dual_group>::exp(&vv).params().clone()
                            };

                        let dx_exp_num_diff =
                            VectorValuedMapFromVector::static_sym_diff_quotient(exp_t, t, 0.0001);
                        let dx_exp_auto_diff =
                            VectorValuedMapFromVector::<$dual_scalar, $batch>::static_fw_autodiff
                            (
                                dual_exp_t,
                                t
                            );

                        assert_relative_eq!(dx_exp_auto_diff, dx_exp_num_diff, epsilon = 0.001);

                        let dx_exp_analytic_diff = Self::dx_exp(&t);
                        assert_relative_eq!(dx_exp_analytic_diff, dx_exp_num_diff, epsilon = 0.001);
                    }
                }

                //dx exp(x) at x=0
                {
                    let exp_t = |t: <$scalar as IsScalar<$batch>>::Vector<DOF>|
                        -> <$scalar as IsScalar<$batch>>::Vector<PARAMS>
                        {
                            *Self::exp(&t).params()
                        };
                    let dual_exp_t = |vv: <$dual_scalar as IsScalar<$batch>>::Vector<DOF>|
                        -> <$dual_scalar as IsScalar<$batch>>::Vector<PARAMS>
                        {
                            <$dual_group>::exp(&vv).params().clone()
                        };

                    let analytic_diff = Self::dx_exp_x_at_0();
                    let num_diff =
                        VectorValuedMapFromVector::static_sym_diff_quotient
                        (
                            exp_t,
                            <$scalar as IsScalar<$batch>>::Vector::zeros(),
                            0.0001
                        );
                    let exp_auto_diff =
                        VectorValuedMapFromVector::<$dual_scalar, $batch>::static_fw_autodiff
                        (
                            dual_exp_t,
                            <$scalar as IsScalar<$batch>>::Vector::zeros()
                        );

                    assert_relative_eq!(exp_auto_diff, num_diff, epsilon = 0.001);
                    assert_relative_eq!(analytic_diff, num_diff, epsilon = 0.001);
                }

                for point in example_points::<$scalar, POINT, $batch>() {
                    let exp_t = |t: <$scalar as IsScalar<$batch>>::Vector<DOF>|
                        -> <$scalar as IsScalar<$batch>>::Vector<POINT>
                        {
                            Self::exp(&t).transform(point)
                        };
                    let dual_exp_t = |vv: <$dual_scalar as IsScalar<$batch>>::Vector<DOF>|
                        -> <$dual_scalar as IsScalar<$batch>>::Vector<POINT>
                        {
                            <$dual_group>::exp(&vv).transform(&<$dual_scalar as IsScalar<$batch>>
                                ::Vector::from_real_vector(point))
                        };

                    let analytic_diff = Self::dx_exp_x_times_point_at_0(point);
                    let num_diff =
                        VectorValuedMapFromVector::static_sym_diff_quotient
                        (
                            exp_t,
                            <$scalar as IsScalar<$batch>>::Vector::zeros(),
                            0.0001
                        );
                    let exp_times_auto_diff =
                        VectorValuedMapFromVector::<$dual_scalar, $batch>::static_fw_autodiff
                        (
                            dual_exp_t,
                            <$scalar as IsScalar<$batch>>::Vector::zeros()
                        );
                    assert_relative_eq!(exp_times_auto_diff, num_diff, epsilon = 0.001);
                    assert_relative_eq!(analytic_diff, num_diff, epsilon = 0.001);
                }

                for g in Self::element_examples() {
                    // dx log(y)
                    {
                        if g.has_shortest_path_ambiguity().any() {
                            // jacobian not uniquely defined, let's skip these cases
                            continue;
                        }

                        let log_x = |t: <$scalar as IsScalar<$batch>>::Vector<DOF>|
                            -> <$scalar as IsScalar<$batch>>::Vector<DOF>
                            {
                                (Self::exp(&t) * g).log()
                            };
                        let o = <$scalar as IsScalar<$batch>>::Vector::zeros();

                        let dual_params = <$dual_scalar as IsScalar<$batch>>::Vector
                            ::from_real_vector(*g.params());
                        let dual_g = <$dual_group>::from_params
                            (
                                &dual_params,
                            );
                        let dual_log_x = |t: <$dual_scalar as IsScalar<$batch>>::Vector<DOF>|
                            -> <$dual_scalar as IsScalar<$batch>>::Vector<DOF>
                            {
                                <$dual_group>::exp(&t).group_mul(&dual_g).log()
                            };

                        let num_diff =
                            VectorValuedMapFromVector::static_sym_diff_quotient(log_x, o, 0.0001);
                        let log_auto_diff =
                            VectorValuedMapFromVector::<$dual_scalar, $batch>::static_fw_autodiff
                            (
                                dual_log_x,
                                o
                            );
                        assert_relative_eq!(log_auto_diff, num_diff, epsilon = 0.001);

                        let dual_log_x = |g: <$dual_scalar as IsScalar<$batch>>::Vector<PARAMS>|
                            -> <$dual_scalar as IsScalar<$batch>>::Vector<DOF>
                            {
                                <$dual_group>::from_params(&g).log()
                            };
                        let auto_diff =
                            VectorValuedMapFromVector::<$dual_scalar, $batch>::static_fw_autodiff(dual_log_x, *g.params());

                        let analytic_diff = Self::dx_log_x(g.params());
                        assert_relative_eq!(analytic_diff, auto_diff, epsilon = 0.001);
                    }
                }

                for a in Self::element_examples() {
                    for b in Self::element_examples() {
                        let dual_params_a =
                            <$dual_scalar as IsScalar<$batch>>::Vector::from_real_vector
                            (
                                *a.clone().params()
                            );
                        let dual_a = <$dual_group>::from_params(&dual_params_a);
                        let dual_params_b =
                            <$dual_scalar as IsScalar<$batch>>::Vector::from_real_vector
                            (
                                *b.params()
                            );
                        let dual_b = <$dual_group>::from_params
                            (
                                &dual_params_b,
                            );
                        let dual_log_x = |t: <$dual_scalar as IsScalar<$batch>>::Vector<DOF>|
                            -> <$dual_scalar as IsScalar<$batch>>::Vector<DOF>
                            {
                                (&dual_a *
                                        &<$dual_group>::exp(&t)
                                        .group_mul(&dual_b)
                                    ).log()
                            };

                        let analytic_diff = Self::dx_log_a_exp_x_b_at_0(&a, &b);
                        let o = <$scalar as IsScalar<$batch>>::Vector::zeros();
                        let auto_diff = VectorValuedMapFromVector::<$dual_scalar, $batch>
                            ::static_fw_autodiff(dual_log_x, o);
                        assert_relative_eq!(auto_diff, analytic_diff, epsilon = 0.001);
                    }
                }
        }


            fn hat_jacobians_tests() {
                use crate::traits::IsLieGroup;
                use sophus_core::manifold::traits::TangentImpl;
                const DOF: usize = <$group>::DOF;
                const AMBIENT: usize = <$group>::AMBIENT;


                for x in <$group>::tangent_examples() {
                    // x == vee(hat(x))
                    let vee_hat_x: <$scalar as IsScalar<$batch>>::Vector<DOF>
                        = <$group>::vee(&<$group>::hat(&x));
                    assert_relative_eq!(x, vee_hat_x, epsilon = 0.0001);

                    // dx hat(x)
                    {
                        let hat_x = |v: <$scalar as IsScalar<$batch>>::Vector<DOF>|
                            -> <$scalar as IsScalar<$batch>>::Matrix<AMBIENT, AMBIENT>
                        {
                            <$group>::hat(&v)
                        };
                        let dual_hat_x = |vv: <$dual_scalar as IsScalar<$batch>>::Vector<DOF>|
                            -> <$dual_scalar as IsScalar<$batch>>::Matrix<AMBIENT, AMBIENT>
                        {
                            <$dual_group>::hat(&vv)
                        };

                        let num_diff =
                            MatrixValuedMapFromVector::sym_diff_quotient(hat_x, x, 0.0001);
                        let auto_diff =
                            MatrixValuedMapFromVector::<$dual_scalar, $batch>::fw_autodiff(
                                dual_hat_x, x);

                        for i in 0..DOF {
                            assert_relative_eq!(
                                auto_diff.get([i]),
                                num_diff.get([i]),
                                epsilon = 0.001
                            );
                        }
                    }

                    // dx vee(y)
                    {
                        let a = Self::hat(&x);
                        let vee_x = |v: <$scalar as IsScalar<$batch>>::Matrix<AMBIENT, AMBIENT>|
                            -> <$scalar as IsScalar<$batch>>::Vector<DOF>
                        {
                            <$group>::vee(&v)
                        };
                        let dual_vee_x =
                            |vv: <$dual_scalar as IsScalar<$batch>>::Matrix<AMBIENT, AMBIENT>|
                            -> <$dual_scalar as IsScalar<$batch>>::Vector<DOF>
                        {
                            <$dual_group>::vee(&vv)
                        };

                        let num_diff =
                            VectorValuedMapFromMatrix::sym_diff_quotient(vee_x, a, 0.0001);
                        let auto_diff =
                            VectorValuedMapFromMatrix::<$dual_scalar, $batch>::fw_autodiff
                            (
                                dual_vee_x,
                                a
                            );

                        for i in 0..AMBIENT {
                            for j in 0..AMBIENT {
                                assert_relative_eq!(
                                    auto_diff.get([i, j]),
                                    num_diff.get([i, j]),
                                    epsilon = 0.001
                                );
                            }
                        }
                    }
                }
            }

            fn mul_jacobians_tests() {
                use crate::traits::IsLieGroup;
                const PARAMS: usize = <$group>::PARAMS;
                for a in Self::element_examples() {
                    for b in Self::element_examples() {
                        let a_dual = a.clone().to_dual_c();
                        let b_dual = b.clone().to_dual_c();

                        let dual_mul_x = |vv: <$dual_scalar as IsScalar<$batch>>::Vector<PARAMS>|
                            -> <$dual_scalar as IsScalar<$batch>>::Vector<PARAMS>
                            {
                                <$dual_group>::from_params(&vv)
                                .group_mul(&b_dual).params().clone()
                            };

                        let auto_diff =
                            VectorValuedMapFromVector::<$dual_scalar, $batch>::static_fw_autodiff
                            (
                                dual_mul_x,
                                *a.clone().params()
                            );
                        let analytic_diff = Self::da_a_mul_b(&a, &b);
                        assert_relative_eq!(analytic_diff, auto_diff, epsilon = 0.001);

                        let dual_mul_x = |vv: <$dual_scalar as IsScalar<$batch>>::Vector<PARAMS>|
                            -> <$dual_scalar as IsScalar<$batch>>::Vector<PARAMS>
                            {
                                a_dual.group_mul(& LieGroup::from_params(&vv)).params().clone()
                            };

                        let auto_diff =
                            VectorValuedMapFromVector::<$dual_scalar, $batch>::static_fw_autodiff
                            (
                                dual_mul_x, *b.clone().params()
                            );
                        let analytic_diff = Self::db_a_mul_b(&a, &b);
                        assert_relative_eq!(analytic_diff, auto_diff, epsilon = 0.001);
                    }
                }
            }

            fn matrix_jacobians_tests() {
                use crate::traits::IsLieGroup;
                let group_examples: alloc::vec::Vec<_> = Self::element_examples();
                const PARAMS: usize = <$group>::PARAMS;
                const POINT: usize = <$group>::POINT;
                const DOF: usize = <$group>::DOF;
                const AMBIENT: usize = <$group>::AMBIENT;

                for foo_from_bar in &group_examples {
                    for i in 0..AMBIENT{

                        let params =  foo_from_bar.params();
                        let  dual_fn = |
                            v: <$dual_scalar as IsScalar<$batch>>::Vector<PARAMS>,
                        | -> <$dual_scalar as IsScalar<$batch>>::Vector<POINT> {
                            let m =  <$dual_group>::from_params(&v).compact();
                            m.get_col_vec(i)
                        };
                        let auto_d = VectorValuedMapFromVector::<$dual_scalar, $batch>
                        ::static_fw_autodiff(dual_fn, params.clone());

                        approx::assert_relative_eq!(
                            auto_d,
                            foo_from_bar.dparams_matrix(i),
                            epsilon = 0.0001,
                        );

                        {
                            let  dual_fn = |
                                v: <$dual_scalar as IsScalar<$batch>>::Vector<DOF>,
                            | -> <$dual_scalar as IsScalar<$batch>>::Vector<POINT> {
                                let m =   ( <$dual_group>::exp(v)* foo_from_bar.to_dual_c()).compact();
                                m.get_col_vec(i)
                            };
                            let o = <$scalar as IsScalar<$batch>>::Vector::zeros();
                            let auto_d = VectorValuedMapFromVector::<$dual_scalar, $batch>
                            ::static_fw_autodiff(dual_fn,o);

                            approx::assert_relative_eq!(
                                auto_d,
                                foo_from_bar.dx_exp_x_time_matrix_at_0(i),
                                epsilon = 0.0001,
                            );
                        }
                    }
                }

            }


            fn interpolation_test() {
                let group_examples: alloc::vec::Vec<_> = Self::element_examples();

                for foo_from_bar in &group_examples {
                    for foo_from_daz in &group_examples {
                        let foo_from_bar0 = foo_from_bar.interpolate(foo_from_daz, <$scalar>::zeros());
                        approx::assert_relative_eq!(
                            foo_from_bar.matrix(),
                            foo_from_bar0.matrix(),
                            epsilon = 0.0001
                        );

                        let foo_from_daz1 = foo_from_bar.interpolate(foo_from_daz, <$scalar>::ones());
                        approx::assert_relative_eq!(
                            foo_from_daz.matrix(),
                            foo_from_daz1.matrix(),
                            epsilon = 0.0001
                        );
                    }
                }
                for alpha in [<$scalar>::from_f64(0.1), <$scalar>::from_f64(0.5), <$scalar>::from_f64(0.99)] {
                    for foo_from_bar in &group_examples {
                        for foo_from_daz in &group_examples {
                            if foo_from_bar.inverse().group_mul(foo_from_daz).has_shortest_path_ambiguity().any() {
                                // interpolation not uniquely defined, let's skip these cases
                                continue;
                            }

                            let foo_from_quiz = foo_from_bar.interpolate(foo_from_daz, alpha);

                            // test left-invariance:
                            //
                            // dash_from_foo * interp(foo_from_bar, foo_from_daz)
                            //   == interp(dash_from_foo * foo_from_bar,
                            //             dash_from_foo * foo_from_daz)

                            for dash_from_foo in &group_examples {

                                let dash_from_quiz = dash_from_foo.group_mul(foo_from_bar).interpolate(
                                    &dash_from_foo.group_mul(foo_from_daz), alpha);

                                    approx::assert_relative_eq!(
                                        dash_from_quiz.matrix(),
                                        dash_from_foo.group_mul(&foo_from_quiz).matrix(),
                                        epsilon = 0.0001
                                    );
                            }

                            // test inverse-invariance:
                            //
                            // interp(foo_from_bar, foo_from_daz).inverse()
                            //   == interp(dash_from_foo.inverse(),
                            //             dash_from_foo.inverse())

                            let quiz_from_foo = foo_from_bar.inverse().interpolate(&foo_from_daz.inverse(), alpha);

                            approx::assert_relative_eq!(
                                foo_from_quiz.inverse().matrix(),
                                quiz_from_foo.matrix(),
                                epsilon = 0.0001
                            );
                        }
                    }
                    for a_from_bar in &group_examples {
                        for b_from_bar in &group_examples {
                            if a_from_bar.group_mul(&b_from_bar.inverse()).has_shortest_path_ambiguity().any() {
                                // interpolation not uniquely defined, let's skip these cases
                                continue;
                            }

                            let c_from_bar = a_from_bar.interpolate(b_from_bar, alpha);

                             // test right-invariance:
                            //
                            // interp(a_from_bar, b_from_bar) * bar_from_foo
                            //   == interp(a_from_bar * bar_from_foo,
                            //             b_from_bar * bar_from_foo)
                            for bar_from_foo in &group_examples {

                                let c_from_foo = a_from_bar.group_mul(bar_from_foo).interpolate(
                                    &b_from_bar.group_mul(bar_from_foo), alpha);

                                    approx::assert_relative_eq!(
                                        c_from_foo.matrix(),
                                        c_from_bar.group_mul(&bar_from_foo).matrix(),
                                        epsilon = 0.0001
                                    );
                            }

                        }
                    }
                }
            }

        }
    };
}

def_real_group_test_template!(f64, DualScalar, Rotation2<f64, 1>, Rotation2<DualScalar, 1>,  1);
#[cfg(feature = "simd")]
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8>,
    Rotation2<BatchScalarF64<8>, 8>,
    Rotation2<DualBatchScalar<8>, 8>,
    8
);

def_real_group_test_template!(f64, DualScalar, Isometry2F64, Isometry2<DualScalar, 1>,  1);
#[cfg(feature = "simd")]
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8>,
    Isometry2<BatchScalarF64<8>, 8>,
    Isometry2<DualBatchScalar<8>, 8>,
    8
);

def_real_group_test_template!(f64, DualScalar, Rotation3<f64, 1>, Rotation3<DualScalar, 1>,  1);
#[cfg(feature = "simd")]
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8>,
    Rotation3<BatchScalarF64<8>, 8>,
    Rotation3<DualBatchScalar<8>, 8>,
    8
);

def_real_group_test_template!(f64, DualScalar, Isometry3F64, Isometry3<DualScalar, 1>,  1);
#[cfg(feature = "simd")]
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8>,
    Isometry3<BatchScalarF64<8>, 8>,
    Isometry3<DualBatchScalar<8>, 8>,
    8
);
