use core::{
    borrow::Borrow,
    fmt::{
        Display,
        Formatter,
    },
};

use approx::assert_relative_eq;
use log::info;
use nalgebra::SVector;
use rand::{
    Rng,
    SeedableRng,
};
use rand_chacha::ChaCha12Rng;
#[cfg(feature = "simd")]
use sophus_autodiff::dual::DualBatchScalar;
#[cfg(feature = "simd")]
use sophus_autodiff::linalg::BatchScalarF64;
use sophus_autodiff::{
    dual::DualScalar,
    maps::{
        MatrixValuedVectorMap,
        VectorValuedMatrixMap,
        VectorValuedVectorMap,
    },
};

use crate::{
    IsLieGroupImpl,
    IsRealLieGroupImpl,
    Isometry2,
    Isometry2F64,
    Isometry3,
    Isometry3F64,
    Rotation2,
    Rotation2F64,
    Rotation3,
    Rotation3F64,
    lie_group::LieGroup,
    prelude::*,
};

extern crate alloc;

impl<
    S: IsRealScalar<BATCH, RealScalar = S>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH: usize,
    G: IsRealLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH>,
> LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, 0, 0, G>
where
    SVector<S, DOF>: IsVector<S, DOF, BATCH, 0, 0>,
{
    /// derivative of exponential map at the identity
    pub fn dx_exp_x_at_0() -> S::Matrix<PARAMS, DOF> {
        G::dx_exp_x_at_0()
    }

    /// derivative of exponential map times point at the identity
    pub fn dx_exp_x_times_point_at_0(point: S::Vector<POINT>) -> S::Matrix<POINT, DOF> {
        G::dx_exp_x_times_point_at_0(point)
    }

    /// are there multiple shortest paths to the identity?
    pub fn has_shortest_path_ambiguity(&self) -> S::Mask {
        G::has_shortest_path_ambiguity(&self.params)
    }

    /// derivative of exponential map
    pub fn dx_exp(tangent: S::Vector<DOF>) -> S::Matrix<PARAMS, DOF> {
        G::dx_exp(tangent)
    }

    /// dual representation of the group
    pub fn to_dual_c<const M: usize, const N: usize>(
        self,
    ) -> LieGroup<S::DualScalar<M, N>, DOF, PARAMS, POINT, AMBIENT, BATCH, M, N, G::DualG<M, N>>
    {
        LieGroup::from_params(self.params.to_dual_const())
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
        Self::inv_left_jacobian(ab.log()).mat_mul(Self::adj(a))
    }

    /// derivative of group multiplication with respect to the first argument
    pub fn da_a_mul_b(a: Self, b: Self) -> S::Matrix<PARAMS, PARAMS> {
        G::da_a_mul_b(*a.params(), *b.params())
    }

    /// derivative of group multiplication with respect to the second argument
    pub fn db_a_mul_b(a: Self, b: Self) -> S::Matrix<PARAMS, PARAMS> {
        G::db_a_mul_b(*a.params(), *b.params())
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
            .mat_mul(Self::da_a_mul_b(Self::identity(), *self))
            .mat_mul(Self::dx_exp_x_at_0())
    }

    /// left
    pub fn left_jacobian(tangent: S::Vector<DOF>) -> S::Matrix<DOF, DOF> {
        G::left_jacobian(tangent)
    }

    /// right
    pub fn inv_left_jacobian(tangent: S::Vector<DOF>) -> S::Matrix<DOF, DOF> {
        G::inv_left_jacobian(tangent)
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN>,
> Display for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
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
        Self::left_jacobian_tests();
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

    /// left jac tests
    fn left_jacobian_tests();
}

macro_rules! def_real_group_test_template {
    ($scalar:ty, $dual_scalar:ty, $dual_scalar_p:ty, $dual_scalar_g:ty, $group: ty, $dual_group: ty,$dual_group_p: ty, $dual_group_g: ty, $batch:literal
) => {

        impl RealLieGroupTest for $group {

            fn left_jacobian_tests() {
                use crate::IsLieGroup;
                const DOF: usize = <$group>::DOF;

                for xi in <$group>::tangent_examples() {
                    // 1) algebraic inverse check  J · J⁻¹ ≈ I
                    let jl  = <$group>::left_jacobian(xi);
                    let jli = <$group>::inv_left_jacobian(xi);
                    let eye = jl.mat_mul(jli);
                    assert_relative_eq!(
                        eye,
                        <$scalar as IsScalar<$batch,0,0>>
                            ::Matrix::<DOF,DOF>::identity(),
                        epsilon = 1e-6
                    );

                    // 2) first–order BCH check  exp(xi)·exp(δ) ≈ exp(xi + J δ)
                    //    with a tiny random δ
                    let mut rng = ChaCha12Rng::from_seed(Default::default());
                    let mut delta =
                        <$scalar as IsScalar<$batch,0,0>>::Vector::<DOF>::zeros();
                    for k in 0..DOF {
                        *sophus_autodiff::linalg::IsVector::elem_mut(&mut delta, k as usize)=
                            <$scalar>::from_f64(rng.random_range(-1e-4..1e-4));
                    }

                    let g1   = <$group>::exp(xi)
                                 .group_mul(<$group>::exp(delta));
                    let pred = jl * delta;
                    let g2   = <$group>::exp(xi)
                                 .group_mul(<$group>::identity())
                                 .group_mul(<$group>::exp(pred)); // == exp(xi+Jδ)

                    assert_relative_eq!(
                        g1.params(),
                        g2.params(),
                        epsilon = 1e-3    // O(‖δ‖²) error expected
                    );
                }
            }

            fn adjoint_jacobian_tests() {
                use crate::IsLieGroup;
                const DOF: usize = <$group>::DOF;
                use sophus_autodiff::manifold::IsTangent;

                let tangent_examples: alloc::vec::Vec<<$scalar as IsScalar<$batch,0,0>>::Vector<DOF>>
                    = <$group>::tangent_examples();

                for a in tangent_examples.clone() {
                    let ad_a: <$scalar as IsScalar<$batch,0,0>>::Matrix<DOF, DOF> = <$group>::ad(a);

                    for b in tangent_examples.clone() {
                        if DOF > 0 {
                            let lambda = |x: <$scalar as IsScalar<$batch,0,0>>::Vector<DOF>| {
                                let lhs = <$group>::hat(a).mat_mul(<$group>::hat(x));
                                let rhs = <$group>::hat(x).mat_mul(<$group>::hat(a));
                                <$group>::vee((lhs - rhs))
                            };

                            let num_diff_ad_a =
                                VectorValuedVectorMap::<$scalar, $batch>
                                    ::sym_diff_quotient_jacobian
                                (
                                    |x| {
                                        lambda(x)
                                    },
                                    b.real_vector(),
                                    0.0001,
                                );
                            approx::assert_relative_eq!(
                                ad_a.real_matrix(),
                                num_diff_ad_a.real_matrix(),
                                epsilon = 0.0001
                            );

                            let dual_a =
                                <$dual_scalar as IsScalar<$batch, DOF,1>>::Vector::from_real_vector
                                (
                                    a
                                );

                             let ad_fn =   |x| {
                                    let hat_x = <$dual_group>::hat(x);
                                    let hat_a = <$dual_group>::hat(dual_a);
                                    let mul = hat_a.mat_mul(hat_x)
                                        - hat_x.mat_mul(hat_a);
                                    <$dual_group>::vee(mul)
                                };

                            let auto_diff_ad_a
                                = ad_fn(<$dual_scalar>::vector_var(b)).jacobian();


                            assert_relative_eq!(
                                    ad_a,
                                    auto_diff_ad_a,
                                    epsilon = 0.001
                                );

                        }
                    }
                }
            }

            fn exp_log_jacobians_tests(){
                use crate::IsLieGroup;
                const DOF: usize = <$group>::DOF;
                const POINT: usize = <$group>::POINT;
                const PARAMS: usize = <$group>::PARAMS;

                use sophus_autodiff::manifold::IsTangent;
                use sophus_autodiff::points::example_points;

                for t in <$group>::tangent_examples() {
                    // x == log(exp(x))

                    let log_exp_t: <$scalar as IsScalar<$batch,0,0>>::Vector<DOF> =
                        Self::log(&Self::exp(t));
                    assert_relative_eq!(t, log_exp_t, epsilon = 0.0001);

                    // dx exp(x).matrix
                    {
                        let exp_t = |t: <$scalar as IsScalar<$batch,0,0>>::Vector<DOF>|
                            -> <$scalar as IsScalar<$batch,0,0>>::Vector<PARAMS>
                            {
                                *Self::exp(t).params()
                            };
                        let dual_exp_t = |vv: <$dual_scalar as IsScalar<$batch, DOF,1>>::Vector<DOF>|
                            -> <$dual_scalar as IsScalar<$batch, DOF,1>>::Vector<PARAMS>
                            {
                                <$dual_group>::exp(vv).params().clone()
                            };

                        let dx_exp_num_diff =
                            VectorValuedVectorMap::sym_diff_quotient_jacobian(exp_t, t, 0.0001);
                        let dx_exp_auto_diff = dual_exp_t(<$dual_scalar>::vector_var(t)).jacobian();

                        assert_relative_eq!(dx_exp_auto_diff, dx_exp_num_diff, epsilon = 0.001);

                        let dx_exp_analytic_diff = Self::dx_exp(t);
                        assert_relative_eq!(dx_exp_analytic_diff, dx_exp_num_diff, epsilon = 0.001);
                    }
                }

                //dx exp(x) at x=0
                {
                    let exp_t = |t: <$scalar as IsScalar<$batch, 0, 0>>::Vector<DOF>|
                        -> <$scalar as IsScalar<$batch, 0, 0>>::Vector<PARAMS>
                        {
                            *Self::exp(t).params()
                        };
                    let dual_exp_t = |vv: <$dual_scalar as IsScalar<$batch, DOF, 1>>::Vector<DOF>|
                        -> <$dual_scalar as IsScalar<$batch, DOF, 1>>::Vector<PARAMS>
                        {
                            <$dual_group>::exp(vv).params().clone()
                        };

                    let analytic_diff = Self::dx_exp_x_at_0();
                    let num_diff =
                        VectorValuedVectorMap::sym_diff_quotient_jacobian
                        (
                            exp_t,
                            <$scalar as IsScalar<$batch, 0, 0>>::Vector::zeros(),
                            0.0001
                        );
                    let exp_auto_diff = dual_exp_t(<$dual_scalar>::vector_var(
                        <<$scalar as IsScalar<$batch, 0, 0>>::Vector<DOF>>::zeros())).jacobian();

                    assert_relative_eq!(exp_auto_diff, num_diff, epsilon = 0.001);
                    assert_relative_eq!(analytic_diff, num_diff, epsilon = 0.001);
                }

                for point in example_points::<$scalar, POINT, $batch, 0,0>() {
                    let exp_t = |t: <$scalar as IsScalar<$batch, 0,0>>::Vector<DOF>|
                        -> <$scalar as IsScalar<$batch,0,0>>::Vector<POINT>
                        {
                            Self::exp(t).transform(point)
                        };
                    let dual_exp_t = |vv: <$dual_scalar as IsScalar<$batch,DOF,1>>::Vector<DOF>|
                        -> <$dual_scalar as IsScalar<$batch,DOF,1>>::Vector<POINT>
                        {
                            <$dual_group>::exp(vv).transform(<$dual_scalar as IsScalar<$batch, DOF,1>>
                                ::Vector::from_real_vector(point))
                        };

                    let analytic_diff = Self::dx_exp_x_times_point_at_0(point);
                    let num_diff =
                        VectorValuedVectorMap::sym_diff_quotient_jacobian
                        (
                            exp_t,
                            <$scalar as IsScalar<$batch, 0, 0>>::Vector::zeros(),
                            0.0001
                        );
                    let exp_times_auto_diff = dual_exp_t(<$dual_scalar>::vector_var(
                        <<$scalar as IsScalar<$batch, 0, 0>>::Vector<DOF>>::zeros())).jacobian();
                    assert_relative_eq!(exp_times_auto_diff, num_diff, epsilon = 0.001);
                    assert_relative_eq!(analytic_diff, num_diff, epsilon = 0.001);
                }

                for a in Self::element_examples() {
                    for b in Self::element_examples() {

                        if ((a.group_mul(b).has_shortest_path_ambiguity()).any()){
                            info!("Skipping test for a,b due to ambiguity.");
                            continue;
                        }

                        let dual_params_a =
                            <$dual_scalar as IsScalar<$batch, DOF, 1>>::Vector::from_real_vector
                            (
                                *a.params()
                            );
                        let dual_a = <$dual_group>::from_params(dual_params_a);
                        let dual_params_b =
                            <$dual_scalar as IsScalar<$batch, DOF, 1>>::Vector::from_real_vector
                            (
                                *b.params()
                            );
                        let dual_b = <$dual_group>::from_params
                            (
                                dual_params_b,
                            );
                        let dual_log_x = |t: <$dual_scalar as IsScalar<$batch, DOF, 1>>::Vector<DOF>|
                            -> <$dual_scalar as IsScalar<$batch, DOF, 1>>::Vector<DOF>
                            {
                                (dual_a *
                                        &<$dual_group>::exp(t)
                                        .group_mul(dual_b)
                                    ).log()
                            };



                        let log_x_analytic_diff = Self::dx_log_a_exp_x_b_at_0(&a, &b);
                        let o = <$scalar as IsScalar<$batch, 0, 0>>::Vector::zeros();
                        let log_x_auto_diff = dual_log_x(<$dual_scalar>::vector_var(o)).jacobian();

                        assert_relative_eq!(log_x_auto_diff, log_x_analytic_diff, epsilon = 0.001);
                    }
                }
        }


            fn hat_jacobians_tests() {
                use crate::IsLieGroup;
                use sophus_autodiff::manifold::IsTangent;
                const DOF: usize = <$group>::DOF;
                const AMBIENT: usize = <$group>::AMBIENT;


                for x in <$group>::tangent_examples() {
                    // x == vee(hat(x))
                    let vee_hat_x: <$scalar as IsScalar<$batch,0,0>>::Vector<DOF>
                        = <$group>::vee(<$group>::hat(x));
                    assert_relative_eq!(x, vee_hat_x, epsilon = 0.0001);

                    // dx hat(x)
                    {
                        let hat_x = |v|
                        {
                            <$group>::hat(v)
                        };
                        let dual_hat_x = |vv|
                        {
                            <$dual_group>::hat(vv)
                        };

                        let num_diff =
                            MatrixValuedVectorMap::sym_diff_quotient(hat_x, x, 0.0001);
                        let auto_diff =
                             dual_hat_x( <$dual_scalar>::vector_var(x) ).derivative();


                        for i in 0..DOF {
                            assert_relative_eq!(
                                auto_diff.out_mat[i],
                                num_diff.out_mat[i],
                                epsilon = 0.001
                            );
                        }
                    }

                    // dx vee(y)
                    {
                        let a = Self::hat(x);
                        let vee_x = |v: <$scalar as IsScalar<$batch,0,0>>::Matrix<AMBIENT, AMBIENT>|
                            -> <$scalar as IsScalar<$batch,0,0>>::Vector<DOF>
                        {
                            <$group>::vee(v)
                        };
                        let dual_vee_x =
                            |vv: <$dual_scalar_g as IsScalar<$batch, AMBIENT, AMBIENT>>::Matrix<AMBIENT, AMBIENT>|
                            -> <$dual_scalar_g as IsScalar<$batch, AMBIENT, AMBIENT>>::Vector<DOF>
                        {
                            <$dual_group_g>::vee(vv)
                        };

                        let num_diff =
                            VectorValuedMatrixMap::sym_diff_quotient(vee_x, a, 0.0001);
                        let auto_diff =
                                 dual_vee_x(<$dual_scalar_g>::matrix_var(a)).derivative();


                        for i in 0..DOF {

                                assert_relative_eq!(
                                    auto_diff.out_vec[i],
                                    num_diff.out_vec[i],
                                    epsilon = 0.001
                                );

                        }
                    }
                }
            }

            fn mul_jacobians_tests() {
                use crate::IsLieGroup;
                const PARAMS: usize = <$group>::PARAMS;
                for a in Self::element_examples() {
                    for b in Self::element_examples() {
                        let a_dual = a.to_dual_c();
                        let b_dual = b.to_dual_c();

                        let dual_mul_x = |vv: <$dual_scalar_p as IsScalar<$batch,PARAMS,1>>::Vector<PARAMS>|
                            -> <$dual_scalar_p as IsScalar<$batch,PARAMS,1>>::Vector<PARAMS>
                            {
                                <$dual_group_p>::from_params(vv)
                                .group_mul(b_dual).params().clone()
                            };

                        let auto_diff =
                            dual_mul_x(<$dual_scalar_p>::vector_var(*a.params())).jacobian();

                        let analytic_diff = Self::da_a_mul_b(a, b);
                        assert_relative_eq!(analytic_diff, auto_diff, epsilon = 0.001);

                        let dual_mul_x = |vv: <$dual_scalar_p as IsScalar<$batch, PARAMS, 1>>::Vector<PARAMS>|
                            -> <$dual_scalar_p as IsScalar<$batch, PARAMS, 1>>::Vector<PARAMS>
                            {
                                a_dual.group_mul(LieGroup::from_params(vv)).params().clone()
                            };

                        let auto_diff =
                            dual_mul_x(<$dual_scalar_p>::vector_var(*b.params())).jacobian();

                        let analytic_diff = Self::db_a_mul_b(a, b);
                        assert_relative_eq!(analytic_diff, auto_diff, epsilon = 0.001);
                    }
                }
            }

            fn matrix_jacobians_tests() {
                use crate::IsLieGroup;
                let group_examples: alloc::vec::Vec<_> = Self::element_examples();
                const PARAMS: usize = <$group>::PARAMS;
                const POINT: usize = <$group>::POINT;
                const DOF: usize = <$group>::DOF;
                const AMBIENT: usize = <$group>::AMBIENT;

                for foo_from_bar in &group_examples {
                    for i in 0..AMBIENT{

                        let params =  foo_from_bar.params();
                        let  dual_fn = |
                            v: <$dual_scalar_p as IsScalar<$batch, PARAMS, 1>>::Vector<PARAMS>,
                        | -> <$dual_scalar_p as IsScalar<$batch, PARAMS, 1>>::Vector<POINT> {
                            let m =  <$dual_group_p>::from_params(v).compact();
                            m.get_col_vec(i)
                        };
                        let auto_d =
                            dual_fn(<$dual_scalar_p>::vector_var(params.clone())).jacobian();

                        approx::assert_relative_eq!(
                            auto_d,
                            foo_from_bar.dparams_matrix(i),
                            epsilon = 0.0001,
                        );

                        {
                            let  dual_fn = |
                                v: <$dual_scalar as IsScalar<$batch, DOF, 1>>::Vector<DOF>,
                            | -> <$dual_scalar as IsScalar<$batch, DOF, 1>>::Vector<POINT> {
                                let m =   ( <$dual_group>::exp(v)* foo_from_bar.to_dual_c()).compact();
                                m.get_col_vec(i)
                            };
                            let o = <$scalar as IsScalar<$batch, 0, 0>>::Vector::zeros();
                            let auto_d =
                                dual_fn(<$dual_scalar>::vector_var(o)).jacobian();

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
                    for foo_from_bar in group_examples.clone() {
                        for foo_from_daz in group_examples.clone() {
                            if foo_from_bar.inverse().group_mul(foo_from_daz).has_shortest_path_ambiguity().any() {
                                // interpolation not uniquely defined, let's skip these cases
                                continue;
                            }

                            let foo_from_quiz = foo_from_bar.interpolate(&foo_from_daz, alpha);

                            // test left-invariance:
                            //
                            // dash_from_foo * interp(foo_from_bar, foo_from_daz)
                            //   == interp(dash_from_foo * foo_from_bar,
                            //             dash_from_foo * foo_from_daz)

                            for dash_from_foo in group_examples.clone() {

                                let dash_from_quiz = dash_from_foo.group_mul(foo_from_bar).interpolate(
                                    &dash_from_foo.group_mul(foo_from_daz), alpha);

                                    approx::assert_relative_eq!(
                                        dash_from_quiz.matrix(),
                                        dash_from_foo.group_mul(foo_from_quiz).matrix(),
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
                    for a_from_bar in group_examples.clone() {
                        for b_from_bar in group_examples.clone() {
                            if a_from_bar.group_mul(b_from_bar.inverse()).has_shortest_path_ambiguity().any() {
                                // interpolation not uniquely defined, let's skip these cases
                                continue;
                            }

                            let c_from_bar = a_from_bar.interpolate(&b_from_bar, alpha);

                             // test right-invariance:
                            //
                            // interp(a_from_bar, b_from_bar) * bar_from_foo
                            //   == interp(a_from_bar * bar_from_foo,
                            //             b_from_bar * bar_from_foo)
                            for bar_from_foo in group_examples.clone() {

                                let c_from_foo = a_from_bar.group_mul(bar_from_foo).interpolate(
                                    &b_from_bar.group_mul(bar_from_foo), alpha);

                                    approx::assert_relative_eq!(
                                        c_from_foo.matrix(),
                                        c_from_bar.group_mul(bar_from_foo).matrix(),
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

def_real_group_test_template!(
    f64,
    DualScalar<1,1>,
    DualScalar<2,1>,
    DualScalar<2,2>,
    Rotation2F64,
    Rotation2<DualScalar<1,1>, 1, 1, 1>,
    Rotation2<DualScalar<2,1>, 1, 2, 1>,
    Rotation2<DualScalar<2,2>, 1, 2, 2>,
    1);
#[cfg(feature = "simd")]
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8, 1, 1>,
    DualBatchScalar<8, 2, 1>,
    DualBatchScalar<8, 2, 2>,
    Rotation2<BatchScalarF64<8>, 8, 0, 0>,
    Rotation2<DualBatchScalar<8, 1, 1>, 8, 1, 1>,
    Rotation2<DualBatchScalar<8, 2, 1>, 8, 2, 1>,
    Rotation2<DualBatchScalar<8, 2, 2>, 8, 2, 2>,
    8
);

def_real_group_test_template!(
    f64,
    DualScalar<3, 1>,
    DualScalar<4, 1>,
    DualScalar<3, 3>,
    Isometry2F64,
    Isometry2<DualScalar<3,1>, 1, 3, 1>,
    Isometry2<DualScalar<4,1>, 1, 4, 1>,
    Isometry2<DualScalar<3,3>, 1, 3, 3>,
    1);
#[cfg(feature = "simd")]
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8, 3, 1>,
    DualBatchScalar<8, 4, 1>,
    DualBatchScalar<8, 3, 3>,
    Isometry2<BatchScalarF64<8>, 8, 0, 0>,
    Isometry2<DualBatchScalar<8, 3, 1>, 8, 3, 1>,
    Isometry2<DualBatchScalar<8, 4, 1>, 8, 4, 1>,
    Isometry2<DualBatchScalar<8, 3, 3>, 8, 3, 3>,
    8
);

def_real_group_test_template!(
    f64,
    DualScalar<3, 1>,
    DualScalar<4, 1>,
    DualScalar<3, 3>,
    Rotation3F64,
    Rotation3<DualScalar<3, 1>, 1, 3, 1>,
    Rotation3<DualScalar<4, 1>, 1, 4, 1>,
    Rotation3<DualScalar<3, 3>, 1, 3, 3>,
    1);
#[cfg(feature = "simd")]
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8, 3, 1>,
    DualBatchScalar<8, 4, 1>,
    DualBatchScalar<8, 3, 3>,
    Rotation3<BatchScalarF64<8>, 8, 0, 0>,
    Rotation3<DualBatchScalar<8, 3, 1>, 8, 3, 1>,
    Rotation3<DualBatchScalar<8, 4, 1>, 8, 4, 1>,
    Rotation3<DualBatchScalar<8, 3, 3>, 8, 3, 3>,
    8
);

def_real_group_test_template!(f64,
    DualScalar<6, 1>,
    DualScalar<7, 1>,
    DualScalar<4, 4>,
    Isometry3F64,
    Isometry3<DualScalar<6, 1>, 1, 6, 1>,
    Isometry3<DualScalar<7, 1>, 1, 7, 1>,
    Isometry3<DualScalar<4, 4>, 1, 4, 4>,
    1);
#[cfg(feature = "simd")]
def_real_group_test_template!(
    BatchScalarF64<8>,
    DualBatchScalar<8, 6, 1>,
    DualBatchScalar<8, 7, 1>,
    DualBatchScalar<8, 4, 4>,
    Isometry3<BatchScalarF64<8>, 8, 0, 0>,
    Isometry3<DualBatchScalar<8, 6, 1>, 8, 6, 1>,
    Isometry3<DualBatchScalar<8, 7, 1>, 8, 7, 1>,
    Isometry3<DualBatchScalar<8, 4, 4>, 8, 4, 4>,
    8
);
