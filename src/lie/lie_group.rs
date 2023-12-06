use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

use approx::assert_relative_eq;
use assertables::assert_le_as_result;

use crate::calculus::dual::dual_matrix::DualM;
use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::dual::dual_vector::DualV;
use crate::calculus::maps::matrix_valued_maps::MatrixValuedMapFromVector;
use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromMatrix;
use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
use crate::calculus::points::example_points;
use crate::calculus::types::matrix::IsMatrix;
use crate::calculus::types::params::HasParams;
use crate::calculus::types::params::ParamsImpl;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::vector::IsVectorLike;
use crate::calculus::types::M;
use crate::calculus::types::V;
use crate::manifold::traits::IsManifold;
use crate::manifold::traits::TangentImpl;
use crate::tensor::view::IsTensorLike;

use super::traits::IsF64LieGroupImpl;
use super::traits::IsLieGroup;
use super::traits::IsLieGroupImpl;

#[derive(Debug, Copy, Clone)]
pub struct LieGroup<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>,
> {
    params: S::Vector<PARAMS>,
    phantom: std::marker::PhantomData<G>,
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>,
    > ParamsImpl<S, PARAMS> for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, G>
{
    fn are_params_valid(params: &<S as IsScalar>::Vector<PARAMS>) -> bool {
        G::are_params_valid(params)
    }

    fn params_examples() -> Vec<S::Vector<PARAMS>> {
        G::params_examples()
    }

    fn invalid_params_examples() -> Vec<S::Vector<PARAMS>> {
        G::invalid_params_examples()
    }
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>,
    > HasParams<S, PARAMS> for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, G>
{
    fn from_params(params: &S::Vector<PARAMS>) -> Self {
        assert!(
            G::are_params_valid(params),
            "Invalid parameters for {}",
            params.real()
        );
        Self {
            params: params.clone(),
            phantom: std::marker::PhantomData,
        }
    }

    fn set_params(&mut self, params: &S::Vector<PARAMS>) {
        self.params = params.clone();
    }

    fn params(&self) -> &S::Vector<PARAMS> {
        &self.params
    }
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>,
    > TangentImpl<S, DOF> for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, G>
{
    fn tangent_examples() -> Vec<<S as IsScalar>::Vector<DOF>> {
        G::tangent_examples()
    }
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>,
    > IsLieGroup<S, DOF, PARAMS, POINT, AMBIENT> for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, G>
{
    type G = G;
    type GenG<S2: IsScalar> = G::GenG<S2>;
    type RealG = G::RealG;
    type DualG = G::DualG;

    type GenGroup<S2: IsScalar, G2: IsLieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT>> =
        LieGroup<S2, DOF, PARAMS, POINT, AMBIENT, G2>;
    type RealGroup = Self::GenGroup<f64, G::RealG>;
    type DualGroup = Self::GenGroup<Dual, G::DualG>; 
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>,
    > LieGroup<S, DOF, PARAMS, POINT, AMBIENT, G>
{
    pub fn adj(&self) -> S::Matrix<DOF, DOF> {
        G::adj(&self.params)
    }

    pub fn exp(omega: &S::Vector<DOF>) -> Self {
        Self::from_params(&G::exp(omega))
    }

    pub fn log(&self) -> S::Vector<DOF> {
        G::log(&self.params)
    }

    pub fn hat(omega: &S::Vector<DOF>) -> S::Matrix<AMBIENT, AMBIENT> {
        G::hat(omega)
    }

    pub fn vee(xi: &S::Matrix<AMBIENT, AMBIENT>) -> S::Vector<DOF> {
        G::vee(xi)
    }

    pub fn identity() -> Self {
        Self::from_params(&G::identity_params())
    }

    pub fn group_mul(&self, other: &Self) -> Self {
        Self::from_params(&G::group_mul(&self.params, &other.params))
    }

    pub fn inverse(&self) -> Self {
        Self::from_params(&G::inverse(&self.params))
    }

    pub fn transform(&self, point: &S::Vector<POINT>) -> S::Vector<POINT> {
        G::transform(&self.params, point)
    }

    pub fn to_ambient(point: &S::Vector<POINT>) -> S::Vector<AMBIENT> {
        G::to_ambient(point)
    }

    pub fn compact(&self) -> S::Matrix<POINT, AMBIENT> {
        G::compact(&self.params)
    }

    pub fn matrix(&self) -> S::Matrix<AMBIENT, AMBIENT> {
        G::matrix(&self.params)
    }

    pub fn ad(tangent: &S::Vector<DOF>) -> S::Matrix<DOF, DOF> {
        G::ad(tangent)
    }

    pub fn has_shortest_path_ambiguity(&self) -> bool {
        G::has_shortest_path_ambiguity(&self.params)
    }

    pub fn element_examples() -> Vec<Self> {
        let mut elements = vec![];
        for params in Self::params_examples() {
            elements.push(Self::from_params(&params));
        }
        elements
    }

    pub fn presentability_tests() {
        if G::IS_ORIGIN_PRESERVING {
            for g in &Self::element_examples() {
                let o = S::Vector::<POINT>::zero();
                assert_relative_eq!(g.transform(&o).real(), o.real());
            }
        } else {
            let mut num_preserves = 0;
            let mut num = 0;
            for g in &Self::element_examples() {
                let o = S::Vector::<POINT>::zero();
                let o_transformed = g.transform(&o);
                if (o_transformed.real()).norm() < 0.0001 {
                    num_preserves += 1;
                }
                num += 1;
            }
            let percentage = num_preserves as f64 / num as f64;
            assertables::assert_le!(percentage, 0.75);
        }
    }

    fn adjoint_tests() {
        let group_examples: Vec<_> = Self::element_examples();
        let tangent_examples: Vec<S::Vector<DOF>> = G::tangent_examples();

        for g in &group_examples {
            let mat: S::Matrix<AMBIENT, AMBIENT> = g.matrix();
            let mat_adj = g.adj();

            for x in &tangent_examples {
                let mat_adj_x = mat_adj.clone() * x.clone();

                let inv_mat: S::Matrix<AMBIENT, AMBIENT> = g.inverse().matrix();
                let mat_adj_x2 = Self::vee(&mat.mat_mul(Self::hat(x).mat_mul(inv_mat)));
                assert_relative_eq!(mat_adj_x.real(), mat_adj_x2.real(), epsilon = 0.0001);
            }
        }
        for a in &tangent_examples {
            for b in &tangent_examples {
                let ad_a = Self::ad(a);
                let ad_a_b = ad_a * b.clone();
                let hat_ab = Self::hat(a).mat_mul(Self::hat(b));
                let hat_ba = Self::hat(b).mat_mul(Self::hat(a));

                let lie_bracket_a_b = Self::vee(&(hat_ab - hat_ba));
                assert_relative_eq!(ad_a_b.real(), lie_bracket_a_b.real(), epsilon = 0.0001);
            }
        }
    }

    fn exp_tests() {
        let group_examples: Vec<_> = Self::element_examples();
        let tangent_examples: Vec<S::Vector<DOF>> = G::tangent_examples();

        for g in &group_examples {
            let matrix_before = *g.compact().real();
            let matrix_after = *Self::exp(&g.log()).compact().real();

            assert_relative_eq!(matrix_before, matrix_after, epsilon = 0.0001);

            let t = *g.clone().inverse().log().real();
            let t2 = -g.log().real();
            assert_relative_eq!(t, t2, epsilon = 0.0001);
        }
        for omega in &tangent_examples {
            let exp_inverse = Self::exp(omega).inverse();
            let neg_omega = -omega.clone();

            let exp_neg_omega = Self::exp(&neg_omega);
            assert_relative_eq!(
                exp_inverse.compact().real(),
                exp_neg_omega.compact().real(),
                epsilon = 0.0001
            );
        }
    }

    fn hat_tests() {
        let tangent_examples: Vec<S::Vector<DOF>> = G::tangent_examples();

        for omega in &tangent_examples {
            assert_relative_eq!(
                omega.real(),
                Self::vee(&Self::hat(omega)).real(),
                epsilon = 0.0001
            );
        }
    }

    fn group_operation_tests() {
        let group_examples: Vec<_> = Self::element_examples();

        for g1 in &group_examples {
            for g2 in &group_examples {
                for g3 in &group_examples {
                    let left_hugging = (g1.group_mul(g2)).group_mul(g3);
                    let right_hugging = g1.group_mul(&g2.group_mul(g3));
                    assert_relative_eq!(
                        left_hugging.compact().real(),
                        right_hugging.compact().real(),
                        epsilon = 0.0001
                    );
                }
            }
        }
        for g1 in &group_examples {
            for g2 in &group_examples {
                let daz_from_foo_transform_1 = g2.inverse().group_mul(&g1.inverse());
                let daz_from_foo_transform_2 = g1.group_mul(g2).inverse();
                assert_relative_eq!(
                    daz_from_foo_transform_1.compact().real(),
                    daz_from_foo_transform_2.compact().real(),
                    epsilon = 0.0001
                );
            }
        }
    }

    pub fn test_suite() {
        // Most tests will trivially pass if there are no examples. So first we make sure we have at least three per group.
        let group_examples: Vec<_> = Self::element_examples();
        assert!(group_examples.len() >= 3);
        let tangent_examples: Vec<S::Vector<DOF>> = G::tangent_examples();
        assert!(tangent_examples.len() >= 3);

        Self::presentability_tests();
        Self::group_operation_tests();
        Self::hat_tests();
        Self::exp_tests();
        Self::adjoint_tests();
    }
}

#[derive(Debug, Clone)]
struct LeftGroupManifold<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>,
> {
    group: LieGroup<S, DOF, PARAMS, POINT, AMBIENT, G>,
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT> + Clone + Debug,
    > ParamsImpl<S, PARAMS> for LeftGroupManifold<S, DOF, PARAMS, POINT, AMBIENT, G>
{
    fn are_params_valid(params: &<S as IsScalar>::Vector<PARAMS>) -> bool {
        G::are_params_valid(params)
    }

    fn params_examples() -> Vec<<S as IsScalar>::Vector<PARAMS>> {
        G::params_examples()
    }

    fn invalid_params_examples() -> Vec<<S as IsScalar>::Vector<PARAMS>> {
        G::invalid_params_examples()
    }
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT> + Clone + Debug,
    > HasParams<S, PARAMS> for LeftGroupManifold<S, DOF, PARAMS, POINT, AMBIENT, G>
{
    fn from_params(params: &<S as IsScalar>::Vector<PARAMS>) -> Self {
        Self {
            group: LieGroup::from_params(params),
        }
    }

    fn set_params(&mut self, params: &<S as IsScalar>::Vector<PARAMS>) {
        self.group.set_params(params)
    }

    fn params(&self) -> &<S as IsScalar>::Vector<PARAMS> {
        self.group.params()
    }
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT> + Clone + Debug,
    > IsManifold<S, PARAMS, DOF> for LeftGroupManifold<S, DOF, PARAMS, POINT, AMBIENT, G>
{
    fn oplus(&self, tangent: &<S as IsScalar>::Vector<DOF>) -> Self {
        Self {
            group: LieGroup::<S, DOF, PARAMS, POINT, AMBIENT, G>::exp(tangent)
                .group_mul(&self.group),
        }
    }

    fn ominus(&self, rhs: &Self) -> <S as IsScalar>::Vector<DOF> {
        self.group.inverse().group_mul(&rhs.group).log()
    }

    fn params(&self) -> &<S as IsScalar>::Vector<PARAMS> {
        self.group.params()
    }
}

impl<
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: IsF64LieGroupImpl<DOF, PARAMS, POINT, AMBIENT>,
    > LieGroup<f64, DOF, PARAMS, POINT, AMBIENT, G>
{
    pub fn dx_exp_x_at_0() -> M<PARAMS, DOF> {
        G::dx_exp_x_at_0()
    }

    pub fn dx_exp_x_times_point_at_0(point: V<POINT>) -> M<POINT, DOF> {
        G::dx_exp_x_times_point_at_0(point)
    }

    pub fn to_dual_c(self) -> LieGroup<Dual, DOF, PARAMS, POINT, AMBIENT, G::DualG> {
        let dual_params = DualV::<PARAMS>::c(self.params);
        LieGroup::from_params(&dual_params)
    }

    fn adjoint_jacobian_tests() {
        let tangent_examples: Vec<V<DOF>> = G::tangent_examples();

        for a in &tangent_examples {
            let ad_a: M<DOF, DOF> = Self::ad(a);

            for b in &tangent_examples {
                if DOF > 0 {
                    let num_diff_ad_a = VectorValuedMapFromVector::sym_diff_quotient(
                        |x: V<DOF>| {
                            Self::vee(
                                &(&Self::hat(a) * Self::hat(&x) - Self::hat(&x) * &Self::hat(a)),
                            )
                        },
                        *b,
                        0.0001,
                    );
                    for i in 0..DOF {
                        assert_relative_eq!(
                            ad_a.get_col_vec(i),
                            num_diff_ad_a.get([i]),
                            epsilon = 0.001
                        );
                    }

                    let auto_diff_ad_a = VectorValuedMapFromVector::fw_autodiff(
                        |x: DualV<DOF>| {
                            // Self::vee(
                            //     &(&Self::hat(a) * Self::hat(&x) - Self::hat(&x) * &Self::hat(a)),
                            // )
                            let hat_x =
                                <LieGroup<f64, DOF, PARAMS, POINT, AMBIENT, G> as IsLieGroup<
                                    f64,
                                    DOF,
                                    PARAMS,
                                    POINT,
                                    AMBIENT,
                                >>::DualGroup::hat(&x);
                            let hat_a =
                                <LieGroup<f64, DOF, PARAMS, POINT, AMBIENT, G> as IsLieGroup<
                                    f64,
                                    DOF,
                                    PARAMS,
                                    POINT,
                                    AMBIENT,
                                >>::DualGroup::hat(&DualV::c(*a));
                            let mul = hat_a.mat_mul(hat_x.clone()) - hat_x.mat_mul(hat_a);
                            <LieGroup<f64, DOF, PARAMS, POINT, AMBIENT, G> as IsLieGroup<
                                f64,
                                DOF,
                                PARAMS,
                                POINT,
                                AMBIENT,
                            >>::DualGroup::vee(&mul)
                            //hat_x
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

    fn test_hat_jacobians() {
        for x in G::tangent_examples() {
            // x == vee(hat(x))
            let vee_hat_x: V<DOF> = Self::vee(&Self::hat(&x));
            assert_relative_eq!(x, vee_hat_x, epsilon = 0.0001);

            // dx hat(x)
            {
                let hat_x = |v: V<DOF>| -> M<AMBIENT, AMBIENT> { Self::hat(&v) };
                let dual_hat_x = |vv: DualV<DOF>| -> DualM<AMBIENT, AMBIENT> { G::DualG::hat(&vv) };

                let num_diff = MatrixValuedMapFromVector::sym_diff_quotient(hat_x, x, 0.0001);
                let auto_diff = MatrixValuedMapFromVector::fw_autodiff(dual_hat_x, x);

                for i in 0..DOF {
                    assert_relative_eq!(auto_diff.get([i]), num_diff.get([i]), epsilon = 0.001);
                }
            }

            // dx vee(y)
            {
                let a = Self::hat(&x);
                let vee_x = |v: M<AMBIENT, AMBIENT>| -> V<DOF> { Self::vee(&v) };
                let dual_vee_x = |vv: DualM<AMBIENT, AMBIENT>| -> DualV<DOF> { G::DualG::vee(&vv) };

                let num_diff = VectorValuedMapFromMatrix::sym_diff_quotient(vee_x, a, 0.0001);
                let auto_diff = VectorValuedMapFromMatrix::fw_autodiff(dual_vee_x, a);

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

    fn test_exp_log_jacobians() {
        for t in G::tangent_examples() {
            // x == log(exp(x))

            let log_exp_t: V<DOF> = Self::log(&Self::exp(&t));
            assert_relative_eq!(t, log_exp_t, epsilon = 0.0001);

            // dx exp(x).matrix
            {
                let exp_t = |t: V<DOF>| -> M<AMBIENT, AMBIENT> { Self::exp(&t).matrix() };
                let dual_exp_t = |vv: DualV<DOF>| -> DualM<AMBIENT, AMBIENT> {
                    LieGroup::<Dual, DOF, PARAMS, POINT, AMBIENT, G::DualG>::exp(&vv).matrix()
                };

                let num_diff = MatrixValuedMapFromVector::sym_diff_quotient(exp_t, t, 0.0001);
                let auto_diff = MatrixValuedMapFromVector::fw_autodiff(dual_exp_t, t);

                for i in 0..DOF {
                    assert_relative_eq!(auto_diff.get([i]), num_diff.get([i]), epsilon = 0.001);
                }
            }
        }

        //dx exp(x) at x=0
        {
            let exp_t = |t: V<DOF>| -> V<PARAMS> { *Self::exp(&t).params() };
            let dual_exp_t = |vv: DualV<DOF>| -> DualV<PARAMS> {
                LieGroup::<Dual, DOF, PARAMS, POINT, AMBIENT, G::DualG>::exp(&vv)
                    .params()
                    .clone()
            };

            let analytic_diff = Self::dx_exp_x_at_0();
            let num_diff =
                VectorValuedMapFromVector::static_sym_diff_quotient(exp_t, V::zeros(), 0.0001);
            let auto_diff = VectorValuedMapFromVector::static_fw_autodiff(dual_exp_t, V::zeros());

            assert_relative_eq!(auto_diff, num_diff, epsilon = 0.001);
            assert_relative_eq!(analytic_diff, num_diff, epsilon = 0.001);
        }

        for point in example_points::<f64, POINT>() {
            let exp_t = |t: V<DOF>| -> V<POINT> { Self::exp(&t).transform(&point) };
            let dual_exp_t = |vv: DualV<DOF>| -> DualV<POINT> {
                LieGroup::<Dual, DOF, PARAMS, POINT, AMBIENT, G::DualG>::exp(&vv)
                    .transform(&DualV::c(point))
            };

            let analytic_diff = Self::dx_exp_x_times_point_at_0(point);
            let num_diff =
                VectorValuedMapFromVector::static_sym_diff_quotient(exp_t, V::zeros(), 0.0001);
            let auto_diff = VectorValuedMapFromVector::static_fw_autodiff(dual_exp_t, V::zeros());

            assert_relative_eq!(auto_diff, num_diff, epsilon = 0.001);
            assert_relative_eq!(analytic_diff, num_diff, epsilon = 0.001);
        }

        for g in Self::element_examples() {
            // dx log(y)
            {
                if g.has_shortest_path_ambiguity() {
                    // jacobian not uniquely defined, let's skip these cases
                    continue;
                }

                let log_x = |t: V<DOF>| -> V<DOF> { Self::exp(&t).group_mul(&g).log() };
                let o = V::zeros();

                let dual_params = DualV::c(*g.params());
                let dual_g = LieGroup::<Dual, DOF, PARAMS, POINT, AMBIENT, G::DualG>::from_params(
                    &dual_params,
                );
                let dual_log_x = |t: DualV<DOF>| -> DualV<DOF> {
                    LieGroup::<Dual, DOF, PARAMS, POINT, AMBIENT, G::DualG>::exp(&t)
                        .group_mul(&dual_g)
                        .log()
                };

                let num_diff = VectorValuedMapFromVector::sym_diff_quotient(log_x, o, 0.0001);
                let auto_diff = VectorValuedMapFromVector::fw_autodiff(dual_log_x, o);

                for i in 0..DOF {
                    assert_relative_eq!(auto_diff.get([i]), num_diff.get([i]), epsilon = 0.001);
                }
            }
        }
    }

    pub fn real_test_suite() {
        Self::adjoint_jacobian_tests();
        Self::test_hat_jacobians();
        Self::test_exp_log_jacobians();
    }
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>,
    > Display for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, G>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.compact().real())
    }
}
