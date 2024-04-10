use super::traits::IsLieGroupImpl;
use crate::prelude::*;
use approx::assert_relative_eq;
use assertables::assert_le_as_result;
use sophus_core::manifold::traits::TangentImpl;
use sophus_core::params::ParamsImpl;
use std::fmt::Debug;

/// Lie group
#[derive(Debug, Copy, Clone, Default)]
pub struct LieGroup<
    S: IsScalar<BATCH_SIZE>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH_SIZE: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
> {
    pub(crate) params: S::Vector<PARAMS>,
    phantom: std::marker::PhantomData<G>,
}

impl<
        S: IsScalar<BATCH_SIZE>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
    > ParamsImpl<S, PARAMS, BATCH_SIZE>
    for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    fn are_params_valid(params: &<S as IsScalar<BATCH_SIZE>>::Vector<PARAMS>) -> S::Mask {
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
        S: IsScalar<BATCH_SIZE>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
    > HasParams<S, PARAMS, BATCH_SIZE> for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    fn from_params(params: &S::Vector<PARAMS>) -> Self {
        assert!(
            G::are_params_valid(params).all(),
            "Invalid parameters for {:?}",
            params.real_vector()
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
        S: IsScalar<BATCH_SIZE>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
    > TangentImpl<S, DOF, BATCH_SIZE> for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    fn tangent_examples() -> Vec<<S as IsScalar<BATCH_SIZE>>::Vector<DOF>> {
        G::tangent_examples()
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
    > IsLieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>
    for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    type G = G;
    type GenG<S2: IsScalar<BATCH_SIZE>> = G::GenG<S2>;
    type RealG = G::RealG;
    type DualG = G::DualG;

    type GenGroup<
        S2: IsScalar<BATCH_SIZE>,
        G2: IsLieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
    > = LieGroup<S2, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G2>;
    type RealGroup = Self::GenGroup<S::RealScalar, G::RealG>;
    type DualGroup = Self::GenGroup<S::DualScalar, G::DualG>;

    const DOF: usize = DOF;

    const PARAMS: usize = PARAMS;

    const POINT: usize = POINT;

    const AMBIENT: usize = AMBIENT;
}

impl<
        S: IsScalar<BATCH_SIZE>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
    > LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    /// group adjoint
    pub fn adj(&self) -> S::Matrix<DOF, DOF> {
        G::adj(&self.params)
    }

    /// exponential map
    pub fn exp(omega: &S::Vector<DOF>) -> Self {
        Self::from_params(&G::exp(omega))
    }

    /// logarithmic map
    pub fn log(&self) -> S::Vector<DOF> {
        G::log(&self.params)
    }

    /// hat operator: hat: R^d -> R^{a x a}
    pub fn hat(omega: &S::Vector<DOF>) -> S::Matrix<AMBIENT, AMBIENT> {
        G::hat(omega)
    }

    /// vee operator: vee: R^{a x a} -> R^d
    pub fn vee(xi: &S::Matrix<AMBIENT, AMBIENT>) -> S::Vector<DOF> {
        G::vee(xi)
    }

    /// identity element
    pub fn identity() -> Self {
        Self::from_params(&G::identity_params())
    }

    /// group multiplication
    pub fn group_mul(&self, other: &Self) -> Self {
        Self::from_params(&G::group_mul(&self.params, &other.params))
    }

    /// group inverse
    pub fn inverse(&self) -> Self {
        Self::from_params(&G::inverse(&self.params))
    }

    /// transform a point
    pub fn transform(&self, point: &S::Vector<POINT>) -> S::Vector<POINT> {
        G::transform(&self.params, point)
    }

    /// convert point to ambient space
    pub fn to_ambient(point: &S::Vector<POINT>) -> S::Vector<AMBIENT> {
        G::to_ambient(point)
    }

    /// return compact matrix representation
    pub fn compact(&self) -> S::Matrix<POINT, AMBIENT> {
        G::compact(&self.params)
    }

    /// return square matrix representation
    pub fn matrix(&self) -> S::Matrix<AMBIENT, AMBIENT> {
        G::matrix(&self.params)
    }

    /// algebra adjoint
    pub fn ad(tangent: &S::Vector<DOF>) -> S::Matrix<DOF, DOF> {
        G::ad(tangent)
    }

    /// group element examples
    pub fn element_examples() -> Vec<Self> {
        let mut elements = vec![];
        for params in Self::params_examples() {
            elements.push(Self::from_params(&params));
        }
        elements
    }

    fn presentability_tests() {
        if G::IS_ORIGIN_PRESERVING {
            for g in &Self::element_examples() {
                let o = S::Vector::<POINT>::zeros();

                approx::assert_abs_diff_eq!(
                    g.transform(&o).real_vector(),
                    o.real_vector(),
                    epsilon = 0.0001
                );
            }
        } else {
            let mut num_preserves = 0;
            let mut num = 0;
            for g in &Self::element_examples() {
                let o = S::Vector::<POINT>::zeros();
                let o_transformed = g.transform(&o);
                let mask = (o_transformed.real_vector())
                    .norm()
                    .less_equal(&S::RealScalar::from_f64(0.0001));

                num_preserves += mask.count();
                num += S::Mask::all_true().count();
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
                assert_relative_eq!(
                    mat_adj_x.real_vector(),
                    mat_adj_x2.real_vector(),
                    epsilon = 0.0001
                );
            }
        }
        for a in &tangent_examples {
            for b in &tangent_examples {
                let ad_a = Self::ad(a);
                let ad_a_b = ad_a * b.clone();
                let hat_ab = Self::hat(a).mat_mul(Self::hat(b));
                let hat_ba = Self::hat(b).mat_mul(Self::hat(a));

                let lie_bracket_a_b = Self::vee(&(hat_ab - hat_ba));
                assert_relative_eq!(
                    ad_a_b.real_vector(),
                    lie_bracket_a_b.real_vector(),
                    epsilon = 0.0001
                );
            }
        }
    }

    fn exp_tests() {
        let group_examples: Vec<_> = Self::element_examples();
        let tangent_examples: Vec<S::Vector<DOF>> = G::tangent_examples();

        for g in &group_examples {
            let matrix_before = *g.compact().real_matrix();
            let matrix_after = *Self::exp(&g.log()).compact().real_matrix();

            assert_relative_eq!(matrix_before, matrix_after, epsilon = 0.0001);

            let t = *g.clone().inverse().log().real_vector();
            let t2 = -(*g.log().real_vector());
            assert_relative_eq!(t, t2, epsilon = 0.0001);
        }
        for omega in &tangent_examples {
            let exp_inverse = Self::exp(omega).inverse();
            let neg_omega = -omega.clone();

            let exp_neg_omega = Self::exp(&neg_omega);
            assert_relative_eq!(
                exp_inverse.compact(),
                exp_neg_omega.compact(),
                epsilon = 0.0001
            );
        }
    }

    fn hat_tests() {
        let tangent_examples: Vec<S::Vector<DOF>> = G::tangent_examples();

        for omega in &tangent_examples {
            assert_relative_eq!(
                omega.real_vector(),
                Self::vee(&Self::hat(omega)).real_vector(),
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
                        left_hugging.compact(),
                        right_hugging.compact(),
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
                    daz_from_foo_transform_1.compact(),
                    daz_from_foo_transform_2.compact(),
                    epsilon = 0.0001
                );
            }
        }
    }

    /// run all tests
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
