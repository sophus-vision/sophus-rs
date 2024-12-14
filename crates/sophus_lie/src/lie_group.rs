use super::traits::IsLieGroupImpl;
use crate::prelude::*;
use approx::assert_relative_eq;
use core::borrow::Borrow;
use core::fmt::Debug;
use sophus_core::manifold::traits::TangentImpl;
use sophus_core::params::ParamsImpl;

extern crate alloc;

/// Lie group average
pub mod average;
/// Group multiplication
pub mod group_mul;
/// Lie group as a manifold
pub mod lie_group_manifold;
/// Real lie group
pub mod real_lie_group;

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
    phantom: core::marker::PhantomData<G>,
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
    fn are_params_valid<P>(params: P) -> S::Mask
    where
        P: for<'a> Borrow<S::Vector<PARAMS>>,
    {
        G::are_params_valid(params)
    }

    fn params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>> {
        G::params_examples()
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>> {
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
    fn from_params<P>(params: P) -> Self
    where
        P: for<'a> Borrow<S::Vector<PARAMS>>,
    {
        let params = params.borrow();

        assert!(
            G::are_params_valid(params).all(),
            "Invalid parameters for {:?}",
            params.real_vector()
        );
        Self {
            params: G::disambiguate(params.clone()),
            phantom: core::marker::PhantomData,
        }
    }

    fn set_params<P>(&mut self, params: P)
    where
        P: for<'a> Borrow<S::Vector<PARAMS>>,
    {
        let params = params.borrow();
        self.params = G::disambiguate(params.clone());
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
    fn tangent_examples() -> alloc::vec::Vec<<S as IsScalar<BATCH_SIZE>>::Vector<DOF>> {
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
    pub fn exp<T>(omega: T) -> Self
    where
        T: Borrow<S::Vector<DOF>>,
    {
        Self::from_params(G::exp(omega.borrow()))
    }

    /// Interpolate between "(w-1) * self" and "w * other".
    ///
    /// w is typically in [0, 1]. If w=0, self is returned. If w=1 other is returned.
    ///
    pub fn interpolate(&self, other: &Self, w: S) -> Self {
        self * &Self::exp((self.inverse() * other).log().scaled(w))
    }

    /// logarithmic map
    pub fn log(&self) -> S::Vector<DOF> {
        G::log(&self.params)
    }

    /// hat operator: hat: R^d -> R^{a x a}
    pub fn hat<T>(omega: T) -> S::Matrix<AMBIENT, AMBIENT>
    where
        T: Borrow<S::Vector<DOF>>,
    {
        G::hat(omega.borrow())
    }

    /// vee operator: vee: R^{a x a} -> R^d
    pub fn vee<M>(xi: M) -> S::Vector<DOF>
    where
        M: Borrow<S::Matrix<AMBIENT, AMBIENT>>,
    {
        G::vee(xi.borrow())
    }

    /// identity element
    pub fn identity() -> Self {
        Self::from_params(G::identity_params())
    }

    /// group multiplication
    pub fn group_mul(&self, other: &Self) -> Self {
        Self::from_params(G::group_mul(&self.params, &other.params))
    }

    /// group inverse
    pub fn inverse(&self) -> Self {
        Self::from_params(G::inverse(&self.params))
    }

    /// transform a point
    pub fn transform<T>(&self, point: T) -> S::Vector<POINT>
    where
        T: Borrow<S::Vector<POINT>>,
    {
        G::transform(&self.params, point.borrow())
    }

    /// convert point to ambient space
    pub fn to_ambient<P>(point: P) -> S::Vector<AMBIENT>
    where
        P: Borrow<S::Vector<POINT>>,
    {
        G::to_ambient(point.borrow())
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
    pub fn ad<T>(tangent: T) -> S::Matrix<DOF, DOF>
    where
        T: Borrow<S::Vector<DOF>>,
    {
        G::ad(tangent.borrow())
    }

    /// group element examples
    pub fn element_examples() -> alloc::vec::Vec<Self> {
        let mut elements = alloc::vec![];
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
                let o_transformed = g.transform(o);
                let mask = (o_transformed.real_vector())
                    .norm()
                    .less_equal(&S::RealScalar::from_f64(0.0001));

                num_preserves += mask.count();
                num += S::Mask::all_true().count();
            }
            let percentage = num_preserves as f64 / num as f64;
            assert!(percentage <= 0.75, "{} <= 0.75", percentage);
        }
    }

    fn adjoint_tests() {
        let group_examples: alloc::vec::Vec<_> = Self::element_examples();
        let tangent_examples: alloc::vec::Vec<S::Vector<DOF>> = G::tangent_examples();

        for g in &group_examples {
            let mat: S::Matrix<AMBIENT, AMBIENT> = g.matrix();
            let mat_adj = g.adj();

            for x in &tangent_examples {
                let mat_adj_x = mat_adj.clone() * x.clone();

                let inv_mat: S::Matrix<AMBIENT, AMBIENT> = g.inverse().matrix();
                let mat_adj_x2 = Self::vee(mat.mat_mul(Self::hat(x).mat_mul(inv_mat)));
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
        let group_examples: alloc::vec::Vec<_> = Self::element_examples();
        let tangent_examples: alloc::vec::Vec<S::Vector<DOF>> = G::tangent_examples();

        for g in &group_examples {
            let matrix_before = *g.compact().real_matrix();
            let matrix_after = *Self::exp(g.log()).compact().real_matrix();

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
        let tangent_examples: alloc::vec::Vec<S::Vector<DOF>> = G::tangent_examples();

        for omega in &tangent_examples {
            assert_relative_eq!(
                omega.real_vector(),
                Self::vee(Self::hat(omega)).real_vector(),
                epsilon = 0.0001
            );
        }
    }

    fn group_operation_tests() {
        let group_examples: alloc::vec::Vec<_> = Self::element_examples();

        for g1 in &group_examples {
            for g2 in &group_examples {
                for g3 in &group_examples {
                    let left_hugging = (g1 * g2) * g3;
                    let right_hugging = g1 * (g2 * g3);
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
                let daz_from_foo_transform_1 = g2.inverse() * g1.inverse();
                let daz_from_foo_transform_2 = (g1 * g2).inverse();
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
        let group_examples: alloc::vec::Vec<_> = Self::element_examples();
        assert!(group_examples.len() >= 3);
        let tangent_examples: alloc::vec::Vec<S::Vector<DOF>> = G::tangent_examples();
        assert!(tangent_examples.len() >= 3);

        Self::presentability_tests();
        Self::group_operation_tests();
        Self::hat_tests();
        Self::exp_tests();
        Self::adjoint_tests();
    }
}
