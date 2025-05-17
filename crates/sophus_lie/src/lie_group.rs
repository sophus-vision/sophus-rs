pub(crate) mod average;
pub(crate) mod factor_lie_group;
pub(crate) mod group_mul;
pub(crate) mod lie_group_manifold;
pub(crate) mod real_lie_group;

use core::{
    borrow::Borrow,
    fmt::Debug,
};

use approx::assert_relative_eq;
use sophus_autodiff::{
    manifold::IsTangent,
    params::{
        HasParams,
        IsParamsImpl,
    },
};

use crate::{
    IsLieGroupImpl,
    prelude::*,
};

extern crate alloc;

/// Lie group
///
/// A Lie group is a group - i.e. fulfills all the group axioms (of multiplication) - which is
/// also a manifold. A manifold is a space which looks locally Euclidean, but globally might
/// have a very different structure.
///
/// Most Lie groups (and all Lie groups in the context of the crate) are Matrix Lie groups:
/// Matrix Lie groups are simply groups of invertible NxN matrices. For instance:
///   - The group of all invertible NxN matrices is a Lie group. It is called the General Linear
///     Group - short GL(N).
///   - The group of orthogonal 3x3 matrices with positive determinate (and hence invertible) is a
///     Lie - called Special Orthogonal Group 3 - short SO(3). It is a sub-group of GL(3).
///     Geometrically, it is the group of rotations in 3d - see the Rotation3 struct.
///
/// This struct uses the following two generic type parameter:
///
///  * S
///    - the underlying scalar such as f64 or [sophus_autodiff::dual::DualScalar].
///  * G
///    - concrete Lie group implementation (e.g. for [crate::Rotation2], or [crate::Isometry3])
///
/// and the following const generic:
///
///  * DOF
///    - Degrees of freedom of the transformation, aka dimension of the tangent space
///  * PARAMS
///     - Number of parameters.
///  * POINT
///     - Dimension of the points the transformation is acting on.
///  * AMBIENT
///      - Dimension of the ambient space. If the matrix is represented as a square matrix then its
///        dimension is AMBIENT x AMBIENT. Note that either AMBIENT==POINT or AMBIENT==POINT+1. In
///        the latter case, the matrix acts on homogeneous points.
///  * BATCH
///     - Batch dimension. If S is f64 or [sophus_autodiff::dual::DualScalar] then BATCH=1.
///  * DM, DN
///    - DM x DN is the static shape of the Jacobian to be computed if S == DualScalar<DM, DN>. If S
///      == f64, then DM==0, DN==0.
///
/// Note on the API: Ideally, we would be able to deduce most const generics, e.g. BATCH, DM, DN
///                  from S and DOF, PARAMS, POINT, AMBIENT from G. Some such API simplifications
///                  might be possible once "min_generic_const_args" is merged / stable:
///                  <https://rust-lang.github.io/rust-project-goals/2024h2/min_generic_const_arguments.html>
#[derive(Debug, Copy, Clone, Default)]
pub struct LieGroup<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN>,
> {
    pub(crate) params: S::Vector<PARAMS>,
    phantom: core::marker::PhantomData<G>,
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
> LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
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
            elements.push(Self::from_params(params));
        }
        elements
    }

    fn presentability_tests() {
        if G::IS_ORIGIN_PRESERVING {
            for g in &Self::element_examples() {
                let o = S::Vector::<POINT>::zeros();

                approx::assert_abs_diff_eq!(
                    g.transform(o).real_vector(),
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
            assert!(percentage <= 0.75, "{percentage} <= 0.75");
        }
    }

    fn adjoint_tests() {
        let group_examples = Self::element_examples();
        let basis: alloc::vec::Vec<S::Vector<DOF>> = (0..DOF)
            .map(|i| {
                let mut e = S::Vector::<DOF>::zeros();
                *e.elem_mut(i) = S::ones();
                e
            })
            .collect();

        for g in &group_examples {
            let mat_g = g.matrix();
            let inv_mat_g = g.inverse().matrix();

            let mut ad_ref = S::Matrix::<DOF, DOF>::zeros();

            for i in 0..DOF {
                let col_i = Self::vee(mat_g.mat_mul(Self::hat(basis[i]).mat_mul(inv_mat_g)));
                ad_ref.set_col_vec(i, col_i);
            }

            let ad_impl = g.adj();
            assert_relative_eq!(
                ad_impl.real_matrix(),
                ad_ref.real_matrix(),
                epsilon = 0.0001
            );
        }
        let tangent_examples: alloc::vec::Vec<S::Vector<DOF>> = G::tangent_examples();
        for a in &tangent_examples {
            for b in &tangent_examples {
                let ad_a = Self::ad(a);
                let ad_a_b = ad_a * *b;
                let hat_ab = Self::hat(a).mat_mul(Self::hat(b));
                let hat_ba = Self::hat(b).mat_mul(Self::hat(a));

                let lie_bracket_a_b = Self::vee(hat_ab - hat_ba);
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
            let matrix_before = g.compact().real_matrix();
            let matrix_after = Self::exp(g.log()).compact().real_matrix();

            assert_relative_eq!(matrix_before, matrix_after, epsilon = 0.0001);

            let t = g.inverse().log().real_vector();
            let t2 = -(g.log().real_vector());
            assert_relative_eq!(t, t2, epsilon = 0.0001);
        }
        for omega in &tangent_examples {
            let exp_inverse = Self::exp(omega).inverse();
            let neg_omega = -*omega;

            let exp_neg_omega = Self::exp(neg_omega);
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
        // Most tests will trivially pass if there are no examples. So first we make sure we have at
        // least three examples per group.
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
> IsParamsImpl<S, PARAMS, BATCH, DM, DN>
    for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    fn are_params_valid(params: S::Vector<PARAMS>) -> S::Mask {
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
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN>,
> HasParams<S, PARAMS, BATCH, DM, DN>
    for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    fn from_params(params: S::Vector<PARAMS>) -> Self {
        assert!(
            G::are_params_valid(params).all(),
            "Invalid parameters for {:?}",
            params.real_vector()
        );
        Self {
            params: G::disambiguate(params),
            phantom: core::marker::PhantomData,
        }
    }

    fn set_params(&mut self, params: S::Vector<PARAMS>) {
        self.params = G::disambiguate(params);
    }

    fn params(&self) -> &S::Vector<PARAMS> {
        &self.params
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
> IsTangent<S, DOF, BATCH, DM, DN> for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    fn tangent_examples() -> alloc::vec::Vec<<S as IsScalar<BATCH, DM, DN>>::Vector<DOF>> {
        G::tangent_examples()
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
> IsLieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN>
    for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    type G = G;
    type GenG<S2: IsScalar<BATCH, DM, DN>> = G::GenG<S2>;
    type RealG = G::RealG;
    type DualG<const M: usize, const N: usize> = G::DualG<M, N>;

    type GenGroup<
        S2: IsScalar<BATCH, DM, DN>,
        G2: IsLieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN>,
    > = LieGroup<S2, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G2>;

    const DOF: usize = DOF;

    const PARAMS: usize = PARAMS;

    const POINT: usize = POINT;

    const AMBIENT: usize = AMBIENT;
}
