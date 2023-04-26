use std::fmt::{Display, Formatter};

use approx::assert_relative_eq;
use assertables::assert_le_as_result;

use crate::calculus::numeric_diff::VectorField;

use super::traits::LieGroupImpl;

type V<const N: usize> = nalgebra::SVector<f64, N>;
type M<const N: usize, const O: usize> = nalgebra::SMatrix<f64, N, O>;

#[derive(Debug, Copy, Clone)]
pub struct LieGroup<
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    G: LieGroupImpl<DOF, PARAMS, POINT, AMBIENT>,
> {
    params: V<PARAMS>,
    phantom: std::marker::PhantomData<G>,
}

impl<
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: LieGroupImpl<DOF, PARAMS, POINT, AMBIENT>,
    > LieGroup<DOF, PARAMS, POINT, AMBIENT, G>
{
    pub fn from_params(params: &V<PARAMS>) -> Self {
        assert!(
            G::are_params_valid(params),
            "Invalid parameters for {}",
            params
        );
        Self {
            params: params.clone(),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn set_params(&mut self, params: &V<PARAMS>) {
        self.params = params.clone();
    }

    pub fn params(&self) -> &V<PARAMS> {
        &self.params
    }

    pub fn params_examples() -> Vec<V<PARAMS>> {
        G::params_examples()
    }

    pub fn invalid_params_examples() -> Vec<V<PARAMS>> {
        G::invalid_params_examples()
    }

    pub fn adj(&self) -> M<DOF, DOF> {
        G::adj(&self.params)
    }

    pub fn exp(omega: &V<DOF>) -> Self {
        Self::from_params(&G::exp(omega))
    }

    pub fn log(&self) -> V<DOF> {
        G::log(&self.params)
    }

    pub fn hat(omega: &V<DOF>) -> M<AMBIENT, AMBIENT> {
        G::hat(omega)
    }

    pub fn vee(xi: &M<AMBIENT, AMBIENT>) -> V<DOF> {
        G::vee(xi)
    }

    pub fn identity() -> Self {
        Self::from_params(&G::identity_params())
    }

    pub fn multiply(&self, other: &Self) -> Self {
        Self::from_params(&G::multiply(&self.params, &other.params))
    }

    pub fn inverse(&self) -> Self {
        Self::from_params(&G::inverse(&self.params))
    }

    pub fn transform(&self, point: &V<POINT>) -> V<POINT> {
        G::transform(&self.params, point)
    }

    pub fn to_ambient(point: &V<POINT>) -> V<AMBIENT> {
        G::to_ambient(point)
    }

    pub fn compact(&self) -> M<POINT, AMBIENT> {
        G::compact(&self.params)
    }

    pub fn matrix(&self) -> M<AMBIENT, AMBIENT> {
        G::matrix(&self.params)
    }

    pub fn ad(tangent: &V<DOF>) -> M<DOF, DOF> {
        G::ad(&tangent)
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
                let o = V::<POINT>::zeros();
                assert_relative_eq!(g.transform(&o), o);
            }
        } else {
            let mut num_preserves = 0;
            let mut num = 0;
            for g in &Self::element_examples() {
                let o = V::<POINT>::zeros();
                let o_transformed = g.transform(&o);
                if (o_transformed).norm() < 0.0001 {
                    num_preserves += 1;
                }
                num += 1;
            }
            let percentage = num_preserves as f64 / num as f64;
            assertables::assert_le!(percentage, 0.75);
        }
    }

    fn adjoint_tests() {
        for g in &Self::element_examples() {
            let mat = g.matrix();
            let mat_adj = g.adj();
            for x in &G::tangent_examples() {
                let mat_adj_x = mat_adj * x;
                let mat_adj_x2 = Self::vee(&(&mat * &Self::hat(x) * &g.inverse().matrix()));
                assert_relative_eq!(mat_adj_x, mat_adj_x2, epsilon = 0.0001);
            }
        }
        for a in &G::tangent_examples() {
            for b in &G::tangent_examples() {
                let ad_a = Self::ad(a);
                let ad_a_b = ad_a * b;
                let lie_bracket_a_b =
                    Self::vee(&(&Self::hat(a) * &Self::hat(b) - &Self::hat(b) * &Self::hat(a)));
                assert_relative_eq!(ad_a_b, lie_bracket_a_b, epsilon = 0.0001);
                if DOF > 0 {
                    let num_diff_ad_a = VectorField::numeric_diff(
                        |x| {
                            Self::vee(
                                &(&Self::hat(a) * &Self::hat(x) - &Self::hat(x) * &Self::hat(a)),
                            )
                        },
                        *b,
                        0.0001,
                    );
                    assert_relative_eq!(ad_a, num_diff_ad_a, epsilon = 0.0001);
                }
            }
        }
    }

    fn exp_tests() {
        for g in &Self::element_examples() {
            let matrix_before = g.compact();
            let matrix_after = Self::exp(&g.log()).compact();
            assert_relative_eq!(matrix_before, matrix_after, epsilon = 0.0001);
        }
        for omega in &G::tangent_examples() {
            let exp_inverse = Self::exp(&omega).inverse();
            let exp_neg_omega = Self::exp(&-omega);
            assert_relative_eq!(
                exp_inverse.compact(),
                exp_neg_omega.compact(),
                epsilon = 0.0001
            );
        }
    }

    fn hat_tests() {
        for omega in &G::tangent_examples() {
            assert_relative_eq!(*omega, Self::vee(&Self::hat(omega)), epsilon = 0.0001);
        }
    }

    fn group_operation_tests() {
        for g1 in &Self::element_examples() {
            for g2 in &Self::element_examples() {
                for g3 in &Self::element_examples() {
                    let left_hugging = &(g1 * g2) * g3;
                    let right_hugging = g1 * &(g2 * g3);
                    assert_relative_eq!(
                        left_hugging.compact(),
                        right_hugging.compact(),
                        epsilon = 0.0001
                    );
                }
            }
        }
        for g1 in &Self::element_examples() {
            for g2 in &Self::element_examples() {
                let daz_from_foo_transform_1 = &g2.inverse() * &g1.inverse();
                let daz_from_foo_transform_2 = (g1 * g2).inverse();
                assert_relative_eq!(
                    daz_from_foo_transform_1.compact(),
                    daz_from_foo_transform_2.compact(),
                    epsilon = 0.0001
                );
            }
        }
    }

    pub fn test_suite() {
        Self::presentability_tests();
        Self::group_operation_tests();
        Self::hat_tests();
        Self::exp_tests();
        Self::adjoint_tests();
    }
}

impl<
        'a,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: LieGroupImpl<DOF, PARAMS, POINT, AMBIENT>,
    > std::ops::Mul<&'a LieGroup<DOF, PARAMS, POINT, AMBIENT, G>>
    for &'a LieGroup<DOF, PARAMS, POINT, AMBIENT, G>
{
    type Output = LieGroup<DOF, PARAMS, POINT, AMBIENT, G>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply(rhs)
    }
}

impl<
        'a,
        'b,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: LieGroupImpl<DOF, PARAMS, POINT, AMBIENT>,
    > std::ops::Mul<&'a V<POINT>> for &'b LieGroup<DOF, PARAMS, POINT, AMBIENT, G>
{
    type Output = V<POINT>;

    fn mul(self, point: &'a V<POINT>) -> Self::Output {
        self.transform(point)
    }
}

impl<
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: LieGroupImpl<DOF, PARAMS, POINT, AMBIENT>,
    > Display for LieGroup<DOF, PARAMS, POINT, AMBIENT, G>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.compact())
    }
}
