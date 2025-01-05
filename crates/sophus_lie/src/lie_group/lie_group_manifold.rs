use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::traits::IsLieGroupImpl;
use core::borrow::Borrow;
use core::fmt::Debug;
use sophus_core::params::ParamsImpl;

extern crate alloc;

/// Left group manifold
///
/// A ⊕ t := exp(t) * A
/// A ⊖ B :=  log(inv(A) * B)
#[derive(Debug, Clone)]
pub struct LeftGroupManifold<
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
    group: LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>,
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
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN> + Clone + Debug,
    > ParamsImpl<S, PARAMS, BATCH, DM, DN>
    for LeftGroupManifold<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    fn are_params_valid<P>(params: P) -> S::Mask
    where
        P: for<'a> Borrow<<S as IsScalar<BATCH, DM, DN>>::Vector<PARAMS>>,
    {
        G::are_params_valid(params)
    }

    fn params_examples() -> alloc::vec::Vec<<S as IsScalar<BATCH, DM, DN>>::Vector<PARAMS>> {
        G::params_examples()
    }

    fn invalid_params_examples() -> alloc::vec::Vec<<S as IsScalar<BATCH, DM, DN>>::Vector<PARAMS>>
    {
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
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN> + Clone + Debug,
    > HasParams<S, PARAMS, BATCH, DM, DN>
    for LeftGroupManifold<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    fn from_params<P>(params: P) -> Self
    where
        P: for<'a> Borrow<<S as IsScalar<BATCH, DM, DN>>::Vector<PARAMS>>,
    {
        Self {
            group: LieGroup::from_params(params),
        }
    }

    fn set_params<P>(&mut self, params: P)
    where
        P: for<'a> Borrow<<S as IsScalar<BATCH, DM, DN>>::Vector<PARAMS>>,
    {
        self.group.set_params(params)
    }

    fn params(&self) -> &<S as IsScalar<BATCH, DM, DN>>::Vector<PARAMS> {
        self.group.params()
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
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN> + Clone + Debug,
    > IsManifold<S, PARAMS, DOF, BATCH, DM, DN>
    for LeftGroupManifold<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    fn oplus(&self, tangent: &<S as IsScalar<BATCH, DM, DN>>::Vector<DOF>) -> Self {
        Self {
            group: &LieGroup::<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>::exp(tangent)
                * &self.group,
        }
    }

    fn ominus(&self, rhs: &Self) -> <S as IsScalar<BATCH, DM, DN>>::Vector<DOF> {
        (&self.group.inverse() * &rhs.group).log()
    }
}
