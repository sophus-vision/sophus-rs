use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::traits::IsLieGroupImpl;
use sophus_core::params::ParamsImpl;
use std::fmt::Debug;

/// Left group manifold
#[derive(Debug, Clone)]
pub struct LeftGroupManifold<
    S: IsScalar<BATCH_SIZE>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH_SIZE: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
> {
    group: LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>,
}

impl<
        S: IsScalar<BATCH_SIZE>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE> + Clone + Debug,
    > ParamsImpl<S, PARAMS, BATCH_SIZE>
    for LeftGroupManifold<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    fn are_params_valid(params: &<S as IsScalar<BATCH_SIZE>>::Vector<PARAMS>) -> S::Mask {
        G::are_params_valid(params)
    }

    fn params_examples() -> Vec<<S as IsScalar<BATCH_SIZE>>::Vector<PARAMS>> {
        G::params_examples()
    }

    fn invalid_params_examples() -> Vec<<S as IsScalar<BATCH_SIZE>>::Vector<PARAMS>> {
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
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE> + Clone + Debug,
    > HasParams<S, PARAMS, BATCH_SIZE>
    for LeftGroupManifold<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    fn from_params(params: &<S as IsScalar<BATCH_SIZE>>::Vector<PARAMS>) -> Self {
        Self {
            group: LieGroup::from_params(params),
        }
    }

    fn set_params(&mut self, params: &<S as IsScalar<BATCH_SIZE>>::Vector<PARAMS>) {
        self.group.set_params(params)
    }

    fn params(&self) -> &<S as IsScalar<BATCH_SIZE>>::Vector<PARAMS> {
        self.group.params()
    }
}

impl<
        S: IsScalar<BATCH_SIZE>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE> + Clone + Debug,
    > IsManifold<S, PARAMS, DOF, BATCH_SIZE>
    for LeftGroupManifold<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    fn oplus(&self, tangent: &<S as IsScalar<BATCH_SIZE>>::Vector<DOF>) -> Self {
        Self {
            group: LieGroup::<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>::exp(tangent)
                .group_mul(&self.group),
        }
    }

    fn ominus(&self, rhs: &Self) -> <S as IsScalar<BATCH_SIZE>>::Vector<DOF> {
        self.group.inverse().group_mul(&rhs.group).log()
    }

    fn params(&self) -> &<S as IsScalar<BATCH_SIZE>>::Vector<PARAMS> {
        self.group.params()
    }
}
