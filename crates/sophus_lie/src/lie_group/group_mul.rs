use core::ops::Mul;

use sophus_autodiff::prelude::IsScalar;

use crate::{
    lie_group::LieGroup,
    traits::IsLieGroupImpl,
};

// a * b
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
    > Mul<LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>>
    for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    type Output = LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.group_mul(&rhs)
    }
}

// a * &b
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
    > Mul<&LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>>
    for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    type Output = LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>;

    fn mul(self, rhs: &Self) -> Self::Output {
        self.group_mul(rhs)
    }
}

// &a * &b
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
    > Mul<LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>>
    for &LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    type Output = LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>;

    fn mul(self, rhs: LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>) -> Self::Output {
        self.group_mul(&rhs)
    }
}

// a * &b
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
    > Mul<&LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>>
    for &LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    type Output = LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>;

    fn mul(self, rhs: &LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>) -> Self::Output {
        self.group_mul(rhs)
    }
}
