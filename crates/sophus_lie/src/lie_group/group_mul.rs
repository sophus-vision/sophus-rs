use crate::lie_group::LieGroup;
use crate::traits::IsLieGroupImpl;
use core::ops::Mul;
use sophus_core::prelude::IsScalar;

// a * b
impl<
        S: IsScalar<BATCH_SIZE>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
    > Mul<LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>>
    for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    type Output = LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.group_mul(&rhs)
    }
}

// a * &b
impl<
        S: IsScalar<BATCH_SIZE>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
    > Mul<&LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>>
    for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    type Output = LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>;

    fn mul(self, rhs: &Self) -> Self::Output {
        self.group_mul(rhs)
    }
}

// &a * &b
impl<
        S: IsScalar<BATCH_SIZE>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
    > Mul<LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>>
    for &LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    type Output = LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>;

    fn mul(self, rhs: LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>) -> Self::Output {
        self.group_mul(&rhs)
    }
}

// a * &b
impl<
        S: IsScalar<BATCH_SIZE>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const BATCH_SIZE: usize,
        G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>,
    > Mul<&LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>>
    for &LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>
{
    type Output = LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>;

    fn mul(self, rhs: &LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE, G>) -> Self::Output {
        self.group_mul(rhs)
    }
}
