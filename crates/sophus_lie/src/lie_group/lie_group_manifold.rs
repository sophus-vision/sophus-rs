use core::fmt::Debug;

use sophus_autodiff::{
    linalg::VecF64,
    manifold::IsVariable,
};

use crate::{
    IsLieGroupImpl,
    lie_group::LieGroup,
    prelude::*,
};

extern crate alloc;

// Lie group as Left group manifold
//
// A ⊕ t := exp(t) * A
// A ⊖ B :=  log(inv(A) * B)
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
    for LieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>
{
    fn oplus(&self, tangent: &<S as IsScalar<BATCH, DM, DN>>::Vector<DOF>) -> Self {
        LieGroup::<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN, G>::exp(*tangent) * self
    }

    fn ominus(&self, rhs: &Self) -> <S as IsScalar<BATCH, DM, DN>>::Vector<DOF> {
        (self.inverse() * rhs).log()
    }
}

impl<
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    G: IsLieGroupImpl<f64, DOF, PARAMS, POINT, AMBIENT, 1, 0, 0>
        + Clone
        + Debug
        + Send
        + Sync
        + 'static,
> IsVariable for LieGroup<f64, DOF, PARAMS, POINT, AMBIENT, 1, 0, 0, G>
{
    const NUM_DOF: usize = DOF;

    fn update(&mut self, delta: nalgebra::DVectorView<f64>) {
        assert_eq!(delta.len(), DOF);
        let mut tangent = VecF64::<DOF>::zeros();
        for d in 0..DOF {
            tangent[d] = delta[d];
        }
        *self = self.oplus(&tangent);
    }
}
