use sophus_lie::prelude::{
    IsScalar,
    IsSingleScalar,
    IsVector,
};

use crate::unit_vector::UnitVector;

/// Ray in `ℝⁿ`.
///
/// ## Generic parameters:
///
///  * S
///    - the underlying scalar such as [f64] or [sophus_autodiff::dual::DualScalar].
///  * DOF
///    - Degrees of freedom of the unit direction vector.
///  * DIM
///    - Dimension of the ray. It holds that DIM == DOF + 1.
///  * BATCH
///    - Batch dimension. If S is [f64] or [sophus_autodiff::dual::DualScalar] then BATCH=1.
///  * DM, DN
///    - DM x DN is the static shape of the Jacobian to be computed if S == DualScalar<DM, DN>. If S
///      == f64, then DM==0, DN==0.
#[derive(Clone, Debug)]
pub struct Ray<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    /// The ray origin.
    pub origin: S::Vector<DIM>,
    /// The unit direction vector of the ray.
    pub dir: UnitVector<S, DOF, DIM, BATCH, DM, DN>,
}

impl<
    S: IsSingleScalar<DM, DN> + PartialOrd,
    const DOF: usize,
    const DIM: usize,
    const DM: usize,
    const DN: usize,
> Ray<S, DOF, DIM, 1, DM, DN>
{
    /// Returns point on ray.
    ///
    /// For a given t, it returns  "origin + t * dir".
    pub fn at(&self, t: S) -> S::Vector<DIM> {
        self.origin + self.dir.vector().scaled(t)
    }
}

/// 2d ray.
pub type Ray2<S, const B: usize, const DM: usize, const DN: usize> = Ray<S, 1, 2, B, DM, DN>;
/// 3d ray.
pub type Ray3<S, const B: usize, const DM: usize, const DN: usize> = Ray<S, 2, 3, B, DM, DN>;
