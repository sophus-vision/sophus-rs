use sophus_core::unit_vector::UnitVector;
use sophus_lie::prelude::IsScalar;
use sophus_lie::prelude::IsSingleScalar;
use sophus_lie::prelude::IsVector;

/// Ray
#[derive(Clone, Debug)]
pub struct Ray<S: IsScalar<BATCH_SIZE>, const DOF: usize, const DIM: usize, const BATCH_SIZE: usize>
{
    /// origin of the ray
    pub origin: S::Vector<DIM>,
    /// unit direction vector of the ray
    pub dir: UnitVector<S, DOF, DIM, BATCH_SIZE>,
}

impl<S: IsSingleScalar + PartialOrd, const DOF: usize, const DIM: usize> Ray<S, DOF, DIM, 1> {
    /// returns point on ray: origin + t * dir for a given t.
    pub fn at(&self, t: S) -> S::Vector<DIM> {
        self.origin.clone() + self.dir.vector().scaled(t)
    }
}

/// 2d ray
pub type Ray2<S, const B: usize> = Ray<S, 1, 2, B>;
/// 3d ray
pub type Ray3<S, const B: usize> = Ray<S, 2, 3, B>;
