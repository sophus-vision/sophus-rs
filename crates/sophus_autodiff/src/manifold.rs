use crate::{
    linalg::VecF64,
    params::HasParams,
    prelude::*,
};
extern crate alloc;
use core::fmt::Debug;

/// A trait for a manifold.
///
/// A manifold is a space that locally resembles Euclidean space `ℝ^DOF``. This
/// trait generalizes addition and subtraction via [IsManifold::oplus] and [IsManifold::ominus].
///
/// # Generic parameters
///
///  * S
///    - the underlying scalar such as f64 or [crate::dual::DualScalar].
///  * PARAMS
///     - Number of parameters.
///  * DOF
///    - Degrees of freedom of the transformation, aka dimension of the tangent space
///  * BATCH
///     - Batch dimension. If S is f64 or [crate::dual::DualScalar] then BATCH=1.
///  * DM, DN
///    - DM x DN is the static shape of the Jacobian to be computed if S == DualScalar<DM, DN>. If S
///      == f64, then DM==0, DN==0.
///
/// # Examples
/// ```
/// use sophus_autodiff::{
///     linalg::VecF64,
///     manifold::IsManifold,
/// };
///
/// // Trivial manifold example - IsManifold is implemented for VecF64.
///
/// let v1 = VecF64::<3>::new(1.0, 2.0, 3.0);
/// let v2 = VecF64::<3>::new(0.5, 0.5, 0.5);
/// let sum = v1.oplus(&v2); // sum = (1.5, 2.5, 3.5)
/// let diff = sum.ominus(&v1); // diff = v2
/// ```
pub trait IsManifold<
    S: IsScalar<BATCH, DM, DN>,
    const PARAMS: usize,
    const DOF: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: HasParams<S, PARAMS, BATCH, DM, DN> + core::fmt::Debug + Clone
{
    /// The o-plus operator.
    fn oplus(&self, tangent: &S::Vector<DOF>) -> Self;
    /// The o-minus operator.
    fn ominus(&self, rhs: &Self) -> S::Vector<DOF>;
}

impl<const N: usize> IsManifold<f64, N, N, 1, 0, 0> for VecF64<N> {
    fn oplus(&self, tangent: &VecF64<N>) -> Self {
        self + tangent
    }

    fn ominus(&self, rhs: &Self) -> VecF64<N> {
        self - rhs
    }
}
impl<const N: usize> IsVariable for VecF64<N> {
    const NUM_DOF: usize = N;

    fn update(&mut self, delta: nalgebra::DVectorView<f64>) {
        assert_eq!(delta.len(), N);
        for d in 0..N {
            self[d] += delta[d];
        }
    }
}

/// Trait for a tangent vector implementation.
///
/// A tangent vector is a vector that lives in the tangent space of a manifold.
/// It is used to represent the derivative of a curve on the manifold.
///
/// # Generic parameters
///
///  * S
///    - the underlying scalar such as [f64] or [crate::dual::DualScalar].
///  * DOF
///    - Degrees of freedom of the transformation, i.e. the dimension of the tangent space
///  * BATCH
///     - Batch dimension. If S is [f64] or [crate::dual::DualScalar] then BATCH=1.
///  * DM, DN
///    - DM x DN is the static shape of the Jacobian to be computed if S == DualScalar<DM, DN>. If S
///      == f64, then DM==0, DN==0.
pub trait IsTangent<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>
{
    /// Non-examples vector of examples of tangent vectors.
    fn tangent_examples() -> alloc::vec::Vec<S::Vector<DOF>>;
}

/// Trait for a decision variable.
///
/// A decision variable is a variable that is modified by an numerical
/// optimization routine. A decision variable can be simply a [crate::linalg::VecF64], but
/// more generally it is a point on a manifold.
///
/// In contrast to [IsManifold], this trait is not generic over the scalar type and dimensions.
pub trait IsVariable: Clone + Debug + Send + Sync + 'static {
    /// Number of degrees of freedom. This is the dimension of the tangent space.
    const NUM_DOF: usize;

    /// Update the variable in-place.
    fn update(&mut self, delta: nalgebra::DVectorView<f64>);
}
