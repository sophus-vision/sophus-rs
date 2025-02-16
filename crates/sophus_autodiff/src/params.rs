use alloc::{
    vec,
    vec::Vec,
};
use core::borrow::Borrow;

use crate::{
    linalg::VecF64,
    points::example_points,
    prelude::IsScalar,
};

extern crate alloc;

/// A trait for a struct that have an internal parameter vector representation.
///
/// # Generic parameters
///
///  * S
///    - The underlying scalar such as f64 or [crate::dual::DualScalar].
///  * PARAMS
///     - Number of parameters.
///  * BATCH
///     - Batch dimension. If S is f64 or [crate::dual::DualScalar] then BATCH=1.
///  * DM, DN
///    - DM x DN is the static shape of the Jacobian to be computed if S == DualScalar<DM, DN>. If S
///      == f64, then DM==0, DN==0.
///
/// # Example
///
/// Such types can be created from parameters and have their parameters set and retrieved:
///
/// ```ignore
/// let params = vec![1.0, 2.0, 3.0];
/// let mut instance = MyType::from_params(&params);
/// let new_params = vec![4.0, 5.0, 6.0];
/// instance.set_params(&new_params);
/// ```
pub trait HasParams<
    S: IsScalar<BATCH, DM, DN>,
    const PARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: IsParamsImpl<S, PARAMS, BATCH, DM, DN>
{
    /// Create instance from parameters.
    fn from_params<P>(params: P) -> Self
    where
        P: Borrow<S::Vector<PARAMS>>;
    /// Set parameter values.
    fn set_params<P>(&mut self, params: P)
    where
        P: Borrow<S::Vector<PARAMS>>;
    /// Get parameter values.
    fn params(&self) -> &S::Vector<PARAMS>;
}

/// Trait for parameter vector implementation.
///
/// # Generic parameters
///
///  * S
///    - The underlying scalar such as f64 or [crate::dual::DualScalar].
///  * PARAMS
///     - Number of parameters.
///  * BATCH
///     - Batch dimension. If S is f64 or [crate::dual::DualScalar] then BATCH=1.
///  * DM, DN
///    - DM x DN is the static shape of the Jacobian to be computed if S == DualScalar<DM, DN>. If S
///      == f64, then DM==0, DN==0.
pub trait IsParamsImpl<
    S: IsScalar<BATCH, DM, DN>,
    const PARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>
{
    /// Is the parameter vector valid?
    fn are_params_valid<P>(params: P) -> S::Mask
    where
        P: Borrow<S::Vector<PARAMS>>;
    /// Returns examples of valid parameters.
    ///
    /// This is useful for internal and external unit/regression/integration tests.
    /// In general, the list should not be empty but also not contain more than a few dozen
    /// examples. Ideally, the examples should cover the entire parameter space.
    fn params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>>;
    /// Examples of invalid parameter vectors, might be empty.
    ///
    /// This is useful for internal and external unit/regression/integration tests. If all finite
    /// parameter vectors are valid, then this should return an empty list. If there is a constraint
    /// on the parameter space, then this should return a list of parameter vectors that violate
    /// the constraint.
    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>>;
}

impl<const N: usize> IsParamsImpl<f64, N, 1, 0, 0> for VecF64<N> {
    fn are_params_valid<P>(_params: P) -> bool
    where
        P: Borrow<VecF64<N>>,
    {
        true
    }

    fn params_examples() -> Vec<VecF64<N>> {
        example_points::<f64, N, 1, 0, 0>()
    }

    fn invalid_params_examples() -> Vec<VecF64<N>> {
        vec![]
    }
}

impl<const N: usize> HasParams<f64, N, 1, 0, 0> for VecF64<N> {
    fn from_params<P>(params: P) -> Self
    where
        P: Borrow<VecF64<N>>,
    {
        *params.borrow()
    }

    fn set_params<P>(&mut self, params: P)
    where
        P: Borrow<VecF64<N>>,
    {
        *self = *params.borrow();
    }

    fn params(&self) -> &VecF64<N> {
        self
    }
}
