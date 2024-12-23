use core::borrow::Borrow;

use crate::linalg::VecF64;
use crate::points::example_points;
use crate::prelude::*;
use alloc::vec;
use alloc::vec::Vec;

extern crate alloc;

/// Parameter implementation.
pub trait ParamsImpl<S: IsScalar<BATCH_SIZE>, const PARAMS: usize, const BATCH_SIZE: usize> {
    /// Is the parameter vector valid?
    fn are_params_valid<P>(params: P) -> S::Mask
    where
        P: Borrow<S::Vector<PARAMS>>;
    /// Examples of valid parameter vectors.
    fn params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>>;
    /// Examples of invalid parameter vectors.
    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>>;
}

/// A trait for linalg that have parameters.
pub trait HasParams<S: IsScalar<BATCH_SIZE>, const PARAMS: usize, const BATCH_SIZE: usize>:
    ParamsImpl<S, PARAMS, BATCH_SIZE>
{
    /// Create from parameters.
    fn from_params<P>(params: P) -> Self
    where
        P: Borrow<S::Vector<PARAMS>>;
    /// Set parameters.
    fn set_params<P>(&mut self, params: P)
    where
        P: Borrow<S::Vector<PARAMS>>;
    /// Get parameters.
    fn params(&self) -> &S::Vector<PARAMS>;
}

impl<const N: usize> ParamsImpl<f64, N, 1> for VecF64<N> {
    fn are_params_valid<P>(_params: P) -> bool
    where
        P: Borrow<VecF64<N>>,
    {
        true
    }

    fn params_examples() -> Vec<VecF64<N>> {
        example_points::<f64, N, 1>()
    }

    fn invalid_params_examples() -> Vec<VecF64<N>> {
        vec![]
    }
}

impl<const N: usize> HasParams<f64, N, 1> for VecF64<N> {
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
