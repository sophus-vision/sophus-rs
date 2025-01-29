use core::fmt::Debug;

/// Builder for variable families
pub mod var_builder;
/// Set of variable families
pub mod var_families;
/// Variable family
pub mod var_family;
/// A tuple of variables
pub mod var_tuple;

extern crate alloc;

/// Variable kind
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum VarKind {
    /// free variable (will be updated during optimization)
    Free,
    /// conditioned variable (will not be fixed during optimization)
    Conditioned,
    /// marginalized variable (will be updated during using the Schur complement trick)
    Marginalized,
}
