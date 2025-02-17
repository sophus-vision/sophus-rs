use core::fmt::Debug;

mod var_builder;
mod var_families;
mod var_family;
mod var_tuple;

pub use var_builder::*;
pub use var_families::*;
pub use var_family::*;
pub use var_tuple::*;

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
