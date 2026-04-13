//! Inequality constraint strategies for the unified `Optimizer`.
//!
//! Currently only the `None` variant is available (standard NLLS / equality-constrained).

/// Inequality constraint strategy for the unified optimizer.
pub enum IneqStrategy {
    /// No inequality constraints (standard NLLS / equality-constrained).
    None,
}
