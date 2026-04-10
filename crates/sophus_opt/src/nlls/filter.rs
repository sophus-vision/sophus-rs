//! Filter for the filter line-search interior point method (Wächter & Biegler, 2006).
//!
//! The filter maintains a set of (θ, φ) pairs where:
//! - θ = constraint violation measure (infeasibility)
//! - φ = barrier objective value
//!
//! A trial point is acceptable if it is not dominated by any filter entry
//! and provides sufficient decrease in either θ or φ.
//!
//! # Convergence guarantee
//!
//! The filter prevents cycling: once a point (θ_k, φ_k) is added to the filter,
//! future iterates must strictly improve over it. Combined with the restoration
//! phase, this ensures global convergence to a KKT point or Fritz John point
//! of the infeasibility measure (Theorem 3.1, Wächter & Biegler 2006).

extern crate alloc;

/// A filter entry: (θ, φ) pair representing infeasibility and barrier objective.
#[derive(Clone, Debug)]
struct FilterEntry {
    /// Constraint violation measure: `θ = max(0, max_j(-h_j))`.
    theta: f64,
    /// Barrier objective: `φ = f(x) - μ Σ ln(h_j)` (or elastic merit during restoration).
    phi: f64,
}

/// Filter line-search acceptance filter (Wächter & Biegler, 2006).
///
/// Maintains a set of (θ, φ) pairs. A trial point is acceptable to the filter if:
/// 1. It is not dominated by any existing entry (θ_trial < θ_k OR φ_trial < φ_k for all k)
/// 2. It provides sufficient decrease in either θ or φ relative to the current iterate
///
/// The filter is augmented (new entry added) when accepting a "θ-type" step that
/// improves feasibility but not necessarily the objective. This prevents the algorithm
/// from cycling back to previously visited infeasible regions.
#[derive(Clone, Debug)]
pub struct Filter {
    entries: alloc::vec::Vec<FilterEntry>,
    /// Sufficient decrease factor for θ (infeasibility). Typical: 1e-5.
    gamma_theta: f64,
    /// Sufficient decrease factor for φ (objective). Typical: 1e-5.
    gamma_phi: f64,
    /// Switching condition exponent: use φ-type step when θ < θ_min^{s_theta}.
    /// Typical: s_theta = 1.1 (slightly superlinear).
    s_theta: f64,
    /// Switching condition factor: use φ-type step when the predicted φ decrease
    /// exceeds delta * θ^{s_phi}. Typical: delta = 1.0, s_phi = 2.3.
    s_phi: f64,
    delta: f64,
    /// Upper bound on θ for the filter. Entries with θ > theta_max are not added.
    theta_max: f64,
}

impl Filter {
    /// Create a new empty filter with IPOPT default parameters.
    pub fn new(theta_max: f64) -> Self {
        Self {
            entries: alloc::vec![],
            gamma_theta: 1e-5,
            gamma_phi: 1e-5,
            s_theta: 1.1,
            s_phi: 2.3,
            delta: 1.0,
            theta_max: theta_max.max(1e-4),
        }
    }

    /// Check whether a trial point (θ_trial, φ_trial) is acceptable to the filter.
    ///
    /// A point is acceptable if it is NOT dominated by any existing filter entry.
    /// Dominated means: θ_trial >= θ_k AND φ_trial >= φ_k for some entry k.
    pub fn is_acceptable(&self, theta_trial: f64, phi_trial: f64) -> bool {
        for entry in &self.entries {
            if theta_trial >= entry.theta && phi_trial >= entry.phi {
                return false;
            }
        }
        true
    }

    /// Determine the step type and whether the trial point should be accepted.
    ///
    /// Returns `Some(true)` for a φ-type step (sufficient φ decrease),
    /// `Some(false)` for a θ-type step (sufficient θ decrease),
    /// or `None` if the step should be rejected.
    ///
    /// # Arguments
    /// - `theta_current`: infeasibility at the current iterate
    /// - `phi_current`: barrier objective at the current iterate
    /// - `theta_trial`: infeasibility at the trial point
    /// - `phi_trial`: barrier objective at the trial point
    /// - `predicted_phi_decrease`: model-predicted decrease in φ (from linearized model)
    pub fn check_step(
        &self,
        theta_current: f64,
        phi_current: f64,
        theta_trial: f64,
        phi_trial: f64,
        predicted_phi_decrease: f64,
    ) -> Option<bool> {
        // First check: trial must be acceptable to the filter.
        if !self.is_acceptable(theta_trial, phi_trial) {
            return None;
        }

        // Switching condition: use φ-type step when current iterate is nearly feasible
        // and the model predicts good objective decrease.
        let use_phi_type = theta_current < self.theta_max
            && predicted_phi_decrease > self.delta * theta_current.powf(self.s_phi);

        if use_phi_type {
            // φ-type (Armijo-like): require sufficient decrease in φ.
            // φ_trial ≤ φ_current - γ_φ · θ_current
            if phi_trial <= phi_current - self.gamma_phi * theta_current {
                return Some(true); // φ-type accepted
            }
        } else {
            // θ-type: require sufficient decrease in either θ or φ.
            let theta_ok = theta_trial <= (1.0 - self.gamma_theta) * theta_current;
            let phi_ok = phi_trial <= phi_current - self.gamma_phi * theta_current;
            if theta_ok || phi_ok {
                return Some(false); // θ-type accepted
            }
        }

        None // rejected
    }

    /// Augment the filter with the current iterate (θ, φ).
    ///
    /// Called when accepting a θ-type step. Adds the *current* (not trial) point
    /// to the filter, strengthened by γ_θ and γ_φ margins. Also removes any
    /// existing entries that are dominated by the new one.
    pub fn augment(&mut self, theta: f64, phi: f64) {
        let new_entry = FilterEntry {
            theta: (1.0 - self.gamma_theta) * theta,
            phi: phi - self.gamma_phi * theta,
        };

        // Remove dominated entries.
        self.entries
            .retain(|e| e.theta < new_entry.theta || e.phi < new_entry.phi);

        self.entries.push(new_entry);
    }

    /// Reset the filter (e.g., when μ changes).
    pub fn reset(&mut self) {
        self.entries.clear();
    }

    /// Number of entries in the filter.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Update theta_max (called when μ decreases).
    pub fn set_theta_max(&mut self, theta_max: f64) {
        self.theta_max = theta_max.max(1e-4);
    }
}
