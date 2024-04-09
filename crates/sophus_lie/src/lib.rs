#![feature(portable_simd)]
#![deny(missing_docs)]
//! # Lie groups module

/// Lie groups
pub mod groups;

/// Lie groups
pub mod lie_group;

/// Lie groups
pub mod factor_lie_group;

/// Lie group as a manifold
pub mod lie_group_manifold;

/// Lie group traits
pub mod traits;

/// Real lie group
pub mod real_lie_group;
