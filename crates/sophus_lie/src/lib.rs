#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![no_std]
//! Lie groups crate - part of the sophus-rs project

/// Lie groups
pub mod groups;
pub use crate::groups::isometry2::Isometry2;
pub use crate::groups::isometry2::Isometry2F64;
pub use crate::groups::isometry3::Isometry3;
pub use crate::groups::isometry3::Isometry3F64;
pub use crate::groups::rotation2::Rotation2;
pub use crate::groups::rotation2::Rotation2F64;
pub use crate::groups::rotation3::Rotation3;
pub use crate::groups::rotation3::Rotation3F64;

/// Lie groups
pub mod lie_group;
pub use crate::lie_group::LieGroup;

/// Lie groups
pub mod factor_lie_group;

/// Lie group traits
pub mod traits;

/// sophus_lie prelude
pub mod prelude {
    pub use crate::traits::IsLieGroup;
    pub use crate::traits::IsTranslationProductGroup;
    pub use sophus_core::prelude::*;
}
