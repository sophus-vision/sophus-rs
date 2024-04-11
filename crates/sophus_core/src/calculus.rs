#![deny(missing_docs)]
//! # Calculus module

/// dual numbers - for automatic differentiation
pub mod dual;

/// curves, scalar-valued, vector-valued, and matrix-valued maps
pub mod maps;

/// intervals and regions
pub mod region;
pub use crate::calculus::region::IInterval;
pub use crate::calculus::region::IRegion;
pub use crate::calculus::region::Interval;
pub use crate::calculus::region::Region;

/// splines
pub mod spline;
pub use crate::calculus::spline::CubicBSpline;
pub use crate::calculus::spline::CubicBSplineParams;
