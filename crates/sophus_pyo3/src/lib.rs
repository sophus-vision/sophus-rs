#![deny(missing_docs)]
//! pyo3 bindings for sophus-rs

/// python wrapper
#[cfg(feature = "pyo3")]
pub mod pyo3;

#[cfg(feature = "pyo3")]
use crate::pyo3::lie_groups::PyIsometry2;
#[cfg(feature = "pyo3")]
use crate::pyo3::lie_groups::PyIsometry3;
#[cfg(feature = "pyo3")]
use crate::pyo3::lie_groups::PyRotation2;
#[cfg(feature = "pyo3")]
use crate::pyo3::lie_groups::PyRotation3;
#[cfg(feature = "pyo3")]
use numpy::pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
#[cfg(feature = "pyo3")]
fn sophus_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRotation2>()?;
    m.add_class::<PyIsometry2>()?;
    m.add_class::<PyRotation3>()?;
    m.add_class::<PyIsometry3>()?;

    Ok(())
}
