#![deny(missing_docs)]
//! # Pyo3 module

/// python wrapper
pub mod pyo3;

use crate::pyo3::lie_groups::{PyIsometry2, PyIsometry3, PyRotation2, PyRotation3};
use numpy::pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn sophus_pyo3(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRotation2>()?;
    m.add_class::<PyIsometry2>()?;
    m.add_class::<PyRotation3>()?;
    m.add_class::<PyIsometry3>()?;

    Ok(())
}
