pub mod calculus;
pub mod image;
pub mod lie;
pub mod manifold;
pub mod opt;
pub mod sensor;
pub mod tensor;
pub mod viewer;

pub use hollywood;

use lie::pyo3::lie_groups::{PyIsometry2, PyIsometry3, PyRotation2, PyRotation3};
use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn sophus(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRotation2>()?;
    m.add_class::<PyIsometry2>()?;
    m.add_class::<PyRotation3>()?;
    m.add_class::<PyIsometry3>()?;

    Ok(())
}
