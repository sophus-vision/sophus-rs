use numpy::PyArray1;
use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::PyOSError;
use pyo3::Bound;
use pyo3::PyErr;
use std::fmt;

/// Error for mismatched array dimensions
#[derive(Debug)]
pub struct PyArray1DimMismatch {
    expected: usize,
    actual: usize,
    file: &'static str,
    line: u32,
}

impl std::error::Error for PyArray1DimMismatch {}

impl fmt::Display for PyArray1DimMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}   Expected array of dimension {}, got {}",
            self.file, self.line, self.expected, self.actual
        )
    }
}

impl std::convert::From<PyArray1DimMismatch> for PyErr {
    fn from(err: PyArray1DimMismatch) -> PyErr {
        PyOSError::new_err(err.to_string())
    }
}

/// Check if array has expected dimension
pub fn check_array1_dim_impl(
    array: &Bound<PyArray1<f64>>,
    expected: usize,
    file: &'static str,
    line: u32,
) -> Result<(), PyArray1DimMismatch> {
    if array.readonly().len() == expected {
        Ok(())
    } else {
        Err(PyArray1DimMismatch {
            expected,
            actual: array.len(),
            file,
            line,
        })
    }
}
