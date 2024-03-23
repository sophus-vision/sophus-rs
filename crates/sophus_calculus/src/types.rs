use simba::simd::AutoSimd;

/// matrices
pub mod matrix;
/// parameters
pub mod params;
/// scalars
pub mod scalar;
/// vectors
pub mod vector;

/// f64 vector
pub type VecF64<const ROWS: usize> = nalgebra::SMatrix<f64, ROWS, 1>;
/// f64 matrix
pub type MatF64<const ROWS: usize, const COLS: usize> = nalgebra::SMatrix<f64, ROWS, COLS>;

/// batch of f64 scalars
pub type BatchF64<const B: usize> = nalgebra::SMatrix<AutoSimd<[f64; B]>, 1, 1>;
/// batch of f64 vectors
pub type BatchVecF64<const B: usize, const ROWS: usize> =
    nalgebra::SMatrix<AutoSimd<[f64; B]>, ROWS, 1>;
/// batch of f64 matrices
pub type BatchMatF64<const B: usize, const ROWS: usize, const COLS: usize> =
    nalgebra::SMatrix<AutoSimd<[f64; B]>, ROWS, COLS>;
