use simba::simd::AutoSimd;

pub type V<const ROWS: usize> = nalgebra::SMatrix<f64, ROWS, 1>;
pub type M<const ROWS: usize, const COLS: usize> = nalgebra::SMatrix<f64, ROWS, COLS>;

pub type BatchS<const B: usize> = nalgebra::SMatrix<AutoSimd<[f64; B]>, 1, 1>;
pub type BatchV<const B: usize, const ROWS: usize> = nalgebra::SMatrix<AutoSimd<[f64; B]>, ROWS, 1>;
pub type BatchM<const B: usize, const ROWS: usize, const COLS: usize> =
    nalgebra::SMatrix<AutoSimd<[f64; B]>, ROWS, COLS>;

pub mod matrix;
pub mod scalar;
pub mod vector;
