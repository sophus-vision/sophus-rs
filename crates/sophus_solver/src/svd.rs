use nalgebra::{
    DMatrix,
    DMatrixView,
};

/// Calculate the (Moore-Penrose) pseudo-inverse of a dense matrix - given a relative tolerance.
pub fn pseudo_inverse_with_tol(mat_a: DMatrixView<f64>, rel_tol: f64) -> DMatrix<f64> {
    let (m, n) = (mat_a.nrows(), mat_a.ncols());

    let mat_a: DMatrix<f64> = mat_a.to_owned().into();
    let svd = nalgebra::SVD::new(mat_a, true, true);
    let (mat_u, singular_vector, mat_v_t) = (svd.u.unwrap(), svd.singular_values, svd.v_t.unwrap());
    let smax = singular_vector
        .iter()
        .fold(0.0, |mx: f64, &v| mx.max(v.abs()))
        .max(1.0);
    let tol = smax * rel_tol;
    let mut singular_vector_pseudoinverse = DMatrix::<f64>::zeros(n, m);
    for i in 0..singular_vector.len() {
        if singular_vector[i].abs() > tol {
            singular_vector_pseudoinverse[(i, i)] = 1.0 / singular_vector[i];
        }
    }
    mat_v_t.transpose() * singular_vector_pseudoinverse * mat_u.transpose()
}

/// Calculate the (Moore-Penrose) pseudo-inverse of a dense matrix.
///
/// A default relative tolerance of `1e-12` is used.
pub fn pseudo_inverse(a: DMatrixView<f64>) -> DMatrix<f64> {
    pseudo_inverse_with_tol(a, 1e-12)
}
