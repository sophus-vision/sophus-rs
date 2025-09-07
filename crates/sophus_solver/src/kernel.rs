use nalgebra::{
    DMatrixView,
    DMatrixViewMut,
    DVectorView,
    DVectorViewMut,
};

/// `y := (L)^{-1} y`, where `L` is lower-triangular with unit diagonal.
#[inline]
pub fn lower_solve_inplace(l: &DMatrixView<'_, f64>, y: &mut DVectorViewMut<'_, f64>) {
    let n = l.nrows();
    for i in 0..n {
        let mut s = y[i];
        for k in 0..i {
            s -= l[(i, k)] * y[k];
        }
        // unit diag: no division
        debug_assert!((l[(i, i)] - 1.0).abs() <= 1e-14);
        y[i] = s;
    }
}

/// `y := (Lᵀ)^{-1} y`, where `L` is lower-triangular with unit diagonal.
#[inline]
pub fn lower_transpose_solve_inplace(l: &DMatrixView<'_, f64>, y: &mut DVectorViewMut<'_, f64>) {
    let n = l.nrows();
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[(k, i)] * y[k];
        }
        // unit diag: no division
        debug_assert!((l[(i, i)] - 1.0).abs() <= 1e-14);
        y[i] = s;
    }
}

/// `y := diag(d)^{-1} y`
///
/// Semi-definite aware: If `d[i] ~ 0`, then `y[i] = 0`.
#[inline]
pub fn diag_solve_inplaced(d: DVectorView<'_, f64>, y: &mut DVectorViewMut<'_, f64>, rel_tol: f64) {
    let max_abs = d.iter().fold(0.0f64, |mx, &v| mx.max(v.abs())).max(1.0);
    let dzero = max_abs * rel_tol;
    for i in 0..y.len() {
        let di = d[i];
        y[i] = if di.abs() <= dzero { 0.0 } else { y[i] / di };
    }
}

/// `X := Y * L^{-1}`, `L` lower-triangular (row-wise, in place).
#[inline]
pub fn lower_right_solve_inplace(l: DMatrixView<'_, f64>, x: &mut DMatrixViewMut<'_, f64>) {
    let (m, n) = x.shape();
    debug_assert_eq!(l.nrows(), n);
    debug_assert_eq!(l.ncols(), n);

    for r in 0..m {
        for j in (0..n).rev() {
            let mut s = x[(r, j)];
            for t in (j + 1)..n {
                s -= x[(r, t)] * l[(t, j)];
            }
            // unit diag: no division
            debug_assert!((l[(j, j)] - 1.0).abs() <= 1e-14);
            x[(r, j)] = s;
        }
    }
}

/// `X := Y * (Lᵀ)^{-1}`, `L` is lower-triangular with unit diagonal.
#[inline]
pub fn lower_transpose_rightsolve_inplace(
    l: DMatrixView<'_, f64>,
    x: &mut DMatrixViewMut<'_, f64>,
) {
    let (m, n) = x.shape();
    debug_assert_eq!(l.nrows(), n);
    debug_assert_eq!(l.ncols(), n);

    for r in 0..m {
        for j in 0..n {
            let mut s = x[(r, j)];
            for t in 0..j {
                s -= x[(r, t)] * l[(j, t)];
            }
            // unit diag: no division
            debug_assert!((l[(j, j)] - 1.0).abs() <= 1e-14);
            x[(r, j)] = s;
        }
    }
}

/// `Y := Y * diag(d)^{-1}`
///
/// Semi-definite aware: If `d[j] ~ 0`, then `Y[:,j] = 0`.
#[inline]
pub fn diag_rightsolve_inplace(
    d: DVectorView<'_, f64>,
    mat_y: &mut DMatrixViewMut<'_, f64>,
    rel_tol: f64,
) {
    let (m, n) = mat_y.shape();
    debug_assert_eq!(d.len(), n);

    let max_abs = d.iter().fold(0.0f64, |mx, &v| mx.max(v.abs())).max(1.0);
    let dzero = max_abs * rel_tol;
    for j in 0..n {
        let inv = if d[j].abs() <= dzero { 0.0 } else { 1.0 / d[j] };
        for r in 0..m {
            mat_y[(r, j)] *= inv;
        }
    }
}

/// `y -= A * diag(d) * Bᵀ`
#[inline]
pub fn sub_mat_diag_mat_t_in_place(
    y: &mut DMatrixViewMut<'_, f64>, // m×n
    mat_a: DMatrixView<'_, f64>,     // m×k
    d: DVectorView<'_, f64>,         // k
    mat_b: DMatrixView<'_, f64>,     // n×k
) {
    debug_assert_eq!(y.nrows(), mat_a.nrows());
    debug_assert_eq!(y.ncols(), mat_b.nrows());
    debug_assert_eq!(mat_a.ncols(), mat_b.ncols());
    debug_assert_eq!(mat_a.ncols(), d.len());

    let m = y.nrows();
    let n = y.ncols();
    let k = mat_a.ncols();

    for t in 0..k {
        let d_t = d[t];
        let vec_a_t = mat_a.column(t); // m
        let vec_b_t = mat_b.column(t); // n
        for j in 0..n {
            let bj = vec_b_t[j] * d_t;
            for i in 0..m {
                y[(i, j)] -= vec_a_t[i] * bj;
            }
        }
    }
}

/// `y -= L * b`
#[inline]
pub fn sub_mat_plus_vec_in_place(
    y: &mut [f64],               // m
    mat_a: DMatrixView<'_, f64>, // m×n
    b: &[f64],                   // n
) {
    debug_assert_eq!(y.len(), mat_a.nrows());
    debug_assert_eq!(mat_a.ncols(), b.len());

    for r in 0..mat_a.nrows() {
        let mut acc = 0.0;
        for c in 0..mat_a.ncols() {
            acc += mat_a[(r, c)] * b[c];
        }
        y[r] -= acc;
    }
}

/// `x -= Lᵀ * z`
pub fn sub_mat_transpose_plus_vec_in_place(
    x: &mut [f64],               // n
    mat_a: DMatrixView<'_, f64>, // m×n
    z: &[f64],                   // m
) {
    debug_assert_eq!(x.len(), mat_a.ncols());
    debug_assert_eq!(mat_a.nrows(), z.len());

    for c in 0..mat_a.ncols() {
        let mut acc = 0.0;
        for r in 0..mat_a.nrows() {
            acc += mat_a[(r, c)] * z[r];
        }
        x[c] -= acc;
    }
}
