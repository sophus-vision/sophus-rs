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

/// `Y := L^{-1} * Y`, where `L` is lower-triangular with unit diagonal.
#[inline]
pub fn lower_matsolve_inplace(l: &DMatrixView<'_, f64>, y: &mut DMatrixViewMut<'_, f64>) {
    let n = l.nrows();
    debug_assert_eq!(l.ncols(), n);
    debug_assert_eq!(y.nrows(), n);

    let (_rows, nrhs) = y.shape();
    for c in 0..nrhs {
        for i in 0..n {
            let mut s = y[(i, c)];
            for k in 0..i {
                s -= l[(i, k)] * y[(k, c)];
            }
            // unit diag: no division
            debug_assert!((l[(i, i)] - 1.0).abs() <= 1e-14);
            y[(i, c)] = s;
        }
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

/// `Y := (Lᵀ)^{-1} * Y`, where `L` is lower-triangular with unit diagonal.
#[inline]
pub fn lower_transpose_matsolve_inplace(l: &DMatrixView<'_, f64>, y: &mut DMatrixViewMut<'_, f64>) {
    let n = l.nrows();
    debug_assert_eq!(l.ncols(), n);
    debug_assert_eq!(y.nrows(), n);

    let (_rows, nrhs) = y.shape();
    for c in 0..nrhs {
        for i in (0..n).rev() {
            let mut s = y[(i, c)];
            for k in (i + 1)..n {
                s -= l[(k, i)] * y[(k, c)];
            }
            // unit diag: no division
            debug_assert!((l[(i, i)] - 1.0).abs() <= 1e-14);
            y[(i, c)] = s;
        }
    }
}

/// `y := diag(d)^{-1} y`
///
/// Semi-definite aware: If `d[i] ~ 0`, then `y[i] = 0`.
#[inline]
pub fn diag_solve_inplaced(d: DVectorView<'_, f64>, y: &mut DVectorViewMut<'_, f64>) {
    for i in 0..y.len() {
        let di = d[i];
        y[i] = if di == 0.0 { 0.0 } else { y[i] / di };
    }
}

/// `Y := diag(d)^{-1} * Y`
///
/// Semi-definite aware: If `d[i] ~ 0`, then the i-th row of `Y` becomes 0.
#[inline]
pub fn diag_matsolve_inplaced(d: DVectorView<'_, f64>, y: &mut DMatrixViewMut<'_, f64>) {
    let n = y.nrows();
    debug_assert_eq!(d.len(), n);

    let (_rows, nrhs) = y.shape();
    for i in 0..n {
        let inv = if d[i] == 0.0 { 0.0 } else { 1.0 / d[i] };
        for c in 0..nrhs {
            y[(i, c)] *= inv;
        }
    }
}

/// `X := Y * L^{-1}`, `L` lower-triangular with unit diagonal (column-oriented).
#[inline]
pub fn lower_right_solve_inplace(l: DMatrixView<'_, f64>, x: &mut DMatrixViewMut<'_, f64>) {
    let (m, n) = x.shape();
    debug_assert_eq!(l.nrows(), n);
    debug_assert_eq!(l.ncols(), n);

    // Column-oriented with unchecked access.
    for j in (0..n).rev() {
        for t in (j + 1)..n {
            // SAFETY: indices are within bounds (checked by loop/debug_assert).
            let l_tj = unsafe { *l.get_unchecked((t, j)) };
            for r in 0..m {
                // SAFETY: indices are within bounds (guarded by loop bounds and debug asserts).
                unsafe {
                    *x.get_unchecked_mut((r, j)) -= *x.get_unchecked((r, t)) * l_tj;
                }
            }
        }
    }
}

/// `X := Y * (Lᵀ)^{-1}`, `L` is lower-triangular with unit diagonal (column-oriented).
#[inline]
pub fn lower_transpose_rightsolve_inplace(
    l: DMatrixView<'_, f64>,
    x: &mut DMatrixViewMut<'_, f64>,
) {
    let (m, n) = x.shape();
    debug_assert_eq!(l.nrows(), n);
    debug_assert_eq!(l.ncols(), n);

    // Column-oriented with unchecked access.
    for j in 0..n {
        for t in 0..j {
            // SAFETY: indices are within bounds (checked by loop/debug_assert).
            let l_jt = unsafe { *l.get_unchecked((j, t)) };
            for r in 0..m {
                // SAFETY: indices are within bounds (guarded by loop bounds and debug asserts).
                unsafe {
                    *x.get_unchecked_mut((r, j)) -= *x.get_unchecked((r, t)) * l_jt;
                }
            }
        }
    }
}

/// `Y := Y * diag(d)^{-1}`
///
/// Semi-definite aware: If `d[j] ~ 0`, then `Y[:,j] = 0`.
#[inline]
pub fn diag_right_matsolve_inplace(d: DVectorView<'_, f64>, mat_y: &mut DMatrixViewMut<'_, f64>) {
    let (m, n) = mat_y.shape();
    debug_assert_eq!(d.len(), n);

    for j in 0..n {
        let inv = if d[j] == 0.0 { 0.0 } else { 1.0 / d[j] };
        for r in 0..m {
            mat_y[(r, j)] *= inv;
        }
    }
}

/// `y -= A * diag(d) * Bᵀ`
#[inline]
pub fn sub_mat_diag_mat_t_inplace(
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

    // Column-oriented with unchecked access.
    for t in 0..k {
        let d_t = d[t];
        let col_a = mat_a.column(t);
        let col_b = mat_b.column(t);
        for j in 0..n {
            // SAFETY: indices are within bounds (checked by loop/debug_assert).
            let bj = unsafe { *col_b.get_unchecked(j) } * d_t;
            for i in 0..m {
                // SAFETY: indices are within bounds (guarded by loop bounds and debug asserts).
                unsafe {
                    *y.get_unchecked_mut((i, j)) -= *col_a.get_unchecked(i) * bj;
                }
            }
        }
    }
}

/// `y += A * diag(d) * Bᵀ``
pub fn add_mat_diag_mat_t_inplace(
    y: &mut DMatrixViewMut<'_, f64>,
    mat_a: DMatrixView<'_, f64>,
    d: DVectorView<'_, f64>,
    mat_b: DMatrixView<'_, f64>,
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
                y[(i, j)] += vec_a_t[i] * bj;
            }
        }
    }
}

/// `y -= L * b`
#[inline]
pub fn sub_mat_plus_vec_inplace(
    y: &mut [f64],               // m
    mat_a: DMatrixView<'_, f64>, // m×n
    b: &[f64],                   // n
) {
    debug_assert_eq!(y.len(), mat_a.nrows());
    debug_assert_eq!(mat_a.ncols(), b.len());

    // Column-oriented: y -= sum_c A[:,c] * b[c] (cache-friendly for col-major A).
    let m = mat_a.nrows();
    for c in 0..mat_a.ncols() {
        let bc = b[c];
        let col = mat_a.column(c);
        for r in 0..m {
            // SAFETY: indices are within bounds (checked by loop/debug_assert).
            y[r] -= unsafe { *col.get_unchecked(r) } * bc;
        }
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

    let m = mat_a.nrows();
    for c in 0..mat_a.ncols() {
        let col = mat_a.column(c);
        let mut acc = 0.0;
        for r in 0..m {
            // SAFETY: indices are within bounds (checked by loop/debug_assert).
            acc += unsafe { *col.get_unchecked(r) } * z[r];
        }
        x[c] -= acc;
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{
        DMatrix,
        DVector,
    };

    /// Test all unsafe kernels on a known 4×4 system to verify correctness.
    /// A = L * D * Lᵀ where L is unit lower triangular.
    #[test]
    fn kernel_correctness_4x4() {
        // L (unit lower triangular)
        let l = DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.2, 0.3, 1.0, 0.0, 0.1, 0.4, 0.6, 1.0,
            ],
        );

        // Test sub_mat_diag_mat_t_inplace (Y -= A * diag(d) * Bᵀ)
        let mat_a_small = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let d_small = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
        let mat_b_small = DMatrix::from_row_slice(2, 3, &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
        let mut y = DMatrix::from_row_slice(2, 2, &[100.0, 200.0, 300.0, 400.0]);
        let y_orig = y.clone();

        super::sub_mat_diag_mat_t_inplace(
            &mut y.as_view_mut(),
            mat_a_small.as_view(),
            d_small.as_view(),
            mat_b_small.as_view(),
        );

        // Verify: y = y_orig - A * diag(d) * Bᵀ
        let expected =
            &y_orig - &mat_a_small * &DMatrix::from_diagonal(&d_small) * mat_b_small.transpose();
        approx::assert_abs_diff_eq!(y, expected, epsilon = 1e-10);

        // Test lower_right_solve_inplace: X * L = B → X = B * L⁻¹
        let mut x_mat = DMatrix::from_row_slice(
            3,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let x_orig = x_mat.clone();
        super::lower_right_solve_inplace(l.as_view(), &mut x_mat.as_view_mut());
        // Verify: x_mat * L ≈ x_orig
        let reconstructed = &x_mat * &l;
        approx::assert_abs_diff_eq!(reconstructed, x_orig, epsilon = 1e-10);

        // Test lower_transpose_rightsolve_inplace: X * Lᵀ = B → X = B * L⁻ᵀ
        let mut x_mat2 = x_orig.clone();
        super::lower_transpose_rightsolve_inplace(l.as_view(), &mut x_mat2.as_view_mut());
        let reconstructed2 = &x_mat2 * l.transpose();
        approx::assert_abs_diff_eq!(reconstructed2, x_orig, epsilon = 1e-10);

        // Test sub_mat_plus_vec_inplace: y -= A * b
        let mut y_vec = vec![10.0, 20.0];
        let mat_mv = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b_mv = vec![1.0, 1.0, 1.0];
        super::sub_mat_plus_vec_inplace(&mut y_vec, mat_mv.as_view(), &b_mv);
        approx::assert_abs_diff_eq!(y_vec[0], 10.0 - 6.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(y_vec[1], 20.0 - 15.0, epsilon = 1e-10);

        // Test sub_mat_transpose_plus_vec_in_place: x -= Aᵀ * z
        let mut x_vec = vec![10.0, 20.0, 30.0];
        let z_vec = vec![1.0, 1.0];
        super::sub_mat_transpose_plus_vec_in_place(&mut x_vec, mat_mv.as_view(), &z_vec);
        approx::assert_abs_diff_eq!(x_vec[0], 10.0 - 5.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(x_vec[1], 20.0 - 7.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(x_vec[2], 30.0 - 9.0, epsilon = 1e-10);
    }

    /// Test sub_mat_diag_mat_t_inplace with large blocks.
    #[test]
    fn kernel_large_block() {
        let n = 10;
        let k = 8;
        let mat_a = DMatrix::from_fn(n, k, |r, c| (r * k + c) as f64 * 0.1);
        let d = DVector::from_fn(k, |i, _| (i + 1) as f64);
        let mat_b = DMatrix::from_fn(n, k, |r, c| ((r + 1) * (c + 1)) as f64 * 0.05);
        let mut y = DMatrix::from_fn(n, n, |r, c| (r + c) as f64);
        let y_orig = y.clone();

        super::sub_mat_diag_mat_t_inplace(
            &mut y.as_view_mut(),
            mat_a.as_view(),
            d.as_view(),
            mat_b.as_view(),
        );

        let expected = &y_orig - &mat_a * &DMatrix::from_diagonal(&d) * mat_b.transpose();
        approx::assert_abs_diff_eq!(y, expected, epsilon = 1e-8);
    }

    /// Edge case: 1×1 matrices.
    #[test]
    fn kernel_1x1() {
        let l = DMatrix::from_row_slice(1, 1, &[1.0]);
        let mut x = DMatrix::from_row_slice(1, 1, &[5.0]);
        super::lower_right_solve_inplace(l.as_view(), &mut x.as_view_mut());
        approx::assert_abs_diff_eq!(x[(0, 0)], 5.0, epsilon = 1e-10);

        let d = DVector::from_row_slice(&[3.0]);
        let a = DMatrix::from_row_slice(1, 1, &[2.0]);
        let b = DMatrix::from_row_slice(1, 1, &[4.0]);
        let mut y = DMatrix::from_row_slice(1, 1, &[100.0]);
        super::sub_mat_diag_mat_t_inplace(
            &mut y.as_view_mut(),
            a.as_view(),
            d.as_view(),
            b.as_view(),
        );
        approx::assert_abs_diff_eq!(y[(0, 0)], 100.0 - 2.0 * 3.0 * 4.0, epsilon = 1e-10);
    }
}
