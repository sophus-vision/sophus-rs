use nalgebra::{
    DMatrix,
    DMatrixView,
    DMatrixViewMut,
    DVector,
    DVectorView,
    DVectorViewMut,
};

use crate::{
    IsDenseLinearSystem,
    LinearSolverError,
};

/// In-place LU factorization with partial pivoting: PA = LU.
/// - A's strict lower tri stores L (unit diag not stored),
/// - A's upper tri (incl diag) stores U.
/// On return, `piv[k]` is the pivot row chosen at step k (the row swapped with k).
pub fn lu_in_place(mut a: DMatrixViewMut<'_, f64>, piv: &mut [usize]) -> Result<(), &'static str> {
    let n = a.nrows();
    if n != a.ncols() || piv.len() != n {
        return Err("dimension mismatch");
    }

    // A small relative pivot tolerance.
    let rel_tol: f64 = 1e-12_f64;

    for k in 0..n {
        // 1) Choose pivot row p = argmax_{i>=k} |A[i,k]|
        let mut p = k;
        let mut max_abs = a[(k, k)].abs();
        for i in (k + 1)..n {
            let v = a[(i, k)].abs();
            if v > max_abs {
                max_abs = v;
                p = i;
            }
        }

        // 2) Check near-singularity
        let tau = f64::max(max_abs, 1.0_f64) * rel_tol;
        if max_abs <= tau {
            return Err("LU failed: near-singular pivot");
        }

        // 3) Swap rows if needed, and record pivot
        piv[k] = p;
        if p != k {
            for j in 0..n {
                let tmp = a[(k, j)];
                a[(k, j)] = a[(p, j)];
                a[(p, j)] = tmp;
            }
        }

        // 4) Eliminate below pivot
        let akk = a[(k, k)];
        for i in (k + 1)..n {
            a[(i, k)] /= akk; // L(i,k)
            let lik = a[(i, k)];
            // A[i, k+1..] -= L(i,k) * U(k, k+1..)
            for j in (k + 1)..n {
                a[(i, j)] -= lik * a[(k, j)];
            }
        }
    }

    Ok(())
}

/// Solve A x = b using the compact LU in `a_lu` and the pivot vector `piv`.
/// Overwrites `x` in-place (x := solution).
pub fn lu_solve_in_place(
    a_lu: DMatrixView<'_, f64>,     // LU in compact form (from lu_in_place)
    piv: &[usize],                  // pivot indices per step
    mut x: DVectorViewMut<'_, f64>, // RHS on input; solution on output
) {
    let n = a_lu.nrows();
    debug_assert_eq!(n, a_lu.ncols());
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(piv.len(), n);

    // Apply the same row swaps to the RHS.
    for k in 0..n {
        let p = piv[k];
        if p != k {
            let tmp = x[k];
            x[k] = x[p];
            x[p] = tmp;
        }
    }

    // Forward solve: L y = Pb  (L has unit diag, stored in strict lower triangle)
    for i in 0..n {
        let mut yi = x[i];
        for k in 0..i {
            yi -= a_lu[(i, k)] * x[k];
        }
        x[i] = yi;
    }

    // Backward solve: U x = y  (U is upper triangular incl diag)
    for i in (0..n).rev() {
        let mut xi = x[i];
        for k in (i + 1)..n {
            xi -= a_lu[(i, k)] * x[k];
        }
        x[i] = xi / a_lu[(i, i)];
    }
}

/// Compute A^{-1} into `a_inv` using LU (n solves). `a_inv` must be n×n.
pub fn lu_inverse_into(
    a_lu: DMatrixView<'_, f64>,
    piv: &[usize],
    mut a_inv: DMatrixViewMut<'_, f64>,
) {
    let n = a_lu.nrows();
    debug_assert_eq!(n, a_lu.ncols());
    debug_assert!(a_inv.nrows() == n && a_inv.ncols() == n);

    let mut work = DVector::<f64>::zeros(n);
    for j in 0..n {
        // rhs = e_j
        work.fill(0.0);
        work[j] = 1.0;

        // Solve A x = e_j  -> (writes solution into work)
        lu_solve_in_place(a_lu, piv, work.rows_mut(0, n));
        for i in 0..n {
            a_inv[(i, j)] = work[i];
        }
    }
}

/// Reconstruct A = Pᵀ L U into `out` (for validation/debug).
/// `a_lu` holds compact LU; `piv` are the elimination row-swaps.
pub fn lu_reconstruct_into(
    a_lu: DMatrixView<'_, f64>,
    piv: &[usize],
    mut out: DMatrixViewMut<'_, f64>,
) {
    let n = a_lu.nrows();
    debug_assert_eq!(n, a_lu.ncols());
    debug_assert!(out.nrows() == n && out.ncols() == n);
    debug_assert_eq!(piv.len(), n);

    // temp = L * U
    let mut temp = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for k in 0..=i {
            let lik = if i == k { 1.0 } else { a_lu[(i, k)] };
            for j in k..n {
                temp[(i, j)] += lik * a_lu[(k, j)];
            }
        }
    }

    // Build final row permutation P such that P A = L U.
    // We want A = Pᵀ * (L U). The inverse row-permutation maps:
    // inv_perm[i] = row in temp that becomes row i of A.
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let p = piv[k];
        if p != k {
            perm.swap(k, p);
        }
    }
    let mut inv_perm = vec![0usize; n];
    for (i, &pi) in perm.iter().enumerate() {
        inv_perm[pi] = i;
    }

    // out = Pᵀ * temp (row-permute by inv_perm)
    for i in 0..n {
        let src = inv_perm[i];
        for j in 0..n {
            out[(i, j)] = temp[(src, j)];
        }
    }
}

/// Dense solver using LU.
pub struct DenseLU {}

impl IsDenseLinearSystem for DenseLU {
    fn solve_dense(
        &self,
        mat_a: DMatrix<f64>,
        b: &DVector<f64>,
    ) -> Result<DVector<f64>, LinearSolverError> {
        let n = mat_a.nrows();
        if n != mat_a.ncols() || b.len() != n {
            return Err(LinearSolverError::DimensionMismatch);
        }

        let mut a = mat_a; // take ownership to reuse storage
        let mut piv = vec![0usize; n];
        lu_in_place(a.view_mut((0, 0), (n, n)), &mut piv)
            .map_err(|_| LinearSolverError::FactorizationFailed)?;

        let mut x = b.clone();
        lu_solve_in_place(a.view((0, 0), (n, n)), &piv, x.rows_mut(0, n));
        Ok(x)
    }
}
