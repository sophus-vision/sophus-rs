use nalgebra::{
    DMatrix,
    DMatrixView,
    DMatrixViewMut,
    DVector,
    DVectorView,
    DVectorViewMut,
};

use crate::{
    IsCompressableMatrix,
    IsLinearSolver,
    LinearSolverError,
};

/// Dense SPD solver using LDLᵀ.
pub struct DenseLdlt {}

impl IsCompressableMatrix for DMatrix<f64> {
    type Compressed = DMatrix<f64>;

    fn compress(&self) -> Self::Compressed {
        self.clone()
    }
}

impl IsLinearSolver for DenseLdlt {
    type Matrix = DMatrix<f64>;

    fn solve_in_place(
        &self,
        mat_a: &Self::Matrix,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let n = mat_a.nrows();
        if n != mat_a.ncols() || b.len() != n {
            return Err(LinearSolverError::DimensionMismatch);
        }

        // Factor A in place: A = L D Lᵀ (L stored in A's lower, D in `d`)
        let mut d = DVector::<f64>::zeros(n);

        let mut lu_storage = mat_a.clone();

        ldlt_spd_in_place(
            lu_storage.view_mut((0, 0), (n, n)), // <— matrix mutable view
            d.rows_mut(0, n),                    // <— vector mutable view
        )
        .map_err(|_| LinearSolverError::FactorizationFailed)?;

        // Solve A x = b using the factor
        ldlt_solve_in_place(
            lu_storage.view((0, 0), (n, n)), // <— matrix immutable view
            d.rows(0, n),                    // <— vector immutable view
            b.rows_mut(0, n),                // <— vector mutable view
        );

        Ok(())
    }
}

/// LDLᵀ (SPD) factorization in-place on A's lower triangle.
/// On return:
///   - `a_lower` stores unit-lower L (diag set to 1.0; upper triangle is ignored)
///   - `d` is filled with the diagonal D
pub fn ldlt_spd_in_place(
    mut a_lower: DMatrixViewMut<'_, f64>,
    mut d: DVectorViewMut<'_, f64>,
) -> Result<(), &'static str> {
    let n = a_lower.nrows();
    if n != a_lower.ncols() || d.len() != n {
        return Err("dimension mismatch");
    }

    // Relative safeguard against tiny/negative pivots.
    let mut max_abs_pivot: f64 = 0.0_f64;

    for j in 0..n {
        // L(j,k) for k < j
        for k in 0..j {
            let mut s = a_lower[(j, k)];
            for p in 0..k {
                s -= a_lower[(j, p)] * d[p] * a_lower[(k, p)];
            }
            a_lower[(j, k)] = s / d[k];
        }

        // D[j]
        let mut djj = a_lower[(j, j)];
        for p in 0..j {
            let ljp = a_lower[(j, p)];
            djj -= d[p] * ljp * ljp;
        }
        if !djj.is_finite() {
            return Err("LDLt (SPD) failed: non-finite pivot");
        }

        max_abs_pivot = f64::max(max_abs_pivot, djj.abs());
        // tiny relative tolerance; tune if needed
        let tau: f64 = f64::max(max_abs_pivot, 1.0_f64) * 1e-12_f64;

        if djj <= tau {
            return Err("LDLt (SPD) failed: non-positive/too-small pivot");
        }

        d[j] = djj;
        a_lower[(j, j)] = 1.0;
    }
    Ok(())
}

/// Solve A x = b using (L,D). Overwrites x in-place.
pub fn ldlt_solve_in_place(
    a_lower: DMatrixView<'_, f64>,  // L in lower triangle, unit diag
    d: DVectorView<'_, f64>,        // diagonal
    mut x: DVectorViewMut<'_, f64>, // RHS on input, solution on output
) {
    let n = a_lower.nrows();
    debug_assert_eq!(n, a_lower.ncols());
    debug_assert_eq!(d.len(), n);
    debug_assert_eq!(x.len(), n);

    // Forward: L y = b  (unit-lower)
    for i in 0..n {
        let mut yi = x[i];
        for k in 0..i {
            yi -= a_lower[(i, k)] * x[k];
        }
        x[i] = yi;
    }

    // Diagonal: z = D^{-1} y
    for i in 0..n {
        x[i] /= d[i];
    }

    // Backward: Lᵀ x = z  (unit-upper)
    for i in (0..n).rev() {
        let mut xi = x[i];
        for k in (i + 1)..n {
            // Lᵀ(i,k) = L(k,i)
            xi -= a_lower[(k, i)] * x[k];
        }
        x[i] = xi;
    }
}

/// Compute A^{-1} into a provided matrix view (column-major), using n solves.
/// `a_inv` must be n×n. Allocates one temporary vector once.
pub fn ldlt_inverse_into(
    a_lower: DMatrixView<'_, f64>,
    d: DVectorView<'_, f64>,
    mut a_inv: DMatrixViewMut<'_, f64>,
) {
    let n = a_lower.nrows();
    debug_assert_eq!(n, a_lower.ncols());
    debug_assert_eq!(d.len(), n);
    debug_assert!(a_inv.nrows() == n && a_inv.ncols() == n);

    let mut work = DVector::<f64>::zeros(n); // reused
    for j in 0..n {
        // rhs = e_j in `work`
        work.fill(0.0);
        work[j] = 1.0;

        // solve A x = e_j   (result goes to column j of a_inv)
        ldlt_solve_in_place(
            a_lower,
            d,
            work.rows_mut(0, work.len()), // <— mutable vector view
        );
        for i in 0..n {
            a_inv[(i, j)] = work[i];
        }
    }
}

/// Reconstruct A = L D Lᵀ into `out` (useful for debugging/validation).
pub fn ldlt_reconstruct_into(
    a_lower: DMatrixView<'_, f64>,
    d: DVectorView<'_, f64>,
    mut out: DMatrixViewMut<'_, f64>,
) {
    let n = a_lower.nrows();
    debug_assert_eq!(n, a_lower.ncols());
    debug_assert_eq!(d.len(), n);
    debug_assert!(out.nrows() == n && out.ncols() == n);

    out.fill(0.0);

    // out = Σ_p D[p] * col_p(L) * col_p(L)ᵀ
    for p in 0..n {
        let dp = d[p];
        if dp == 0.0 {
            continue;
        }
        for i in p..n {
            let lip = if i == p { 1.0 } else { a_lower[(i, p)] };
            for j in p..n {
                let ljp = if j == p { 1.0 } else { a_lower[(j, p)] };
                out[(i, j)] += dp * lip * ljp;
            }
        }
    }
    // Symmetrize upper (optional)
    for j in 0..n {
        for i in 0..j {
            out[(i, j)] = out[(j, i)];
        }
    }
}

/// Optional: inertia of D (counts of +, -, ~0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseInertia {
    /// pos
    pub pos: usize,
    /// neg
    pub neg: usize,
    /// zero
    pub zero: usize,
}

/// ldlt
pub fn ldlt_inertia_from_d(d: DVectorView<'_, f64>, tol_rel: f64) -> DenseInertia {
    let mut max_abs: f64 = 0.0_f64;
    for i in 0..d.len() {
        max_abs = f64::max(max_abs, d[i].abs());
    }
    let tau: f64 = f64::max(max_abs, 1.0_f64) * tol_rel;

    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut zero = 0usize;
    for i in 0..d.len() {
        let di = d[i];
        if di > tau {
            pos += 1;
        } else if di < -tau {
            neg += 1;
        } else {
            zero += 1;
        }
    }
    DenseInertia { pos, neg, zero }
}
