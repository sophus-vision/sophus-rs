use nalgebra::{
    DMatrix,
    DMatrixView,
    DMatrixViewMut,
    DVector,
    DVectorView,
    DVectorViewMut,
};

use crate::{
    LinearSolverError,
    SymmetricMatrixBuilderEnum,
    dense::DenseSymmetricMatrixBuilder,
    prelude::*,
};

/// Dense solver using LDLᵀ decomposition.
#[derive(Copy, Clone, Debug)]
pub struct DenseLdlt {}

impl IsLinearSolver for DenseLdlt {
    type Matrix = DMatrix<f64>;

    const NAME: &'static str = "dense LDLᵀ";

    fn matrix_builder(&self, partitions: &[crate::PartitionSpec]) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::Dense(DenseSymmetricMatrixBuilder::zero(partitions))
    }

    fn solve_in_place(
        &self,
        _parallelize: bool,
        mat_a: &Self::Matrix,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let n = mat_a.nrows();
        if n != mat_a.ncols() || b.len() != n {
            return Err(LinearSolverError::DimensionMismatch);
        }

        // Factor A in place: A = L D Lᵀ (L stored in A's lower; D is diagonal.)
        let mut diag = DVector::<f64>::zeros(n);
        let mut mat_l = mat_a.clone();

        const EPS: f64 = 1e-12;

        ldlt_decompose_in_place(mat_l.as_view_mut(), diag.as_view_mut(), EPS)
            .map_err(|_| LinearSolverError::FactorizationFailed)?;

        // Solve A x = b.
        ldlt_solve_in_place(mat_l.as_view(), diag.as_view(), b.as_view_mut(), EPS);

        Ok(())
    }
}

/// LDLᵀ factorization in-place on A's lower triangle.
///
/// On return:
///   - `mat_a_lower` stores unit-lower L (diag set to 1.0; upper triangle ignored)
///   - `mat_d` is filled with the diagonal D (entries may be zero for PSD)
///
/// We assume that A is positive semi-definite. Hence some pivots may be ≈ 0 and are set to exactly
/// 0.
///
/// Returns matrix rank on success.
pub fn ldlt_decompose_in_place(
    mut mat_a_lower: DMatrixViewMut<'_, f64>,
    mut mat_d: DVectorViewMut<'_, f64>,
    rel_tol: f64,
) -> Result<u64, &'static str> {
    let n = mat_a_lower.nrows();
    if n != mat_a_lower.ncols() || mat_d.len() != n {
        return Err("dimension mismatch");
    }

    // Tracks the scale of seen (non-tiny) pivots to set a *relative* zero threshold.
    // Start at 1.0 so the threshold is at least absolute EPS.
    let mut max_abs_pivot: f64 = 1.0;

    let mut rank = 0;

    for j in 0..n {
        for k in 0..j {
            // Left-looking update: s = A(j,k) - sum_{p<k} L(j,p) * D[p] * L(k,p)
            let mut s = mat_a_lower[(j, k)];
            for p in 0..k {
                s -= mat_a_lower[(j, p)] * mat_d[p] * mat_a_lower[(k, p)];
            }

            // If D[k] is tiny/zero (semi-definite pivot), we must not divide by it.
            // In that case, that column contributes nothing, so set L[j,k] = 0.
            let scale = f64::max(max_abs_pivot, 1.0);
            let zero_tol = scale * rel_tol;

            if mat_d[k].abs() <= zero_tol {
                mat_a_lower[(j, k)] = 0.0;
            } else {
                mat_a_lower[(j, k)] = s / mat_d[k];
            }
        }

        // D[j] = A(j,j) - sum_{p<j} D[p] * L(j,p)^2
        let mut djj = mat_a_lower[(j, j)];
        for p in 0..j {
            let ljp = mat_a_lower[(j, p)];
            djj -= mat_d[p] * ljp * ljp;
        }

        if !djj.is_finite() {
            return Err("LDLᵀ (PSD) failed: non-finite pivot");
        }

        // Relative zero threshold w.r.t. largest seen pivot (or 1.0).
        let scale = f64::max(max_abs_pivot, 1.0);
        let zero_tol = scale * rel_tol;

        if djj.abs() <= zero_tol {
            // Treat as exactly zero (semi-definite pivot).
            mat_d[j] = 0.0;
        } else if djj < 0.0 {
            return Err("LDLᵀ failed: negative pivot (matrix appears indefinite)");
        } else {
            // Strictly positive pivot.
            mat_d[j] = djj;
            max_abs_pivot = max_abs_pivot.max(djj.abs());
            rank += 1;
        }

        // Unit diagonal for L
        mat_a_lower[(j, j)] = 1.0;
    }
    Ok(rank)
}

/// Solve A x = b using A = L D Lᵀ decomposition.
pub fn ldlt_solve_in_place(
    mat_l: DMatrixView<'_, f64>,
    mat_d: DVectorView<'_, f64>,
    mut x: DVectorViewMut<'_, f64>,
    rel_tol: f64,
) {
    let n = mat_l.nrows();
    debug_assert_eq!(n, mat_l.ncols());
    debug_assert_eq!(mat_d.len(), n);
    debug_assert_eq!(x.len(), n);

    // Forward solve: L y = b  (unit lower)
    for i in 0..n {
        let mut yi = x[i];
        for k in 0..i {
            yi -= mat_l[(i, k)] * x[k];
        }
        x[i] = yi;
    }

    // Scale for tolerances based on D
    let max_abs_d = mat_d.iter().fold(0.0f64, |m, &v| m.max(v.abs())).max(1.0);
    let dzero = max_abs_d * rel_tol;

    // If D[i] ~ 0, then the equation enforces y[i] ~ 0 for consistency.
    for i in 0..n {
        let di = mat_d[i];
        let yi = x[i];
        if di.abs() <= dzero {
            x[i] = 0.0;
        } else {
            x[i] = yi / di;
        }
    }

    // Backward solve: Lᵀ x = z  (unit upper)
    for i in (0..n).rev() {
        let mut xi = x[i];
        for k in (i + 1)..n {
            xi -= mat_l[(k, i)] * x[k];
        }
        x[i] = xi;
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{
        DMatrix,
        DVector,
    };

    use crate::{
        assert_le,
        ldlt_decompose_in_place,
    };
    const EPS: f64 = 1e-12;

    fn reconstruct_from_ldlt(mat_l: &DMatrix<f64>, diag: &DVector<f64>) -> DMatrix<f64> {
        let n = mat_l.nrows();
        // Build L from the unit-lower stored in a_fact.
        let mut l = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    l[(i, j)] = 1.0;
                } else {
                    l[(i, j)] = mat_l[(i, j)];
                }
            }
        }
        // Build diagonal matrix D from vector d.
        let mut mat_d = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            mat_d[(i, i)] = diag[i];
        }
        &l * mat_d * l.transpose()
    }

    #[test]
    fn matrix_reconstructs() {
        // A = M^T M + mu I  (rank full)
        let m =
            DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let mu = 0.1;
        let a_spd = m.transpose() * &m + mu * DMatrix::<f64>::identity(3, 3);

        let mut a_fact = a_spd.clone();
        let mut d = DVector::<f64>::zeros(3);
        let rank = ldlt_decompose_in_place(a_fact.as_view_mut(), d.as_view_mut(), EPS).unwrap();
        assert_eq!(rank, 3);

        let a_rec = reconstruct_from_ldlt(&a_fact, &d);
        let err = (&a_rec - &a_spd).norm();
        assert_le!(err, 1e-9, "SPD reconstruction error too large: {}", err);
        assert!(d.iter().all(|&di| di > 0.0));
    }

    #[test]
    fn rank_deficient_reconstructs_and_has_zero_pivot() {
        // semi-definite case: A = M^T M with rank(M)=2 for n=3.
        let mat_m = DMatrix::<f64>::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let mat_a = mat_m.transpose() * mat_m;

        let mut mat_l = mat_a.clone();
        let mut d = DVector::<f64>::zeros(3);
        let rank = ldlt_decompose_in_place(mat_l.as_view_mut(), d.as_view_mut(), EPS).unwrap();
        assert_eq!(rank, 2);

        let a_rec = reconstruct_from_ldlt(&mat_l, &d);
        let err = (&a_rec - &mat_a).norm();
        assert_le!(err, 1e-9, "PSD reconstruction error too large: {}", err);

        // Expect exactly one zero pivot.
        let zero_count = d.iter().filter(|&&di| di == 0.0).count();
        assert!(zero_count == 1, "expected at least one zero pivot for PSD");
    }

    #[test]
    fn indefinite_matrix_errors_on_negative_pivot() {
        // Symmetric indefinite (eigs: 3, -1) — not positive sem-definite.
        let mat_a = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let mut mat_l = mat_a.clone();
        let mut d = DVector::<f64>::zeros(2);

        let res = ldlt_decompose_in_place(mat_l.as_view_mut(), d.as_view_mut(), EPS);
        assert!(
            res.is_err(),
            "expected failure on negative pivot (indefinite)"
        );
    }

    #[test]
    fn ldlt_dimension_mismatch_is_error() {
        let mat_a = DMatrix::<f64>::identity(3, 3);
        let mut mat_l = mat_a.clone();
        let mut d = DVector::<f64>::zeros(2); // wrong length

        let res = ldlt_decompose_in_place(mat_l.as_view_mut(), d.as_view_mut(), EPS);
        assert!(res.is_err(), "expected dimension mismatch error");
        assert!(format!("{res:?}").contains("dimension mismatch"));
    }

    #[test]
    fn solve_min_norm() {
        use super::{
            ldlt_decompose_in_place,
            ldlt_solve_in_place,
        };
        const EPS: f64 = 1e-12;

        // Rank-deficient: A = MᵀM, rank 2 < n=3
        let mat_m = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let mat_a = mat_m.transpose() * &mat_m;

        // Factor
        let mut mat_l = mat_a.clone();
        let mut d = DVector::zeros(3);
        let rank = ldlt_decompose_in_place(mat_l.as_view_mut(), d.as_view_mut(), EPS).unwrap();
        assert_eq!(rank, 2);

        // Choose b in Range(A): b = A * u
        let u = DVector::from_row_slice(&[0.7, -0.4, 0.3]);
        let b = &mat_a * &u;

        // Solve A x = b (min-norm)
        let mut x = b.clone();
        ldlt_solve_in_place(mat_l.as_view(), d.as_view(), x.as_view_mut(), EPS);

        // Check Ax ≈ b
        let r = &mat_a * &x - &b;
        assert!(r.norm() < 1e-9, "residual too large: {}", r.norm());
    }
}
