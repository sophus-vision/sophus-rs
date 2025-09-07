use nalgebra::{
    DMatrix,
    DMatrixView,
    DMatrixViewMut,
    DVector,
    DVectorView,
    DVectorViewMut,
};
use snafu::prelude::*;

use crate::{
    DenseLdltSnafu,
    LdltDecompositionError,
    LinearSolverError,
    kernel::{
        diag_rightsolve_inplace,
        diag_solve_inplaced,
        lower_right_solve_inplace,
        lower_solve_inplace,
        lower_transpose_rightsolve_inplace,
        lower_transpose_solve_inplace,
    },
    matrix::{
        DenseSymmetricMatrixBuilder,
        PartitionSpec,
        SymmetricMatrixBuilderEnum,
    },
    prelude::*,
};

/// Dense solver using LDLᵀ decomposition.
#[derive(Copy, Clone, Debug)]
pub struct DenseLdlt {
    /// relative tolerance
    pub tol_rel: f64,
}

impl Default for DenseLdlt {
    fn default() -> Self {
        DenseLdlt { tol_rel: 1e-12_f64 }
    }
}

/// Dense LDLᵀ factors: unit-lower `L` in the lower triangle of `mat_l`, diagonal in `diag_d`.
pub struct LdltFactors {
    mat_l: DMatrix<f64>,
    diag_d: DVector<f64>,
    tol_rel: f64,
}

impl LdltFactors {
    #[inline]
    fn new(mat_l: DMatrix<f64>, tol_rel: f64) -> Self {
        debug_assert_eq!(mat_l.nrows(), mat_l.ncols());
        let n = mat_l.nrows();
        Self {
            mat_l,
            diag_d: DVector::<f64>::zeros(n),
            tol_rel,
        }
    }

    #[inline]
    fn decompose_in_place(&mut self) -> Result<u64, LdltDecompositionError> {
        ldlt_decompose_inplace(
            self.mat_l.as_view_mut(),
            self.diag_d.as_view_mut(),
            self.tol_rel,
        )
    }

    #[inline]
    fn solve_in_place(&self, b_in_x_out: &mut DVector<f64>) {
        ldlt_solve_in_place(
            self.mat_l.as_view(),
            self.diag_d.as_view(),
            b_in_x_out.as_view_mut(),
            self.tol_rel,
        );
    }
}

impl IsLinearSolver for DenseLdlt {
    type Matrix = DMatrix<f64>;

    const NAME: &'static str = "dense LDLᵀ";

    fn matrix_builder(&self, partitions: &[PartitionSpec]) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::Dense(DenseSymmetricMatrixBuilder::zero(partitions))
    }

    fn solve_in_place(
        &self,
        _parallelize: bool,
        mat_a: &Self::Matrix,
        b_in_x_out: &mut DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let n = mat_a.nrows();
        assert_eq!(n, mat_a.ncols());
        assert_eq!(n, b_in_x_out.len());

        let mut ldlt = LdltFactors::new(mat_a.clone(), self.tol_rel);
        ldlt.decompose_in_place().context(DenseLdltSnafu)?;
        ldlt.solve_in_place(b_in_x_out);
        Ok(())
    }
}

/// LDLᵀ factorization in-place on A's lower triangle.
///
/// On return:
///   - `mat_a_lower` stores unit-lower L (diag set to 1.0; upper triangle ignored)
///   - `mat_d` is filled with the diagonal D (entries may be zero for PSD)
///
/// Returns matrix rank on success.
pub fn ldlt_decompose_inplace(
    mut mat_a_lower: DMatrixViewMut<'_, f64>,
    mut mat_d: DVectorViewMut<'_, f64>,
    rel_tol: f64,
) -> Result<u64, LdltDecompositionError> {
    let n = mat_a_lower.nrows();

    assert_eq!(n, mat_a_lower.ncols());
    assert_eq!(n, mat_d.len());

    // Tracks largest positive pivot to set a relative zero threshold.
    let mut max_abs_pivot: f64 = 1.0;
    let mut rank: u64 = 0;

    for j in 0..n {
        // Compute column j below the diagonal
        for k in 0..j {
            // s = A(j,k) - sum_{p<k} L(j,p) * D[p] * L(k,p)
            let mut s = mat_a_lower[(j, k)];
            for p in 0..k {
                s -= mat_a_lower[(j, p)] * mat_d[p] * mat_a_lower[(k, p)];
            }

            let zero_tol = max_abs_pivot.max(1.0) * rel_tol;

            // PSD-aware: if D[k] ≈ 0, that column contributes nothing
            mat_a_lower[(j, k)] = if mat_d[k].abs() <= zero_tol {
                0.0
            } else {
                s / mat_d[k]
            };
        }

        // d[j] = A(j,j) - sum_{p<j} d[p] * L(j,p)^2
        let mut djj = mat_a_lower[(j, j)];
        for p in 0..j {
            let ljp = mat_a_lower[(j, p)];
            djj -= mat_d[p] * ljp * ljp;
        }

        if !djj.is_finite() {
            return Err(LdltDecompositionError::NonFinitePivot { j, d_jj: djj });
        }

        let zero_tol = max_abs_pivot.max(1.0) * rel_tol;
        if djj.abs() <= zero_tol {
            // PSD zero pivot
            mat_d[j] = 0.0;
        } else if djj < 0.0 {
            return Err(LdltDecompositionError::NegativeFinitePivot { j, d_jj: djj });
        } else {
            mat_d[j] = djj;
            max_abs_pivot = max_abs_pivot.max(djj.abs());
            rank += 1;
        }

        // Unit diagonal for L
        mat_a_lower[(j, j)] = 1.0;
    }
    Ok(rank)
}

/// Solve A x = b using A = L D Lᵀ decomposition (in place on `x`).
pub fn ldlt_solve_in_place(
    mat_l: DMatrixView<'_, f64>,
    mat_d: DVectorView<'_, f64>,
    mut x: DVectorViewMut<'_, f64>,
    rel_tol: f64,
) {
    debug_assert_eq!(mat_l.nrows(), mat_l.ncols());
    debug_assert_eq!(mat_d.len(), mat_l.nrows());
    debug_assert_eq!(x.len(), mat_l.nrows());

    // Forward: L y = b
    lower_solve_inplace(&mat_l, &mut x);

    // z = D^{-1} y   (PSD-aware)
    diag_solve_inplaced(mat_d, &mut x, rel_tol);

    // Backward: Lᵀ x = z
    lower_transpose_solve_inplace(&mat_l, &mut x);
}

/// In-place block right solve: X ← X * (Lᵈᵀ)^{-1} * diag(d)^{-1} * (Lᵈ)^{-1}
#[inline]
pub fn ldlt_right_solve_inplace(
    ld: DMatrixView<'_, f64>,       // unit-lower Lᵈ
    d: DVectorView<'_, f64>,        // diagonal d
    mut x: DMatrixViewMut<'_, f64>, // in/out
    rel_tol: f64,
) {
    lower_transpose_rightsolve_inplace(ld.as_view(), &mut x);
    diag_rightsolve_inplace(d, &mut x, rel_tol);
    lower_right_solve_inplace(ld, &mut x);
}

#[cfg(test)]
mod tests {
    use nalgebra::{
        DMatrix,
        DVector,
    };

    use super::{
        ldlt_decompose_inplace,
        ldlt_solve_in_place,
    };
    use crate::assert_le;

    const LDLT_EPS: f64 = 1e-12;

    fn reconstruct_from_ldlt(mat_l: &DMatrix<f64>, diag: &DVector<f64>) -> DMatrix<f64> {
        let n = mat_l.nrows();
        // Build L (unit-lower) from the stored lower triangle of mat_l.
        let mut l = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                l[(i, j)] = if i == j { 1.0 } else { mat_l[(i, j)] };
            }
        }
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
        let rank = ldlt_decompose_inplace(a_fact.as_view_mut(), d.as_view_mut(), LDLT_EPS).unwrap();
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
        let rank = ldlt_decompose_inplace(mat_l.as_view_mut(), d.as_view_mut(), LDLT_EPS).unwrap();
        assert_eq!(rank, 2);

        let a_rec = reconstruct_from_ldlt(&mat_l, &d);
        let err = (&a_rec - &mat_a).norm();
        assert_le!(err, 1e-9, "PSD reconstruction error too large: {}", err);

        // Expect exactly one zero pivot.
        let zero_count = d.iter().filter(|&&di| di == 0.0).count();
        assert_eq!(zero_count, 1, "expected exactly one zero pivot for PSD");
    }

    #[test]
    fn indefinite_matrix_errors_on_negative_pivot() {
        // Symmetric indefinite (eigs: 3, -1) — not PSD.
        let mat_a = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let mut mat_l = mat_a.clone();
        let mut d = DVector::<f64>::zeros(2);

        let res = ldlt_decompose_inplace(mat_l.as_view_mut(), d.as_view_mut(), LDLT_EPS);
        assert!(
            res.is_err(),
            "expected failure on negative pivot (indefinite)"
        );
    }

    #[test]
    fn solve_min_norm() {
        // Rank-deficient: A = MᵀM, rank 2 < n=3
        let mat_m = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let mat_a = mat_m.transpose() * &mat_m;

        // Factor
        let mut mat_l = mat_a.clone();
        let mut d = DVector::zeros(3);
        let rank = ldlt_decompose_inplace(mat_l.as_view_mut(), d.as_view_mut(), LDLT_EPS).unwrap();
        assert_eq!(rank, 2);

        // Choose b in Range(A): b = A * u
        let u = DVector::from_row_slice(&[0.7, -0.4, 0.3]);
        let b = &mat_a * &u;

        // Solve A x = b (min-norm)
        let mut x = b.clone();
        ldlt_solve_in_place(mat_l.as_view(), d.as_view(), x.as_view_mut(), LDLT_EPS);

        // Check Ax ≈ b
        let r = &mat_a * &x - &b;
        assert!(r.norm() < 1e-9, "residual too large: {}", r.norm());
    }
}
