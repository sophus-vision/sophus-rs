use nalgebra::{
    DMatrixView,
    DMatrixViewMut,
    DVector,
    DVectorView,
    DVectorViewMut,
};
use snafu::prelude::*;

use crate::{
    Definiteness,
    DenseLdltSnafu,
    LdltDecompositionError,
    LdltResult,
    LinearSolverEnum,
    LinearSolverError,
    kernel::{
        diag_matsolve_inplaced,
        diag_right_matsolve_inplace,
        diag_solve_inplaced,
        lower_matsolve_inplace,
        lower_right_solve_inplace,
        lower_solve_inplace,
        lower_transpose_matsolve_inplace,
        lower_transpose_rightsolve_inplace,
        lower_transpose_solve_inplace,
    },
    ldlt::{
        BK_FALLBACK_THRESHOLD,
        IsFactor,
    },
    matrix::{
        PartitionSet,
        SymmetricMatrixBuilderEnum,
        dense::{
            DenseSymmetricMatrix,
            DenseSymmetricMatrixBuilder,
        },
    },
    prelude::*,
};

/// Dense LDLᵀ solver for symmetric systems.
///
/// Handles positive-definite, positive semi-definite (rank-deficient), and
/// indefinite (KKT) matrices. For indefinite blocks with poor pivot condition
/// (< 1e-10), automatically falls back to Bunch-Kaufman pivoting.
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
///
/// If the matrix is indefinite with poor pivot condition, a Bunch-Kaufman fallback is stored
/// and used for solves instead.
#[derive(Clone, Debug)]
pub struct DenseLdltFactor {
    pub(crate) mat_l: DenseSymmetricMatrix,
    pub(crate) diag_d: DVector<f64>,
    pub(crate) ldlt_result: LdltResult,
    pub(crate) tol_rel: f64,
    /// BK fallback factor (used when standard LDLᵀ has poor pivot condition on indefinite input).
    bk_factor: Option<super::dense_bunch_kaufman::BunchKaufmanFactor>,
}

impl DenseLdltFactor {
    /// Return rank of matrix A.
    pub fn rank(&self) -> usize {
        self.ldlt_result.rank
    }
}

impl IsFactor for DenseLdltFactor {
    type Matrix = DenseSymmetricMatrix;

    fn solve_inplace(
        &self,
        b_in_x_out: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        if let Some(ref bk) = self.bk_factor {
            bk.solve_slice_inplace(b_in_x_out.as_mut_slice());
        } else {
            ldlt_solve_inplace(
                self.mat_l.view(),
                self.diag_d.as_view(),
                b_in_x_out.as_view_mut(),
            );
        }
        Ok(())
    }
}

impl IsLinearSolver for DenseLdlt {
    type SymmetricMatrixBuilder = DenseSymmetricMatrixBuilder;

    const NAME: &'static str = "dense LDLᵀ";

    fn zero(&self, partitions: PartitionSet) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::Dense(
            DenseSymmetricMatrixBuilder::zero(partitions),
            LinearSolverEnum::DenseLdlt(*self),
        )
    }

    type Factor = DenseLdltFactor;

    fn factorize(&self, mat_a: &DenseSymmetricMatrix) -> Result<Self::Factor, LinearSolverError> {
        let mut diag_d = DVector::<f64>::zeros(mat_a.scalar_dimension());

        let mut mat_l = mat_a.clone();
        let mut ldlt_result =
            ldlt_decompose_inplace(mat_l.view_mut(), diag_d.as_view_mut(), self.tol_rel)
                .context(DenseLdltSnafu)?;

        // BK fallback for ill-conditioned indefinite matrices.
        let bk_factor = if ldlt_result.definiteness == Definiteness::Indefinite
            && ldlt_result.pivot_condition < BK_FALLBACK_THRESHOLD
        {
            let a_dense = mat_a.view().clone_owned();
            let bk = super::dense_bunch_kaufman::factorize(&a_dense).map_err(|_| {
                LinearSolverError::DenseLdltError {
                    source: LdltDecompositionError::NonFinitePivot {
                        j: 0,
                        d_jj: ldlt_result.pivot_condition,
                    },
                }
            })?;
            ldlt_result.used_bk_fallback = true;
            Some(bk)
        } else {
            None
        };

        Ok(DenseLdltFactor {
            mat_l,
            diag_d,
            ldlt_result,
            tol_rel: self.tol_rel,
            bk_factor,
        })
    }

    /// Does not support parallel execution.
    fn set_parallelize(&mut self, _parallelize: bool) {
        // no-op
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
) -> Result<LdltResult, LdltDecompositionError> {
    let n = mat_a_lower.nrows();

    assert_eq!(n, mat_a_lower.ncols());
    assert_eq!(n, mat_d.len());

    // Tracks largest pivot (absolute) to set a relative zero threshold.
    let mut max_abs_pivot: f64 = 1.0;
    let mut min_abs_pivot: f64 = f64::MAX;
    let mut rank: usize = 0;
    let mut has_zero_pivot = false;
    let mut has_negative_pivot = false;

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
            // PSD zero pivot — rank-deficient
            mat_d[j] = 0.0;
            has_zero_pivot = true;
        } else {
            // PD (djj > 0) or indefinite (djj < 0) — both are valid pivots.
            if djj < 0.0 {
                has_negative_pivot = true;
            }
            mat_d[j] = djj;
            let abs_djj = djj.abs();
            max_abs_pivot = max_abs_pivot.max(abs_djj);
            min_abs_pivot = min_abs_pivot.min(abs_djj);
            rank += 1;
        }

        // Unit diagonal for L
        mat_a_lower[(j, j)] = 1.0;
    }

    // Zero the strict upper part of L[j, j].
    for r in 0..mat_a_lower.nrows() {
        for c in (r + 1)..mat_a_lower.ncols() {
            mat_a_lower[(r, c)] = 0.0;
        }
    }
    let definiteness = if has_negative_pivot {
        Definiteness::Indefinite
    } else if has_zero_pivot {
        Definiteness::PositiveSemiDefinite
    } else {
        Definiteness::PositiveDefinite
    };
    let pivot_condition = if rank == 0 {
        0.0
    } else {
        min_abs_pivot / max_abs_pivot
    };
    Ok(LdltResult {
        rank,
        definiteness,
        pivot_condition,
        used_bk_fallback: false,
    })
}

/// Solve A x = b using A = L D Lᵀ decomposition (in place on `x`).
pub fn ldlt_solve_inplace(
    mat_l: DMatrixView<'_, f64>,
    mat_d: DVectorView<'_, f64>,
    mut x: DVectorViewMut<'_, f64>,
) {
    debug_assert_eq!(mat_l.nrows(), mat_l.ncols());
    debug_assert_eq!(mat_d.len(), mat_l.nrows());
    debug_assert_eq!(x.len(), mat_l.nrows());

    // Forward: L y = b
    lower_solve_inplace(&mat_l, &mut x);

    // z = D^{-1} y   (PSD-aware)
    diag_solve_inplaced(mat_d, &mut x);

    // Backward: Lᵀ x = z
    lower_transpose_solve_inplace(&mat_l, &mut x);
}

/// Solve A X = B using A = L D Lᵀ decomposition (in place on `x`).
pub fn ldlt_matsolve_inplace(
    mat_l: DMatrixView<'_, f64>,
    mat_d: DVectorView<'_, f64>,
    mut mat_x: DMatrixViewMut<'_, f64>,
) {
    debug_assert_eq!(mat_l.nrows(), mat_l.ncols());
    debug_assert_eq!(mat_d.len(), mat_l.nrows());
    debug_assert_eq!(mat_x.nrows(), mat_l.nrows());

    // Forward: L Y = B
    lower_matsolve_inplace(&mat_l, &mut mat_x);

    // Z = D^{-1} Y   (PSD-aware)
    diag_matsolve_inplaced(mat_d, &mut mat_x);

    // Backward: Lᵀ X = Z
    lower_transpose_matsolve_inplace(&mat_l, &mut mat_x);
}

/// In-place block right solve: X ← X * (Lᵈᵀ)^{-1} * diag(d)^{-1} * (Lᵈ)^{-1}
#[inline]
pub fn ldlt_right_matsolve_inplace(
    ld: DMatrixView<'_, f64>,       // unit-lower Lᵈ
    d: DVectorView<'_, f64>,        // diagonal d
    mut x: DMatrixViewMut<'_, f64>, // in/out
) {
    lower_transpose_rightsolve_inplace(ld.as_view(), &mut x);
    diag_right_matsolve_inplace(d, &mut x);
    lower_right_solve_inplace(ld, &mut x);
}

#[cfg(test)]
mod tests {
    use nalgebra::{
        DMatrix,
        DVector,
    };
    use sophus_assert::assert_le;

    use super::{
        ldlt_decompose_inplace,
        ldlt_solve_inplace,
        *,
    };

    const LDLT_TOL: f64 = 1e-12;

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
        let result =
            ldlt_decompose_inplace(a_fact.as_view_mut(), d.as_view_mut(), LDLT_TOL).unwrap();
        assert_eq!(result.rank, 3);
        assert_eq!(result.definiteness, Definiteness::PositiveDefinite);
        assert!(
            result.pivot_condition > 0.01,
            "pivot_condition={}",
            result.pivot_condition
        );

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
        let result =
            ldlt_decompose_inplace(mat_l.as_view_mut(), d.as_view_mut(), LDLT_TOL).unwrap();
        assert_eq!(result.rank, 2);
        assert_eq!(result.definiteness, Definiteness::PositiveSemiDefinite);

        let a_rec = reconstruct_from_ldlt(&mat_l, &d);
        let err = (&a_rec - &mat_a).norm();
        assert_le!(err, 1e-9, "PSD reconstruction error too large: {}", err);

        // Expect exactly one zero pivot.
        let zero_count = d.iter().filter(|&&di| di == 0.0).count();
        assert_eq!(zero_count, 1, "expected exactly one zero pivot for PSD");
    }

    #[test]
    fn indefinite_matrix_solves_correctly() {
        // Symmetric indefinite (eigs: 3, -1) — not PSD but LDLᵀ handles it.
        let mat_a = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let mut mat_l = mat_a.clone();
        let mut d = DVector::<f64>::zeros(2);

        let result =
            ldlt_decompose_inplace(mat_l.as_view_mut(), d.as_view_mut(), LDLT_TOL).unwrap();
        assert_eq!(result.rank, 2);
        assert_eq!(result.definiteness, Definiteness::Indefinite);
        assert!(
            result.pivot_condition > 0.1,
            "pivot_condition={}",
            result.pivot_condition
        );

        // d should have one positive and one negative pivot.
        assert!(d[0] > 0.0);
        assert!(d[1] < 0.0);

        // Verify solve via L D Lᵀ: A x = b.
        let b = DVector::from_row_slice(&[1.0, 2.0]);
        let mut x = b.clone();
        let n = x.len();
        ldlt_solve_inplace(mat_l.as_view(), d.as_view(), x.rows_mut(0, n));

        let x_ref = mat_a.clone().lu().solve(&b).unwrap();
        approx::assert_abs_diff_eq!(x, x_ref, epsilon = 1e-10);
    }

    #[test]
    fn solve_min_norm() {
        // Rank-deficient: A = MᵀM, rank 2 < n=3
        let mat_m = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let mat_a = mat_m.transpose() * &mat_m;

        // Factor
        let mut mat_l = mat_a.clone();
        let mut d = DVector::zeros(3);
        let result =
            ldlt_decompose_inplace(mat_l.as_view_mut(), d.as_view_mut(), LDLT_TOL).unwrap();
        assert_eq!(result.rank, 2);

        // Choose b in Range(A): b = A * u
        let u = DVector::from_row_slice(&[0.7, -0.4, 0.3]);
        let b = &mat_a * &u;

        // Solve A x = b (min-norm)
        let mut x = b.clone();
        ldlt_solve_inplace(mat_l.as_view(), d.as_view(), x.as_view_mut());

        // Check Ax ≈ b
        let r = &mat_a * &x - &b;
        assert!(r.norm() < 1e-9, "residual too large: {}", r.norm());
    }

    #[test]
    fn left_right_equivalence_on_spd() {
        use nalgebra::{
            DMatrix,
            DVector,
        };
        let n = 6;
        let k = 3;

        // make a random unit-lower L and positive D
        let mut l = DMatrix::<f64>::identity(n, n);
        for i in 1..n {
            for j in 0..i {
                l[(i, j)] = (i + 2 * j + 1) as f64 / 17.0;
            }
        }
        let d = DVector::from_fn(n, |i, _| 1.0 + 0.2 * (i as f64));

        // random RHS B (n×k)
        let b = DMatrix::<f64>::from_fn(n, k, |i, j| ((3 * i + 5 * j + 1) as f64) / 13.0);

        // left: solve H X = B
        let mut left = b.clone();
        lower_matsolve_inplace(&l.as_view(), &mut left.as_view_mut());
        diag_matsolve_inplaced(d.as_view(), &mut left.as_view_mut());
        lower_transpose_matsolve_inplace(&l.as_view(), &mut left.as_view_mut());

        // right: compute Bᵀ H^{-1} then transpose
        let mut right_t = b.transpose();
        lower_transpose_rightsolve_inplace(l.as_view(), &mut right_t.as_view_mut());
        diag_right_matsolve_inplace(d.as_view(), &mut right_t.as_view_mut());
        lower_right_solve_inplace(l.as_view(), &mut right_t.as_view_mut());
        let right = right_t.transpose();

        assert!((&left - &right).norm() < 1e-12);
    }

    #[test]
    fn dense_ldlt_bk_fallback_produces_correct_result() {
        use crate::{
            IsFactor,
            IsLinearSolver,
            matrix::dense::DenseSymmetricMatrix,
        };

        // Ill-conditioned indefinite matrix where standard LDLᵀ loses precision
        // but BK fallback recovers the correct answer.
        //
        // A = [[ε, 1], [1, -ε]] with ε = 1e-12.
        // Standard LDLᵀ: d[0] = ε (tiny), L[1,0] = 1/ε (huge), d[1] ≈ -1/ε (huge).
        // Pivot condition ≈ ε² ≈ 1e-24 → catastrophic loss of precision.
        //
        // BK pivoting rearranges to avoid the tiny pivot, giving a stable result.
        let eps = 1e-12;
        let raw = DMatrix::from_row_slice(2, 2, &[eps, 1.0, 1.0, -eps]);
        let mat_a = DenseSymmetricMatrix::new(
            raw.clone(),
            crate::matrix::PartitionSet::new(vec![crate::matrix::PartitionSpec {
                eliminate_last: false,
                block_count: 1,
                block_dim: 2,
            }]),
        );
        let b = DVector::from_row_slice(&[1.0, 2.0]);

        // Reference solution via LU (numerically stable).
        let x_ref = raw.clone().lu().solve(&b).unwrap();

        // Solve with DenseLdlt (which should trigger BK fallback).
        let solver = DenseLdlt { tol_rel: 1e-15 };
        let factor = solver.factorize(&mat_a).unwrap();

        assert!(
            factor.ldlt_result.used_bk_fallback,
            "expected BK fallback for ill-conditioned indefinite matrix, \
             pivot_condition={}",
            factor.ldlt_result.pivot_condition,
        );

        let mut x = b.clone();
        factor.solve_inplace(&mut x).unwrap();

        // BK fallback should match LU to high precision.
        approx::assert_abs_diff_eq!(x, x_ref, epsilon = 1e-6);

        // Verify that WITHOUT BK fallback, the result would be poor.
        // (Use raw LDLᵀ solve with the stored L and d.)
        let mut x_no_bk = b.clone();
        ldlt_solve_inplace(
            factor.mat_l.view(),
            factor.diag_d.as_view(),
            x_no_bk.as_view_mut(),
        );
        let error_no_bk = (&x_no_bk - &x_ref).norm();
        let error_with_bk = (&x - &x_ref).norm();
        assert!(
            error_with_bk < error_no_bk * 0.01,
            "BK should be much more accurate: bk_err={:.2e}, ldlt_err={:.2e}",
            error_with_bk,
            error_no_bk,
        );
    }
}
