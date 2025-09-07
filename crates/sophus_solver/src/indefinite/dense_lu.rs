use nalgebra::{
    DMatrix,
    DMatrixView,
    DMatrixViewMut,
    DVectorViewMut,
};
use snafu::prelude::*;

use crate::{
    DenseLuSnafu,
    LuDecompositionError,
    error::LinearSolverError,
    matrix::{
        DenseSymmetricMatrixBuilder,
        PartitionSpec,
        SymmetricMatrixBuilderEnum,
    },
    prelude::*,
};

/// Dense solver using LU decomposition.
#[derive(Copy, Clone, Debug)]

pub struct DenseLu {}

impl IsLinearSolver for DenseLu {
    type Matrix = DMatrix<f64>;

    const NAME: &'static str = "dense LU";

    fn matrix_builder(&self, partitions: &[PartitionSpec]) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::Dense(DenseSymmetricMatrixBuilder::zero(partitions))
    }

    fn solve_in_place(
        &self,
        _parallelize: bool,
        mat_a: &Self::Matrix,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let n = mat_a.nrows();
        assert_eq!(n, mat_a.ncols());
        assert_eq!(b.len(), n);

        let mut a = mat_a.clone();
        let mut piv = vec![0usize; n];
        const EPS: f64 = 1e-12;

        lu_in_place(a.as_view_mut(), &mut piv, EPS).context(DenseLuSnafu)?;

        lu_solve_in_place(a.as_view(), &piv, b.as_view_mut());
        Ok(())
    }
}

/// In-place LU factorization with partial pivoting: PA = LU.
/// - A's strict lower tri stores L (unit diag not stored),
/// - A's upper tri (incl diag) stores U.
///
/// On return, `piv[k]` is the pivot row chosen at step k (the row swapped with k).
pub fn lu_in_place(
    mut mat_a: DMatrixViewMut<'_, f64>,
    piv: &mut [usize],
    rel_tol: f64,
) -> Result<(), LuDecompositionError> {
    puffin::profile_scope!("LU fact");

    let n = mat_a.nrows();
    assert_eq!(n, mat_a.ncols());
    assert_eq!(piv.len(), n);

    for k in 0..n {
        // 1) Choose pivot row p = argmax_{i>=k} |A[i,k]|
        let mut p = k;
        let mut max_abs = mat_a[(k, k)].abs();
        for i in (k + 1)..n {
            let v = mat_a[(i, k)].abs();
            if v > max_abs {
                max_abs = v;
                p = i;
            }
        }

        // 2) Check near-singularity
        let tau = f64::max(max_abs, 1.0_f64) * rel_tol;
        if max_abs <= tau {
            return Err(LuDecompositionError::NearSingularPivot);
        }

        // 3) Swap rows if needed, and record pivot
        piv[k] = p;
        if p != k {
            for j in 0..n {
                let tmp = mat_a[(k, j)];
                mat_a[(k, j)] = mat_a[(p, j)];
                mat_a[(p, j)] = tmp;
            }
        }

        // 4) Eliminate below pivot
        let akk = mat_a[(k, k)];
        for i in (k + 1)..n {
            mat_a[(i, k)] /= akk; // L(i,k)
            let v_ik = mat_a[(i, k)];
            // A[i, k+1..] -= L(i,k) * U(k, k+1..)
            for j in (k + 1)..n {
                mat_a[(i, j)] -= v_ik * mat_a[(k, j)];
            }
        }
    }

    Ok(())
}

/// Solve A x = b using the compact LU storage and the pivot vector `piv`.
/// Overwrites `x` in-place.
pub fn lu_solve_in_place(
    mat_lu: DMatrixView<'_, f64>,
    piv: &[usize],
    mut x: DVectorViewMut<'_, f64>,
) {
    puffin::profile_scope!("LU solve");

    let n = mat_lu.nrows();
    assert_eq!(n, mat_lu.ncols());
    assert_eq!(x.len(), n);
    assert_eq!(piv.len(), n);

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
            yi -= mat_lu[(i, k)] * x[k];
        }
        x[i] = yi;
    }

    // Backward solve: U x = y  (U is upper triangular incl diag)
    for i in (0..n).rev() {
        let mut xi = x[i];
        for k in (i + 1)..n {
            xi -= mat_lu[(i, k)] * x[k];
        }
        x[i] = xi / mat_lu[(i, i)];
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{
        DMatrix,
        DVector,
    };

    use super::{
        lu_in_place,
        lu_solve_in_place,
    };
    use crate::assert_le;

    const EPS: f64 = 1e-12;

    /// Expand compact LU storage into (L, U) where:
    /// - L is unit-lower (diag = 1) from strict lower of `mat_lu`
    /// - U is upper-triangular (incl diag) from upper of `mat_lu`
    fn expand_lu(lu_storage: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        let n = lu_storage.nrows();
        let mut mat_l = DMatrix::<f64>::zeros(n, n);
        let mut mat_u = DMatrix::<f64>::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                if i > j {
                    mat_l[(i, j)] = lu_storage[(i, j)];
                } else {
                    mat_u[(i, j)] = lu_storage[(i, j)];
                }
            }
            mat_l[(i, i)] = 1.0;
        }
        (mat_l, mat_u)
    }

    /// Apply the recorded partial-pivot row swaps to a copy of A.
    fn apply_row_swaps(mut a: DMatrix<f64>, piv: &[usize]) -> DMatrix<f64> {
        let n = a.nrows();
        assert_eq!(n, a.ncols());
        assert_eq!(piv.len(), n);

        for k in 0..n {
            let p = piv[k];
            if p != k {
                for j in 0..n {
                    a.swap((k, j), (p, j));
                }
            }
        }
        a
    }

    fn residual_norm(a: &DMatrix<f64>, x: &DVector<f64>, b: &DVector<f64>) -> f64 {
        (a * x - b).norm()
    }

    #[test]
    fn lu_reconstructs_pa_equals_lu() {
        // General (nonsingular) test matrix
        let mat_a = DMatrix::<f64>::from_row_slice(
            4,
            4,
            &[
                2.0, -1.0, 0.0, 3.0, 4.0, 1.0, -1.0, 0.0, 0.0, 5.0, 2.0, -2.0, 1.0, 0.0, 3.0, 1.0,
            ],
        );

        let mut lu_storage = mat_a.clone();
        let mut piv = vec![0usize; 4];
        lu_in_place(lu_storage.as_view_mut(), &mut piv, EPS)
            .expect("LU factorization should succeed");

        // Build L and U from compact storage
        let (l, u) = expand_lu(&lu_storage);

        // Apply pivot history to A to form P*A
        let p_times_a = apply_row_swaps(mat_a.clone(), &piv);

        let err = (&l * &u - p_times_a).norm();
        assert_le!(err, 1e-10);
    }

    #[test]
    fn lu_solve_matches_nalgebra() {
        let mat_a = DMatrix::<f64>::from_row_slice(
            4,
            4,
            &[
                3.0, 2.0, -1.0, 4.0, //
                2.0, 1.0, 5.0, -2.0, //
                -1.0, 0.0, 3.0, 1.0, //
                4.0, -2.0, 1.0, 2.0,
            ],
        );
        let b = DVector::from_row_slice(&[1.0, 2.0, -1.0, 0.5]);

        let mut lu_storage = mat_a.clone();
        let mut piv = vec![0usize; 4];
        lu_in_place(lu_storage.as_view_mut(), &mut piv, EPS).expect("LU should succeed");
        let mut x1 = b.clone();
        lu_solve_in_place(lu_storage.as_view(), &piv, x1.as_view_mut());

        // nalgebra reference
        let x2 = mat_a
            .clone()
            .lu()
            .solve(&b)
            .expect("reference LU should succeed");

        let nrm = (x1.clone() - x2).norm();
        assert_le!(nrm, 1e-10);
        assert!(residual_norm(&mat_a, &x1, &b) < 1e-10, "residual too large");
    }

    #[test]
    fn lu_detects_singular_matrix() {
        // Singular: duplicate first two rows
        let mat_a =
            DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0, 1.0, 4.0]);
        let mut lu_storage = mat_a.clone();
        let mut piv = vec![0usize; 3];
        let res = lu_in_place(lu_storage.as_view_mut(), &mut piv, EPS);
        assert!(res.is_err(), "expected near-singular detection");
    }

    #[test]
    fn lu_identity_trivial() {
        let a = DMatrix::<f64>::identity(3, 3);
        let mut lu = a.clone();
        let mut piv = vec![0usize; 3];
        lu_in_place(lu.as_view_mut(), &mut piv, EPS).unwrap();

        // Solve Ax=b should return b
        let b = DVector::from_row_slice(&[1.0, -2.0, 3.0]);
        let mut x = b.clone();
        lu_solve_in_place(lu.as_view(), &piv, x.as_view_mut());
        assert!((x - b).norm() == 0.0);
    }

    #[test]
    fn pivots_are_valid_partial_pivots() {
        let mat_a = DMatrix::<f64>::from_row_slice(
            5,
            5,
            &[
                1.0, 4.0, 7.0, 0.0, -1.0, //
                3.0, -2.0, 5.0, 2.0, 0.0, //
                0.0, 1.0, 1.0, 3.0, 2.0, //
                -2.0, 0.0, 4.0, 1.0, 1.0, //
                5.0, 2.0, 0.0, -3.0, 2.0,
            ],
        );

        let mut lu_storage = mat_a.clone();
        let mut piv = vec![0usize; 5];
        lu_in_place(lu_storage.as_view_mut(), &mut piv, EPS).unwrap();

        let n = mat_a.nrows();
        for k in 0..n {
            assert!(
                piv[k] >= k && piv[k] < n,
                "piv[{}] out of range: {}",
                k,
                piv[k]
            );
        }
    }

    #[test]
    fn lu_solve_low_residual_positive_definite_matrix() {
        // positive definiteness is not required, but tends to produce well-conditioned systems.
        let mat_m = DMatrix::<f64>::from_row_slice(
            3,
            3,
            &[
                1.0, 2.0, 3.0, //
                0.0, 1.0, 4.0, //
                5.0, 6.0, 0.0,
            ],
        );
        let mat_a: DMatrix<f64> = mat_m.transpose() * &mat_m + 0.1 * DMatrix::identity(3, 3);

        let x_true = DVector::from_row_slice(&[0.7, -0.2, 0.9]);
        let b = &mat_a * &x_true;

        let mut lu_storage = mat_a.clone();
        let mut piv = vec![0usize; 3];
        lu_in_place(lu_storage.as_view_mut(), &mut piv, EPS).unwrap();

        let mut x = b.clone();
        lu_solve_in_place(lu_storage.as_view(), &piv, x.as_view_mut());
        assert_le!(residual_norm(&mat_a, &x, &b), 1e-11);
        let nrm = (x - x_true).norm();
        assert_le!(nrm, 1e-8);
    }
}
