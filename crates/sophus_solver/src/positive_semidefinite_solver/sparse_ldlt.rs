use nalgebra::DVector;

use crate::{
    LinearSolverError,
    SymmetricMatrixBuilderEnum,
    positive_semidefinite_solver::elimination_tree::elimination_tree_upper,
    prelude::*,
    sparse::{
        CscMatrix,
        CscPattern,
        SparseSymmetricMatrixBuilder,
        TripletMatrix,
    },
};

/// Public: a sparse LDLᵀ solver that matches your trait and dense API.
/// For SPD matrices; no pivoting; identity ordering (add AMD later if needed).
#[derive(Clone, Copy, Debug)]

pub struct SparseLdlt {
    /// relative tolerance
    pub tol_rel: f64,
}

impl Default for SparseLdlt {
    fn default() -> Self {
        SparseLdlt { tol_rel: 1e-12_f64 }
    }
}

impl IsLinearSolver for SparseLdlt {
    const NAME: &'static str = "sparse LDLt";

    type Matrix = TripletMatrix;

    fn solve_in_place(
        &self,
        parallelize: bool,
        a_lower: &CscMatrix,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        // Numeric LDLᵀ (SPD)
        let (mat_l, d) = ldlt_numeric_spd(&a_lower, self.tol_rel)
            .map_err(|_| LinearSolverError::FactorizationFailed)?;

        // Identity permutation solve (P = I)
        ldlt_solve_csc_in_place(&mat_l, &d, b);

        Ok(())
    }

    fn matrix_builder(
        &self,
        partitions: &[crate::PartitionSpec],
    ) -> crate::SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::SparseLower(SparseSymmetricMatrixBuilder::zero(partitions))
    }
}

/// foo
fn ldlt_numeric_spd(
    mat_a_lower: &CscMatrix, // lower triangle with values
    tol_rel: f64,
) -> Result<(CscMatrix, Vec<f64>), &'static str> {
    let n = mat_a_lower.row_count();
    let mat_a_upper_pattern = mat_a_lower.pattern().transpose();

    // A = LDL

    // Output: D
    let mut mat_a_diag = vec![0.0f64; n];
    // Output: L - see mat_l below.

    // Workspace: reused for is col j:
    // accumulator to calculate diagonal D.
    let mut y = vec![0.0f64; n];
    // vector of stamps.
    let mut stamp_col_j = vec![0usize; n];
    let mut touched_col_j: Vec<usize> = Vec::with_capacity(n);

    let mut etree = elimination_tree_upper(&mat_a_upper_pattern);
    // L as vector-of-columns; append (row, val); rows need not be sorted.
    let mut mat_l_cols: Vec<Vec<(usize, f64)>> =
        (0..n).map(|_| Vec::<(usize, f64)>::new()).collect();

    let mut max_abs_pivot = 0.0f64;

    // EXAMPLE
    //
    //     | 2 1 0 0 0 1 |
    //     | 1 2 1 0 0 0 |
    // A = | 0 1 2 0 0 0 |
    //     | 0 0 0 1 0 0 |
    //     | 0 0 0 0 1 0 |
    //     | 1 0 0 0 0 1 |
    //
    //            | 2 0 0 0 0 0 |
    //            | 1 2 0 0 0 0 |
    // lower(A) = | 0 1 2 0 0 0 |
    //            | 0 0 0 1 0 0 |
    //            | 0 0 0 0 1 0 |
    //            | 1 0 0 0 0 1 |
    //
    // mat_a_lower:
    // . col_ptr = [0, 3, 5, 6, 7, 8, 9]
    // . row_ind:  [0, 1, 5, 1, 2, 2, 3, 4, 5]
    // . values:   [2, 1, 1, 1, 2, 2, 1, 1, 1]

    // for each column j of A
    for j in 0..n {
        let stamp = j + 1;

        // EXAMPLE j=0
        // |     y       | stamp | stamped     | touched |
        // -----------------------------------------------
        // | 0,0,0,0,0,0 |   1   | 0,0,0,0,0,0 |         |
        //
        // EXAMPLE j=1
        // |     y       | stamp | stamped     | touched |
        // -----------------------------------------------
        // | 0,0,0,0,0,0 |   2   | 1,1,0,0,0,1 |         |
        for p in mat_a_lower.col_ptr()[j]..mat_a_lower.col_ptr()[j + 1] {
            let i = mat_a_lower.row_idx()[p];
            let v = mat_a_lower.values()[p];
            if stamp_col_j[i] != stamp {
                y[i] = v;
                stamp_col_j[i] = stamp;
                touched_col_j.push(i);
            } else {
                y[i] += v;
                panic!("fff {}", v);
            }
        }
        // example j=0
        // |     y       | stamp | stamped     | touched |
        // -----------------------------------------------
        // | 0,2,1,0,0,0 |   1   | 1,1,0,0,0,1 | 0,1,5   |

        // --- Symbolic reach to find contributing columns k < j
        let top = etree.reach(&mat_a_upper_pattern, j);

        // --- Apply updates from each k in topological order (root -> leaf)
        for idx in top..n {
            let k = etree.reach_buf[idx];

            // Look up lij = L(j,k) directly from the already-finalized column k.
            // (Lcols[k] stores pairs (row, val) with rows > k.)
            let mut lij_opt: Option<f64> = None;
            for &(row, val) in &mat_l_cols[k] {
                if row == j {
                    lij_opt = Some(val);
                    break;
                }
            }

            {
                if lij_opt.is_none() {
                    // We *thought* k contributes to column j (it’s in the ereach),
                    // yet L(:,k) has no row j. This is usually a symbolics bug.
                    eprintln!(
                        "[LDLt] ereach says k={k} contributes to j={j}, but L(:,{k}) has no row {j}"
                    );
                }
            }

            // If L(j,k) is structurally zero (no row j in column k), nothing to do.
            let lij = match lij_opt {
                Some(v) if v != 0.0 => v,
                _ => {
                    // Nothing to scatter from column k into column j.
                    continue;
                }
            };

            // Scatter: y[i] -= lij * D[k] * L(i,k)
            let alpha = lij * mat_a_diag[k];
            let mut hit_diag = false;

            for &(i, lik) in &mat_l_cols[k] {
                if i == j {
                    hit_diag = true;
                }
                let delta = alpha * lik;
                if stamp_col_j[i] != stamp {
                    y[i] = -delta;
                    stamp_col_j[i] = stamp;
                    touched_col_j.push(i);
                } else {
                    y[i] -= delta;
                }
            }

            // If column k oddly doesn't contain row j, still subtract the diagonal term.
            if !hit_diag {
                if stamp_col_j[j] != stamp {
                    y[j] = 0.0;
                    stamp_col_j[j] = stamp;
                    touched_col_j.push(j);
                }
                // y[j] -= lij * D[k] * lij
                y[j] -= alpha * lij;
            }
        }

        // --- Pivot
        let djj = if stamp_col_j[j] == stamp { y[j] } else { 0.0 };
        if !djj.is_finite() {
            #[cfg(debug_assertions)]
            eprintln!("[LDLt] non-finite pivot at j={j}");
            return Err("LDLt (SPD) failed: non-finite pivot");
        }
        max_abs_pivot = f64::max(max_abs_pivot, djj.abs());
        let tau = f64::max(max_abs_pivot, 1.0) * tol_rel;
        if djj <= tau {
            #[cfg(debug_assertions)]
            eprintln!(
                "[LDLt] non-positive/too-small pivot at j={j} (djj={djj:.3e}, tau={tau:.3e})"
            );
            return Err("LDLt (SPD) failed: non-positive/too-small pivot");
        }
        mat_a_diag[j] = djj;

        // --- Finalize column j of L: L(i,j) = y[i]/d[j] for i>j
        for &i in &touched_col_j {
            if i > j {
                let lij = y[i] / djj;
                if lij != 0.0 {
                    mat_l_cols[j].push((i, lij));
                }
            }
        }

        // --- Clear y for all touched indices
        for i in touched_col_j.drain(..) {
            if stamp_col_j[i] == stamp {
                y[i] = 0.0;
            }
        }
    }

    // Compress mat_l_cols -> CSC
    let mut col_ptr = vec![0usize; n + 1];
    for j in 0..n {
        col_ptr[j + 1] = col_ptr[j] + mat_l_cols[j].len();
    }
    let nnz = col_ptr[n];
    let mut row_ind = vec![0usize; nnz];
    let mut values = vec![0f64; nnz];
    let mut base = 0usize;
    for j in 0..n {
        for (k, &(i, v)) in mat_l_cols[j].iter().enumerate() {
            row_ind[base + k] = i;
            values[base + k] = v;
        }
        base += mat_l_cols[j].len();
    }
    let mat_l = CscMatrix::new(CscPattern::new(n, n, col_ptr, row_ind), values);
    Ok((mat_l, mat_a_diag))
}

/// Solve with L (unit-lower in CSC), D (diag), Lᵀ. Identity permutation here.
/// x is returned (does not overwrite b).
fn ldlt_solve_csc_in_place(mat_l: &CscMatrix, d: &[f64], b: &mut DVector<f64>) {
    let n = mat_l.row_count();
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(d.len(), n);

    // Forward: L y = b (unit-lower)
    for j in 0..n {
        let t = b[j];
        for p in mat_l.col_ptr()[j]..mat_l.col_ptr()[j + 1] {
            let i = mat_l.row_idx()[p]; // i > j
            b[i] -= mat_l.values()[p] * t;
        }
    }

    // Diagonal: z = D^{-1} y
    for i in 0..n {
        b[i] /= d[i];
    }

    // Backward: Lᵀ x = z
    for j in (0..n).rev() {
        for p in mat_l.col_ptr()[j]..mat_l.col_ptr()[j + 1] {
            let i = mat_l.row_idx()[p]; // i > j
            b[j] -= mat_l.values()[p] * b[i];
        }
    }
}
