use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    IsLinearSolver,
    LinearSolverError,
    sparse::{
        CscMatrix,
        LowerTripletsMatrix,
    },
};

/// Public: a sparse LDLᵀ solver that matches your trait and dense API.
/// For SPD matrices; no pivoting; identity ordering (add AMD later if needed).
pub struct SparseLdlt {
    /// relative tolerance
    pub tol_rel: f64,
}

impl IsLinearSolver for SparseLdlt {
    type Matrix = LowerTripletsMatrix;

    fn solve_in_place(
        &self,
        a_lower: &CscMatrix,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let at = csc_transpose(a_lower);

        // Elimination tree from the **upper** structure
        let parent = elimination_tree_upper(&at);

        // Numeric LDLᵀ (SPD)
        let (mat_l, d) = ldlt_numeric_spd(a_lower, &at, &parent, self.tol_rel)
            .map_err(|_| LinearSolverError::FactorizationFailed)?;

        // Identity permutation solve (P = I)
        ldlt_solve_csc_in_place(&mat_l, &d, b);
        Ok(())
    }
}

const INVALID: usize = usize::MAX;

/// Transpose a CSC (including values).
fn csc_transpose(a: &CscMatrix) -> CscMatrix {
    let n = a.n;
    let nnz = a.nnz();

    // Count by future column (== current row indices)
    let mut row_counts = vec![0usize; n];
    for &r in &a.row_ind {
        row_counts[r] += 1;
    }

    // Prefix sum -> col_ptr_t
    let mut col_ptr_t = vec![0usize; n + 1];
    for i in 0..n {
        col_ptr_t[i + 1] = col_ptr_t[i] + row_counts[i];
    }
    let mut next = col_ptr_t.clone();

    let mut row_ind_t = vec![0usize; nnz];
    let mut values_t = vec![0f64; nnz];

    for j in 0..n {
        for p in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_ind[p];
            let dst = next[i];
            row_ind_t[dst] = j;
            values_t[dst] = a.values[p];
            next[i] += 1;
        }
    }
    CscMatrix::new(n, col_ptr_t, row_ind_t, values_t)
}

/// Elimination tree using the **upper** structure (from Aᵗ).
/// parent[v] = parent column of v, or INVALID if root.
/// Davis, "Direct Methods...", Alg. 4.1 (upper form).
fn elimination_tree_upper(at_upper: &CscMatrix) -> Vec<usize> {
    let n = at_upper.n;
    let Ap = &at_upper.col_ptr;
    let Ai = &at_upper.row_ind;

    let mut parent = vec![INVALID; n];
    let mut ancestor = vec![INVALID; n];

    for j in 0..n {
        for p in Ap[j]..Ap[j + 1] {
            let i0 = Ai[p];
            if i0 >= j {
                continue;
            } // strictly upper: i < j
            let mut i = i0;
            while i != INVALID && i != j {
                let next = ancestor[i];
                ancestor[i] = j;
                if next == INVALID {
                    parent[i] = j;
                    break;
                }
                i = next;
            }
        }
    }
    parent
}

/// Symbolic reach on the **upper** structure (Aᵗ).
/// Returns top-of-stack index into `stack`, so `stack[top..n]` are the columns
/// k (< j) in topological order. Pre-marks `j` to avoid returning it.
fn ereach_upper(
    at_upper: &CscMatrix,
    j: usize,
    parent: &[usize],
    w: &mut [usize],     // stamp marks
    stack: &mut [usize], // length n
) -> usize {
    let n = at_upper.n;
    let Ap = &at_upper.col_ptr;
    let Ai = &at_upper.row_ind;

    let mark = j + 1;
    let mut top = n;

    // Block j from appearing in the reach
    w[j] = mark;

    for p in Ap[j]..Ap[j + 1] {
        let mut i = Ai[p];
        if i >= j {
            continue;
        } // strictly upper start points
        while i != INVALID && w[i] != mark {
            stack[top - 1] = i;
            top -= 1;
            w[i] = mark;
            i = parent[i];
        }
    }
    top
}

#[cfg(any(test, debug_assertions))]
fn reconstruct_dense_from_ldl(L: &CscMatrix, d: &[f64]) -> DMatrix<f64> {
    let n = L.n;
    let mut a = DMatrix::<f64>::zeros(n, n);

    // For each column p, build the list of (row, val) including the implicit 1.0 at (p,p)
    for p in 0..n {
        // collect rows in this column: rlist = [p] ∪ {rows > p}
        let mut rows: Vec<usize> = Vec::with_capacity(L.col_ptr[p + 1] - L.col_ptr[p] + 1);
        let mut vals: Vec<f64> = Vec::with_capacity(rows.capacity());
        rows.push(p);
        vals.push(1.0);
        for q in L.col_ptr[p]..L.col_ptr[p + 1] {
            rows.push(L.row_ind[q]);
            vals.push(L.values[q]);
        }

        // Accumulate D[p] * col_p * col_p^T
        let dp = d[p];
        if dp == 0.0 {
            continue;
        }
        for rix in 0..rows.len() {
            let i = rows[rix];
            let vi = vals[rix];
            for rjx in 0..=rix {
                let j = rows[rjx];
                let vj = vals[rjx];
                let add = dp * vi * vj;
                a[(i, j)] += add;
                if i != j {
                    a[(j, i)] += add;
                }
            }
        }
    }
    a
}

/// Numeric simplicial LDLᵀ (SPD), left-looking.
/// Uses A_lower for numerics and Aᵗ (upper) for symbolics **and** to gather A(k,j).
/// Numeric simplicial LDLᵀ (SPD), left-looking.
/// Uses A_lower for numerics and Aᵗ (upper) for symbolics **and** to gather A(k,j).
fn ldlt_numeric_spd(
    a_lower: &CscMatrix,  // lower triangle with values
    at_upper: &CscMatrix, // transpose(a_lower) with values (=> upper)
    parent: &[usize],
    tol_rel: f64,
) -> Result<(CscMatrix, Vec<f64>), &'static str> {
    let n = a_lower.n;
    debug_assert_eq!(n, at_upper.n);
    debug_assert_eq!(parent.len(), n);

    // Workspace
    let mut d = vec![0.0f64; n];
    let mut y = vec![0.0f64; n]; // dense accumulator
    let mut ymark = vec![0usize; n]; // stamping marks for y
    let mut touched: Vec<usize> = Vec::with_capacity(n);

    let mut w = vec![0usize; n]; // ereach marks (per column stamping)
    let mut stk = vec![0usize; n]; // stack buffer

    // L as vector-of-columns; append (row, val); rows need not be sorted.
    let mut Lcols: Vec<Vec<(usize, f64)>> = (0..n).map(|_| Vec::<(usize, f64)>::new()).collect();

    let mut max_abs_pivot = 0.0f64;

    for j in 0..n {
        let stamp = j + 1;

        // --- Gather LOWER: A(i >= j, j) -> y[i]
        for p in a_lower.col_ptr[j]..a_lower.col_ptr[j + 1] {
            let i = a_lower.row_ind[p]; // i >= j
            let v = a_lower.values[p];
            if ymark[i] != stamp {
                y[i] = v;
                ymark[i] = stamp;
                touched.push(i);
            } else {
                y[i] += v;
            }
        }

        // --- Symbolic reach to find contributing columns k < j
        let top = ereach_upper(at_upper, j, parent, &mut w, &mut stk);

        // --- Apply updates from each k in topological order (root -> leaf)
        for idx in top..n {
            let k = stk[idx];

            // Look up lij = L(j,k) directly from the already-finalized column k.
            // (Lcols[k] stores pairs (row, val) with rows > k.)
            let mut lij_opt: Option<f64> = None;
            for &(row, val) in &Lcols[k] {
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
            let alpha = lij * d[k];
            let mut hit_diag = false;

            for &(i, lik) in &Lcols[k] {
                if i == j {
                    hit_diag = true;
                }
                let delta = alpha * lik;
                if ymark[i] != stamp {
                    y[i] = -delta;
                    ymark[i] = stamp;
                    touched.push(i);
                } else {
                    y[i] -= delta;
                }
            }

            // If column k oddly doesn't contain row j, still subtract the diagonal term.
            if !hit_diag {
                if ymark[j] != stamp {
                    y[j] = 0.0;
                    ymark[j] = stamp;
                    touched.push(j);
                }
                // y[j] -= lij * D[k] * lij
                y[j] -= alpha * lij;
            }
        }

        // --- Pivot
        let djj = if ymark[j] == stamp { y[j] } else { 0.0 };
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
        d[j] = djj;

        // --- Finalize column j of L: L(i,j) = y[i]/d[j] for i>j
        for &i in &touched {
            if i > j {
                let lij = y[i] / djj;
                if lij != 0.0 {
                    Lcols[j].push((i, lij));
                }
            }
        }

        // --- Clear y for all touched indices
        for i in touched.drain(..) {
            if ymark[i] == stamp {
                y[i] = 0.0;
            }
        }
    }

    // Compress Lcols -> CSC
    let mut col_ptr = vec![0usize; n + 1];
    for j in 0..n {
        col_ptr[j + 1] = col_ptr[j] + Lcols[j].len();
    }
    let nnz = col_ptr[n];
    let mut row_ind = vec![0usize; nnz];
    let mut values = vec![0f64; nnz];
    let mut base = 0usize;
    for j in 0..n {
        for (k, &(i, v)) in Lcols[j].iter().enumerate() {
            row_ind[base + k] = i;
            values[base + k] = v;
        }
        base += Lcols[j].len();
    }
    let L = CscMatrix::new(n, col_ptr, row_ind, values);
    Ok((L, d))
}

/// Solve with L (unit-lower in CSC), D (diag), Lᵀ. Identity permutation here.
/// x is returned (does not overwrite b).
fn ldlt_solve_csc_in_place(L: &CscMatrix, d: &[f64], b: &mut DVector<f64>) {
    let n = L.n;
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(d.len(), n);

    // Forward: L y = b (unit-lower)
    for j in 0..n {
        let t = b[j];
        for p in L.col_ptr[j]..L.col_ptr[j + 1] {
            let i = L.row_ind[p]; // i > j
            b[i] -= L.values[p] * t;
        }
    }

    // Diagonal: z = D^{-1} y
    for i in 0..n {
        b[i] /= d[i];
    }

    // Backward: Lᵀ x = z
    for j in (0..n).rev() {
        for p in L.col_ptr[j]..L.col_ptr[j + 1] {
            let i = L.row_ind[p]; // i > j
            b[j] -= L.values[p] * b[i];
        }
    }
}
