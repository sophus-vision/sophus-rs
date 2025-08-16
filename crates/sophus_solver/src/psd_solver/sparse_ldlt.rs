use nalgebra::DVector;

use crate::{
    IsLinearSolver,
    LinearSolverError,
    sparse::{
        CscMatrix,
        CscStruct,
        LowerCscMatrix,
        LowerTripletsMatrix,
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

    type Matrix = LowerTripletsMatrix;

    fn solve_in_place(
        &self,
        a_lower: &LowerCscMatrix,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let at = a_lower.mat.structure.transpose();

        // Elimination tree from the **upper** structure
        let parent = elimination_tree_upper(&at);

        // Numeric LDLᵀ (SPD)
        let (mat_l, d) = ldlt_numeric_spd(&a_lower.mat, &at, &parent, self.tol_rel)
            .map_err(|_| LinearSolverError::FactorizationFailed)?;

        // Identity permutation solve (P = I)
        ldlt_solve_csc_in_place(&mat_l, &d, b);
        Ok(())
    }
}

const INVALID: usize = usize::MAX;

/// Elimination tree using the **upper** structure (from Aᵗ).
/// parent[v] = parent column of v, or INVALID if root.
/// Davis, "Direct Methods...", Alg. 4.1 (upper form).
fn elimination_tree_upper(at_upper: &CscStruct) -> Vec<usize> {
    puffin::profile_function!(format!("{}/elimination_tree_upper", SparseLdlt::NAME));

    let n = at_upper.n;
    let mat_a_p = &at_upper.col_ptr;
    let mat_a_i = &at_upper.row_ind;

    let mut parent = vec![INVALID; n];
    let mut ancestor = vec![INVALID; n];

    for j in 0..n {
        for p in mat_a_p[j]..mat_a_p[j + 1] {
            let i0 = mat_a_i[p];
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
    at_upper: &CscStruct,
    j: usize,
    parent: &[usize],
    w: &mut [usize],     // stamp marks
    stack: &mut [usize], // length n
) -> usize {
    puffin::profile_function!(format!("{}/ereach_upper", SparseLdlt::NAME));

    let n = at_upper.n;
    let mat_a_p = &at_upper.col_ptr;
    let mat_a_i = &at_upper.row_ind;

    let mark = j + 1;
    let mut top = n;

    // Block j from appearing in the reach
    w[j] = mark;

    for p in mat_a_p[j]..mat_a_p[j + 1] {
        let mut i = mat_a_i[p];
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

/// Numeric simplicial LDLᵀ (SPD), left-looking.
/// Uses A_lower for numerics and Aᵗ (upper) for symbolics **and** to gather A(k,j).
/// Numeric simplicial LDLᵀ (SPD), left-looking.
/// Uses A_lower for numerics and Aᵗ (upper) for symbolics **and** to gather A(k,j).
fn ldlt_numeric_spd(
    a_lower: &CscMatrix,  // lower triangle with values
    at_upper: &CscStruct, // transpose(a_lower) with values (=> upper)
    parent: &[usize],
    tol_rel: f64,
) -> Result<(CscMatrix, Vec<f64>), &'static str> {
    puffin::profile_function!(format!("{}/ldlt_numeric_spd", SparseLdlt::NAME));

    let n = a_lower.structure.n;
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
    let mut l_cols: Vec<Vec<(usize, f64)>> = (0..n).map(|_| Vec::<(usize, f64)>::new()).collect();

    let mut max_abs_pivot = 0.0f64;

    for j in 0..n {
        let stamp = j + 1;

        // --- Gather LOWER: A(i >= j, j) -> y[i]
        for p in a_lower.structure.col_ptr[j]..a_lower.structure.col_ptr[j + 1] {
            let i = a_lower.structure.row_ind[p]; // i >= j
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
        let top = ereach_upper(&at_upper, j, parent, &mut w, &mut stk);

        // --- Apply updates from each k in topological order (root -> leaf)
        for idx in top..n {
            let k = stk[idx];

            // Look up lij = L(j,k) directly from the already-finalized column k.
            // (Lcols[k] stores pairs (row, val) with rows > k.)
            let mut lij_opt: Option<f64> = None;
            for &(row, val) in &l_cols[k] {
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

            for &(i, lik) in &l_cols[k] {
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
                    l_cols[j].push((i, lij));
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
        col_ptr[j + 1] = col_ptr[j] + l_cols[j].len();
    }
    let nnz = col_ptr[n];
    let mut row_ind = vec![0usize; nnz];
    let mut values = vec![0f64; nnz];
    let mut base = 0usize;
    for j in 0..n {
        for (k, &(i, v)) in l_cols[j].iter().enumerate() {
            row_ind[base + k] = i;
            values[base + k] = v;
        }
        base += l_cols[j].len();
    }
    let mat_l = CscMatrix::new(n, col_ptr, row_ind, values);
    Ok((mat_l, d))
}

/// Solve with L (unit-lower in CSC), D (diag), Lᵀ. Identity permutation here.
/// x is returned (does not overwrite b).
fn ldlt_solve_csc_in_place(mat_l: &CscMatrix, d: &[f64], b: &mut DVector<f64>) {
    puffin::profile_function!(format!("{}/ldlt_solve_csc_in_place", SparseLdlt::NAME));

    let n = mat_l.structure.n;
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(d.len(), n);

    // Forward: L y = b (unit-lower)
    for j in 0..n {
        let t = b[j];
        for p in mat_l.structure.col_ptr[j]..mat_l.structure.col_ptr[j + 1] {
            let i = mat_l.structure.row_ind[p]; // i > j
            b[i] -= mat_l.values[p] * t;
        }
    }

    // Diagonal: z = D^{-1} y
    for i in 0..n {
        b[i] /= d[i];
    }

    // Backward: Lᵀ x = z
    for j in (0..n).rev() {
        for p in mat_l.structure.col_ptr[j]..mat_l.structure.col_ptr[j + 1] {
            let i = mat_l.structure.row_ind[p]; // i > j
            b[j] -= mat_l.values[p] * b[i];
        }
    }
}
