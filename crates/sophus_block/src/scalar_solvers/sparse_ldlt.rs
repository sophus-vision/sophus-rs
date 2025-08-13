use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    IsDenseLinearSystem,
    LinearSolverError,
};

const INVALID: usize = usize::MAX;

/// Simple CSC container (column-compressed sparse).
#[derive(Clone, Debug)]
struct CscMatrix {
    n: usize,            // square n x n
    col_ptr: Vec<usize>, // len = n+1
    row_ind: Vec<usize>, // len = nnz
    values: Vec<f64>,    // len = nnz
}

impl CscMatrix {
    fn new(n: usize, col_ptr: Vec<usize>, row_ind: Vec<usize>, values: Vec<f64>) -> Self {
        debug_assert_eq!(col_ptr.len(), n + 1);
        debug_assert_eq!(row_ind.len(), values.len());
        Self {
            n,
            col_ptr,
            row_ind,
            values,
        }
    }
    fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Extract lower-triangle triplets (i >= j) from a dense matrix.
fn lower_triplets_from_dense(a: &DMatrix<f64>) -> (usize, Vec<usize>, Vec<usize>, Vec<f64>) {
    let n = a.nrows();
    assert_eq!(n, a.ncols());
    let reserve = n * (n + 1) / 2;
    let mut ii = Vec::with_capacity(reserve);
    let mut jj = Vec::with_capacity(reserve);
    let mut xx = Vec::with_capacity(reserve);

    for j in 0..n {
        for i in j..n {
            let v = a[(i, j)];
            if v != 0.0 {
                ii.push(i);
                jj.push(j);
                xx.push(v);
            }
        }
    }
    (n, ii, jj, xx)
}

/// Compress triplets to CSC; assumes indices in range [0, n).
/// Coalesces duplicates by summing; keeps lower-only structure as provided.
fn triplets_to_csc(
    n: usize,
    mut ii: Vec<usize>,
    mut jj: Vec<usize>,
    mut xx: Vec<f64>,
) -> CscMatrix {
    let nnz = ii.len();
    let mut idx: Vec<usize> = (0..nnz).collect();

    // Sort by (col, row)
    idx.sort_unstable_by(|&a, &b| {
        let ca = jj[a].cmp(&jj[b]);
        if ca == std::cmp::Ordering::Equal {
            ii[a].cmp(&ii[b])
        } else {
            ca
        }
    });

    // Count per column
    let mut col_counts = vec![0usize; n];
    for &k in &idx {
        col_counts[jj[k]] += 1;
    }

    // Prefix sum -> col_ptr
    let mut col_ptr = vec![0usize; n + 1];
    for j in 0..n {
        col_ptr[j + 1] = col_ptr[j] + col_counts[j];
    }

    // Fill Ai/Ax, coalescing duplicates
    let mut Ai = vec![0usize; nnz];
    let mut Ax = vec![0f64; nnz];
    let mut next = col_ptr.clone();

    for &k in &idx {
        let j = jj[k];
        let i = ii[k];
        let x = xx[k];
        let pos = next[j];
        if pos > col_ptr[j] && Ai[pos - 1] == i {
            Ax[pos - 1] += x; // coalesce
        } else {
            Ai[pos] = i;
            Ax[pos] = x;
            next[j] += 1;
        }
    }

    // Compact columns (remove holes from coalescing)
    let mut write_ptr = 0usize;
    for j in 0..n {
        let start = col_ptr[j];
        let stop = next[j];
        if write_ptr != start {
            Ai.copy_within(start..stop, write_ptr);
            Ax.copy_within(start..stop, write_ptr);
        }
        col_ptr[j] = write_ptr;
        write_ptr += stop - start;
    }
    col_ptr[n] = write_ptr;
    Ai.truncate(write_ptr);
    Ax.truncate(write_ptr);

    CscMatrix::new(n, col_ptr, Ai, Ax)
}

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

#[cfg(any(test, debug_assertions))]
fn rel_frob(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    let nrm = b.norm();
    let diff = (a - b).norm();
    diff / nrm.max(1.0)
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

        // // --- Gather UPPER: A(k < j, j) from at_upper -> y[k]
        // for p in at_upper.col_ptr[j]..at_upper.col_ptr[j + 1] {
        //     let k = at_upper.row_ind[p];
        //     if k >= j {
        //         continue;
        //     } // strictly upper only
        //     let v = at_upper.values[p]; // equals A(k,j) == A(j,k)
        //     if ymark[k] != stamp {
        //         y[k] = v;
        //         ymark[k] = stamp;
        //         touched.push(k);
        //     } else {
        //         y[k] += v;
        //     }
        // }

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
                        "[LDLt] ereach says k={} contributes to j={}, but L(:,{}) has no row {}",
                        k, j, k, j
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
            eprintln!("[LDLt] non-finite pivot at j={}", j);
            return Err("LDLt (SPD) failed: non-finite pivot");
        }
        max_abs_pivot = f64::max(max_abs_pivot, djj.abs());
        let tau = f64::max(max_abs_pivot, 1.0) * tol_rel;
        if djj <= tau {
            #[cfg(debug_assertions)]
            eprintln!(
                "[LDLt] non-positive/too-small pivot at j={} (djj={:.3e}, tau={:.3e})",
                j, djj, tau
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
fn ldlt_solve_csc(L: &CscMatrix, d: &[f64], b: &DVector<f64>) -> DVector<f64> {
    let n = L.n;
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(d.len(), n);

    let mut x = b.clone_owned();

    // Forward: L y = b (unit-lower)
    for j in 0..n {
        let t = x[j];
        for p in L.col_ptr[j]..L.col_ptr[j + 1] {
            let i = L.row_ind[p]; // i > j
            x[i] -= L.values[p] * t;
        }
    }

    // Diagonal: z = D^{-1} y
    for i in 0..n {
        x[i] /= d[i];
    }

    // Backward: Lᵀ x = z
    for j in (0..n).rev() {
        for p in L.col_ptr[j]..L.col_ptr[j + 1] {
            let i = L.row_ind[p]; // i > j
            x[j] -= L.values[p] * x[i];
        }
    }
    x
}

/// Top-level: factor A (dense) as sparse LDLᵀ and solve.
fn sparse_ldlt_solve_dense_spd(
    mat_a: &DMatrix<f64>,
    b: &DVector<f64>,
    tol_rel: f64,
) -> Result<DVector<f64>, LinearSolverError> {
    let (n, ii, jj, xx) = lower_triplets_from_dense(mat_a);
    if b.len() != n {
        return Err(LinearSolverError::DimensionMismatch);
    }

    // Lower CSC of A (numerics)
    let a_lower = triplets_to_csc(n, ii, jj, xx);

    // Upper via transpose (symbolics + to gather A(k,j))
    let at = csc_transpose(&a_lower);

    // Elimination tree from the **upper** structure
    let parent = elimination_tree_upper(&at);

    // Numeric LDLᵀ (SPD)
    let (L, d) = ldlt_numeric_spd(&a_lower, &at, &parent, tol_rel)
        .map_err(|_| LinearSolverError::FactorizationFailed)?;

    #[cfg(any(test, debug_assertions))]
    {
        let Ahat = reconstruct_dense_from_ldl(&L, &d);
        let rel = rel_frob(&Ahat, mat_a);
        eprintln!("[LDLt] ||Â - A||_F / ||A||_F = {:.3e}", rel);
        // Optional: if rel is large, dump the first mismatching column
        if rel > 1e-10 {
            for j in 0..mat_a.nrows() {
                for k in 0..j {
                    // check whether j is present in L(:,k)
                    // 2) Check the slice of row indices for this column
                    let present = L.row_ind[L.col_ptr[k]..L.col_ptr[k + 1]]
                        .iter()
                        .any(|&r| r == j);
                    if !present {
                        // this is frequently the smoking gun
                        // (only print a few lines)
                        eprintln!(
                            "[LDLt] missing row {} in L(:,{}), A({}, {}) = {:.3e}",
                            j,
                            k,
                            j,
                            k,
                            mat_a[(j, k)]
                        );
                        break;
                    }
                }
            }
        }
    }

    // Identity permutation solve (P = I)
    let x = ldlt_solve_csc(&L, &d, b);
    Ok(x)
}

/// Public: a sparse LDLᵀ solver that matches your trait and dense API.
/// For SPD matrices; no pivoting; identity ordering (add AMD later if needed).
pub struct SparseLDLt;

impl IsDenseLinearSystem for SparseLDLt {
    fn solve_dense(
        &self,
        mat_a: DMatrix<f64>,
        b: &DVector<f64>,
    ) -> Result<DVector<f64>, LinearSolverError> {
        // Tiny relative tolerance akin to your dense solver
        let tol_rel = 1e-12_f64;
        sparse_ldlt_solve_dense_spd(&mat_a, b, tol_rel)
    }
}

/* ===========================
Optional helpers & tests
=========================== */

#[cfg(test)]
mod tests {
    use nalgebra::{
        DMatrix,
        DVector,
    };

    use super::*;

    fn make_spd_from_dense(n: usize, lam: f64) -> DMatrix<f64> {
        // Deterministic dense R
        let mut r = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                r[(i, j)] = ((i + 1) as f64) * ((j + 2) as f64) / ((i + j + 1) as f64);
            }
        }
        r.transpose() * &r + lam * DMatrix::<f64>::identity(n, n)
    }

    #[test]
    fn sparse_ldlt_spd_basic() {
        let n = 16;
        let a = make_spd_from_dense(n, 1e-3);
        let mut b = DVector::<f64>::zeros(n);
        for i in 0..n {
            b[i] = (i as f64) - 0.5;
        }

        let solver = SparseLDLt;
        let x = solver.solve_dense(a.clone(), &b).unwrap();

        // Residual
        let r = &a * &x - &b;
        let rel_res = r.norm() / b.norm().max(1.0);
        assert!(rel_res < 1e-9, "residual too large: {}", rel_res);
    }
}
