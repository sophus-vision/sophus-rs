use std::collections::HashMap;

use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    BlockSparseLowerMatrixBuilder,
    IsLinearSolver,
    block_csc_matrix::{
        BlockCscMatrix,
        BlockRegion,
    },
    grid::Grid,
    psd_solver::elimination_tree::{
        EliminationTree,
        elimination_tree_upper,
    },
    sparse::CscStruct,
};

#[derive(Copy, Clone, Debug)]
pub struct BlockSparseLdlt2 {}

// impl IsLinearSolver for BlockSparseLdlt2 {
//     type Matrix = BlockSparseLowerMatrixBuilder;

//     const NAME: &'static str = "block sparse LDLt";

//     fn solve_in_place(
//         &self,
//         a_lower: &BlockSparseLowerCompressedMatrix,
//         b: &mut nalgebra::DVector<f64>,
//     ) -> Result<(), LinearSolverError> {
//         phase("solve_in_place/before");

//         let mut ldlt = BlockLDLt::structure_from_cbm(&a_lower.mat);
//         phase("solve_in_place/ldlt");

//         ldlt.factorize_left_looking(&a_lower.mat);
//         phase("solve_in_place/factorize_left_looking");

//         ldlt.solve_in_place(b);
//         phase("solve_in_place/solve_in_place");

//         Ok(())
//     }
// }

#[inline]
fn check_stride_ok(region: &BlockRegion, base: usize, need: usize, where_: &str) {
    debug_assert!(
        base + need <= region.storage.len(),
        "[{}] region payload OOB: base={} need={} len={}",
        where_,
        base,
        need,
        region.storage.len()
    );
}

/// y = A * x, where `a` stores **lower** blocks only.
/// - Diagonal: y_j += A_jj * x_j
/// - Off-diag (i>j): y_i += A_ij * x_j  and  y_j += A_ij^T * x_i
pub fn sym_block_mv(a: &BlockCscMatrix, x: &DVector<f64>) -> DVector<f64> {
    assert_eq!(x.len(), a.nrows, "x length must match matrix rows");
    let nb = a.col_splits.len() - 1;
    assert_eq!(nb, a.row_splits.len() - 1, "square block grid required");

    let mut y = DVector::zeros(a.nrows);

    for j in 0..nb {
        let h_j = a.row_splits[j + 1] - a.row_splits[j];
        let xj = x.rows(a.row_splits[j], h_j);

        // ---------- Diagonal block A_jj ----------
        if let Ok(pos) = a.row_ind[a.col_ptr[j]..a.col_ptr[j + 1]].binary_search(&(j as u32)) {
            let k = a.col_ptr[j] + pos;
            let rj = a.row_class_of_br[j] as usize;
            let cj = a.col_class_of_bc[j] as usize;

            let stride = h_j * h_j;
            let base = (a.region_loc[k] as usize) * stride;
            let reg = a.regions.get(&[rj, cj]);
            check_stride_ok(reg, base, stride, "sym_block_mv diag");

            let Ajj = DMatrix::from_column_slice(h_j, h_j, &reg.storage[base..base + stride]);
            let mut yj = y.rows_mut(a.row_splits[j], h_j);
            yj.gemv(1.0, &Ajj, &xj, 1.0);
        }

        // ---------- Off-diagonals A_ij with i > j (stored in column j) ----------
        let start = a.col_ptr[j];
        let end = a.col_ptr[j + 1];
        for k in start..end {
            let i = a.row_ind[k] as usize;
            if i <= j {
                continue;
            }

            let h_i = a.row_splits[i + 1] - a.row_splits[i];
            let ri = a.row_class_of_br[i] as usize;
            let cj = a.col_class_of_bc[j] as usize;

            let stride = h_i * h_j;
            let base = (a.region_loc[k] as usize) * stride;
            let reg = a.regions.get(&[ri, cj]);
            check_stride_ok(reg, base, stride, "sym_block_mv offdiag");

            let Aij = DMatrix::from_column_slice(h_i, h_j, &reg.storage[base..base + stride]);

            // y_i += A_ij * x_j
            {
                let mut yi = y.rows_mut(a.row_splits[i], h_i);
                yi.gemv(1.0, &Aij, &xj, 1.0);
            }

            // y_j += A_ij^T * x_i  (upper contribution)
            {
                let xi = x.rows(a.row_splits[i], h_i);
                let mut yj = y.rows_mut(a.row_splits[j], h_j);
                yj.gemv(1.0, &Aij.transpose(), &xi, 1.0);
            }
        }
    }

    y
}

fn get_L_block<'a>(lf: &'a BlockLowerFactor, i: usize, k: usize) -> Option<DMatrix<f64>> {
    let L = &lf.L_off;

    // Column k in L_off is sorted by row (lbuilder_finish did that).
    let (start, end) = (L.col_ptr[k], L.col_ptr[k + 1]);
    if start == end {
        return None;
    }
    let rows = &L.row_ind[start..end];
    let pos = rows.binary_search(&(i as u32)).ok()?;
    let idx = start + pos;

    // L_off regions are indexed by (i, k) in an nb×nb grid (NOT class-bucketed).
    let h_i = L.row_splits[i + 1] - L.row_splits[i];
    let h_k = L.row_splits[k + 1] - L.row_splits[k];
    let stride = h_i * h_k;
    let base = (L.region_loc[idx] as usize) * stride;
    let reg = L.regions.get(&[i, k]); // nb×nb for L
    debug_assert!(base + stride <= reg.storage.len(), "L block OOB in factor");
    Some(DMatrix::from_column_slice(
        h_i,
        h_k,
        &reg.storage[base..base + stride],
    ))
}

fn reconstruct_A_block_from_factor(
    lf: &BlockLowerFactor,
    d: &BlockDiagLdlt,
    i: usize,
    j: usize,
) -> DMatrix<f64> {
    let A = &lf.L_off;
    let h_i = A.row_splits[i + 1] - A.row_splits[i];
    let h_j = A.row_splits[j + 1] - A.row_splits[j];
    let mut ahat = DMatrix::<f64>::zeros(h_i, h_j);

    let min_k = usize::min(i, j);

    for k in 0..=min_k {
        let Ldk = &d.chol[k];
        // S_{pk} := L_{pk} * Ldk  with unit diagonal for p == k (=> I * Ldk)
        let Sik = if i == k {
            Ldk.clone_owned()
        } else if let Some(Lik) = get_L_block(lf, i, k) {
            &Lik * Ldk
        } else {
            continue; // L_{ik} structurally zero => contributes nothing
        };

        let Sjk = if j == k {
            Ldk.clone_owned()
        } else if let Some(Ljk) = get_L_block(lf, j, k) {
            &Ljk * Ldk
        } else {
            continue;
        };

        ahat += &Sik * Sjk.transpose();
    }

    ahat
}

pub fn debug_compare_block_identities(
    a: &BlockCscMatrix,
    lf: &BlockLowerFactor,
    d: &BlockDiagLdlt,
    pairs: &[(usize, usize)],
) {
    for &(i, j) in pairs {
        let aij = a_block_copy(a, i, j).expect("A(i,j) present");
        let ahat = reconstruct_A_block_from_factor(lf, d, i, j);

        let diff = &ahat - &aij;
        eprintln!(
            "[CHK] A({i},{j})  ||A||_F={:.3e}  ||Â||_F={:.3e}  ||Â-A||_F={:.3e}  max|·|={:.3e}",
            aij.norm(),
            ahat.norm(),
            diff.norm(),
            diff.amax()
        );
        if diff.amax() > 1e-10 {
            eprintln!("  A(i,j)=\n{aij}");
            eprintln!("  Â(i,j)=\n{ahat}");
            eprintln!("  Δ=\n{diff}");
        }
    }
}

/// Off-diagonal part of the block-lower factor L_b (unit diagonal).
#[derive(Debug)]
pub struct BlockLowerFactor {
    pub L_off: BlockCscMatrix, // same partitions as A; only i>j blocks populated
}

/// Diagonal for LDLᵀ: store chol(D_j) (lower-triangular) per block column j.
#[derive(Debug)]
pub struct BlockDiagLdlt {
    pub chol: Vec<DMatrix<f64>>, // L_{D_j}, s.t. D_j = L_{D_j} L_{D_j}^T
}
#[derive(Debug)]
pub enum BlockLdltError {
    NonSpdPivot { j: usize },
    MissingDiagonal { j: usize },
}

/// Strictly-upper *block* structure of A (no values); n = #global block columns.
/// Strictly-upper *block* structure of A built from the **transpose** of lower storage.
/// For every stored (i>j) in lower column j, we insert j into **column i** of the upper.
fn build_block_upper(a: &BlockCscMatrix) -> CscStruct {
    let n = a.col_splits.len() - 1;
    assert_eq!(
        n,
        a.row_splits.len() - 1,
        "square block partitions required"
    );

    // Count entries for each upper column 'i'
    let mut counts = vec![0usize; n];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_ind[k] as usize;
            if i > j {
                counts[i] += 1;
            } // upper column i gets row j
        }
    }

    // Prefix sum
    let mut col_ptr = vec![0usize; n + 1];
    for i in 0..n {
        col_ptr[i + 1] = col_ptr[i] + counts[i];
    }
    let mut next = col_ptr.clone();
    let mut row_ind = vec![0usize; col_ptr[n]];

    // Fill: column i collects all j<i with A(i,j) present
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_ind[k] as usize;
            if i > j {
                let dst = next[i];
                row_ind[dst] = j; // strictly upper row index
                next[i] += 1;
            }
        }
    }
    CscStruct {
        n,
        col_ptr,
        row_ind,
    }
}

// /// Block elimination tree (upper structure).
// fn elimination_tree_upper_struct(at: &CscStruct) -> Vec<usize> {
//     let n = at.n;
//     let ap = &at.col_ptr;
//     let ai = &at.row_ind;
//     let mut parent = vec![usize::MAX; n];
//     let mut ancestor = vec![usize::MAX; n];

//     for j in 0..n {
//         for p in ap[j]..ap[j + 1] {
//             let mut i = ai[p]; // i<j
//             while i != usize::MAX && i != j {
//                 let next = ancestor[i];
//                 ancestor[i] = j;
//                 if next == usize::MAX {
//                     parent[i] = j;
//                     break;
//                 }
//                 i = next;
//             }
//         }
//     }
//     parent
// }

// /// Symbolic reach on upper structure; returns top so stack[top..n] are contributors k<j in topo
// /// order.
// fn ereach_upper_struct(
//     at: &CscStruct,
//     j: usize,
//     parent: &[usize],
//     w: &mut [usize],
//     stack: &mut [usize],
// ) -> usize {
//     let n = at.n;
//     let ap = &at.col_ptr;
//     let ai = &at.row_ind;
//     let mark = j + 1;
//     let mut top = n;

//     w[j] = mark;
//     for p in ap[j]..ap[j + 1] {
//         let mut i = ai[p];
//         while i != usize::MAX && w[i] != mark {
//             stack[top - 1] = i;
//             top -= 1;
//             w[i] = mark;
//             i = parent[i];
//         }
//     }
//     top
// }

/// Copy dense block A_{ij} (h_i × h_j) if present.
/// If i<j, mirrors from lower: A_{ij} = A_{ji}^T.
/// Copy dense block A_{ij} (h_i × h_j) if present.
/// If i<j, mirrors from lower: A_{ij} = A_{ji}^T.
fn a_block_copy(a: &BlockCscMatrix, i: usize, j: usize) -> Option<DMatrix<f64>> {
    let (ii, jj, transpose) = if i >= j { (i, j, false) } else { (j, i, true) };

    let h = a.row_splits[ii + 1] - a.row_splits[ii];
    let w = a.col_splits[jj + 1] - a.col_splits[jj];

    // Find block (ii, jj) in block column jj
    let (start, end) = (a.col_ptr[jj], a.col_ptr[jj + 1]);
    let br = &a.row_ind[start..end];
    let pos = br.binary_search(&(ii as u32)).ok()?; // not found -> None
    let k = start + pos;

    // *** Map global block indices -> region grid indices ***
    let ri = a.row_class_of_br[ii] as usize; // 0..R-1
    let cj = a.col_class_of_bc[jj] as usize; // 0..C-1

    // Region-local payload
    let base = (a.region_loc[k] as usize) * (h * w);
    let reg = a.regions.get(&[ri, cj]);
    debug_assert!(
        base + h * w <= reg.storage.len(),
        "region_loc points past region storage (stride mismatch?)"
    );

    let dat = &reg.storage[base..base + h * w];

    let m = DMatrix::from_column_slice(h, w, dat);
    Some(if !transpose { m } else { m.transpose() })
}

#[derive(Debug)]
struct LBuilder {
    col_rows: Vec<Vec<usize>>,             // per block column j: rows i>j
    col_loc: Vec<Vec<u32>>,                // per j: region_loc for each i
    col_row2loc: Vec<HashMap<usize, u32>>, // for lookup L_{jk} inside column k
    regions: Grid<BlockRegion>,            // payloads for L (off-diagonals)
}

fn lbuilder_empty_like(a: &BlockCscMatrix) -> LBuilder {
    let nb = a.col_splits.len() - 1;
    LBuilder {
        col_rows: vec![Vec::new(); nb],
        col_loc: vec![Vec::new(); nb],
        col_row2loc: vec![HashMap::new(); nb],
        regions: Grid::new(
            [nb, nb],
            BlockRegion {
                storage: Vec::new(),
            },
        ),
    }
}

/// Append L_{ij} (i>j), `lij` is h_i×h_j (column-major).
fn lbuilder_append(a: &BlockCscMatrix, lb: &mut LBuilder, i: usize, j: usize, lij: &DMatrix<f64>) {
    let h = a.row_splits[i + 1] - a.row_splits[i];
    let w = a.col_splits[j + 1] - a.col_splits[j];
    debug_assert_eq!(lij.nrows(), h);
    debug_assert_eq!(lij.ncols(), w);

    let reg = lb.regions.get_mut(&[i, j]);
    let loc = (reg.storage.len() / (h * w)) as u32;
    for c in 0..w {
        reg.storage.extend_from_slice(lij.column(c).as_slice());
    }
    lb.col_rows[j].push(i);
    lb.col_loc[j].push(loc);
    lb.col_row2loc[j].insert(i, loc);
}

/// Finalize builder -> compressed BlockCscMatrix (off-diagonal only, rows sorted within columns).
fn lbuilder_finish(a: &BlockCscMatrix, lb: LBuilder) -> BlockCscMatrix {
    let nbcols = a.col_splits.len() - 1;

    // Build col_ptr, row_ind, region_loc
    let mut col_ptr = Vec::with_capacity(nbcols + 1);
    col_ptr.push(0);

    let mut row_ind = Vec::<u32>::new();
    let mut region_loc = Vec::<u32>::new();

    for j in 0..nbcols {
        // If you appended rows in arbitrary order, sort (i,loc) before emitting:
        let mut pairs: Vec<(usize, u32)> = lb.col_rows[j]
            .iter()
            .cloned()
            .zip(lb.col_loc[j].iter().cloned())
            .collect();
        pairs.sort_unstable_by_key(|p| p.0);

        for (i, loc) in pairs {
            row_ind.push(i as u32);
            region_loc.push(loc);
        }
        col_ptr.push(row_ind.len());
    }

    BlockCscMatrix {
        // scalar dims & splits match the input
        nrows: a.nrows,
        ncols: a.ncols,
        row_splits: a.row_splits.clone(),
        col_splits: a.col_splits.clone(),

        // block-level CSC
        col_ptr,
        row_ind,
        region_loc,

        // *** the two missing fields: reuse the same class maps as the input ***
        row_class_of_br: a.row_class_of_br.clone(),
        col_class_of_bc: a.col_class_of_bc.clone(),

        // region payloads for the factor (off-diagonal cells contain L_ij)
        regions: lb.regions, // diagonal cells unused for L (they live in D)
    }
}

/// X := Y * U^{-1}, U upper-triangular (solve x_r U = y_r per row, forward in j)
fn right_solve_upper(U: &DMatrix<f64>, Y: &DMatrix<f64>) -> DMatrix<f64> {
    let m = Y.nrows();
    let n = Y.ncols();
    debug_assert_eq!(U.nrows(), n);
    debug_assert_eq!(U.ncols(), n);
    let mut X = DMatrix::<f64>::zeros(m, n);
    for r in 0..m {
        let y = Y.row(r);
        let mut x = vec![0.0f64; n];
        for j in 0..n {
            // FORWARD for U
            let mut s = y[j];
            for t in 0..j {
                s -= x[t] * U[(t, j)];
            }
            x[j] = s / U[(j, j)];
        }
        for j in 0..n {
            X[(r, j)] = x[j];
        }
    }
    X
}

/// X := Y * L^{-1}, L lower-triangular (solve x_r L = y_r per row, backward in j)
fn right_solve_lower(L: &DMatrix<f64>, Y: &DMatrix<f64>) -> DMatrix<f64> {
    let m = Y.nrows();
    let n = Y.ncols();
    debug_assert_eq!(L.nrows(), n);
    debug_assert_eq!(L.ncols(), n);
    let mut X = DMatrix::<f64>::zeros(m, n);
    for r in 0..m {
        let y = Y.row(r);
        let mut x = vec![0.0f64; n];
        for j in (0..n).rev() {
            // BACKWARD for L
            let mut s = y[j];
            for t in (j + 1)..n {
                s -= x[t] * L[(t, j)];
            }
            x[j] = s / L[(j, j)];
        }
        for j in 0..n {
            X[(r, j)] = x[j];
        }
    }
    X
}

/// Factorization: A = L_b D_b L_b^T, storing L_b off-diagonals and chol(D_j).
pub fn block_ldlt(
    a: &BlockCscMatrix,
    tol_rel: f64,
) -> Result<(BlockLowerFactor, BlockDiagLdlt), BlockLdltError> {
    let nb = a.col_splits.len() - 1;
    assert_eq!(
        nb,
        a.row_splits.len() - 1,
        "square block partitions required"
    );

    // Symbolics
    let at = build_block_upper(a);
    let parent = elimination_tree_upper(&at);
    let mut w = vec![0usize; nb];
    let mut stk = vec![0usize; nb];

    // Factor under construction
    let mut lb = lbuilder_empty_like(a);
    let mut cholD: Vec<DMatrix<f64>> = Vec::with_capacity(nb); // L_{D_j}

    // Track largest pivot magnitude for a simple relative threshold (optional)
    let mut max_diag_norm = 0.0f64;

    for j in 0..nb {
        let h_j = a.row_splits[j + 1] - a.row_splits[j];

        // --- 1) Seed diagonal & below-diagonal from A ---
        let mut Yjj = a_block_copy(a, j, j).ok_or(BlockLdltError::MissingDiagonal { j })?;

        // Off-diagonal accumulators: Y(i,j) for i>j
        let mut Ymap: HashMap<usize, DMatrix<f64>> = HashMap::new();
        let mut touched: Vec<usize> = Vec::new();

        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_ind[k] as usize;
            if i == j {
                continue;
            }
            if i > j {
                let aij = a_block_copy(a, i, j).expect("present by construction (lower)");
                let entry = Ymap.entry(i).or_insert_with(|| {
                    touched.push(i);
                    let h_i = a.row_splits[i + 1] - a.row_splits[i];
                    DMatrix::<f64>::zeros(h_i, h_j)
                });
                *entry += aij;
            }
        }

        // --- 2) Symbolic reach: contributing earlier columns k ---
        let top = EliminationTree::reach(&at, j, &parent.parent, &mut w, &mut stk);

        // --- 3) Apply updates from each k in topo order ---
        for idx in top..nb {
            let k = stk[idx];

            // L_{jk} must already be computed (since k<j). Look it up in column k.
            let l_jk = match lb.col_row2loc[k].get(&j) {
                None => continue, // structurally zero; nothing to subtract
                Some(&lc) => {
                    let h_k = a.row_splits[k + 1] - a.row_splits[k];
                    let stride = h_j * h_k;
                    let base = (lc as usize) * stride;
                    let reg = lb.regions.get(&[j, k]);
                    let dat = &reg.storage[base..base + stride];
                    DMatrix::from_column_slice(h_j, h_k, dat) // L_{jk}
                }
            };

            // Get L_{D_k} (chol of D_k)
            let Ldk = &cholD[k]; // h_k × h_k, lower-triangular

            // Q_jk = L_{jk} * L_{D_k}
            let Qjk = &l_jk * Ldk;

            // (a) Diagonal update: Yjj -= Q_jk * Q_jk^T
            Yjj = &Yjj - &(&Qjk * Qjk.transpose());

            // (b) Off-diagonal updates: for all i>k in column k, Y(i) -= (L_{ik} Ldk) (Qjk)^T
            let rows_k = &lb.col_rows[k];
            let locs_k = &lb.col_loc[k];
            for (&i, &loc_ik) in rows_k.iter().zip(locs_k.iter()) {
                if i < j {
                    continue;
                }
                let h_i = a.row_splits[i + 1] - a.row_splits[i];
                let h_k = a.row_splits[k + 1] - a.row_splits[k];
                let stride = h_i * h_k;
                let base = (loc_ik as usize) * stride;
                let reg = lb.regions.get(&[i, k]);
                let dat = &reg.storage[base..base + stride];
                let l_ik = DMatrix::from_column_slice(h_i, h_k, dat); // L_{ik}

                // Q_ik = L_{ik} * Ldk
                let Qik = &l_ik * Ldk;

                let entry = Ymap.entry(i).or_insert_with(|| {
                    touched.push(i);
                    DMatrix::<f64>::zeros(h_i, h_j)
                });
                *entry = &*entry - &(&Qik * Qjk.transpose());
            }
        }

        // --- 4) Pivot: D_j = Yjj (must be SPD). Store chol(D_j).
        // Simple robustness: relative threshold check on diag norm
        let diag_norm = Yjj.diagonal().amax();
        max_diag_norm = max_diag_norm.max(diag_norm.abs());
        let tau = max_diag_norm.max(1.0) * tol_rel;
        // (you can add a tiny diagonal damping here if desired)
        let chol = Yjj.cholesky().ok_or(BlockLdltError::NonSpdPivot { j })?;
        let Ldj = chol.l().clone_owned(); // lower-triangular
        cholD.push(Ldj.clone());

        // --- 5) Finalize off-diagonals: L_{ij} = Y(i) * D_j^{-1}
        // Use two right solves: X = Y(i) * inv(Ldj^T);  L_{ij} = X * inv(Ldj)
        // Ensure ascending i before appending (helps downstream)
        touched.sort_unstable();
        for &i in &touched {
            if i <= j {
                continue;
            }
            let Yij = Ymap.remove(&i).unwrap(); // h_i × h_j

            let X = right_solve_upper(&Ldj.transpose(), &Yij); // X = Y * inv(Ldj^T)
            let Lij = right_solve_lower(&Ldj, &X); // Lij = X * inv(Ldj)

            lbuilder_append(a, &mut lb, i, j, &Lij);
        }

        // ---- DEBUG at end of column j ----
        if cfg!(debug_assertions) {
            eprintln!(
                "[DBG] col {j}: D_j (chol) diag = {:?}",
                Ldj.diagonal().transpose()
            );
            for &i in &touched {
                if i > j {
                    let h_i = a.row_splits[i + 1] - a.row_splits[i];
                    let h_j = a.row_splits[j + 1] - a.row_splits[j];
                    let reg = lb.regions.get(&[i, j]);
                    let loc = *lb.col_row2loc[j].get(&i).expect("loc present after append");
                    let base = (loc as usize) * (h_i * h_j);
                    check_stride_ok(reg, base, h_i * h_j, "factor col dbg");
                    let Lij =
                        DMatrix::from_column_slice(h_i, h_j, &reg.storage[base..base + h_i * h_j]);
                    eprintln!(
                        "  L[{i},{j}] frob={:.3e} max={:.3e}",
                        Lij.norm(),
                        Lij.amax()
                    );
                }
            }
        }
    }

    Ok((
        BlockLowerFactor {
            L_off: lbuilder_finish(a, lb),
        },
        BlockDiagLdlt { chol: cholD },
    ))
}

pub fn factor_apply(lf: &BlockLowerFactor, d: &BlockDiagLdlt, x: &DVector<f64>) -> DVector<f64> {
    let a = &lf.L_off;
    let nb = a.col_splits.len() - 1;

    // y = L^T x
    let mut y = x.clone();
    for j in (0..nb).rev() {
        let h_j = a.row_splits[j + 1] - a.row_splits[j];
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_ind[k] as usize;
            if i <= j {
                continue;
            }
            let h_i = a.row_splits[i + 1] - a.row_splits[i];
            let stride = h_i * h_j;
            let base = (a.region_loc[k] as usize) * stride;
            let reg = a.regions.get(&[i, j]);
            let Lij = DMatrix::from_column_slice(h_i, h_j, &reg.storage[base..base + stride]);

            // y_j += L_ij^T * x_i  (use x_i, not y_i; we cloned x into y, so y_i==x_i here)
            let xi = y.rows(a.row_splits[i], h_i).into_owned();
            let mut yj = y.rows_mut(a.row_splits[j], h_j);
            yj.gemv(1.0, &Lij.transpose(), &xi, 1.0);
        }
    }

    // u = D y   (D_j = Ldj Ldj^T)
    let mut u = y.clone();
    for j in 0..nb {
        let Ldj = &d.chol[j];
        let h_j = a.row_splits[j + 1] - a.row_splits[j];
        let yj = y.rows(a.row_splits[j], h_j).into_owned();
        let w = Ldj.transpose() * yj;
        let uj = Ldj * w;
        u.rows_mut(a.row_splits[j], h_j).copy_from(&uj);
    }

    // v = L u
    let mut v = u.clone();
    for j in 0..nb {
        let h_j = a.row_splits[j + 1] - a.row_splits[j];
        let uj = v.rows(a.row_splits[j], h_j).into_owned();
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_ind[k] as usize;
            if i <= j {
                continue;
            }
            let h_i = a.row_splits[i + 1] - a.row_splits[i];
            let stride = h_i * h_j;
            let base = (a.region_loc[k] as usize) * stride;
            let reg = a.regions.get(&[i, j]);
            let Lij = DMatrix::from_column_slice(h_i, h_j, &reg.storage[base..base + stride]);

            // v_i += L_ij * u_j
            let mut vi = v.rows_mut(a.row_splits[i], h_i);
            vi.gemv(1.0, &Lij, &uj, 1.0);
        }
    }
    v
}

/// Compare A*x with (L D L^T)*x for a few test vectors; print max discrepancy.
pub fn debug_check_factor_matches_a(
    a: &BlockCscMatrix,
    lf: &BlockLowerFactor,
    df: &BlockDiagLdlt,
    dense_a: Option<&DMatrix<f64>>, // if you have a dense reference, pass it
) {
    let nb = a.col_splits.len() - 1;
    let mut probes: Vec<DVector<f64>> = Vec::new();

    // 3 probes: random-ish and basis on last block
    let mut v1 = DVector::from_element(a.nrows, 0.0);
    v1.iter_mut()
        .enumerate()
        .for_each(|(k, x)| *x = ((k * 97) % 13) as f64 * 0.1 + 1.0);
    probes.push(v1);
    let mut v2 = DVector::from_element(a.nrows, 0.0);
    let j_last = nb - 1;
    let (r0, r1) = (a.row_splits[j_last], a.row_splits[j_last + 1]);
    for t in r0..r1 {
        v2[t] = 1.0;
    }
    probes.push(v2);
    let mut v3 = DVector::from_element(a.nrows, 0.0);
    v3[0] = 1.0;
    probes.push(v3);

    for (pi, v) in probes.iter().enumerate() {
        let lhs = if let Some(DA) = dense_a {
            DA * v
        } else {
            sym_block_mv(a, v)
        };
        let rhs = factor_apply(lf, df, v);
        let diff = (&lhs - &rhs).norm();
        eprintln!(
            "[DBG] factor_vs_A probe#{pi}: ||(A - LDLT)v|| = {:.3e}",
            diff
        );
        if diff > 1e-8 {
            eprintln!(
                "  (block last, lhs segment) = {:?}",
                lhs.rows(r0, r1 - r0).as_slice()
            );
            eprintln!(
                "  (block last, rhs segment) = {:?}",
                rhs.rows(r0, r1 - r0).as_slice()
            );
        }
    }
}

/// Forward/Backward helpers on vectors
fn forward_solve_lower_in_place(L: &DMatrix<f64>, y: &mut nalgebra::DVectorSliceMut<'_, f64>) {
    let n = L.nrows();
    for i in 0..n {
        let mut s = y[i];
        for k in 0..i {
            s -= L[(i, k)] * y[k];
        }
        y[i] = s / L[(i, i)];
    }
}
fn backward_solve_upper_in_place(U: &DMatrix<f64>, y: &mut nalgebra::DVectorSliceMut<'_, f64>) {
    let n = U.nrows();
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= U[(i, k)] * y[k];
        }
        y[i] = s / U[(i, i)];
    }
}

#[inline]
fn mat_norms(label: &str, m: &DMatrix<f64>) {
    eprintln!(
        "{label}: (m={},n={})  ||·||_F={:.3e}  ||·||_∞={:.3e}",
        m.nrows(),
        m.ncols(),
        m.norm(),
        m.amax()
    );
}

#[inline]
fn vec_block_dump(a: &BlockCscMatrix, label: &str, v: &DVector<f64>) {
    let nb = a.col_splits.len() - 1;
    eprintln!("{label}:");
    for j in 0..nb {
        let (r0, r1) = (a.row_splits[j], a.row_splits[j + 1]);
        eprintln!(
            "  block {j} [{}..{}): {:?}",
            r0,
            r1,
            &v.rows(r0, r1 - r0).as_slice()
        );
    }
}

#[inline]
fn block_range(a: &BlockCscMatrix, j: usize) -> (usize, usize) {
    (a.row_splits[j], a.row_splits[j + 1])
}

#[inline]
fn block_ro<'a>(
    a: &BlockCscMatrix,
    j: usize,
    v: &'a DVector<f64>,
) -> nalgebra::DVectorSlice<'a, f64> {
    let (r0, r1) = block_range(a, j);
    v.rows(r0, r1 - r0)
}

#[inline]
fn block_mut<'a>(
    a: &BlockCscMatrix,
    j: usize,
    v: &'a mut DVector<f64>,
) -> nalgebra::DVectorSliceMut<'a, f64> {
    let (r0, r1) = block_range(a, j);
    v.rows_mut(r0, r1 - r0)
}
pub fn solve_block_ldlt_in_place(f: &BlockLowerFactor, d: &BlockDiagLdlt, b: &mut DVector<f64>) {
    let a = &f.L_off;
    let nb = a.col_splits.len() - 1;
    assert_eq!(a.nrows, a.ncols);
    assert_eq!(b.len(), a.nrows);

    let b_orig = b.clone();

    // -------- 1) Forward: L_b y = b (unit block-lower; NO diagonal here) --------
    for j in 0..nb {
        let yj_owned = block_ro(a, j, b).into_owned();
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_ind[k] as usize;
            if i <= j {
                continue;
            }

            let h_i = a.row_splits[i + 1] - a.row_splits[i];
            let h_j = a.row_splits[j + 1] - a.row_splits[j];
            let stride = h_i * h_j;
            let base = (a.region_loc[k] as usize) * stride;
            let reg = a.regions.get(&[i, j]); // L is nb×nb grid
            check_stride_ok(reg, base, stride, "solve forward");
            let Lij = DMatrix::from_column_slice(h_i, h_j, &reg.storage[base..base + stride]);

            let mut bi = block_mut(a, i, b);
            bi.gemv(-1.0, &Lij, &yj_owned, 1.0);
        }
    }
    if cfg!(debug_assertions) {
        vec_block_dump(a, "[DBG] after forward (y)", b);
    }

    // -------- 2) Diagonal: D_b z = y  (apply D^{-1} to each block) --------
    for j in 0..nb {
        let Ldj = &d.chol[j];
        {
            let mut bj = block_mut(a, j, b);
            forward_solve_lower_in_place(Ldj, &mut bj);
        }
        {
            let mut bj = block_mut(a, j, b);
            backward_solve_upper_in_place(&Ldj.transpose(), &mut bj);
        }
    }
    if cfg!(debug_assertions) {
        vec_block_dump(a, "[DBG] after diagonal (z = D^{-1} y)", b);
    }

    // -------- 3) Backward: L_b^T x = z --------
    for j in (0..nb).rev() {
        let (r0j, r1j) = block_range(a, j);
        let len_j = r1j - r0j;
        let mut delta = DVector::zeros(len_j);

        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_ind[k] as usize;
            if i <= j {
                continue;
            }

            let h_i = a.row_splits[i + 1] - a.row_splits[i];
            let h_j = len_j;
            let stride = h_i * h_j;
            let base = (a.region_loc[k] as usize) * stride;
            let reg = a.regions.get(&[i, j]);
            check_stride_ok(reg, base, stride, "solve backward");
            let Lij = DMatrix::from_column_slice(h_i, h_j, &reg.storage[base..base + stride]);

            let bi = block_ro(a, i, b);
            delta.fill(0.0);
            delta.gemv(1.0, &Lij.transpose(), &bi, 0.0);

            {
                let mut bj = block_mut(a, j, b);
                bj.axpy(-1.0, &delta, 1.0);
            }
        }
    }
    if cfg!(debug_assertions) {
        vec_block_dump(a, "[DBG] after backward (x)", b);
    }

    // -------- final: residuals --------
    if cfg!(debug_assertions) {
        let fx = factor_apply(f, d, b);
        let r2 = &fx - &b_orig;
        eprintln!("[DBG] ‖(LDLᵀ) x - b‖₂ = {:.3e}", r2.norm());
    }
}
