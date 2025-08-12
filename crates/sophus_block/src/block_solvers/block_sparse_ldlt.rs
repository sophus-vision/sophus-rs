use nalgebra::{
    DMatrixView,
    DMatrixViewMut,
    DVectorViewMut,
};

use crate::{
    AssembledCol,
    BlockSparseMatrixBuilder,
    CompressedBlockMatrix,
};

impl BlockSparseMatrixBuilder {
    /// ldlt
    pub fn ldlt_solve(&self, b: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        println!("foo");
        let mut x = b.clone();
        let c = &self.to_compressed();
        let mut ldlt = BlockLDLt::structure_from_cbm(c);
        ldlt.factorize_left_looking(c);
        ldlt.solve_in_place(&mut x);
        x
    }
}

/// Block LDLt
#[derive(Debug)]
pub struct BlockLDLt {
    // n block-columns (== n block-rows); column j has size m_j
    n: usize,
    // per block: m_j
    blk_dim: Vec<usize>,

    // D as one big buffer; diag_ptr[j] is start of D_j (m_j x m_j, col-major)
    diag_ptr: Vec<usize>, // len = n+1 (prefix sum of m_j*m_j)
    d_storage: Vec<f64>,

    scalar_ptr: Vec<usize>,

    // L strictly-lower in block CSC (by column)
    // For column j, rows are i>j in ascending order.
    l_col_ptr: Vec<usize>,      // len = n+1
    l_row_idx: Vec<usize>,      // block row indices (global)
    l_entry_of_pos: Vec<usize>, // map CSC slot -> entry index
    l_storage: Vec<f64>,        // concatenated blocks, col-major (m_i x m_j)
}

impl BlockLDLt {
    /// structure_from_C
    pub fn structure_from_cbm(cbm: &CompressedBlockMatrix) -> Self {
        let n = cbm.sym_block_pattern.num_block_cols;
        // m_j from diagonal regions
        let mut blk_dim = vec![0usize; n];
        let col_off = &cbm.sym_block_pattern.block_index_offset_per_col_partition;
        for rx in 0..(col_off.len() - 1) {
            let reg = cbm.region_grid.get(&[rx, rx]);
            let m = reg.block_shape[0];
            debug_assert_eq!(m, reg.block_shape[1]);
            let j0 = col_off[rx];
            let j1 = col_off[rx + 1];
            for j in j0..j1 {
                blk_dim[j] = m;
            }
        }

        // Diagonal storage prefix sums
        let mut diag_ptr = Vec::with_capacity(n + 1);
        diag_ptr.push(0);
        for j in 0..n {
            let m = blk_dim[j];
            diag_ptr.push(diag_ptr[j] + m * m);
        }

        let mut scalar_ptr = Vec::with_capacity(n + 1);
        scalar_ptr.push(0);
        for j in 0..n {
            scalar_ptr.push(scalar_ptr[j] + blk_dim[j]);
        }

        let last = diag_ptr[n];
        Self {
            n,
            blk_dim,
            diag_ptr,
            d_storage: vec![0.0; last],

            scalar_ptr,

            // L is built numerically; allocate empty and compress later
            l_col_ptr: vec![0; n + 1],
            l_row_idx: Vec::new(),
            l_entry_of_pos: Vec::new(),
            l_storage: Vec::new(),
        }
    }

    /// Solve A x = b using the factor (L,D).
    /// Overwrites `x` in-place; `x` is a scalar vector of size sum(m_j).
    pub fn solve_in_place(&self, x: &mut nalgebra::DVector<f64>) {
        // y = b
        // 1) forward: L y = b
        self.forward_solve_in_place(x);

        // 2) diagonal: D z = y
        self.diag_solve_in_place(x);

        // 3) back: Lᵀ x = z
        self.back_solve_in_place(x);
    }

    fn forward_solve_in_place(&self, bv: &mut nalgebra::DVector<f64>) {
        let x = bv.as_mut_slice();
        for j in 0..self.n {
            let m_j = self.blk_dim[j];
            let j0 = self.scalar_ptr[j]; // see §2
            let (left, mid_right) = x.split_at_mut(j0);
            let (yj, right) = mid_right.split_at_mut(m_j);
            let _ = left;

            let p0 = self.l_col_ptr[j];
            let p1 = self.l_col_ptr[j + 1];
            for pos in p0..p1 {
                let i = self.l_row_idx[pos];
                let mi = self.blk_dim[i];
                let elem_off = self.l_entry_of_pos[pos];
                let lij = &self.l_storage[elem_off..elem_off + (mi * m_j)];

                let i0 = self.scalar_ptr[i];
                debug_assert!(i0 >= j0 + m_j); // L is strictly lower: i > j
                let rel = i0 - (j0 + m_j);
                let yi = &mut right[rel..rel + mi];

                gemv_sub(yi, lij, [mi, m_j], yj);
            }
        }
    }

    fn diag_solve_in_place(&self, bv: &mut nalgebra::DVector<f64>) {
        let x = bv.as_mut_slice();
        for j in 0..self.n {
            let m = self.blk_dim[j];
            let j0 = self.scalar_offset_of_block(j);
            let y = &mut x[j0..j0 + m];

            let dj = self.diag_block(j);

            let mat_a = DMatrixView::from_slice(dj, m, m);
            let mut y = DVectorViewMut::from_slice(y, m);

            println!("A={mat_a}");

            println!("A={mat_a}");

            mat_a.cholesky().unwrap().solve_mut(&mut y);
        }
    }

    fn back_solve_in_place(&self, bv: &mut nalgebra::DVector<f64>) {
        let x = bv.as_mut_slice();
        for j in (0..self.n).rev() {
            let m_j = self.blk_dim[j];
            let j0 = self.scalar_ptr[j];
            let (left, mid_right) = x.split_at_mut(j0);
            let (xj, right) = mid_right.split_at_mut(m_j);

            let p0 = self.l_col_ptr[j];
            let p1 = self.l_col_ptr[j + 1];
            for pos in p0..p1 {
                let i = self.l_row_idx[pos];
                let mi = self.blk_dim[i];
                let elem_off = self.l_entry_of_pos[pos];
                let lij = &self.l_storage[elem_off..elem_off + (mi * m_j)];

                let i0 = self.scalar_ptr[i]; // i > j
                let rel = i0 - (j0 + m_j);
                let xi = &right[rel..rel + mi];

                // xj -= L(i,j)^T * x_i
                gemv_t_sub(xj, lij, mi, m_j, xi);
            }
            let _ = left;
        }
    }

    #[inline]
    fn scalar_offset_of_block(&self, j: usize) -> usize {
        // sum of blk_dim[0..j]
        // (cache if this shows up in profiles)
        let mut s = 0usize;
        for k in 0..j {
            s += self.blk_dim[k];
        }
        s
    }
    #[inline]
    fn diag_block(&self, j: usize) -> &[f64] {
        let p = self.diag_ptr[j];
        let q = self.diag_ptr[j + 1];
        &self.d_storage[p..q]
    }
    #[inline]
    fn diag_block_mut(&mut self, j: usize) -> &mut [f64] {
        let p = self.diag_ptr[j];
        let q = self.diag_ptr[j + 1];
        &mut self.d_storage[p..q]
    }
}

// --------- tiny dense helpers (col-major) ----------

#[inline]
fn gemm_sub_c_ab(c: &mut [f64], mc: usize, nc: usize, a: &[f64], ka: usize, b: &[f64]) {
    debug_assert_eq!(c.len(), mc * nc);
    debug_assert_eq!(a.len(), mc * ka);
    debug_assert_eq!(b.len(), ka * nc);

    for n in 0..nc {
        let col_c = &mut c[n * mc..(n + 1) * mc];
        let col_b = &b[n * ka..(n + 1) * ka]; // length ka
        for k in 0..ka {
            let bk = col_b[k];
            let col_a = &a[k * mc..(k + 1) * mc];
            for m in 0..mc {
                col_c[m] -= col_a[m] * bk;
            }
        }
    }
}

#[inline]
fn gemv_sub(y: &mut [f64], a: &[f64], a_shape: [usize; 2], x: &[f64]) {
    // y[m] -= A[m x n] * x[n]
    for n in 0..a_shape[1] {
        let xn = x[n];
        let col = &a[n * a_shape[0]..(n + 1) * a_shape[0]];
        for m in 0..a_shape[0] {
            y[m] -= col[m] * xn;
        }
    }
}

#[inline]
fn gemv_t_sub(y: &mut [f64], a: &[f64], ma: usize, na: usize, x: &[f64]) {
    // y[n] -= A^T[n x m] * x[m]  (A is m x n)
    for n in 0..na {
        let mut acc = 0.0;
        let col = &a[n * ma..(n + 1) * ma];
        for m in 0..ma {
            acc += col[m] * x[m];
        }
        y[n] -= acc;
    }
}

/// Compact accumulator for off-diagonal blocks W_ij in one column j.
/// Keeps rows sorted, and stores all blocks in one contiguous buffer.
struct RowAcc {
    rows: Vec<usize>, // i's in ascending order
    offs: Vec<usize>, // starting offset in `buf` per row
    mi: Vec<usize>,   // m_i per row
    buf: Vec<f64>,    // concatenated blocks, each size = m_i * m_j
    m_j: usize,       // current column block-size
}

impl RowAcc {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            offs: Vec::new(),
            mi: Vec::new(),
            buf: Vec::new(),
            m_j: 0,
        }
    }
    /// Prepare for a new column j, with optional capacity hints.
    fn begin(&mut self, m_j: usize, row_cap: usize, buf_cap: usize) {
        self.rows.clear();
        self.offs.clear();
        self.mi.clear();
        self.buf.clear();
        self.m_j = m_j;
        self.rows.reserve(row_cap);
        self.offs.reserve(row_cap);
        self.mi.reserve(row_cap);
        self.buf.reserve(buf_cap);
    }
    /// Get a mutable block W_ij (alloc if first seen). Returns a slice of length m_i * m_j.
    fn get_or_alloc(&mut self, i: usize, m_i: usize) -> &mut [f64] {
        match self.rows.binary_search(&i) {
            Ok(pos) => {
                debug_assert_eq!(self.mi[pos], m_i, "m_i changed for row {}", i);
                let off = self.offs[pos];
                let len = self.mi[pos] * self.m_j;
                &mut self.buf[off..off + len]
            }
            Err(pos) => {
                let off = self.buf.len();
                let len = m_i * self.m_j;
                self.buf.resize(off + len, 0.0);
                self.rows.insert(pos, i);
                self.offs.insert(pos, off);
                self.mi.insert(pos, m_i);
                &mut self.buf[off..off + len]
            }
        }
    }
    #[inline]
    fn n_rows(&self) -> usize {
        self.rows.len()
    }
    #[inline]
    fn row(&self, idx: usize) -> usize {
        self.rows[idx]
    }
    #[inline]
    fn block_mut_by_index(&mut self, idx: usize) -> &mut [f64] {
        let off = self.offs[idx];
        let len = self.mi[idx] * self.m_j;
        &mut self.buf[off..off + len]
    }
}

use std::collections::BTreeMap;

impl BlockLDLt {
    /// f
    pub fn factorize_left_looking(&mut self, cbm: &CompressedBlockMatrix) {
        assert_eq!(self.n, cbm.sym_block_pattern.num_block_cols);

        let mut assembled = AssembledCol {
            m_j: 0,
            diag: None,
            entries: Vec::new(),
        };

        // Off-diagonal accumulator for a single column (reused each j)
        let mut acc = RowAcc::new();

        // Reusable temporaries
        let mut mat_m = Vec::new(); // m_k x m_j

        // L columns (same as before)
        let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); self.n];
        let mut col_blocks: Vec<Vec<f64>> = vec![Vec::new(); self.n];

        for j in 0..self.n {
            // 1) assemble A(:,j) lower + diag
            cbm.assemble_numeric_col_lower(j, &mut assembled);
            let m_j = assembled.m_j;

            // Heuristic capacity hints for this column:
            //   - row_cap: at least the structural lower count from assembly
            //   - buf_cap: sum(rdim) * m_j (first-order guess)
            let row_cap = assembled.entries.len();
            let buf_cap = assembled.entries.iter().map(|e| e.rdim).sum::<usize>() * m_j;
            acc.begin(m_j, row_cap, buf_cap);

            // D_j workspace
            {
                let dj = self.diag_block_mut(j);
                if let Some(a_jj) = assembled.diag {
                    dj.copy_from_slice(a_jj);
                } else {
                    dj.fill(0.0);
                    for r in 0..m_j {
                        dj[r * m_j + r] = 1.0;
                    }
                }
            }

            // Initial off-diagonal from A
            for e in &assembled.entries {
                let buf = acc.get_or_alloc(e.i, e.rdim);
                // buf += A(i,j)
                for t in 0..(e.rdim * m_j) {
                    buf[t] += e.a_ij[t];
                }
            }

            // 2) left-looking updates from earlier columns k
            for k in 0..j {
                let rows_k = &col_rows[k];
                if rows_k.is_empty() {
                    continue;
                }
                match rows_k.binary_search(&j) {
                    Err(_) => continue, // no contribution
                    Ok(idx_in_k) => {
                        let m_k = self.blk_dim[k];
                        let off_jk =
                            offset_of_block_in_col(&col_rows[k], idx_in_k, &self.blk_dim, k);
                        let l_jk = &col_blocks[k][off_jk..off_jk + (self.blk_dim[j] * m_k)];

                        // M := D_k * L(j,k)^T  (m_k x m_j)
                        mat_m.resize(m_k * m_j, 0.0);
                        let dk = self.diag_block(k);
                        for n in 0..m_j {
                            for kk in 0..m_k {
                                let mut accn = 0.0;
                                for kk2 in 0..m_k {
                                    let dk_kk_kk2 = dk[kk2 * m_k + kk];
                                    let ljk_n_kk2 = l_jk[kk2 * m_j + n];
                                    accn += dk_kk_kk2 * ljk_n_kk2;
                                }
                                mat_m[n * m_k + kk] = accn;
                            }
                        }

                        // 2a) D_j -= L(j,k) * M
                        {
                            let dj = self.diag_block_mut(j);
                            // dj[m_j x m_j] -= L(j,k)[m_j x m_k] * M[m_k x m_j]
                            gemm_sub_c_ab(dj, m_j, m_j, l_jk, m_k, &mat_m);
                        }

                        // 2b) W(i,j) -= L(i,k) * M  for each i>j in col k
                        let mut run_off = 0usize; // into col_blocks[k]
                        for &i in rows_k.iter() {
                            let m_i = self.blk_dim[i];
                            let blk_sz = m_i * m_k;
                            let lik = &col_blocks[k][run_off..run_off + blk_sz];
                            run_off += blk_sz;

                            if i <= j {
                                continue;
                            }
                            let w_ij = acc.get_or_alloc(i, m_i);
                            gemm_sub_c_ab(w_ij, m_i, m_j, lik, m_k, &mat_m);
                        }
                    }
                }
            }

            // 3) Finalize: L(:,j) = W * D_j^{-1} ; store column j
            let dj = DMatrixView::from_slice(self.diag_block(j), m_j, m_j);
            let chol = dj.cholesky().unwrap(); // (Simple; can be made in-place later)

            let mut rows: Vec<usize> = Vec::with_capacity(acc.n_rows());
            let mut blocks: Vec<f64> = Vec::with_capacity(acc.buf.len());

            for idx in 0..acc.n_rows() {
                let i = acc.row(idx);
                let m_i = self.blk_dim[i];
                rows.push(i);

                // Build W^T view (m_j x m_i) on top of the contiguous block
                let w_ij = acc.block_mut_by_index(idx);
                let mut Wt = DMatrixViewMut::from_slice_with_strides_mut(w_ij, m_j, m_i, m_i, 1);

                // Solve D_j * Y = W^T
                chol.solve_mut(&mut Wt);

                // L_ij = Y^T (m_i x m_j)
                blocks.extend_from_slice(Wt.transpose().as_slice());
            }

            col_rows[j] = rows;
            col_blocks[j] = blocks;
        }

        // 4) compress columns into CSC buffers
        self.compress_columns(col_rows, col_blocks);
    }

    fn compress_columns(&mut self, col_rows: Vec<Vec<usize>>, col_blocks: Vec<Vec<f64>>) {
        self.l_col_ptr.fill(0);
        let n = self.n;

        // count
        let mut nnz = 0usize;
        for j in 0..n {
            self.l_col_ptr[j + 1] = self.l_col_ptr[j] + col_rows[j].len();
            nnz += col_rows[j].len();
        }
        self.l_row_idx = vec![0; nnz];
        self.l_entry_of_pos = vec![0; nnz];

        // compute total storage size
        let mut total_blocks_f64 = 0usize;
        for j in 0..n {
            total_blocks_f64 += col_blocks[j].len();
        }
        self.l_storage = Vec::with_capacity(total_blocks_f64);

        let mut pos = 0usize;
        for j in 0..n {
            let rows = &col_rows[j];
            let blks = &col_blocks[j];

            // write row indices
            for &i in rows {
                self.l_row_idx[pos] = i;
                self.l_entry_of_pos[pos] = 0; // we'll fill after pushing into storage
                pos += 1;
            }

            // append blocks into global storage, fix entry_of_pos
            let p0 = self.l_col_ptr[j];
            let mut off = 0usize;
            for k in 0..rows.len() {
                let m_i = self.blk_dim[rows[k]];
                let m_j = self.blk_dim[j];
                let sz = m_i * m_j;

                let elem_off = self.l_storage.len();
                self.l_storage.extend_from_slice(&blks[off..off + sz]);
                off += sz;
                self.l_entry_of_pos[p0 + k] = elem_off; // entry index in block units
            }
        }
    }
}

#[inline]
fn offset_of_block_in_col(rows_k: &[usize], idx_in_k: usize, blk_dim: &[usize], k: usize) -> usize {
    let m_k = blk_dim[k];
    rows_k[..idx_in_k]
        .iter()
        .fold(0usize, |acc, &i| acc + blk_dim[i] * m_k)
}
