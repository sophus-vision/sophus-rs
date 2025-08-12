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

/// Block LDLᵀ factor (streaming CSC build)
#[derive(Debug)]
pub struct BlockLDLt {
    // number of block-columns (== block-rows)
    n: usize,
    // per block-column size m_j
    blk_dim: Vec<usize>,

    // D: one big buffer; diag_ptr[j] is start of D_j (m_j x m_j, col-major)
    diag_ptr: Vec<usize>, // len = n + 1 (prefix sum of m_j*m_j)
    d_storage: Vec<f64>,

    // scalar offsets for each block-column (prefix sum of m_j)
    scalar_ptr: Vec<usize>,

    // L strictly-lower in block CSC by column (rows sorted ascending within each column)
    l_col_ptr: Vec<usize>,      // len = n + 1
    l_row_idx: Vec<usize>,      // block row indices (global), length = nnz blocks in L
    l_entry_of_pos: Vec<usize>, // maps CSC slot -> offset in l_storage
    l_storage: Vec<f64>,        // concatenated blocks, col-major per block (m_i x m_j)
}

impl BlockLDLt {
    /// Allocate factor structure (sizes) from the compressed matrix C (for blk dims).
    pub fn structure_from_cbm(cbm: &CompressedBlockMatrix) -> Self {
        let n = cbm.sym_block_pattern.num_block_cols;

        // Infer m_j (block sizes) from diagonal regions
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

        // Diagonal storage prefix sums (m_j * m_j elements each)
        let mut diag_ptr = Vec::with_capacity(n + 1);
        diag_ptr.push(0);
        for j in 0..n {
            diag_ptr.push(diag_ptr[j] + blk_dim[j] * blk_dim[j]);
        }

        // Scalar prefix sums (sum of m_j)
        let mut scalar_ptr = Vec::with_capacity(n + 1);
        scalar_ptr.push(0);
        for j in 0..n {
            scalar_ptr.push(scalar_ptr[j] + blk_dim[j]);
        }

        let d_len = diag_ptr[n];
        Self {
            n,
            blk_dim,
            diag_ptr,
            d_storage: vec![0.0; d_len],
            scalar_ptr,
            l_col_ptr: vec![0; n + 1],
            l_row_idx: Vec::new(),
            l_entry_of_pos: Vec::new(),
            l_storage: Vec::new(),
        }
    }

    /// Solve A x = b using the factor (L, D).
    pub fn solve_in_place(&self, x: &mut nalgebra::DVector<f64>) {
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
            let j0 = self.scalar_ptr[j];
            let (_left, mid_right) = x.split_at_mut(j0);
            let (yj, right) = mid_right.split_at_mut(m_j);

            let p0 = self.l_col_ptr[j];
            let p1 = self.l_col_ptr[j + 1];
            for pos in p0..p1 {
                let i = self.l_row_idx[pos];
                let m_i = self.blk_dim[i];
                let elem_off = self.l_entry_of_pos[pos];
                let lij = &self.l_storage[elem_off..elem_off + (m_i * m_j)];

                let i0 = self.scalar_ptr[i];
                debug_assert!(i0 >= j0 + m_j); // strictly lower: i > j
                let rel = i0 - (j0 + m_j);
                let yi = &mut right[rel..rel + m_i];

                gemv_sub(yi, lij, m_i, m_j, yj);
            }
        }
    }

    fn diag_solve_in_place(&self, bv: &mut nalgebra::DVector<f64>) {
        let x = bv.as_mut_slice();
        for j in 0..self.n {
            let m = self.blk_dim[j];
            let j0 = self.scalar_ptr[j];
            let y = &mut x[j0..j0 + m];

            let dj = self.diag_block(j);
            let mat_a = DMatrixView::from_slice(dj, m, m);
            let mut yv = DVectorViewMut::from_slice(y, m);

            // Note: we factor D_j on the fly here.
            // (Optional future optimization: store a triangular factor instead of D_j.)
            mat_a
                .cholesky()
                .expect("D_j must be SPD")
                .solve_mut(&mut yv);
        }
    }

    fn back_solve_in_place(&self, bv: &mut nalgebra::DVector<f64>) {
        let x = bv.as_mut_slice();
        for j in (0..self.n).rev() {
            let m_j = self.blk_dim[j];
            let j0 = self.scalar_ptr[j];
            let (_left, mid_right) = x.split_at_mut(j0);
            let (xj, right) = mid_right.split_at_mut(m_j);

            let p0 = self.l_col_ptr[j];
            let p1 = self.l_col_ptr[j + 1];
            for pos in p0..p1 {
                let i = self.l_row_idx[pos];
                let m_i = self.blk_dim[i];
                let elem_off = self.l_entry_of_pos[pos];
                let lij = &self.l_storage[elem_off..elem_off + (m_i * m_j)];

                let i0 = self.scalar_ptr[i]; // i > j
                let rel = i0 - (j0 + m_j);
                let xi = &right[rel..rel + m_i];

                // xj -= L(i,j)^T * x_i
                gemv_t_sub(xj, lij, m_i, m_j, xi);
            }
        }
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

    // ------------- factorization (streaming CSC build) -------------

    /// find row r in column k (binary search in final CSC)
    #[inline]
    fn find_pos_in_col(&self, k: usize, r: usize) -> Option<usize> {
        let start = self.l_col_ptr[k];
        let end = self.l_col_ptr[k + 1];
        self.l_row_idx[start..end]
            .binary_search(&r)
            .ok()
            .map(|rel| start + rel)
    }

    /// Left-looking numeric factorization; writes L directly into CSC.
    pub fn factorize_left_looking(&mut self, cbm: &CompressedBlockMatrix) {
        assert_eq!(self.n, cbm.sym_block_pattern.num_block_cols);

        let mut assembled = AssembledCol {
            m_j: 0,
            diag: None,
            entries: Vec::new(),
        };

        // per-column accumulators / scratch
        let mut acc = RowAcc::new();
        let mut mat_m = Vec::new(); // size m_k x m_j
        let mut l_jk_scratch = Vec::new(); // size m_j x m_k, reused across k

        // reset CSC buffers
        self.l_col_ptr.fill(0);
        self.l_row_idx.clear();
        self.l_entry_of_pos.clear();
        self.l_storage.clear();

        for j in 0..self.n {
            cbm.assemble_numeric_col_lower(j, &mut assembled);
            let m_j = assembled.m_j;

            let row_cap = assembled.entries.len();
            let buf_cap = assembled.entries.iter().map(|e| e.rdim).sum::<usize>() * m_j;
            acc.begin(m_j, row_cap, buf_cap);

            // D_j <- A_jj (or I)
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

            // seed W(i,j) with A(i,j)
            for e in &assembled.entries {
                let buf = acc.get_or_alloc(e.i, e.rdim);
                for t in 0..(e.rdim * m_j) {
                    buf[t] += e.a_ij[t];
                }
            }

            // left-looking updates from k < j
            for k in 0..j {
                // need L(j,k) — find row j in column k of L
                let pos_jk = if let Some(p) = self.find_pos_in_col(k, j) {
                    p
                } else {
                    continue;
                };
                let m_k = self.blk_dim[k];
                let off_jk = self.l_entry_of_pos[pos_jk];
                let len_jk = m_j * m_k;

                // ---- copy L(j,k) into scratch so we can mut-borrow D_j later
                l_jk_scratch.resize(len_jk, 0.0);
                l_jk_scratch.copy_from_slice(&self.l_storage[off_jk..off_jk + len_jk]);

                // M := D_k * L(j,k)^T  (m_k x m_j)
                mat_m.resize(m_k * m_j, 0.0);
                {
                    let dk = self.diag_block(k); // immutable borrow ends at block end
                    for n in 0..m_j {
                        for kk in 0..m_k {
                            let mut s = 0.0;
                            for kk2 in 0..m_k {
                                s += dk[kk2 * m_k + kk] * l_jk_scratch[kk2 * m_j + n];
                            }
                            mat_m[n * m_k + kk] = s;
                        }
                    }
                }

                // D_j -= L(j,k) * M   (now safe: no immutable borrow of self is alive)
                {
                    let dj = self.diag_block_mut(j);
                    gemm_sub_c_ab(dj, m_j, m_j, &l_jk_scratch, m_k, &mat_m);
                }

                // For each i in column k with i>j: W(i,j) -= L(i,k) * M
                let start_k = self.l_col_ptr[k];
                let end_k = self.l_col_ptr[k + 1];
                for pos in start_k..end_k {
                    let i = self.l_row_idx[pos];
                    if i <= j {
                        continue;
                    }
                    let m_i = self.blk_dim[i];
                    let off = self.l_entry_of_pos[pos];
                    let lik = &self.l_storage[off..off + (m_i * m_k)];
                    let w_ij = acc.get_or_alloc(i, m_i);
                    gemm_sub_c_ab(w_ij, m_i, m_j, lik, m_k, &mat_m);
                }
            }

            // finalize column j: L(:,j) = W * D_j^{-1}
            let dj = DMatrixView::from_slice(self.diag_block(j), m_j, m_j);
            let chol = dj.cholesky().expect("D_j must be SPD");

            let col_start = self.l_row_idx.len();
            for idx in 0..acc.n_rows() {
                let i = acc.row(idx);
                let m_i = self.blk_dim[i];
                let w_ij = acc.block_mut_by_index(idx);

                // W^T view (m_j x m_i) backed by acc.buf
                let mut wt = DMatrixViewMut::from_slice_with_strides_mut(w_ij, m_j, m_i, m_i, 1);

                // Solve D_j * Y = W^T   => wt := Y
                chol.solve_mut(&mut wt);

                self.l_row_idx.push(i);
                let elem_off = self.l_storage.len();
                self.l_entry_of_pos.push(elem_off);

                // Append L_ij = Y^T in col-major without extra temp
                for c in 0..m_j {
                    for r in 0..m_i {
                        let val = wt[(c, r)]; // Y(c,r)
                        self.l_storage.push(val);
                    }
                }
            }
            self.l_col_ptr[j + 1] = self.l_row_idx.len();
        }
    }
}

// --------- tiny dense helpers (col-major) ----------

#[inline]
fn gemm_sub_c_ab(c: &mut [f64], mc: usize, nc: usize, a: &[f64], ka: usize, b: &[f64]) {
    // c[mc x nc] -= a[mc x ka] * b[ka x nc]   (all col-major)
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
fn gemv_sub(y: &mut [f64], a: &[f64], ma: usize, na: usize, x: &[f64]) {
    // y[ma] -= A[ma x na] * x[na]  (col-major)
    for n in 0..na {
        let xn = x[n];
        let col = &a[n * ma..(n + 1) * ma];
        for m in 0..ma {
            y[m] -= col[m] * xn;
        }
    }
}

#[inline]
fn gemv_t_sub(y: &mut [f64], a: &[f64], ma: usize, na: usize, x: &[f64]) {
    // y[na] -= A^T[na x ma] * x[ma]  (A is [ma x na], col-major)
    for n in 0..na {
        let mut acc = 0.0;
        let col = &a[n * ma..(n + 1) * ma];
        for m in 0..ma {
            acc += col[m] * x[m];
        }
        y[n] -= acc;
    }
}

/// Compact accumulator for W(i,j) blocks in one column j.
/// Keeps rows sorted, stores all blocks in one contiguous buffer.
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

#[inline]
fn offset_of_block_in_col(rows_k: &[usize], idx_in_k: usize, blk_dim: &[usize], k: usize) -> usize {
    let m_k = blk_dim[k];
    rows_k[..idx_in_k]
        .iter()
        .fold(0usize, |acc, &i| acc + blk_dim[i] * m_k)
}
