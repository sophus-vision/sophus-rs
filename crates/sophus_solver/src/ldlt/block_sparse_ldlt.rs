use nalgebra::{
    DMatrixView,
    DMatrixViewMut,
    DVectorViewMut,
};

use crate::{
    AssembledCol,
    BlockSparseCompressedMatrix,
    BlockSparseMatrixBuilder,
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

// ====================== Small, allocation-free dense kernels ======================

// In-place Cholesky (lower) of a symmetric matrix A (col-major, size m×m).
// On success, A is overwritten with L (lower), and returns true.
// On failure (not SPD), returns false (A content is undefined).
#[inline]
fn chol_inplace_lower(a: &mut [f64], m: usize) -> bool {
    debug_assert_eq!(a.len(), m * m);
    for j in 0..m {
        // a[j,j] = sqrt(a[j,j] - sum_{k<j} L[j,k]^2)
        let mut d = a[j * m + j];
        for k in 0..j {
            let ljk = a[k * m + j]; // L[j,k] at col=k, row=j
            d -= ljk * ljk;
        }
        if !(d > 0.0) {
            return false;
        }
        let ljj = d.sqrt();
        a[j * m + j] = ljj;

        // For i>j: L[i,j] = (A[i,j] - sum_{k<j} L[i,k] L[j,k]) / L[j,j]
        for i in (j + 1)..m {
            let mut s = a[j * m + i]; // A[i,j] in lower (col=j, row=i)
            for k in 0..j {
                s -= a[k * m + i] * a[k * m + j]; // L[i,k]*L[j,k]
            }
            a[j * m + i] = s / ljj;
        }
        // Zero the upper part in column j (optional, for cleanliness)
        for i in 0..j {
            a[j * m + i] = 0.0;
        }
    }
    // Zero strictly upper triangle (optional, keep lower only)
    for c in 0..m {
        for r in 0..c {
            a[c * m + r] = 0.0;
        }
    }
    true
}

// Solve L x = b, where L is lower (m×m) col-major, b is a single RHS vector
// stored with stride 'inc' between successive rows. Overwrites b with x.
#[inline]
fn trisv_lower_no_trans_in_place(L: &[f64], m: usize, b: &mut [f64], inc: usize) {
    for r in 0..m {
        let mut s = b[r * inc];
        for c in 0..r {
            s -= L[c * m + r] * b[c * inc]; // L[r,c] at col=c,row=r
        }
        b[r * inc] = s / L[r * m + r];
    }
}

// Solve Lᵀ x = b, where L is lower (m×m) col-major, b is a single RHS vector
// with stride 'inc'. Overwrites b with x.
#[inline]
fn trisv_lower_trans_in_place(L: &[f64], m: usize, b: &mut [f64], inc: usize) {
    for r in (0..m).rev() {
        let mut s = b[r * inc];
        // U(r,c) = Lᵀ(r,c) = L(c,r). c>r
        for c in (r + 1)..m {
            s -= L[r * m + c] * b[c * inc]; // L(c,r) is at col=r, row=c → index r*m + c
        }
        b[r * inc] = s / L[r * m + r]; // diag is the same
    }
}

// y[m] -= A[m×n] * x[n]   (all col-major, allocation-free)
#[inline]
fn gemv_sub(y: &mut [f64], a: &[f64], ma: usize, na: usize, x: &[f64]) {
    for n in 0..na {
        let xn = x[n];
        let col = &a[n * ma..(n + 1) * ma];
        for m in 0..ma {
            y[m] -= col[m] * xn;
        }
    }
}

// y[n] -= Aᵀ[n×m] * x[m]  (A is m×n, col-major)
#[inline]
fn gemv_t_sub(y: &mut [f64], a: &[f64], ma: usize, na: usize, x: &[f64]) {
    for n in 0..na {
        let mut acc = 0.0;
        let col = &a[n * ma..(n + 1) * ma];
        for m in 0..ma {
            acc += col[m] * x[m];
        }
        y[n] -= acc;
    }
}

// c[mc×nc] -= a[mc×ka] * b[ka×nc]  (all col-major)
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

// Given L (lower m_k×m_k) and L(j,k) (m_j×m_k, col-major), compute
// M := (L Lᵀ) * L(j,k)ᵀ  (M has shape m_k×m_j, col-major).
// Uses a temporary vector tmp (len ≥ m_k); no heap alloc here.
#[inline]
fn form_M_from_Dfactor_and_LjkT(
    L: &[f64], // L_D(k), lower, size m_k×m_k
    m_k: usize,
    Ljk: &[f64], // L(j,k), size m_j×m_k (col-major)
    m_j: usize,
    M_out: &mut [f64], // output M, size m_k×m_j (col-major)
    tmp: &mut [f64],   // workspace, len ≥ m_k
) {
    debug_assert!(tmp.len() >= m_k);
    // For each column n of M (equivalently each row n of L(j,k))
    for n in 0..m_j {
        // Step 1: t = Lᵀ * b, where b is column n of L(j,k)ᵀ → row n of L(j,k)
        // b[s] = Ljk(scol=s, row=n) = Ljk[s * m_j + n]
        for r in 0..m_k {
            let mut sum = 0.0;
            // sum_{s=r..m_k-1} L[s,r] * b[s]
            let col_r = &L[r * m_k..(r + 1) * m_k]; // column r of L
            for s in r..m_k {
                sum += col_r[s] * Ljk[s * m_j + n];
            }
            tmp[r] = sum;
        }

        // Step 2: m = L * t
        for r in 0..m_k {
            let mut sum = 0.0;
            // sum_{q=0..r} L[r,q] * t[q]  ; L[r,q] at index q*m_k + r
            for q in 0..=r {
                sum += L[q * m_k + r] * tmp[q];
            }
            M_out[n * m_k + r] = sum; // store col-major (m_k×m_j)
        }
    }
}

// ====================== Compact row accumulator (same as before) ======================

struct RowAcc {
    rows: Vec<usize>, // i in ascending order
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
    fn get_or_alloc(&mut self, i: usize, m_i: usize) -> &mut [f64] {
        match self.rows.binary_search(&i) {
            Ok(pos) => {
                debug_assert_eq!(self.mi[pos], m_i);
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

/// ====================== BlockLDLt with D stored as L_D (lower) ======================
#[derive(Debug)]
pub struct BlockLDLt {
    // n block-columns (== block-rows); column j has size m_j
    n: usize,
    // per block: m_j
    blk_dim: Vec<usize>,

    // D is stored as its in-place Cholesky factor L_D (lower).
    // diag_ptr[j] is the start of D_j / L_D_j (m_j×m_j, col-major).
    diag_ptr: Vec<usize>, // len = n+1 (prefix sum of m_j*m_j)
    d_storage: Vec<f64>,

    // scalar offsets per block
    scalar_ptr: Vec<usize>,

    // L strictly-lower in block CSC (by column)
    l_col_ptr: Vec<usize>,      // len = n+1
    l_row_idx: Vec<usize>,      // block row indices (global)
    l_entry_of_pos: Vec<usize>, // map CSC slot -> entry offset
    l_storage: Vec<f64>,        // concatenated blocks, col-major (m_i×m_j)
}

impl BlockLDLt {
    ///s
    pub fn structure_from_cbm(cbm: &BlockSparseCompressedMatrix) -> Self {
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

        // Scalar offsets
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
            d_storage: vec![0.0; last], // will hold L_D after factorize
            scalar_ptr,
            l_col_ptr: vec![0; n + 1],
            l_row_idx: Vec::new(),
            l_entry_of_pos: Vec::new(),
            l_storage: Vec::new(),
        }
    }

    /// Solve A x = b using the factor (L, D) where D is stored as L_D (lower).
    pub fn solve_in_place(&self, x: &mut nalgebra::DVector<f64>) {
        // 1) forward: L y = b
        self.forward_solve_in_place(x);
        // 2) diagonal: D z = y  → with L_D: solve L_D w = y, then L_Dᵀ z = w
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

            let Ld = self.diag_block(j); // L_D (lower)
            // Solve L_D w = y (in-place overwrite y → w)
            trisv_lower_no_trans_in_place(Ld, m, y, 1);
            // Solve L_Dᵀ z = w (in-place overwrite y → z)
            trisv_lower_trans_in_place(Ld, m, y, 1);
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

    // -------------------- Numeric factorization (left-looking) --------------------

    #[inline]
    fn find_pos_in_col(&self, k: usize, r: usize) -> Option<usize> {
        let start = self.l_col_ptr[k];
        let end = self.l_col_ptr[k + 1];
        self.l_row_idx[start..end]
            .binary_search(&r)
            .ok()
            .map(|rel| start + rel)
    }

    /// f
    pub fn factorize_left_looking(&mut self, cbm: &BlockSparseCompressedMatrix) {
        assert_eq!(self.n, cbm.sym_block_pattern.num_block_cols);

        let mut assembled = AssembledCol {
            m_j: 0,
            diag: None,
            entries: Vec::new(),
        };

        let mut acc = RowAcc::new();
        let mut mat_m = Vec::<f64>::new(); // will hold M (m_k×m_j)
        let mut l_jk_scratch = Vec::<f64>::new(); // L(j,k) scratch (m_j×m_k)
        let mut tmp_vec = Vec::<f64>::new(); // length up to max(m_k)

        // one-time reserves for small scratch
        let max_m = self.blk_dim.iter().copied().max().unwrap_or(0);
        mat_m.reserve(max_m * max_m);
        l_jk_scratch.reserve(max_m * max_m);
        tmp_vec.reserve(max_m);

        // Reset CSC buffers (we stream-fill them)
        self.l_col_ptr.fill(0);
        self.l_row_idx.clear();
        self.l_entry_of_pos.clear();
        self.l_storage.clear();

        for j in 0..self.n {
            // Assemble A(:,j) lower + diag
            cbm.assemble_numeric_col_lower(j, &mut assembled);
            let m_j = assembled.m_j;

            // Prepare accumulator for column j
            let row_cap = assembled.entries.len();
            let buf_cap = assembled.entries.iter().map(|e| e.rdim).sum::<usize>() * m_j;
            acc.begin(m_j, row_cap, buf_cap);

            // D_j workspace ← A_jj (or I if missing)
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

            // Seed W(i,j) from A(i,j)
            for e in &assembled.entries {
                let buf = acc.get_or_alloc(e.i, e.rdim);
                for t in 0..(e.rdim * m_j) {
                    buf[t] += e.a_ij[t];
                }
            }

            // Left-looking updates from k<j
            for k in 0..j {
                // Need L(j,k): find row j in column k
                let pos_jk = if let Some(p) = self.find_pos_in_col(k, j) {
                    p
                } else {
                    continue;
                };

                let m_k = self.blk_dim[k];
                let off_jk = self.l_entry_of_pos[pos_jk];
                let len_jk = m_j * m_k;

                // Copy L(j,k) into scratch (so we can mut-borrow D_j later)
                l_jk_scratch.resize(len_jk, 0.0);
                l_jk_scratch[..len_jk].copy_from_slice(&self.l_storage[off_jk..off_jk + len_jk]);

                // M := D_k * L(j,k)^T, with D_k stored as L_D (lower):
                // M = L_D · (L_Dᵀ · L(j,k)ᵀ)
                mat_m.resize(m_k * m_j, 0.0);
                {
                    let Ld_k = self.diag_block(k);
                    let M = &mut mat_m[..m_k * m_j];
                    let Ljk = &l_jk_scratch[..len_jk];
                    // tmp_vec length ≥ m_k
                    if tmp_vec.len() < m_k {
                        tmp_vec.resize(m_k, 0.0);
                    }
                    form_M_from_Dfactor_and_LjkT(Ld_k, m_k, Ljk, m_j, M, &mut tmp_vec);
                }

                // 2a) D_j -= L(j,k) * M   (m_j×m_k) · (m_k×m_j) → (m_j×m_j)
                {
                    let dj = self.diag_block_mut(j);
                    let M = &mat_m[..m_k * m_j];
                    let Ljk = &l_jk_scratch[..len_jk];
                    gemm_sub_c_ab(dj, m_j, m_j, Ljk, m_k, M);
                }

                // 2b) For each i>j in column k: W(i,j) -= L(i,k) * M
                let start_k = self.l_col_ptr[k];
                let end_k = self.l_col_ptr[k + 1];
                let mut run_off = 0usize; // run through blocks in col k
                for pos in start_k..end_k {
                    let i = self.l_row_idx[pos];
                    let m_i = self.blk_dim[i];
                    let off = self.l_entry_of_pos[pos];
                    let lik = &self.l_storage[off..off + (m_i * m_k)];
                    run_off += m_i * m_k;

                    if i <= j {
                        continue;
                    }
                    let w_ij = acc.get_or_alloc(i, m_i);
                    let M = &mat_m[..m_k * m_j];
                    gemm_sub_c_ab(w_ij, m_i, m_j, lik, m_k, M);
                }
            }

            // --- Finalize column j ---
            // Factorize D_j in-place: D_j := L_D (lower)
            {
                let dj = self.diag_block_mut(j);
                let ok = chol_inplace_lower(dj, m_j);
                assert!(ok, "D_j is not SPD");
            }

            // Solve D_j * Y = W^T using L_D (two triangular solves on each RHS of W^T)
            // Then store L(i,j) = Y^T
            let col_start = self.l_row_idx.len();
            for idx in 0..acc.n_rows() {
                let i = acc.row(idx);
                let m_i = self.blk_dim[i];
                let w_ij = acc.block_mut_by_index(idx); // W (m_i×m_j), col-major

                // We need to transform W^T (m_j×m_i) in-place on this same buffer.
                // Each column of W^T is a strided vector with stride = m_i.
                let Ld_j = self.diag_block(j);

                // For each column c of W^T (i.e., row c of W):
                for c in 0..m_i {
                    // forward solve L_D * z = b  (b is column c of W^T with stride m_i)
                    trisv_lower_no_trans_in_place(Ld_j, m_j, &mut w_ij[c..], m_i);
                    // back solve L_Dᵀ * y = z
                    trisv_lower_trans_in_place(Ld_j, m_j, &mut w_ij[c..], m_i);
                }

                // Append L_ij = Y^T into global storage (col-major m_i×m_j)
                self.l_row_idx.push(i);
                let elem_off = self.l_storage.len();
                self.l_entry_of_pos.push(elem_off);

                // Write Y^T (m_i×m_j) from the strided W^T we just solved:
                // For col = 0..m_j, row = 0..m_i : L_ij(row, col) = W^T(col, row)
                for col in 0..m_j {
                    for row in 0..m_i {
                        // W^T(col,row) lives at w_ij[row + col*m_i] because row-stride = 1 across
                        // columns in this loop? Careful: w_ij is W
                        // (m_i×m_j) col-major. W^T(col,row) = W(row,col).
                        // W(row,col) at index col*m_i + row.
                        let val = w_ij[col * m_i + row];
                        self.l_storage.push(val);
                    }
                }
            }
            self.l_col_ptr[j + 1] = self.l_row_idx.len();
        }
    }
}

// If you still use this helper elsewhere:
#[inline]
fn offset_of_block_in_col(rows_k: &[usize], idx_in_k: usize, blk_dim: &[usize], k: usize) -> usize {
    let m_k = blk_dim[k];
    rows_k[..idx_in_k]
        .iter()
        .fold(0usize, |acc, &i| acc + blk_dim[i] * m_k)
}
