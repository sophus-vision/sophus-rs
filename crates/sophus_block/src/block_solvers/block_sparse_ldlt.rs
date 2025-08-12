use nalgebra::{
    DMatrixView,
    DMatrixViewMut,
    DVectorViewMut,
};

use crate::{
    AssembledCol,
    BlockSparseMatrixBuilder,
    BlockVector,
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
        println!("{}", x);
        self.forward_solve_in_place(x);
        println!("{}", x);

        // 2) diagonal: D z = y
        self.diag_solve_in_place(x);
        println!("{}", x);

        // 3) back: Lᵀ x = z
        self.back_solve_in_place(x);
        println!("{}", x);
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
    // c[m x n] -= a[m x k] * b[k x n]
    for n in 0..nc {
        for k in 0..ka {
            let bk = &b[n * ka + k]; // b(k,n)
            let col_a = &a[k * mc..(k + 1) * mc];
            let col_c = &mut c[n * mc..(n + 1) * mc];
            for m in 0..mc {
                col_c[m] -= col_a[m] * (*bk);
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

use std::collections::BTreeMap;

impl BlockLDLt {
    /// factorize
    pub fn factorize_left_looking(&mut self, cbm: &CompressedBlockMatrix) {
        assert_eq!(self.n, cbm.sym_block_pattern.num_block_cols);

        // Temporary workspace per column
        let mut assembled = AssembledCol {
            m_j: 0,
            diag: None,
            entries: Vec::new(),
        };
        // Off-diagonal accumulator: i -> W_ij (m_i x m_j)
        let mut acc: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        // re-usable temporaries
        let mut mat_m = Vec::new(); // m_k x m_j

        // We’ll build L as vectors per column, then compress into CSC at the end.
        let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); self.n];
        let mut col_blocks: Vec<Vec<f64>> = vec![Vec::new(); self.n]; // concatenated blocks per col

        for j in 0..self.n {
            acc.clear();

            // 1) assemble A(:,j) lower + diag
            cbm.assemble_numeric_col_lower(j, &mut assembled);
            let m_j = assembled.m_j;
            // diag workspace D_j
            {
                let dj = self.diag_block_mut(j);
                if let Some(a_jj) = assembled.diag {
                    dj.copy_from_slice(a_jj);
                } else {
                    // treat as identity if absent
                    dj.fill(0.0);
                    for r in 0..m_j {
                        dj[r * m_j + r] = 1.0;
                    }
                }
            }
            // initial off-diagonal from A
            for e in &assembled.entries {
                let buf = acc.entry(e.i).or_insert_with(|| vec![0.0; e.rdim * m_j]);
                // buf += A(i,j)
                for t in 0..(e.rdim * m_j) {
                    buf[t] += e.a_ij[t];
                }
            }

            // 2) left-looking updates from earlier columns k that hit row j,
            // i.e. columns k where L(j,k) exists. We can find them by scanning k<j
            // and binary searching for row j in col_rows[k].
            for k in 0..j {
                // find L(j,k)
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
                        {
                            let dk = self.diag_block(k);
                            // M = D_k * (L_jk)^T
                            // Form L_jk^T implicitly in the multiply
                            // M[:,n] = D_k * (col n of L_jk^T) = D_k * (row n of L_jk)^T
                            // Implement as naive triple loop:
                            for n in 0..m_j {
                                for kk in 0..m_k {
                                    let mut accn = 0.0;
                                    for kk2 in 0..m_k {
                                        // dk(kk,kk2) * L_jk(n, kk2)
                                        let dk_kk_kk2 = dk[kk2 * m_k + kk];
                                        let ljk_n_kk2 = l_jk[kk2 * m_j + n];
                                        accn += dk_kk_kk2 * ljk_n_kk2;
                                    }
                                    mat_m[n * m_k + kk] = accn; // store col-major (kk along rows)
                                }
                            }
                        }

                        // 2a) Diagonal update: D_j -= L(j,k) * M
                        {
                            let dj = self.diag_block_mut(j);
                            // dj[m_j x m_j] -= L(j,k)[m_j x m_k] * M[m_k x m_j]
                            gemm_sub_c_ab(dj, m_j, m_j, l_jk, m_k, &mat_m);
                        }

                        // 2b) Off-diagonal updates: for each i in column k with i>j
                        let rows_k = &col_rows[k];
                        let mut run_off = 0usize; // running offset into col_blocks[k]
                        for &i in rows_k.iter() {
                            let m_i = self.blk_dim[i];
                            let blk_sz = m_i * m_k;
                            let lik = &col_blocks[k][run_off..run_off + blk_sz];
                            run_off += blk_sz;

                            if i <= j {
                                continue;
                            } // only below current pivot row
                            let w_ij = acc.entry(i).or_insert_with(|| vec![0.0; m_i * m_j]);
                            // w_ij -= L(i,k) * M
                            gemm_sub_c_ab(w_ij, m_i, m_j, lik, m_k, &mat_m);
                        }
                    }
                }
            }

            // 3) Finalize: L(:,j) = W * D_j^{-1} ; store column j

            // D_j (m_j x m_j), col-major from your storage
            let dj = DMatrixView::from_slice(self.diag_block(j), m_j, m_j);
            let chol = dj.cholesky().unwrap();

            let mut rows: Vec<usize> = Vec::with_capacity(acc.len());
            let mut blocks: Vec<f64> = Vec::new();

            for (&i, w_ij) in acc.iter_mut() {
                rows.push(i);
                let m_i = self.blk_dim[i];

                // Build W^T (m_j x m_i) from col-major W (m_i x m_j) stored in `w_ij`
                let mut mat_w_transpose =
                    DMatrixViewMut::from_slice_with_strides_mut(w_ij, m_j, m_i, m_i, 1);

                // Solve D_j * Y = W^T  (since D_j == D_j^T)
                chol.solve_mut(&mut mat_w_transpose);

                // L_ij = Y^T (m_i x m_j), still col-major when we push its slice
                blocks.extend_from_slice(mat_w_transpose.transpose().as_slice());
            }

            col_rows[j] = rows;
            col_blocks[j] = blocks;

            // copy D_j we already have in place (kept in self.d_storage)
        }

        println!("{:?}", col_blocks);

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
