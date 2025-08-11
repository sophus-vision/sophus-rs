use crate::{
    BlockVector,
    CompressedBlockMatrix,
};

/// Triangular matrix type.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Triangular {
    /// lower triangular
    Lower,
    /// upper triangular
    Upper,
}

/// standard aor transposed matrix multiplication.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MultOp {
    /// standard: A x
    NoTrans,
    /// transposed: Aᵗ x
    Transpose,
}

impl CompressedBlockMatrix {
    /// Solve T x = b in-place, where T is triangular in *block* pattern.
    /// - `triangle`: Lower or Upper (of T, before transpose)
    /// - `op`: NoTrans (T) or Transpose (Tᵗ)
    ///
    /// Assumptions:
    ///  - Matrix is square.
    ///  - Block-triangular *pattern* holds (off-triangle regions are empty).
    ///  - Diagonal blocks are square and nonsingular. If missing, treated as unit.
    pub fn triangular_solve_in_place(&self, b: &mut BlockVector, triangle: Triangular, op: MultOp) {
        match (op, triangle) {
            (MultOp::NoTrans, Triangular::Lower) => self.solve_lower_no_trans(b),
            (MultOp::NoTrans, Triangular::Upper) => self.solve_upper_no_trans(b),
            (MultOp::Transpose, Triangular::Lower) => self.solve_lower_transpose(b), /* solve Lᵗ
                                                                                       * x */
            // = b
            (MultOp::Transpose, Triangular::Upper) => self.solve_upper_transpose(b), /* solve Uᵗ x = b */
        }
    }

    fn solve_lower_no_trans(&self, b: &mut BlockVector) {
        let [n, m] = self.scalar_shape;
        assert_eq!(n, m, "L must be square");
        assert_eq!(b.scalar_vector().nrows(), n, "RHS size mismatch");

        let np = self.index_offset_per_row_partition.len();
        assert_eq!(np, self.index_offset_per_col_partition.len());

        let x = b.scalar_vector_mut().as_mut_slice();

        for rx in 0..np {
            let reg_diag = self.region_grid.get(&[rx, rx]);
            let m = reg_diag.block_shape[0];
            debug_assert_eq!(m, reg_diag.block_shape[1], "diag blocks must be square");
            let n_block_rows = reg_diag.region_shape[0];

            for br in 0..n_block_rows {
                let i0 = self.index_offset_per_row_partition[rx] + br * m;
                let (left, mid_right) = x.split_at_mut(i0);
                let (y, _right) = mid_right.split_at_mut(m);

                // Left-of-diagonal regions
                for ry in 0..rx {
                    let reg = self.region_grid.get(&[rx, ry]);
                    if reg.num_non_empty_blocks == 0 {
                        continue;
                    }
                    let cols = reg.block_shape[1];
                    for (bc, eidx) in reg.iter_csr_row(br) {
                        let j0 = self.index_offset_per_col_partition[ry] + bc * cols;
                        debug_assert!(j0 + cols <= i0);
                        let xsub = &left[j0..j0 + cols];

                        let a = reg.block_slice(eidx); // m × cols, col-major
                        for c in 0..cols {
                            let xc = xsub[c];
                            let col = &a[c * m..(c + 1) * m];
                            for r in 0..m {
                                y[r] -= col[r] * xc;
                            }
                        }
                    }
                }

                // Inside-diagonal region, strictly left blocks (bc < br)
                {
                    let reg = reg_diag;
                    if reg.num_non_empty_blocks != 0 {
                        let cols = reg.block_shape[1]; // == m
                        for (bc, eidx) in reg.iter_csr_row(br) {
                            if bc >= br {
                                continue;
                            }
                            let j0 = self.index_offset_per_col_partition[rx] + bc * cols;
                            debug_assert!(j0 + cols <= i0);
                            let xsub = &left[j0..j0 + cols];

                            let a = reg.block_slice(eidx);
                            for c in 0..cols {
                                let xc = xsub[c];
                                let col = &a[c * m..(c + 1) * m];
                                for r in 0..m {
                                    y[r] -= col[r] * xc;
                                }
                            }
                        }
                    }
                }

                // Solve diagonal (lower-triangular)
                if let Some(pos) = reg_diag.diag_pos_in_csc.get(br).and_then(|&p| p) {
                    let eidx = reg_diag.csc_entry_of_pos[pos];
                    let d = reg_diag.block_slice(eidx); // m×m col-major
                    // forward substitution
                    for rr in 0..m {
                        let mut rhs = y[rr];
                        for cc in 0..rr {
                            rhs -= d[cc * m + rr] * y[cc]; // A(rr,cc)
                        }
                        let diag = d[rr * m + rr];
                        debug_assert!(diag != 0.0);
                        y[rr] = rhs / diag;
                    }
                } // else: unit diagonal
            }
        }
    }

    fn solve_upper_no_trans(&self, b: &mut BlockVector) {
        let [n, m] = self.scalar_shape;
        assert_eq!(n, m);
        assert_eq!(b.scalar_vector().nrows(), n);

        let np = self.index_offset_per_row_partition.len();
        assert_eq!(np, self.index_offset_per_col_partition.len());

        let x = b.scalar_vector_mut().as_mut_slice();

        for rx in (0..np).rev() {
            let reg_diag = self.region_grid.get(&[rx, rx]);
            let m = reg_diag.block_shape[0];
            debug_assert_eq!(m, reg_diag.block_shape[1]);
            let n_block_rows = reg_diag.region_shape[0];

            for br in (0..n_block_rows).rev() {
                let i0 = self.index_offset_per_row_partition[rx] + br * m;
                let (left, mid_right) = x.split_at_mut(i0);
                let (y, right) = mid_right.split_at_mut(m);
                let _ = left; // unused here

                // Right-of-diagonal regions (ry > rx)
                for ry in (rx + 1)..np {
                    let reg = self.region_grid.get(&[rx, ry]);
                    if reg.num_non_empty_blocks == 0 {
                        continue;
                    }
                    let cols = reg.block_shape[1];
                    for (bc, eidx) in reg.iter_csr_row(br) {
                        let j0 = self.index_offset_per_col_partition[ry] + bc * cols;
                        debug_assert!(j0 >= i0 + m);
                        let rel = j0 - (i0 + m);
                        let xsub = &right[rel..rel + cols];

                        let a = reg.block_slice(eidx); // m × cols
                        for c in 0..cols {
                            let xc = xsub[c];
                            let col = &a[c * m..(c + 1) * m];
                            for r in 0..m {
                                y[r] -= col[r] * xc;
                            }
                        }
                    }
                }

                // Inside-diagonal region, strictly right blocks (bc > br)
                {
                    let reg = reg_diag;
                    if reg.num_non_empty_blocks != 0 {
                        let cols = reg.block_shape[1]; // == m
                        for (bc, eidx) in reg.iter_csr_row(br) {
                            if bc <= br {
                                continue;
                            }
                            let j0 = self.index_offset_per_col_partition[rx] + bc * cols;
                            debug_assert!(j0 >= i0 + m);
                            let rel = j0 - (i0 + m);
                            let xsub = &right[rel..rel + cols];

                            let a = reg.block_slice(eidx);
                            for c in 0..cols {
                                let xc = xsub[c];
                                let col = &a[c * m..(c + 1) * m];
                                for r in 0..m {
                                    y[r] -= col[r] * xc;
                                }
                            }
                        }
                    }
                }

                // Solve diagonal (upper-triangular)
                if let Some(pos) = reg_diag.diag_pos_in_csc.get(br).and_then(|&p| p) {
                    let eidx = reg_diag.csc_entry_of_pos[pos];
                    let d = reg_diag.block_slice(eidx); // m×m col-major
                    // backward substitution
                    for rr in (0..m).rev() {
                        let mut rhs = y[rr];
                        for cc in (rr + 1)..m {
                            rhs -= d[cc * m + rr] * y[cc]; // U(rr,cc)
                        }
                        let diag = d[rr * m + rr];
                        debug_assert!(diag != 0.0);
                        y[rr] = rhs / diag;
                    }
                } // else: unit diagonal
            }
        }
    }

    fn solve_lower_transpose(&self, b: &mut BlockVector) {
        let [n, m_tot] = self.scalar_shape;
        assert_eq!(n, m_tot);
        assert_eq!(b.scalar_vector().nrows(), n);

        let np = self.index_offset_per_row_partition.len();
        assert_eq!(np, self.index_offset_per_col_partition.len());

        let x = b.scalar_vector_mut().as_mut_slice();

        // L^T is upper → sweep partitions backward
        for rx in (0..np).rev() {
            let reg_diag = self.region_grid.get(&[rx, rx]);
            let m = reg_diag.block_shape[0];
            debug_assert_eq!(m, reg_diag.block_shape[1], "diag blocks must be square");
            let n_block_cols = reg_diag.region_shape[1];

            // Inside this column-partition, sweep block-columns backward (upper solve)
            for bc in (0..n_block_cols).rev() {
                // y corresponds to column-partition `rx`, block-col `bc`
                let i0 = self.index_offset_per_col_partition[rx] + bc * m;
                let (left, mid_right) = x.split_at_mut(i0);
                let (y, right) = mid_right.split_at_mut(m);
                let _ = left; // not used

                // --- A) strictly-right blocks inside diagonal region (rx,rx): br_i > bc
                if reg_diag.num_non_empty_blocks != 0 {
                    let rdim = reg_diag.block_shape[0]; // == m
                    for (br_i, eidx) in reg_diag.iter_csc_col(bc) {
                        if br_i <= bc {
                            continue;
                        }
                        // xsub lives in *column* partition rx, block-col = br_i
                        let j0 = self.index_offset_per_col_partition[rx] + br_i * rdim;
                        debug_assert!(j0 >= i0 + m);
                        let rel = j0 - (i0 + m);
                        let xsub = &right[rel..rel + rdim];

                        let a = reg_diag.block_slice(eidx); // rdim × m (this is L), col-major
                        // y -= Aᵗ * xsub
                        for c in 0..m {
                            let col = &a[c * rdim..(c + 1) * rdim];
                            let mut acc = 0.0;
                            for r in 0..rdim {
                                acc += col[r] * xsub[r];
                            }
                            y[c] -= acc;
                        }
                    }
                }

                // --- B) contributions from strictly-below regions (rxx > rx)
                for rxx in (rx + 1)..np {
                    let reg = self.region_grid.get(&[rxx, rx]);
                    if reg.num_non_empty_blocks == 0 {
                        continue;
                    }
                    let rdim = reg.block_shape[0];
                    for (br_i, eidx) in reg.iter_csc_col(bc) {
                        // xsub lives in *column* partition rxx, block-col = br_i
                        let j0 = self.index_offset_per_col_partition[rxx] + br_i * rdim;
                        debug_assert!(j0 >= i0 + m);
                        let rel = j0 - (i0 + m);
                        let xsub = &right[rel..rel + rdim];

                        let a = reg.block_slice(eidx); // rdim × m (this is L), col-major
                        // y -= Aᵗ * xsub
                        for c in 0..m {
                            let col = &a[c * rdim..(c + 1) * rdim];
                            let mut acc = 0.0;
                            for r in 0..rdim {
                                acc += col[r] * xsub[r];
                            }
                            y[c] -= acc;
                        }
                    }
                }

                // --- C) solve (L_iiᵗ) y = y  [upper triangular back-sub]
                if let Some(pos) = reg_diag.diag_pos_in_csc.get(bc).and_then(|&p| p) {
                    let eidx = reg_diag.csc_entry_of_pos[pos];
                    let d = reg_diag.block_slice(eidx); // stores L_ii (lower), m×m, col-major
                    // in solve_lower_transpose: back substitution on U = Lᵗ
                    for rr in (0..m).rev() {
                        let mut rhs = y[rr];
                        for cc in (rr + 1)..m {
                            // U(rr,cc) = L(cc,rr) at (col=rr, row=cc)
                            rhs -= d[rr * m + cc] * y[cc];
                        }
                        let diag = d[rr * m + rr];
                        debug_assert!(diag != 0.0);
                        y[rr] = rhs / diag;
                    }
                } // else: unit-diagonal assumed
            }
        }
    }

    fn solve_upper_transpose(&self, b: &mut BlockVector) {
        let [n, m] = self.scalar_shape;
        assert_eq!(n, m);
        assert_eq!(b.scalar_vector().nrows(), n);

        let np = self.index_offset_per_row_partition.len();
        assert_eq!(np, self.index_offset_per_col_partition.len());

        let x = b.scalar_vector_mut().as_mut_slice();

        for rx in 0..np {
            let reg_diag = self.region_grid.get(&[rx, rx]);
            let m = reg_diag.block_shape[0];
            debug_assert_eq!(m, reg_diag.block_shape[1]);
            let n_block_cols = reg_diag.region_shape[1];

            for bc in 0..n_block_cols {
                let i0 = self.index_offset_per_row_partition[rx] + bc * m;
                let (left, mid_right) = x.split_at_mut(i0);
                let (y, _right) = mid_right.split_at_mut(m);

                // Contributions from blocks *above* the diagonal of U:
                // iterate all row partitions rxx <= rx in column `bc`
                for rxx in 0..=rx {
                    let reg = self.region_grid.get(&[rxx, rx]);
                    if reg.num_non_empty_blocks == 0 {
                        continue;
                    }
                    let rdim = reg.block_shape[0];
                    for (br_i, eidx) in reg.iter_csc_col(bc) {
                        // In U, above-diagonal means (rxx == rx && br_i < bc) OR (rxx < rx).
                        // For Uᵗ solve (forward), these contribute from the *left*.
                        if !(rxx < rx || (rxx == rx && br_i < bc)) {
                            continue;
                        }

                        let j0 = self.index_offset_per_row_partition[rxx] + br_i * rdim;
                        debug_assert!(j0 + rdim <= i0); // left of y
                        let xsub = &left[j0..j0 + rdim];

                        let a = reg.block_slice(eidx); // rdim × m
                        // y -= Aᵗ * xsub
                        for c in 0..m {
                            let col = &a[c * rdim..(c + 1) * rdim];
                            let mut acc = 0.0;
                            for r in 0..rdim {
                                acc += col[r] * xsub[r];
                            }
                            y[c] -= acc;
                        }
                    }
                }

                // Solve (U_iiᵗ) y = y  (lower-triangular)
                if let Some(pos) = reg_diag.diag_pos_in_csc.get(bc).and_then(|&p| p) {
                    let eidx = reg_diag.csc_entry_of_pos[pos];
                    let d = reg_diag.block_slice(eidx); // stores U_ii (upper)
                    // in solve_upper_transpose: forward substitution on L = Uᵗ
                    for rr in 0..m {
                        let mut rhs = y[rr];
                        for cc in 0..rr {
                            // L(rr,cc) = U(cc,rr) at (col=rr, row=cc)
                            rhs -= d[rr * m + cc] * y[cc];
                        }
                        let diag = d[rr * m + rr];
                        debug_assert!(diag != 0.0);
                        y[rr] = rhs / diag;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra as na;

    use super::*;
    use crate::{
        BlockSparseMatrix,
        PartitionSpec,
    };

    fn ps(block_dim: usize, num_blocks: usize) -> PartitionSpec {
        PartitionSpec {
            block_dim,
            num_blocks,
        }
    }

    fn dm_from_cols(r: usize, c: usize, cols: &[f64]) -> na::DMatrix<f64> {
        // `cols` is provided column-major; nalgebra wants column-slice as well.
        na::DMatrix::from_column_slice(r, c, cols)
    }

    fn add_block(
        a: &mut BlockSparseMatrix,
        region: [usize; 2],
        br: usize,
        bc: usize,
        cols_col_major: &[f64],
        rdim: usize,
        cdim: usize,
    ) {
        let m = dm_from_cols(rdim, cdim, cols_col_major);
        a.add_block(&region, [br, bc], &m.as_view());
    }

    /// Build a small block **lower-triangular** matrix with:
    /// - partition 0: block_dim=2, num_blocks=2  (so 4 scalar rows/cols)
    /// - partition 1: block_dim=1, num_blocks=2  (so 2 scalar rows/cols)
    ///
    /// Regions:
    /// (0,0): 2x2 blocks of size 2x2 (fill diag+strictly-left)
    /// (1,0): 2x2 blocks of size 1x2 (fill some)
    /// (1,1): 2x2 blocks of size 1x1 (fill diag)
    fn make_lower_block_matrix() -> BlockSparseMatrix {
        let row_parts = [ps(2, 2), ps(1, 2)];
        let col_parts = [ps(2, 2), ps(1, 2)];
        let mut mat_l = BlockSparseMatrix::zero(&row_parts, &col_parts);

        // Region (0,0): block_shape = [2,2]
        // Diag block (0,0): 2x2 (lower-ish but not strictly needed)
        add_block(&mut mat_l, [0, 0], 0, 0, &[2.0, 0.1, 0.0, 1.5], 2, 2);
        // Strictly-left within region (br=1, bc=0): 2x2
        add_block(&mut mat_l, [0, 0], 1, 0, &[0.5, -0.2, 0.3, 0.7], 2, 2);
        // Diag block (1,1): 2x2
        add_block(&mut mat_l, [0, 0], 1, 1, &[1.7, 0.0, 0.0, 1.2], 2, 2);

        // Region (1,0): block_shape = [1,2]
        // br=0, bc=0: 1x2
        add_block(&mut mat_l, [1, 0], 0, 0, &[0.9, -0.1], 1, 2);
        // br=1, bc=0: 1x2
        add_block(&mut mat_l, [1, 0], 1, 0, &[-0.3, 0.4], 1, 2);
        // br=1, bc=1: 1x2 (strictly-left inside (1,1) will be on region (1,1) below)
        // (leave empty; this region is (1,0), not diagonal)

        // Region (1,1): block_shape = [1,1]
        // Diag blocks (0,0) and (1,1): 1x1 each
        add_block(&mut mat_l, [1, 1], 0, 0, &[1.3], 1, 1);
        add_block(&mut mat_l, [1, 1], 1, 1, &[0.8], 1, 1);

        // Note: We intentionally leave (0,1) empty to preserve lower-triangular pattern.
        mat_l
    }

    fn make_upper_block_matrix() -> BlockSparseMatrix {
        let row_parts = [ps(2, 2), ps(1, 2)];
        let col_parts = [ps(2, 2), ps(1, 2)];
        let mut mat_u = BlockSparseMatrix::zero(&row_parts, &col_parts);

        // Region (0,0): block_shape = [2,2]
        // Diag blocks (now with non-zero top-right to catch transpose indexing bugs):
        // column-major [a00, a10, a01, a11], with a10=0 (upper), a01 ≠ 0
        add_block(&mut mat_u, [0, 0], 0, 0, &[1.6, 0.0, 0.25, 1.4], 2, 2); // a01 = +0.25
        add_block(&mut mat_u, [0, 0], 1, 1, &[1.1, 0.0, -0.35, 1.9], 2, 2); // a01 = -0.35

        // Strictly-right within region (br=0, bc=1) — unchanged
        add_block(&mut mat_u, [0, 0], 0, 1, &[0.2, -0.1, 0.5, 0.3], 2, 2);

        // Region (0,1): block_shape = [2,1]  (upper off-diagonal) — unchanged
        add_block(&mut mat_u, [0, 1], 0, 0, &[0.7, -0.2], 2, 1);
        add_block(&mut mat_u, [0, 1], 1, 1, &[0.4, 0.9], 2, 1);

        // Region (1,1): block_shape = [1,1] (diag) — unchanged
        add_block(&mut mat_u, [1, 1], 0, 0, &[1.2], 1, 1);
        add_block(&mut mat_u, [1, 1], 1, 1, &[0.6], 1, 1);

        mat_u
    }

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() <= tol * (1.0 + x.abs() + y.abs()))
    }

    #[test]
    fn triangular_solve_lower_no_trans_works() {
        // Build mat_l and compress
        let mat_l = make_lower_block_matrix();
        let mat_c = mat_l.to_compressed();

        // Dense reference
        let dense_l = mat_l.to_dense();

        // Make a deterministic x_true
        let n = dense_l.ncols();
        let x_true = na::DVector::from_iterator(n, (0..n).map(|i| 0.5 + 0.1 * (i as f64)));

        // b = L * x_true
        let b_dense = &dense_l * &x_true;

        // Put b into a BlockVector with the same row partitions as L
        let mut b = BlockVector::zero(&[
            ps(2, 2), // 4 scalars
            ps(1, 2), // 2 scalars
        ]);
        b.scalar_vector_mut().copy_from(&b_dense);

        // Solve L x = b
        mat_c.triangular_solve_in_place(&mut b, Triangular::Lower, MultOp::NoTrans);
        let x_sol = b.scalar_vector();

        assert!(approx_eq(x_sol.as_slice(), x_true.as_slice(), 1e-10));
    }

    #[test]
    fn triangular_solve_lower_transpose_works() {
        // Build L and compress
        let mat_l = make_lower_block_matrix();
        let mat_c = mat_l.to_compressed();

        let dense_l = mat_l.to_dense();
        let dense_l_transpose = dense_l.transpose();

        let n = dense_l.ncols();
        let x_true = na::DVector::from_iterator(n, (0..n).map(|i| -0.2 + 0.05 * (i as f64)));

        // b = L^T * x_true
        let b_dense = &dense_l_transpose * &x_true;

        let mut b = BlockVector::zero(&[ps(2, 2), ps(1, 2)]);
        b.scalar_vector_mut().copy_from(&b_dense);

        // Solve L^T x = b
        mat_c.triangular_solve_in_place(&mut b, Triangular::Lower, MultOp::Transpose);
        let x_sol = b.scalar_vector();

        assert!(approx_eq(x_sol.as_slice(), x_true.as_slice(), 1e-10));
    }

    #[test]
    fn triangular_solve_upper_no_trans_works() {
        // Build U and compress
        let dense_u = make_upper_block_matrix();
        let mat_c = dense_u.to_compressed();

        let dense_u = dense_u.to_dense();

        let n = dense_u.ncols();
        let x_true = na::DVector::from_iterator(n, (0..n).map(|i| 1.0 - 0.03 * (i as f64)));

        // b = U * x_true
        let b_dense = &dense_u * &x_true;

        let mut b = BlockVector::zero(&[ps(2, 2), ps(1, 2)]);
        b.scalar_vector_mut().copy_from(&b_dense);

        // Solve U x = b
        mat_c.triangular_solve_in_place(&mut b, Triangular::Upper, MultOp::NoTrans);
        let x_sol = b.scalar_vector();

        assert!(approx_eq(x_sol.as_slice(), x_true.as_slice(), 1e-10));
    }

    #[test]
    fn triangular_solve_upper_transpose_works() {
        // Build U and compress
        let dense_u = make_upper_block_matrix();
        let mat_c = dense_u.to_compressed();

        let dense_u = dense_u.to_dense();
        let dense_u_transpose = dense_u.transpose();

        let n = dense_u.ncols();
        let x_true = na::DVector::from_iterator(n, (0..n).map(|i| 0.25 - 0.02 * (i as f64)));

        // b = U^T * x_true
        let b_dense = &dense_u_transpose * &x_true;

        let mut b = BlockVector::zero(&[ps(2, 2), ps(1, 2)]);
        b.scalar_vector_mut().copy_from(&b_dense);

        // Solve U^T x = b
        mat_c.triangular_solve_in_place(&mut b, Triangular::Upper, MultOp::Transpose);
        let x_sol = b.scalar_vector();

        assert!(approx_eq(x_sol.as_slice(), x_true.as_slice(), 1e-10));
    }
}
