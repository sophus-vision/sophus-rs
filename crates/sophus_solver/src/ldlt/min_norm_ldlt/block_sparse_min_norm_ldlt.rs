use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    ldlt::{
        BlockSparseLdltFactor,
        min_norm_ldlt::{
            IsMinNormLdltBackend,
            MinNormLdlt,
        },
    },
    matrix::{
        BlockRange,
        PartitionBlockIndex,
    },
};

/// f
#[derive(Clone, Debug)]
pub struct BlockSparseBackend {
    ldlt: BlockSparseLdltFactor,
    tol_rel: f64,
    positive_pivot_idx: Vec<usize>,
    positive_pivot_values: DVector<f64>,
}

impl BlockSparseBackend {
    fn scalar_to_block_col(&self, t: usize) -> (usize, usize) {
        // Return (block_col_index, in_block_col_offset).
        // Uses subdivision methods to map scalar column -> (block, offset).
        let sub = &self.ldlt.mat_l.subdivision;
        // Walk blocks to find where 't' falls. If you have a direct helper, use it.
        let mut acc = 0usize;
        for bc in 0..sub.block_count() {
            let w = sub.block_dim(sub.idx(bc).partition);
            if t < acc + w {
                return (bc, t - acc);
            }
            acc += w;
        }
        unreachable!("scalar column out of range");
    }

    fn scalar_offset(&self, block_row: usize) -> usize {
        self.ldlt.mat_l.subdivision.scalar_offset(block_row)
    }
}

impl IsMinNormLdltBackend for BlockSparseBackend {
    fn scalar_dim(&self) -> usize {
        self.ldlt.mat_l.subdivision.scalar_dim()
    }

    fn tol_rel(&self) -> f64 {
        self.tol_rel
    }

    type LdltFactor = BlockSparseLdltFactor;

    fn new(ldlt: Self::LdltFactor) -> Self {
        let sub = &ldlt.mat_l.subdivision;
        let mut positive_pivot_idx = Vec::new();
        let mut positive_pivot_values = Vec::new();
        let mut scalar_col = 0usize;
        for bc in 0..sub.block_count() {
            let col_idx = sub.idx(bc);
            let w = sub.block_dim(col_idx.partition);
            let d_block = ldlt.block_diag.d.get_block(col_idx);
            debug_assert_eq!(d_block.len(), w);
            for c in 0..w {
                if d_block[c] > 0.0 {
                    positive_pivot_idx.push(scalar_col + c);
                    positive_pivot_values.push(d_block[c]);
                }
            }
            scalar_col += w;
        }

        Self {
            ldlt,
            tol_rel: 1e-12,
            positive_pivot_idx,
            positive_pivot_values: DVector::from_vec(positive_pivot_values),
        }
    }

    fn positive_pivot_idx(&self) -> &[usize] {
        &self.positive_pivot_idx
    }

    fn positive_pivot_values(&self) -> &DVector<f64> {
        &self.positive_pivot_values
    }

    fn column_of_mat_e(&self, col_j: usize, out: &mut [f64]) {
        let n = self.scalar_dim();
        assert_eq!(out.len(), n);
        out.fill(0.0);

        let sub = &self.ldlt.mat_l.subdivision;
        let (bj, co) = self.scalar_to_block_col(col_j);

        // Diagonal block L[j,j] and its column 'co'
        let col_idx = sub.idx(bj);
        let l_jj = self.ldlt.block_diag.mat_l.get_block(col_idx); // square, lower-triangular
        let hj = l_jj.nrows(); // == l_jj.ncols() for diagonal blocks
        let row0_j = self.scalar_offset(bj);

        // Copy the diagonal block column into out at rows of block-row j.
        for r in 0..hj {
            out[row0_j + r] = l_jj[(r, co)];
        }

        // Prepare v = L[j,j] * e_co  (this is just the same column we copied above)
        // Reuse it to form off-diagonal contributions: L[i,j] * v.
        let mut v = vec![0.0; hj];
        for r in 0..hj {
            v[r] = l_jj[(r, co)];
        }

        // Off-diagonal blocks in column j
        for e in self.ldlt.mat_l.col(bj).iter() {
            let bi = e.global_block_row_idx;
            debug_assert!(bi > bj);
            let l_ij = e.view; // (h_i × h_j)
            let hi = l_ij.nrows();
            let row0_i = self.scalar_offset(bi);

            // y = L[i,j] * v
            for r in 0..hi {
                let mut acc = 0.0;
                // h_j equals v.len()
                for s in 0..v.len() {
                    acc += l_ij[(r, s)] * v[s];
                }
                out[row0_i + r] = acc;
            }
        }
    }

    fn try_column_of_inverse(&self, col_range: &BlockRange) -> Option<DMatrix<f64>> {
        if self.rank() != self.scalar_dim() {
            return None;
        }
        use crate::IsFactor;

        let n = self.scalar_dim();
        debug_assert!(col_range.start_idx + col_range.block_dim <= n);
        let mut cols = DMatrix::<f64>::zeros(n, col_range.block_dim);

        let mut rhs = nalgebra::DVector::<f64>::zeros(n);
        for k in 0..col_range.block_dim {
            rhs.fill(0.0);
            rhs[col_range.start_idx + k] = 1.0;
            let mut x = rhs.clone();
            self.ldlt.solve_inplace(&mut x).ok()?; // Solve A x = e_{sj+k}
            cols.set_column(k, &x);
        }
        Some(cols)
    }

    fn try_inverse(&self) -> Option<DMatrix<f64>> {
        if self.rank() != self.scalar_dim() {
            return None;
        }
        // Generic SPD via solves (uses your BlockSparseLdltSystem::solve_inplace)
        use crate::IsFactor;
        let n = self.scalar_dim();
        let mut inv = DMatrix::<f64>::zeros(n, n);
        let mut col = nalgebra::DVector::<f64>::zeros(n);
        for j in 0..n {
            col.fill(0.0);
            col[j] = 1.0;
            let mut x = col.clone();
            self.ldlt.solve_inplace(&mut x).ok()?;
            inv.set_column(j, &x);
        }
        Some(inv)
    }

    fn block_range(&self, idx: PartitionBlockIndex) -> BlockRange {
        self.ldlt.mat_l.subdivision.partitions().block_range(idx)
    }
}

/// g
pub type BlockSparseMinNormPsd = MinNormLdlt<BlockSparseBackend>;
