use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    ldlt::{
        SparseLdltFactor,
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

/// s
#[derive(Clone, Debug)]

pub struct SparseBackend {
    ldlt: SparseLdltFactor,
    tol_rel: f64,
    positive_pivot_idx: Vec<usize>,
    positive_pivot_values: DVector<f64>,
}

impl IsMinNormLdltBackend for SparseBackend {
    type LdltFactor = SparseLdltFactor;

    fn new(ldlt: Self::LdltFactor) -> Self {
        let n = ldlt.mat_l.scalar_dim();
        let positive_pivot_idx: Vec<usize> = (0..n).filter(|&t| ldlt.d[t] > 0.0).collect();
        let mut positive_pivot_values = DVector::<f64>::zeros(positive_pivot_idx.len());
        for (c, &t) in positive_pivot_idx.iter().enumerate() {
            positive_pivot_values[c] = ldlt.d[t];
        }
        Self {
            ldlt,
            tol_rel: 1e-12,
            positive_pivot_idx,
            positive_pivot_values,
        }
    }

    fn positive_pivot_idx(&self) -> &[usize] {
        &self.positive_pivot_idx
    }

    fn positive_pivot_values(&self) -> &DVector<f64> {
        &self.positive_pivot_values
    }

    fn scalar_dim(&self) -> usize {
        self.ldlt.mat_l.scalar_dim()
    }

    fn tol_rel(&self) -> f64 {
        self.tol_rel
    }

    fn column_of_mat_e(&self, col_j: usize, out: &mut [f64]) {
        // CSC strict-lower storage + implicit diag 1.0
        let l = &self.ldlt.mat_l;
        for i in 0..col_j {
            out[i] = 0.0;
        }
        out[col_j] = 1.0;
        for i in (col_j + 1)..self.scalar_dim() {
            out[i] = 0.0;
        }

        let c0 = l.storage_idx_by_col()[col_j];
        let c1 = l.storage_idx_by_col()[col_j + 1];
        let rows = &l.row_idx_storage()[c0..c1];
        let vals = &l.value_storage()[c0..c1];
        for (ri, &row) in rows.iter().enumerate() {
            debug_assert!(row > col_j);
            out[row] = vals[ri];
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
        // Generic SPD inverse via column solves:
        use crate::IsFactor; // you already implement solve_inplace for SparseLdltSystem
        let n = self.scalar_dim();
        let mut inv = DMatrix::<f64>::zeros(n, n);
        let mut col = nalgebra::DVector::<f64>::zeros(n);
        for j in 0..n {
            col.fill(0.0);
            col[j] = 1.0;
            let mut x = col.clone();
            // Solve A x = e_j
            self.ldlt.solve_inplace(&mut x).ok()?;
            inv.set_column(j, &x);
        }
        Some(inv)
    }

    fn block_range(&self, idx: PartitionBlockIndex) -> BlockRange {
        self.ldlt.partitions.block_range(idx)
    }
}

/// g
pub type SparseMinNormPsd = MinNormLdlt<SparseBackend>;
