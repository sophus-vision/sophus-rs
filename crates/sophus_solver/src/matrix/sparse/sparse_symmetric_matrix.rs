use crate::matrix::{
    IsSymmetricMatrix,
    PartitionSet,
    sparse::SparseMatrix,
};

/// Symmetric `N x N` matrix in column compressed sparse form.
///
/// Internally it is represented by a lower triangular matrix.
#[derive(Clone, Debug)]
pub struct SparseSymmetricMatrix {
    lower: SparseMatrix,
    partitions: PartitionSet,
}

impl SparseSymmetricMatrix {
    pub(crate) fn new(lower: SparseMatrix, partitions: PartitionSet) -> Self {
        SparseSymmetricMatrix { lower, partitions }
    }

    /// Lower-triangular matrix.
    pub fn lower(&self) -> &SparseMatrix {
        &self.lower
    }

    /// Subtract `nu` from every scalar diagonal entry `M[i,i]` in-place.
    ///
    /// In lower-triangular CSC storage, the diagonal entry for column `j` is the
    /// entry where `row_idx == j`. The sparsity structure is unchanged.
    pub fn subtract_scalar_diagonal(&mut self, nu: f64) {
        let n = self.lower.scalar_dim();
        // Collect storage positions of diagonal entries first to avoid borrow conflicts.
        let mut diag_positions = Vec::with_capacity(n);
        for j in 0..n {
            let start = self.lower.storage_idx_by_col()[j];
            let end = self.lower.storage_idx_by_col()[j + 1];
            let rows = &self.lower.row_idx_storage()[start..end];
            if let Ok(pos) = rows.binary_search(&j) {
                diag_positions.push(start + pos);
            }
        }
        let vals = self.lower.value_storage_mut();
        for storage_idx in diag_positions {
            vals[storage_idx] -= nu;
        }
    }
}

impl IsSymmetricMatrix for SparseSymmetricMatrix {
    fn has_block(
        &self,
        row_idx: crate::matrix::PartitionBlockIndex,
        col_idx: crate::matrix::PartitionBlockIndex,
    ) -> bool {
        let storage_offset_by_col = self.lower.storage_idx_by_col();
        let row_idx_storage = self.lower.row_idx_storage();
        let row_range = self.partitions.block_range(row_idx);
        let col_range = self.partitions.block_range(col_idx);
        // Lower-triangle storage: entries (i, j) with i >= j, rows sorted per column.
        // Case 1: lower-triangle block → scan col_range columns for rows in row_range.
        for j in col_range.start_idx..(col_range.start_idx + col_range.block_dim) {
            let rows = &row_idx_storage[storage_offset_by_col[j]..storage_offset_by_col[j + 1]];
            let pos = rows.partition_point(|&r| r < row_range.start_idx);
            if pos < rows.len() && rows[pos] < row_range.start_idx + row_range.block_dim {
                return true;
            }
        }
        // Case 2: upper-triangle block → scan row_range columns for cols in col_range.
        for i in row_range.start_idx..(row_range.start_idx + row_range.block_dim) {
            let rows = &row_idx_storage[storage_offset_by_col[i]..storage_offset_by_col[i + 1]];
            let pos = rows.partition_point(|&r| r < col_range.start_idx);
            if pos < rows.len() && rows[pos] < col_range.start_idx + col_range.block_dim {
                return true;
            }
        }
        false
    }

    fn get_block(
        &self,
        row_idx: crate::matrix::PartitionBlockIndex,
        col_idx: crate::matrix::PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64> {
        use nalgebra::DMatrix;

        let row_range = self.partitions.block_range(row_idx);
        let col_range = self.partitions.block_range(col_idx);
        let height = row_range.block_dim;
        let width = col_range.block_dim;

        let mut out = DMatrix::<f64>::zeros(height, width);
        if height == 0 || width == 0 {
            return out;
        }

        let storage_offset_by_col = self.lower.storage_idx_by_col();
        let row_idx_storage = self.lower.row_idx_storage();
        let vals = self.lower.value_storage();

        // Rows within each column are strictly sorted — use binary search to skip irrelevant rows.

        // Pass 1: Scan columns j in the requested column-range to extract the entries of
        //         the lower triangular matrix as is, which are inside the block.
        for j in col_range.start_idx..(col_range.start_idx + width) {
            let start = storage_offset_by_col[j];
            let end = storage_offset_by_col[j + 1];
            let rows = &row_idx_storage[start..end];
            let lo = rows.partition_point(|&r| r < row_range.start_idx);
            let hi = rows.partition_point(|&r| r < row_range.start_idx + height);
            for pos in lo..hi {
                let i = rows[pos];
                out[(i - row_range.start_idx, j - col_range.start_idx)] = vals[start + pos];
            }
        }

        // Pass 2: Recover upper part inside the block. Scan columns i in the requested *row*-range.
        for i in row_range.start_idx..(row_range.start_idx + height) {
            let start = storage_offset_by_col[i];
            let end = storage_offset_by_col[i + 1];
            let rows = &row_idx_storage[start..end];
            let lo = rows.partition_point(|&r| r < col_range.start_idx);
            let hi = rows.partition_point(|&r| r < col_range.start_idx + width);
            for pos in lo..hi {
                let j = rows[pos];
                if j > i {
                    out[(i - row_range.start_idx, j - col_range.start_idx)] = vals[start + pos];
                }
            }
        }

        out
    }

    fn partitions(&self) -> &PartitionSet {
        &self.partitions
    }
}
