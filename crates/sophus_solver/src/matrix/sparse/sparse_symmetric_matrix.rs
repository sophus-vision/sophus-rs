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
}

impl IsSymmetricMatrix for SparseSymmetricMatrix {
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

        // Pass 1: Scan columns j in the requested column-range to extract the entries of
        //         the lower triangular matrix as is, which are inside the block.
        for j in col_range.start_idx..(col_range.start_idx + width) {
            let start = storage_offset_by_col[j];
            let end = storage_offset_by_col[j + 1];
            for p in start..end {
                let i = row_idx_storage[p];
                if i >= row_range.start_idx && i < row_range.start_idx + height {
                    out[(i - row_range.start_idx, j - col_range.start_idx)] = vals[p];
                }
            }
        }

        // Pass 2: Recover upper part inside the block. Scan columns i in the requested *row*-range.
        for i in row_range.start_idx..(row_range.start_idx + height) {
            let start = storage_offset_by_col[i];
            let end = storage_offset_by_col[i + 1];
            for p in start..end {
                let j = row_idx_storage[p];
                if j >= col_range.start_idx && j < col_range.start_idx + width && j > i
                {
                    out[(i - row_range.start_idx, j - col_range.start_idx)] = vals[p];
                }
            }
        }

        out
    }

    fn partitions(&self) -> &PartitionSet {
        &self.partitions
    }
}
