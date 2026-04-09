use faer::sparse::Triplet;

use crate::matrix::{
    IsSymmetricMatrix,
    IsSymmetricMatrixBuilder,
    PartitionSet,
    sparse::{
        SparseSymmetricMatrixBuilder,
        TripletMatrix,
    },
};

/// Builder for the sparse symmetric matrix - [FaerSparseSymmetricMatrix].
#[derive(Debug, Clone)]
pub struct FaerSparseSymmetricMatrixBuilder {
    builder: SparseSymmetricMatrixBuilder,
}

impl IsSymmetricMatrixBuilder for FaerSparseSymmetricMatrixBuilder {
    type Matrix = FaerSparseSymmetricMatrix;

    fn zero(partitions: PartitionSet) -> Self {
        FaerSparseSymmetricMatrixBuilder {
            builder: SparseSymmetricMatrixBuilder::zero(partitions),
        }
    }

    fn scalar_dim(&self) -> usize {
        self.builder.scalar_dim()
    }

    fn partitions(&self) -> &PartitionSet {
        self.builder.partitions()
    }

    fn add_lower_block(
        &mut self,
        row_idx: crate::matrix::PartitionBlockIndex,
        col_idx: crate::matrix::PartitionBlockIndex,
        block: &nalgebra::DMatrixView<f64>,
    ) {
        self.builder.add_lower_block(row_idx, col_idx, block);
    }

    fn build(self) -> Self::Matrix {
        self.builder.into_faer_symmetric()
    }
}

/// Sparse symmetric matrix which wraps around [faer::sparse::SparseColMat].
///
/// The symmetric matrix is represented internally as a *upper triangular* matrix.
#[derive(Debug, Clone)]
pub struct FaerSparseSymmetricMatrix {
    pub(crate) upper: faer::sparse::SparseColMat<usize, f64>,
    pub(crate) partitions: PartitionSet,
}

impl FaerSparseSymmetricMatrix {
    pub(crate) fn new(lower_triplets: &TripletMatrix, partitions: PartitionSet) -> Self {
        let mut upper_triplets = Vec::with_capacity(lower_triplets.sorted_triplets().len());

        for &(i, j, v) in lower_triplets.sorted_triplets() {
            // Lower storage usually ensures i >= j, but be robust:
            let (r, c) = if i < j { (i, j) } else { (j, i) };
            upper_triplets.push(Triplet {
                row: r,
                col: c,
                val: v,
            });
        }
        FaerSparseSymmetricMatrix {
            upper: faer::sparse::SparseColMat::try_new_from_triplets(
                partitions.scalar_dim(),
                partitions.scalar_dim(),
                &upper_triplets,
            )
            .unwrap(),
            partitions,
        }
    }
}

impl IsSymmetricMatrix for FaerSparseSymmetricMatrix {
    fn has_block(
        &self,
        row_idx: crate::matrix::PartitionBlockIndex,
        col_idx: crate::matrix::PartitionBlockIndex,
    ) -> bool {
        let col_ptrs = self.upper.col_ptr();
        let row_ind = self.upper.row_idx();
        let row_range = self.partitions.block_range(row_idx);
        let col_range = self.partitions.block_range(col_idx);
        // Upper-triangle storage: entries (r, c) with r <= c, rows sorted per column.
        // Case 1: upper-triangle block (row scalars <= col scalars) → scan col_range columns.
        for j in col_range.start_idx..(col_range.start_idx + col_range.block_dim) {
            let rows = &row_ind[col_ptrs[j]..col_ptrs[j + 1]];
            let pos = rows.partition_point(|&r| r < row_range.start_idx);
            if pos < rows.len() && rows[pos] < row_range.start_idx + row_range.block_dim {
                return true;
            }
        }
        // Case 2: lower-triangle block → scan row_range columns for cols in col_range.
        for i in row_range.start_idx..(row_range.start_idx + row_range.block_dim) {
            let rows = &row_ind[col_ptrs[i]..col_ptrs[i + 1]];
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

        let col_ptrs = self.upper.col_ptr();
        let row_ind = self.upper.row_idx();
        let vals = self.upper.val();

        // Rows within each column are strictly sorted — use binary search to skip irrelevant rows.

        // Pass 1: Scan columns j in the requested column-range to extract the entries of
        //         the upper triangular matrix as is, which are inside the block.
        for j in col_range.start_idx..(col_range.start_idx + width) {
            let col_start = col_ptrs[j];
            let rows = &row_ind[col_start..col_ptrs[j + 1]];
            let lo = rows.partition_point(|&r| r < row_range.start_idx);
            let hi = rows.partition_point(|&r| r < row_range.start_idx + height);
            for pos in lo..hi {
                let i = rows[pos];
                out[(i - row_range.start_idx, j - col_range.start_idx)] = vals[col_start + pos];
            }
        }

        // Pass 2: Recover lower part inside the block. Scan columns i in the requested *row*-range.
        for i in row_range.start_idx..(row_range.start_idx + height) {
            let col_start = col_ptrs[i];
            let rows = &row_ind[col_start..col_ptrs[i + 1]];
            let lo = rows.partition_point(|&r| r < col_range.start_idx);
            let hi = rows.partition_point(|&r| r < col_range.start_idx + width);
            for pos in lo..hi {
                let j = rows[pos];
                if j < i {
                    out[(i - row_range.start_idx, j - col_range.start_idx)] = vals[col_start + pos];
                }
            }
        }

        out
    }

    fn partitions(&self) -> &PartitionSet {
        &self.partitions
    }

    fn block_range(&self, idx: crate::matrix::PartitionBlockIndex) -> crate::matrix::BlockRange {
        self.partitions.block_range(idx)
    }
}
