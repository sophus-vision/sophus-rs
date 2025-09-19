use crate::matrix::{
    IsSymmetricMatrix,
    PartitionSet,
    block_sparse::{
        BlockMatrixSubdivision,
        BlockSparseMatrix,
    },
};

/// Symmetric `N x N` matrix in column compressed block-sparse form.
///
/// It is a newtype over [BlockSparseMatrix]. The symmetric matrix is represented by
/// a lower block-triangular matrix.
#[derive(Clone, Debug)]
pub struct BlockSparseSymmetricMatrix {
    pub(crate) lower: BlockSparseMatrix,
}

impl BlockSparseSymmetricMatrix {
    /// Return the matrix subdivision.
    pub fn subdivision(&self) -> &BlockMatrixSubdivision {
        &self.lower.subdivision
    }

    /// Return the internal lower block-triangular matrix.
    pub fn into_lower(self) -> BlockSparseMatrix {
        self.lower
    }
}

impl IsSymmetricMatrix for BlockSparseSymmetricMatrix {
    fn get_block(
        &self,
        row_idx: crate::matrix::PartitionBlockIndex,
        col_idx: crate::matrix::PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64> {
        let is_lower = (row_idx.partition > col_idx.partition)
            || (row_idx.partition == col_idx.partition && row_idx.block >= col_idx.block);

        if is_lower {
            self.lower.get_block(row_idx, col_idx)
        } else {
            self.lower.get_block(col_idx, row_idx).transpose()
        }
    }

    fn partitions(&self) -> &PartitionSet {
        self.lower.subdivision.partitions()
    }
}
