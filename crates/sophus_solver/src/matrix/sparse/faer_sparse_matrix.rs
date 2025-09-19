use faer::sparse::Triplet;

use crate::{
    matrix::{
        IsSymmetricMatrix,
        PartitionSet,
        sparse::{
            SparseSymmetricMatrixBuilder,
            TripletMatrix,
        },
    },
    prelude::*,
};

/// Builder for the sparse matrix - [FaerSparseMatrix].
/// 
/// The created matrix will be symmetric.
#[derive(Debug, Clone)]
pub struct FaerSparseMatrixBuilder {
    builder: SparseSymmetricMatrixBuilder,
}

impl IsSymmetricMatrixBuilder for FaerSparseMatrixBuilder {
    type Matrix = FaerSparseMatrix;

    fn zero(partitions: PartitionSet) -> Self {
        FaerSparseMatrixBuilder {
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
        self.builder.into_faer()
    }
}

/// Sparse `N x N` matrix which wraps around [faer::sparse::SparseColMat].
#[derive(Debug)]
pub struct FaerSparseMatrix {
    pub(crate) square: faer::sparse::SparseColMat<usize, f64>,
    partitions: PartitionSet,
}

impl FaerSparseMatrix {
    pub(crate) fn new(lower_triplets: &TripletMatrix, partitions: PartitionSet) -> Self {
        let mut triplets = Vec::with_capacity(lower_triplets.sorted_triplets().len() * 2);

        for &(row, col, val) in lower_triplets.sorted_triplets() {
            // Always add the original lower entry
            triplets.push(Triplet { row, col, val });
            // Add the mirrored entry if it's off-diagonal
            if row != col {
                triplets.push(Triplet {
                    row: col,
                    col: row,
                    val,
                });
            }
        }

        FaerSparseMatrix {
            square: faer::sparse::SparseColMat::try_new_from_triplets(
                lower_triplets.scalar_dimension(),
                lower_triplets.scalar_dimension(),
                &triplets,
            )
            .unwrap(),
            partitions,
        }
    }
}

impl IsSymmetricMatrix for FaerSparseMatrix {
    fn get_block(
        &self,
        row_idx: crate::matrix::PartitionBlockIndex,
        col_idx: crate::matrix::PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64> {
        use nalgebra::DMatrix;

        let r = self.partitions.block_range(row_idx);
        let c = self.partitions.block_range(col_idx);
        let (ri, di) = (r.start_idx, r.block_dim);
        let (cj, dj) = (c.start_idx, c.block_dim);

        let mut out = DMatrix::<f64>::zeros(di, dj);
        if di == 0 || dj == 0 {
            return out;
        }

        // Access CSC internals from faer::sparse::SparseColMat
        let col_ptrs = self.square.col_ptr();
        let row_ind = self.square.row_idx();
        let vals = self.square.val();

        for j in cj..(cj + dj) {
            let col_start = col_ptrs[j];
            let col_end = col_ptrs[j + 1];

            for p in col_start..col_end {
                let i = row_ind[p];
                if i >= ri && i < ri + di {
                    out[(i - ri, j - cj)] = vals[p];
                }
            }
        }
        out
    }

    /// Partition set.
    fn partitions(&self) -> &PartitionSet {
        &self.partitions
    }
}
