use nalgebra::{
    DMatrixView,
    DMatrixViewMut,
};

use crate::matrix::{
    PartitionBlockIndex,
    PartitionSpec,
};

/// A region of a block vector.
#[derive(Debug, Clone)]
struct BlockDiagRegion {
    scalar_offset: usize,
    block_dim: usize,
}

/// Block column matrix
///
/// ```ascii
/// -------------------------------------------------------
/// | M0×M0           |                                   |
/// |      .          |                                   |
/// |          .      |                                   |
/// |           M0×M0 |                                   |
/// |-----------------|------------------                 |
/// |                 | M1×M1           |                 |
/// |                 |      .          |                 |
/// |                 |          .      |                 |
/// |                 |           M1×M1 |                 |
/// |                 ------------------------------------|
/// |                                   |                 |
/// |                                   |      *          |
/// |                                   |          *      |
/// |                                   |                 |
/// -------------------------------------------------------
/// ```
#[derive(Debug, Clone)]
pub struct BlockDiag {
    partitions: Vec<BlockDiagRegion>,
    storage: Vec<f64>,
}

impl BlockDiag {
    /// Create a block-diagonal matrix filled with zeros.
    pub fn zero(partition_specs: &[PartitionSpec]) -> Self {
        let mut total_elems = 0usize;
        let mut partitions: Vec<BlockDiagRegion> = Vec::with_capacity(partition_specs.len());

        for ps in partition_specs {
            partitions.push(BlockDiagRegion {
                scalar_offset: total_elems,
                block_dim: ps.block_dim,
            });
            total_elems += ps.block_dim.pow(2) * ps.block_count;
        }

        Self {
            partitions,
            storage: vec![0.0; total_elems],
        }
    }

    /// Add a block in the row/column slot specified by the given partition and (local) block index.
    pub fn add_block(&mut self, idx: PartitionBlockIndex, block: DMatrixView<'_, f64>) {
        use std::ops::AddAssign;
        self.get_block_mut(idx).add_assign(&block)
    }

    /// Get view of a block specified by the given partition and (local) block index.
    pub fn get_block(&self, idx: PartitionBlockIndex) -> DMatrixView<'_, f64> {
        let partition = &self.partitions[idx.partition];
        let block_area = partition.block_dim.pow(2);
        let base = partition.scalar_offset + idx.block * block_area;
        let storage = &self.storage[base..base + block_area];
        DMatrixView::from_slice(storage, partition.block_dim, partition.block_dim)
    }

    /// Get mutable view of a block specified by the given partition and (local) block index.
    pub fn get_block_mut(&mut self, idx: PartitionBlockIndex) -> DMatrixViewMut<'_, f64> {
        let partition = &self.partitions[idx.partition];
        let block_area = partition.block_dim.pow(2);
        let base = partition.scalar_offset + idx.block * block_area;
        let storage = &mut self.storage[base..base + block_area];
        DMatrixViewMut::from_slice(storage, partition.block_dim, partition.block_dim)
    }

    /// Zero out a block.
    #[inline]
    pub fn zero_block(&mut self, idx: PartitionBlockIndex) {
        let mut b = self.get_block_mut(idx);
        b.fill(0.0);
    }
}
