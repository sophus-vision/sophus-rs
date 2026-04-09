use crate::matrix::{
    PartitionSet,
    grid::Grid,
};

/// Block-sparse `N x N` matrix, in triplet form.
#[derive(Debug, Clone)]
pub struct BlockSparseTripletMatrix {
    pub(crate) triplet_grid: Grid<BlockTripletRegion>,
    pub(crate) partitions: PartitionSet,
}

impl BlockSparseTripletMatrix {
    /// Scalar dimensions `N` of the `N x N` matrix.
    #[inline]
    pub fn scalar_dimension(&self) -> usize {
        self.partitions.scalar_dim()
    }

    /// Horizontal (or vertical) partition count `P`.
    #[inline]
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    #[inline]
    pub(crate) fn get_region_mut(&mut self, region_idx: &[usize; 2]) -> &mut BlockTripletRegion {
        self.triplet_grid.get_mut(region_idx)
    }
}

/// A homogeneous region in the block sparse matrix.
///
/// It is represented by a list of index triplets, and a flattened storage of the `H x W` blocks.
#[derive(Debug, Clone)]
pub(crate) struct BlockTripletRegion {
    // Flattened storage of column-major matrix blocks.
    pub(crate) flattened_block_storage: Vec<f64>,
    pub(crate) triplets: Vec<BlockTriplet>,
    // Dimensions (height, width) of each block.
    pub(crate) block_shape: [usize; 2],
}

/// A single `H x W` block in a region of the block sparse matrix.
#[derive(Debug, Clone)]
pub(crate) struct BlockTriplet {
    // Index (row, column) of block within the region.
    pub(crate) block_idx: [usize; 2],
    // Scalar index offset into BlockTripletRegion.flattened_block_storage.
    pub(crate) storage_base: usize,
}
