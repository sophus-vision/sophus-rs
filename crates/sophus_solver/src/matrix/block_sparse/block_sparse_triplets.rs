use crate::matrix::{
    BlockColCompressedMatrix,
    IsCompressibleMatrix,
    PartitionIndexOffsets,
    PartitionSpec,
    grid::Grid,
};

/// Block sparse matrix, in triplet-like form.
#[derive(Debug, Clone)]
pub struct BlockSparseTripletMatrix {
    pub(crate) region_grid: Grid<BlockTripletRegion>,
    pub(crate) index_offsets: PartitionIndexOffsets,
    pub(crate) scalar_shape: [usize; 2],
    pub(crate) row_partitions: Vec<PartitionSpec>,
    pub(crate) col_partitions: Vec<PartitionSpec>,
}

impl BlockSparseTripletMatrix {
    /// scalar dimension of the matrix.
    #[inline]
    pub fn scalar_shape(&self) -> [usize; 2] {
        self.scalar_shape
    }

    /// region grid dimension
    #[inline]
    pub fn region_grid_shape(&self) -> [usize; 2] {
        [
            self.index_offsets.per_row_partition.len(),
            self.index_offsets.per_col_partition.len(),
        ]
    }

    #[inline]
    pub(crate) fn get_region_mut(&mut self, region_idx: &[usize; 2]) -> &mut BlockTripletRegion {
        self.region_grid.get_mut(region_idx)
    }
}

impl IsCompressibleMatrix for BlockSparseTripletMatrix {
    type Compressed = BlockColCompressedMatrix;

    fn compress(&self) -> Self::Compressed {
        self.to_block_col_compressed()
    }
}

/// A single block "AxB" in a region of the block sparse matrix.
#[derive(Debug, Clone)]
pub(crate) struct BlockTriplet {
    // index (row, column) of block within the region
    pub(crate) local_block_idx: [usize; 2],
    // scalar offset into flattened_block_storage
    pub(crate) storage_base: usize,
}

/// A homogeneous region in the block sparse matrix.
///
/// ```ascii
/// | M1×N1 ... M1×N1 |
/// |   .  .      .   |
/// |   .      .  .   |
/// | M1×N1 ... M1×N1 |
/// ```
///
/// It is represented by a list of index triplets, and a flattened storage of the blocks.
///
/// Note that the flattened storage in column-major order does not follow the macro shape of the
/// matrix (except by coincidence) but is purely based on the insertion order of the blocks /
/// add_block() calls.
#[derive(Debug, Clone)]
pub(crate) struct BlockTripletRegion {
    // Flattened storage of column-major matrix blocks
    pub(crate) flattened_block_storage: Vec<f64>,
    pub(crate) triplets: Vec<BlockTriplet>,
    // Dimensions (rows, columns) of each block
    pub(crate) block_shape: [usize; 2],
}
