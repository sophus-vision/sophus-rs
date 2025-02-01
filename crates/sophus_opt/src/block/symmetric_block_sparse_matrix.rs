use std::collections::HashMap;

use super::symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder;

/// A homogeneous region in the block sparse matrix.
#[derive(Debug)]
struct BlockSparseRegion {
    // Flattened storage for all column-major matrix blocks
    flattened_block_storage: Vec<f64>,
    // hashmap of block indices to offsets in flattened_block_storage
    blocks: HashMap<[usize; 2], usize>,
    // Dimensions (rows, columns) of each block
    shape: [usize; 2],
}

/// A read-only block sparse matrix with random access to the blocks.
///
/// It consists of a grid of (N x N) block sparse regions. Within each region, the the blocks have
/// the same shape.
///
/// TODO: This struct is untested / wip.
#[derive(Debug)]
pub struct SymmetricBlockSparseMatrix {
    regions: Vec<BlockSparseRegion>,
    index_offset_per_family: Vec<usize>,
    scalar_dimension: usize,
    _num_families: usize,
}

impl SymmetricBlockSparseMatrix {
    /// Create block sparse matrix from block triplets.
    pub fn from_triplets(triplets: &SymmetricBlockSparseMatrixBuilder) -> Self {
        let _num_families = triplets.scalar_dimension();
        let mut regions = Vec::new();
        for row in 0.._num_families {
            for col in 0.._num_families {
                let triplet_region = triplets.get_region(&[row, col]);
                let mut region = BlockSparseRegion {
                    flattened_block_storage: Vec::new(),
                    blocks: HashMap::new(),
                    shape: triplet_region.shape,
                };
                let dim_sq = triplet_region.shape[0] * triplet_region.shape[1];
                for block_triplet in &triplet_region.triplets {
                    region.blocks.insert(
                        block_triplet.block_idx,
                        region.flattened_block_storage.len(),
                    );
                    region.flattened_block_storage.extend_from_slice(
                        &triplet_region.flattened_block_storage
                            [block_triplet.start_data_idx..block_triplet.start_data_idx + dim_sq],
                    );
                }
                regions.push(region);
            }
        }
        Self {
            regions,
            index_offset_per_family: triplets.index_offset_per_segment.clone(),
            scalar_dimension: triplets.scalar_dimension(),
            _num_families,
        }
    }

    /// Get a block as a slice and block-shape shape.
    pub fn get_dyn_block_slice(
        &self,
        grid_idx: &[usize; 2],
        block_idx: &[usize; 2],
    ) -> (&[f64], [usize; 2]) {
        let region = self.get_region(grid_idx);
        let offset = region.blocks[block_idx];
        (
            &region.flattened_block_storage[offset..offset + region.shape[0] * region.shape[1]],
            region.shape,
        )
    }

    /// Get a block as a dynamic matrix view.
    pub fn get_dyn_block(
        &self,
        grid_idx: &[usize; 2],
        block_idx: &[usize; 2],
    ) -> nalgebra::DMatrixView<f64> {
        let (slice, shape) = self.get_dyn_block_slice(grid_idx, block_idx);
        nalgebra::DMatrixView::from_slice(slice, shape[0], shape[1])
    }

    /// Get a block as a static matrix view.
    pub fn get_block<const M: usize, const N: usize>(
        &self,
        grid_idx: &[usize; 2],
        block_idx: &[usize; 2],
    ) -> nalgebra::SMatrixView<f64, M, N> {
        let (slice, shape) = self.get_dyn_block_slice(grid_idx, block_idx);
        assert_eq!(shape, [M, N]);
        nalgebra::SMatrixView::from_slice(slice)
    }

    /// The number of variable families.
    pub fn num_families(&self) -> usize {
        todo!()
    }

    fn get_region(&self, grid_idx: &[usize; 2]) -> &BlockSparseRegion {
        &self.regions[grid_idx[0] + grid_idx[1] * self.num_families()]
    }

    /// Convert the block sparse matrix to a dense matrix.
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let mut full_matrix =
            nalgebra::DMatrix::from_element(self.scalar_dimension, self.scalar_dimension, 0.0);

        for row in 0..self.num_families() {
            for col in 0..self.num_families() {
                let region = self.get_region(&[row, col]);
                for (block_idx, offset) in &region.blocks {
                    let scalar_row_offset =
                        self.index_offset_per_family[row] + block_idx[0] * region.shape[0];
                    let scalar_col_offset =
                        self.index_offset_per_family[col] + block_idx[1] * region.shape[1];

                    for c in 0..region.shape[1] {
                        for r in 0..region.shape[0] {
                            full_matrix[(scalar_row_offset + r, scalar_col_offset + c)] =
                                region.flattened_block_storage[*offset + c * region.shape[0] + r];
                        }
                    }
                }
            }
        }
        full_matrix
    }
}
