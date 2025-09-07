use std::collections::hash_map::{
    Entry,
    HashMap,
};

use nalgebra::DMatrixView;

/// Block-column compressed matrix.
#[derive(Debug)]
pub struct BlockColCompressedMatrix {
    /// The block matrix subdivision structure.
    pub subdivision: BlockMatrixSubdivision,
    /// Grid of regions (R × C).
    pub regions: Grid<BlockRegion>,
    /// Pattern of non-zero blocks, sorted by block columns.
    pub block_col_pattern: BlockColCompressedPattern,
}

/// Pattern of non-zero blocks in sparse matrix, sorted by block columns.
#[derive(Debug, Clone)]
pub struct BlockColCompressedPattern {
    /// List of indices into nonzero_blocks (len = BC+1).
    pub nonzero_idx_by_block_col: Vec<usize>,

    /// Flat list of non-zero blocks.
    pub nonzero_blocks: Vec<NonZeroBlock>,
}

/// A region of the [BlockColCompressedMatrix].
#[derive(Debug, Clone)]
pub struct BlockRegion {
    /// Back-to-back blocks, column-major.
    pub storage: Vec<f64>,
}

/// A non-zero block of the sparse matrix.
///
/// This struct does not hold the actual data by indexes into [BlockRegion::storage].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NonZeroBlock {
    /// Block-row index (0..BR-1).
    pub block_row_idx: u32,
    /// Scalar offset into [BlockRegion::storage].
    pub storage_base: u32,
}

impl BlockColCompressedMatrix {
    /// Total number of block rows in this matrix: BR.
    #[inline]
    pub fn block_row_count(&self) -> usize {
        self.subdivision.block_row_count()
    }

    /// Total number of block columns in this matrix: BC.
    #[inline]
    pub fn block_col_count(&self) -> usize {
        self.subdivision.block_col_count()
    }

    /// Number of non-zero blocks in sparse matrix.
    #[inline]
    pub fn non_zero_block_count(&self) -> usize {
        self.block_col_pattern.nonzero_blocks.len()
    }

    #[inline]
    fn col_range_by_block(&self, global_block_col_idx: usize) -> std::ops::Range<usize> {
        self.block_col_pattern.nonzero_idx_by_block_col[global_block_col_idx]
            ..self.block_col_pattern.nonzero_idx_by_block_col[global_block_col_idx + 1]
    }

    /// Return non-zero block entry given block column index.
    #[inline]
    pub fn non_zero_block(&self, block_col_idx: usize) -> &[NonZeroBlock] {
        &self.block_col_pattern.nonzero_blocks[self.col_range_by_block(block_col_idx)]
    }

    /// Return column compressed pattern of the upper-triangular blocks of the sparse matrix.
    pub fn build_pattern_upper(&self) -> ColumnCompressedPattern {
        let block_col_count = self.block_col_count();
        assert_eq!(
            block_col_count,
            self.block_row_count(),
            "square block partitions required"
        );

        let mut counts = vec![0usize; block_col_count];
        for col_j in 0..block_col_count {
            for storage_idx in self.col_range_by_block(col_j) {
                let row_i =
                    self.block_col_pattern.nonzero_blocks[storage_idx].block_row_idx as usize;
                if row_i > col_j {
                    counts[row_i] += 1;
                }
            }
        }

        // Prefix sum
        let mut storage_offset_by_col = vec![0usize; block_col_count + 1];
        for i in 0..block_col_count {
            storage_offset_by_col[i + 1] = storage_offset_by_col[i] + counts[i];
        }
        let mut next = storage_offset_by_col.clone();
        let mut row_ind = vec![0usize; storage_offset_by_col[block_col_count]];

        // Fill: column i collects all j<i with A(i,j) present
        for col_j in 0..block_col_count {
            for storage_idx in self.col_range_by_block(col_j) {
                let row_i =
                    self.block_col_pattern.nonzero_blocks[storage_idx].block_row_idx as usize;
                if row_i > col_j {
                    let dst = next[row_i];
                    row_ind[dst] = col_j; // strictly upper row index
                    next[row_i] += 1;
                }
            }
        }
        ColumnCompressedPattern::new(
            block_col_count,
            block_col_count,
            storage_offset_by_col,
            row_ind,
        )
    }

    #[inline]
    /// Return block column slice.
    pub fn col(&self, global_block_col_idx: usize) -> BlockColSlice<'_> {
        BlockColSlice {
            mat_subdivision: &self.subdivision,
            regions: &self.regions,
            block_col_idx: global_block_col_idx,
            slice: self.non_zero_block(global_block_col_idx),
        }
    }

    /// Return k-th block of column `block_col_idx`.
    #[inline]
    pub fn kth_block<'a>(
        &'a self,
        block_col_idx: usize,
        k: usize,
    ) -> (usize, DMatrixView<'a, f64>) {
        let nonzero_idx = self.block_col_pattern.nonzero_idx_by_block_col[block_col_idx] + k;
        let entry: NonZeroBlock = self.block_col_pattern.nonzero_blocks[nonzero_idx];
        let global_block_row = entry.block_row_idx as usize;

        let row_info = self.subdivision.row_info(global_block_row);
        let col_info = self.subdivision.col_info(block_col_idx);
        let region_idx = [row_info.row_partition_idx, col_info.col_partition_idx];
        let base = entry.storage_base as usize;
        let storage = &self.regions.get(&region_idx).storage;
        let block_size = row_info.block_height * col_info.block_width;

        debug_assert_le!(base + block_size, storage.len());
        (
            global_block_row,
            DMatrixView::from_slice(
                &storage[base..base + block_size],
                row_info.block_height,
                col_info.block_width,
            ),
        )
    }
}

/// Compressed sparse matrix in CSC-like form.
#[derive(Debug, Clone)]
pub struct BlockMatrixSubdivision {
    // Scalar matrix dimensions: M * N.
    scalar_shape: [usize; 2],

    // Scalar index offset per block vertically (len = BR+1).
    scalar_row_offset_by_block: Vec<usize>,
    // Scalar index offset per block horizontally (len = BC+1).
    scalar_col_offset_by_block: Vec<usize>,

    // Maps block row index to row partition index (len = BR)
    row_partition_idx_by_block: Vec<u16>,
    // Maps block column index to column partition index (len = BC).
    col_partition_idx_by_block: Vec<u16>,

    // Block index offset per vertical partition (len = R+1).
    block_row_offset_by_partition: Vec<usize>,
    // Block index offset by horizontal partition (len = C+1).
    block_col_offset_by_partition: Vec<usize>,

    // len = R
    row_partitions: Vec<PartitionSpec>,
    // len = C
    col_partitions: Vec<PartitionSpec>,
}

/// Block-row information.
pub struct BlockRowInfo {
    /// The partition this row of blocks is in.
    pub row_partition_idx: usize,
    /// Local index of the row of blocks.
    pub local_block_row_idx: usize,
    /// Height of each block in this partition.
    pub block_height: usize,
}

/// Block-column information.
pub struct BlockColInfo {
    /// The partition this column of blocks is in.
    pub col_partition_idx: usize,
    /// Local index of this column of blocks.
    pub local_col_block_idx: usize,
    /// Width of each block in this partition.
    pub block_width: usize,
}

impl BlockMatrixSubdivision {
    /// Scalar matrix dimensions: M x N.
    #[inline]
    pub fn scalar_shape(&self) -> &[usize; 2] {
        &self.scalar_shape
    }

    /// Region grid dimensions: R x C.
    #[inline]
    pub fn region_grid_shape(&self) -> [usize; 2] {
        [self.row_partitions.len(), self.col_partitions.len()]
    }

    /// Total number of block rows in this matrix: BR.
    #[inline]
    pub fn block_row_count(&self) -> usize {
        self.row_partition_idx_by_block.len()
    }

    /// Total number of block columns in this matrix: BC.
    #[inline]
    pub fn block_col_count(&self) -> usize {
        self.col_partition_idx_by_block.len()
    }

    /// Maximum block height.
    #[inline]
    pub fn max_block_height(&self) -> usize {
        self.row_partitions
            .iter()
            .map(|s| s.block_dimension)
            .max()
            .unwrap()
    }

    /// Maximum block width.
    #[inline]
    pub fn max_block_width(&self) -> usize {
        self.col_partitions
            .iter()
            .map(|s| s.block_dimension)
            .max()
            .unwrap()
    }

    /// Returns scalar row offset into whole matrix given block row index.
    #[inline]
    pub fn scalar_row_offset(&self, block_row_idx: usize) -> usize {
        self.scalar_row_offset_by_block[block_row_idx]
    }

    /// Returns scalar column offset into whole matrix given block column index.
    #[inline]
    pub fn scalar_col_offset(&self, global_col_block_idx: usize) -> usize {
        self.scalar_col_offset_by_block[global_col_block_idx]
    }

    /// List of row partitions.
    #[inline]
    pub fn row_partitons(&self) -> &[PartitionSpec] {
        &self.row_partitions
    }

    /// List of column partitions.
    #[inline]
    pub fn col_partitons(&self) -> &[PartitionSpec] {
        &self.col_partitions
    }

    /// Return block row information given block row index.
    #[inline]
    pub fn row_info(&self, block_row_idx: usize) -> BlockRowInfo {
        let row_partition_idx = self.row_partition_idx_by_block[block_row_idx] as usize;
        BlockRowInfo {
            row_partition_idx,
            local_block_row_idx: block_row_idx
                - self.block_row_offset_by_partition[row_partition_idx],
            block_height: self.row_partitions[row_partition_idx].block_dimension,
        }
    }

    /// Return block column information given block column index.
    #[inline]
    pub fn col_info(&self, global_col_block_idx: usize) -> BlockColInfo {
        let col_partition_idx = self.col_partition_idx_by_block[global_col_block_idx] as usize;
        BlockColInfo {
            col_partition_idx,
            local_col_block_idx: global_col_block_idx
                - self.block_col_offset_by_partition[col_partition_idx],
            block_width: self.col_partitions[col_partition_idx].block_dimension,
        }
    }

    /// Block shape given block index tuple.
    #[inline]
    pub fn block_shape(&self, block_idx: [usize; 2]) -> [usize; 2] {
        let r = self.row_partition_idx_by_block[block_idx[0]] as usize;
        let c = self.col_partition_idx_by_block[block_idx[1]] as usize;
        [
            self.row_partitions[r].block_dimension,
            self.col_partitions[c].block_dimension,
        ]
    }
}

/// Slice of a block-column.
#[derive(Debug)]
pub struct BlockColSlice<'a> {
    mat_subdivision: &'a BlockMatrixSubdivision,
    regions: &'a Grid<BlockRegion>,
    block_col_idx: usize,
    slice: &'a [NonZeroBlock],
}

/// Block entry containing a view to its data and the block-row index.
pub struct BlockEntry<'a> {
    /// Index of the block-row.
    pub block_row_idx: usize,
    /// View of the matrix block.
    pub view: DMatrixView<'a, f64>,
}

impl<'a> BlockColSlice<'a> {
    /// Iterate over blocks in this column of blocks.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = BlockEntry<'a>> + 'a {
        let mat_subdivision = self.mat_subdivision;
        let regions = self.regions;

        let partition_idx_col_j =
            mat_subdivision.col_partition_idx_by_block[self.block_col_idx] as usize;
        let block_width_col_j = mat_subdivision.col_partitions[partition_idx_col_j].block_dimension;

        self.slice.iter().map(move |nonzero_block| {
            let block_row_idx = nonzero_block.block_row_idx as usize;

            let partition_idx_row_i =
                mat_subdivision.row_partition_idx_by_block[block_row_idx] as usize;
            let block_height_row_i =
                mat_subdivision.row_partitions[partition_idx_row_i].block_dimension;
            let base = nonzero_block.storage_base as usize;
            let storage = &regions
                .get(&[partition_idx_row_i, partition_idx_col_j])
                .storage;

            debug_assert!(base + block_height_row_i * block_width_col_j <= storage.len());

            BlockEntry {
                block_row_idx,
                view: DMatrixView::from_slice(
                    &storage[base..base + block_height_row_i * block_width_col_j],
                    block_height_row_i,
                    block_width_col_j,
                ),
            }
        })
    }
}

use crate::{
    debug_assert_le,
    matrix::{
        BlockSparseTripletMatrix,
        BlockTriplet,
        ColumnCompressedPattern,
        PartitionSpec,
        grid::Grid,
    },
};
impl BlockSparseTripletMatrix {
    /// Convert to [BlockColCompressedMatrix].
    pub fn to_block_col_compressed(&self) -> BlockColCompressedMatrix {
        // Number of partition in grid vertically.
        let row_partition_count = self.index_offsets.per_row_partition.len();
        // Number of partitions in grid horizontally.
        let col_partition_count = self.index_offsets.per_col_partition.len();
        let total_row_count = self.scalar_shape[0];
        let total_col_count = self.scalar_shape[1];

        // Blocks-per-partition.
        let mut blocks_per_row_partition = vec![0usize; row_partition_count];
        for row_partition_idx in 0..row_partition_count {
            let start = self.index_offsets.per_row_partition[row_partition_idx];
            let end = if row_partition_idx + 1 < row_partition_count {
                self.index_offsets.per_row_partition[row_partition_idx + 1]
            } else {
                total_row_count
            };
            let block_height = self.row_partitions[row_partition_idx].block_dimension;
            debug_assert_eq!((end - start) % block_height, 0,);
            if block_height > 0 {
                blocks_per_row_partition[row_partition_idx] = (end - start) / block_height;
            }
        }
        let mut blocks_per_col_partition = vec![0usize; col_partition_count];
        for col_partition_idx in 0..col_partition_count {
            let start = self.index_offsets.per_col_partition[col_partition_idx];
            let end = if col_partition_idx + 1 < col_partition_count {
                self.index_offsets.per_col_partition[col_partition_idx + 1]
            } else {
                total_col_count
            };
            let block_width = self.col_partitions[col_partition_idx].block_dimension;
            debug_assert_eq!((end - start) % block_width, 0,);
            if block_width > 0 {
                blocks_per_col_partition[col_partition_idx] = (end - start) / block_width;
            }
        }

        // (Global) block row offset by row partition.
        let mut block_row_offset_by_partition = vec![0usize; row_partition_count + 1];
        for r in 0..row_partition_count {
            block_row_offset_by_partition[r + 1] =
                block_row_offset_by_partition[r] + blocks_per_row_partition[r];
        }

        // (Global) block column offset by row partition.
        let mut block_col_offset_by_partition = vec![0usize; col_partition_count + 1];
        for c in 0..col_partition_count {
            block_col_offset_by_partition[c + 1] =
                block_col_offset_by_partition[c] + blocks_per_col_partition[c];
        }

        let block_row_count = block_row_offset_by_partition[row_partition_count];
        let block_col_count = block_col_offset_by_partition[col_partition_count];

        // Calculate scalar offsets into whole matrix for each block row index.
        let mut scalar_row_offset_by_block = Vec::with_capacity(block_row_count + 1);
        {
            let mut offset = 0usize;
            for row_partition_idx in 0..row_partition_count {
                let block_height = self.row_partitions[row_partition_idx].block_dimension;
                for _ in 0..blocks_per_row_partition[row_partition_idx] {
                    scalar_row_offset_by_block.push(offset);
                    offset += block_height;
                }
            }
            scalar_row_offset_by_block.push(offset);
            debug_assert_eq!(offset, total_row_count);
        }

        // Calculate scalar offsets into whole matrix for each block column index.
        let mut scalar_col_offset_by_block = Vec::with_capacity(block_col_count + 1);
        {
            let mut offset = 0usize;
            for col_partition_idx in 0..col_partition_count {
                let block_width = self.col_partitions[col_partition_idx].block_dimension;
                for _ in 0..blocks_per_col_partition[col_partition_idx] {
                    scalar_col_offset_by_block.push(offset);
                    offset += block_width;
                }
            }
            scalar_col_offset_by_block.push(offset);
            debug_assert_eq!(offset, total_col_count);
        }

        // Fast maps: block row/col → partition row/col.
        let mut row_partition_idx_by_block = vec![0u16; block_row_count];
        for row_partition_idx in 0..row_partition_count {
            for block_row_idx in block_row_offset_by_partition[row_partition_idx]
                ..block_row_offset_by_partition[row_partition_idx + 1]
            {
                row_partition_idx_by_block[block_row_idx] = row_partition_idx as u16;
            }
        }
        let mut col_partition_idx_by_block = vec![0u16; block_col_count];
        for col_partition_idx in 0..col_partition_count {
            for block_col_idx in block_col_offset_by_partition[col_partition_idx]
                ..block_col_offset_by_partition[col_partition_idx + 1]
            {
                col_partition_idx_by_block[block_col_idx] = col_partition_idx as u16;
            }
        }

        // Temporary per-region buffers.
        #[derive(Default, Clone)]
        struct TempRegion {
            storage: Vec<f64>,
            triplets: Vec<BlockTriplet>,
        }
        let mut dedup_storage = Grid::new(
            [row_partition_count, col_partition_count],
            TempRegion::default(),
        );
        let mut storage_base_by_local_block: HashMap<[usize; 2], usize> = HashMap::new();

        for row_partition_idx in 0..row_partition_count {
            let block_height = self.row_partitions[row_partition_idx].block_dimension;
            for col_partition_idx in 0..col_partition_count {
                let block_width = self.col_partitions[col_partition_idx].block_dimension;
                let block_size = block_height * block_width;

                let src_region = self
                    .region_grid
                    .get(&[row_partition_idx, col_partition_idx]);
                if src_region.triplets.is_empty() {
                    continue;
                }

                storage_base_by_local_block.clear();
                storage_base_by_local_block.reserve(src_region.triplets.len());

                // Destination buffers
                let tmp_region = dedup_storage.get_mut(&[row_partition_idx, col_partition_idx]);

                // Walk all triplets and sum duplicates:
                for triplet in &src_region.triplets {
                    let dst_base = match storage_base_by_local_block.entry(triplet.local_block_idx)
                    {
                        Entry::Occupied(e) => *e.get(),
                        Entry::Vacant(vacant_entry) => {
                            // allocate space for a fresh block (zero-initialized)
                            let storage_base = tmp_region.storage.len();
                            tmp_region.storage.resize(storage_base + block_size, 0.0);
                            vacant_entry.insert(storage_base);
                            tmp_region.triplets.push(BlockTriplet {
                                local_block_idx: triplet.local_block_idx,
                                storage_base,
                            });
                            storage_base
                        }
                    };

                    let src_base = triplet.storage_base;
                    let src = &src_region.flattened_block_storage[src_base..src_base + block_size];

                    // Column-major add
                    for j in 0..block_width {
                        let dst_col = &mut tmp_region.storage
                            [dst_base + j * block_height..dst_base + (j + 1) * block_height];
                        let src_col = &src[j * block_height..(j + 1) * block_height];
                        // elementwise +=
                        for i in 0..block_height {
                            dst_col[i] += src_col[i];
                        }
                    }
                }
            }
        }

        // Materialize the output regions grid.
        let mut regions = Grid::new(
            [row_partition_count, col_partition_count],
            BlockRegion {
                storage: Vec::new(),
            },
        );
        for row_partition_idx in 0..row_partition_count {
            for col_partition_idx in 0..col_partition_count {
                regions
                    .get_mut(&[row_partition_idx, col_partition_idx])
                    .storage = std::mem::take(
                    &mut dedup_storage
                        .get_mut(&[row_partition_idx, col_partition_idx])
                        .storage,
                );
            }
        }

        // Collect block column each containing a list of non-zero row entries.
        let mut block_colums: Vec<Vec<NonZeroBlock>> = vec![Vec::new(); block_col_count];

        for row_partition_idx in 0..row_partition_count {
            let block_height = self.row_partitions[row_partition_idx].block_dimension;
            for col_partition_idx in 0..col_partition_count {
                let block_width = self.col_partitions[col_partition_idx].block_dimension;
                let stride = block_height * block_width;

                let unique_region = dedup_storage.get(&[row_partition_idx, col_partition_idx]);
                if unique_region.triplets.is_empty() {
                    continue;
                }

                for triplet in unique_region.triplets.iter() {
                    let block_row_idx = block_row_offset_by_partition[row_partition_idx]
                        + triplet.local_block_idx[0];
                    let block_col_idx = block_col_offset_by_partition[col_partition_idx]
                        + triplet.local_block_idx[1];

                    debug_assert!(
                        block_row_idx < block_row_count && block_col_idx < block_col_count
                    );
                    debug_assert_eq!(
                        regions
                            .get(&[row_partition_idx, col_partition_idx])
                            .storage
                            .len()
                            % stride,
                        0
                    );

                    block_colums[block_col_idx].push(NonZeroBlock {
                        block_row_idx: block_row_idx as u32,
                        storage_base: triplet.storage_base as u32,
                    });
                }
            }
        }

        // Sort each block-column by block-row index.
        for block_col_idx in 0..block_col_count {
            block_colums[block_col_idx].sort_unstable_by_key(|e| e.block_row_idx);
            debug_assert!(
                block_colums[block_col_idx]
                    .windows(2)
                    .all(|w| w[0].block_row_idx < w[1].block_row_idx),
                "Duplicate block rows remained in column {block_col_idx} after dedup"
            );
        }

        // Build non-zero block storage.
        let mut nonzero_idx_by_block_col = Vec::with_capacity(block_col_count + 1);
        nonzero_idx_by_block_col.push(0);
        for block_col_idx in 0..block_col_count {
            nonzero_idx_by_block_col
                .push(nonzero_idx_by_block_col[block_col_idx] + block_colums[block_col_idx].len());
        }

        let mut nonzero_blocks = vec![
            NonZeroBlock {
                block_row_idx: 0,
                storage_base: 0
            };
            nonzero_idx_by_block_col[block_col_count]
        ];
        for block_col_idx in 0..block_col_count {
            let start = nonzero_idx_by_block_col[block_col_idx];
            nonzero_blocks[start..start + block_colums[block_col_idx].len()]
                .copy_from_slice(&block_colums[block_col_idx]);
        }

        BlockColCompressedMatrix {
            block_col_pattern: BlockColCompressedPattern {
                nonzero_idx_by_block_col,
                nonzero_blocks,
            },
            subdivision: BlockMatrixSubdivision {
                scalar_shape: [total_row_count, total_col_count],
                block_row_offset_by_partition: block_row_offset_by_partition.clone(),
                block_col_offset_by_partition: block_col_offset_by_partition.clone(),
                scalar_row_offset_by_block,
                scalar_col_offset_by_block,
                row_partition_idx_by_block,
                col_partition_idx_by_block,
                row_partitions: self.row_partitions.clone(),
                col_partitions: self.col_partitions.clone(),
            },
            regions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{
        PartitionSpec,
        grid::Grid,
    };

    fn make_subdivision(
        row_partitions: &[(usize, usize)],
        col_partitions: &[(usize, usize)],
    ) -> BlockMatrixSubdivision {
        // R, C
        let row_partition_count = row_partitions.len();
        let col_partition_count = col_partitions.len();

        // Block dims per partition.
        let block_height_by_partition: Vec<usize> =
            row_partitions.iter().map(|&(_, h)| h).collect();
        let block_width_by_partition: Vec<usize> = col_partitions.iter().map(|&(_, w)| w).collect();

        // Blocks per partition.
        let blocks_per_row: Vec<usize> = row_partitions.iter().map(|&(bc, _)| bc).collect();
        let blocks_per_col: Vec<usize> = col_partitions.iter().map(|&(bc, _)| bc).collect();

        let mut block_row_offset_by_partition = vec![0usize; row_partition_count + 1];
        for row_partition_idx in 0..row_partition_count {
            block_row_offset_by_partition[row_partition_idx + 1] = block_row_offset_by_partition
                [row_partition_idx]
                + blocks_per_row[row_partition_idx];
        }
        let mut block_col_offset_by_partition = vec![0usize; col_partition_count + 1];
        for col_partition_idx in 0..col_partition_count {
            block_col_offset_by_partition[col_partition_idx + 1] = block_col_offset_by_partition
                [col_partition_idx]
                + blocks_per_col[col_partition_idx];
        }

        let block_row_count = block_row_offset_by_partition[row_partition_count];
        let block_col_count = block_col_offset_by_partition[col_partition_count];

        // Scalar offsets by block (rows)
        let mut scalar_row_offset_by_block = Vec::with_capacity(block_row_count + 1);
        {
            let mut acc = 0usize;
            for row_partition_idx in 0..row_partition_count {
                let block_height = block_height_by_partition[row_partition_idx];
                for _ in 0..blocks_per_row[row_partition_idx] {
                    scalar_row_offset_by_block.push(acc);
                    acc += block_height;
                }
            }
            scalar_row_offset_by_block.push(acc);
        }

        // Scalar offsets by block (cols)
        let mut scalar_col_offset_by_block = Vec::with_capacity(block_col_count + 1);
        {
            let mut acc = 0usize;
            for col_partition_idx in 0..col_partition_count {
                let block_width = block_width_by_partition[col_partition_idx];
                for _ in 0..blocks_per_col[col_partition_idx] {
                    scalar_col_offset_by_block.push(acc);
                    acc += block_width;
                }
            }
            scalar_col_offset_by_block.push(acc);
        }

        // Partition maps per global block idx.
        let mut row_partition_idx_by_block = vec![0u16; block_row_count];
        for row_partition_idx in 0..row_partition_count {
            for block_row_idx in block_row_offset_by_partition[row_partition_idx]
                ..block_row_offset_by_partition[row_partition_idx + 1]
            {
                row_partition_idx_by_block[block_row_idx] = row_partition_idx as u16;
            }
        }
        let mut col_partition_idx_by_block = vec![0u16; block_col_count];
        for col_partition_idx in 0..col_partition_count {
            for block_col_idx in block_col_offset_by_partition[col_partition_idx]
                ..block_col_offset_by_partition[col_partition_idx + 1]
            {
                col_partition_idx_by_block[block_col_idx] = col_partition_idx as u16;
            }
        }

        // Convert raw (count,dim) into PartitionSpec
        let row_partitions = row_partitions
            .iter()
            .map(|&(block_count, block_dimension)| PartitionSpec {
                block_count,
                block_dimension,
            })
            .collect();
        let col_partitions = col_partitions
            .iter()
            .map(|&(bc, dim)| PartitionSpec {
                block_count: bc,
                block_dimension: dim,
            })
            .collect();

        BlockMatrixSubdivision {
            scalar_shape: [
                *scalar_row_offset_by_block.last().unwrap(),
                *scalar_col_offset_by_block.last().unwrap(),
            ],
            scalar_row_offset_by_block,
            scalar_col_offset_by_block,
            row_partition_idx_by_block,
            col_partition_idx_by_block,
            block_row_offset_by_partition,
            block_col_offset_by_partition,
            row_partitions,
            col_partitions,
        }
    }

    fn make_matrix_from(
        subdivision: BlockMatrixSubdivision,
        nonzero_blocks_per_col: Vec<Vec<NonZeroBlock>>,
        regions: Grid<BlockRegion>,
    ) -> BlockColCompressedMatrix {
        let block_col_count = subdivision.block_col_count();
        assert_eq!(nonzero_blocks_per_col.len(), block_col_count);

        let mut nonzero_idx_by_block_col = Vec::with_capacity(block_col_count + 1);
        nonzero_idx_by_block_col.push(0);
        for block_col_idx in 0..block_col_count {
            nonzero_idx_by_block_col.push(
                nonzero_idx_by_block_col[block_col_idx]
                    + nonzero_blocks_per_col[block_col_idx].len(),
            );
        }

        let mut nonzero_blocks = vec![
            NonZeroBlock {
                block_row_idx: 0,
                storage_base: 0
            };
            nonzero_idx_by_block_col[block_col_count]
        ];
        for block_col_idx in 0..block_col_count {
            let start = nonzero_idx_by_block_col[block_col_idx];
            nonzero_blocks[start..start + nonzero_blocks_per_col[block_col_idx].len()]
                .copy_from_slice(&nonzero_blocks_per_col[block_col_idx]);
        }

        BlockColCompressedMatrix {
            block_col_pattern: BlockColCompressedPattern {
                nonzero_idx_by_block_col,
                nonzero_blocks,
            },
            subdivision,
            regions,
        }
    }

    fn make_regions(row_partition_count: usize, col_partition_count: usize) -> Grid<BlockRegion> {
        Grid::new(
            [row_partition_count, col_partition_count],
            BlockRegion {
                storage: Vec::new(),
            },
        )
    }

    #[test]
    fn iter_single_block_nonsymmetric_dims() {
        // Row partitions:    2 blocks of height 2  +  1 block  of height 1  => total 5 rows
        // Column partitions: 1 block  of width  3  +  2 blocks of width  2  => total 7 cols
        let mat_subdivision = make_subdivision(&[(2, 2), (1, 1)], &[(1, 3), (2, 2)]);
        let [row_partition_count, col_partition_count] = mat_subdivision.region_grid_shape();
        assert_eq!(row_partition_count, 2);
        assert_eq!(col_partition_count, 2);

        let mut regions = make_regions(row_partition_count, col_partition_count);
        let block_height = 2;
        let block_width = 3;
        let payload: [f64; 6] = [
            1.0, 2.0, // col 0
            3.0, 4.0, // col 1
            5.0, 6.0, // col 2
        ];
        let base = regions.get_mut(&[0, 0]).storage.len();
        regions.get_mut(&[0, 0]).storage.extend_from_slice(&payload);

        let nonzero_blocks_per_col = vec![
            vec![NonZeroBlock {
                block_row_idx: 1,
                storage_base: base as u32,
            }], // block col 0
            vec![], // block col 1
            vec![], // block col 2
        ];
        let mat_m = make_matrix_from(mat_subdivision, nonzero_blocks_per_col, regions);

        // Iterate column 0
        let mut it = mat_m.col(0).iter();
        let e: BlockEntry<'_> = it.next().expect("one block expected");
        assert_eq!(e.block_row_idx, 1);
        assert_eq!(e.view.nrows(), block_height);
        assert_eq!(e.view.ncols(), block_width);
        let expected = DMatrixView::from_slice(&payload, block_height, block_width);
        assert!((e.view - expected).amax() < 1e-12);

        assert!(it.next().is_none());
    }

    #[test]
    fn kth_block_matches_iter_order_multi_blocks() {
        // Simple symmetric-ish sizing but non-symmetric block placement.
        // 1 row-part (h=2, 2 block rows), 1 col-part (w=2, 3 block cols)
        let mat_subdivision = make_subdivision(&[(2, 2)], &[(3, 2)]);
        let [row_partition_count, col_partition_count] = mat_subdivision.region_grid_shape();
        let mut regions = make_regions(row_partition_count, col_partition_count);

        // Blocks at (row=0, col=0) and (row=1, col=0), each 2x2
        let b0 = [1.0, 0.0, 0.0, 1.0];
        let b1: [f64; 4] = [2.0, 3.0, 4.0, 5.0];
        let base0 = regions.get_mut(&[0, 0]).storage.len();
        regions.get_mut(&[0, 0]).storage.extend_from_slice(&b0);
        let base1 = regions.get_mut(&[0, 0]).storage.len();
        regions.get_mut(&[0, 0]).storage.extend_from_slice(&b1);

        // Column 0 holds two blocks: row 0 then row 1
        let per_col = vec![
            vec![
                NonZeroBlock {
                    block_row_idx: 0,
                    storage_base: base0 as u32,
                },
                NonZeroBlock {
                    block_row_idx: 1,
                    storage_base: base1 as u32,
                },
            ],
            vec![],
            vec![],
        ];
        let mat_m = make_matrix_from(mat_subdivision, per_col, regions);

        let col_0: Vec<_> = mat_m.col(0).iter().collect();
        assert_eq!(col_0.len(), 2);
        assert_eq!(col_0[0].block_row_idx, 0);
        assert_eq!(col_0[1].block_row_idx, 1);

        // kth_block() matches
        let (r0, k0) = mat_m.kth_block(0, 0);
        let (r1, k1) = mat_m.kth_block(0, 1);
        assert_eq!(r0, 0);
        assert_eq!(r1, 1);
        assert!((k0 - col_0[0].view).amax() < 1e-12);
        assert!((k1 - col_0[1].view).amax() < 1e-12);
    }

    #[test]
    fn views_have_correct_shapes_with_mixed_partitions() {
        // 2 row-parts: [(1,3), (2,1)] => block heights 3 and 1
        // 2 col-parts: [(2,2), (1,4)] => block widths  2 and 4
        let mat_subdivision = make_subdivision(&[(1, 3), (2, 1)], &[(2, 2), (1, 4)]);
        let [row_partition_count, col_partition_count] = mat_subdivision.region_grid_shape();
        assert_eq!((row_partition_count, col_partition_count), (2, 2));
        let mut regions = make_regions(row_partition_count, col_partition_count);

        // Place non-zero blocks:
        //   col 0 (col-part 0, width 2): rows 0 (row-part0, h=3), 2 (row-part1, h=1)
        //   col 2 (col-part 1, width 4): row 1 (row-part1, h=1)
        // Compute bases and push payloads of right sizes
        let b00 = vec![1.0; 3 * 2]; // (row=0,col=0) 3x2
        let base00 = regions.get_mut(&[0, 0]).storage.len();
        regions.get_mut(&[0, 0]).storage.extend_from_slice(&b00);

        let b20 = vec![2.0; 2]; // (row=2,col=0) 1x2
        let base20 = regions.get_mut(&[1, 0]).storage.len();
        regions.get_mut(&[1, 0]).storage.extend_from_slice(&b20);

        let b12 = vec![3.0; 4]; // (row=1,col=2) 1x4
        let base12 = regions.get_mut(&[1, 1]).storage.len();
        regions.get_mut(&[1, 1]).storage.extend_from_slice(&b12);

        let nonzero_blocks_per_col = vec![
            vec![
                NonZeroBlock {
                    block_row_idx: 0,
                    storage_base: base00 as u32,
                },
                NonZeroBlock {
                    block_row_idx: 2,
                    storage_base: base20 as u32,
                },
            ],
            vec![], // col 1 empty
            vec![NonZeroBlock {
                block_row_idx: 1,
                storage_base: base12 as u32,
            }],
        ];
        let mat_m = make_matrix_from(mat_subdivision, nonzero_blocks_per_col, regions);

        // check col 0 shapes
        let blocks0: Vec<_> = mat_m.col(0).iter().collect();
        assert_eq!(blocks0.len(), 2);
        assert_eq!(blocks0[0].view.nrows(), 3);
        assert_eq!(blocks0[0].view.ncols(), 2);
        assert_eq!(blocks0[1].view.nrows(), 1);
        assert_eq!(blocks0[1].view.ncols(), 2);

        // check col 2 shape (from col-part1 width 4)
        let blocks2: Vec<_> = mat_m.col(2).iter().collect();
        assert_eq!(blocks2.len(), 1);
        assert_eq!(blocks2[0].block_row_idx, 1);
        assert_eq!(blocks2[0].view.nrows(), 1);
        assert_eq!(blocks2[0].view.ncols(), 4);
    }
}
