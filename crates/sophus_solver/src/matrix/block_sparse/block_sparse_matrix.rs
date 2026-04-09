use std::collections::hash_map::{
    Entry,
    HashMap,
};

use nalgebra::DMatrixView;
use sophus_assert::debug_assert_le;

use crate::matrix::{
    Grid,
    PartitionBlockIndex,
    PartitionSet,
    block_sparse::{
        BlockSparseTripletMatrix,
        BlockTriplet,
    },
    sparse::ColumnCompressedPattern,
};

/// `N x N` matrix in column compressed block sparse dorm
///
/// The matrix is partitioned into regions which form an `P x P` grid. The total
/// number of matrix blocks horizontally (and vertically) is `B`.
#[derive(Clone, Debug)]
pub struct BlockSparseMatrix {
    /// The block matrix subdivision structure.
    pub subdivision: BlockMatrixSubdivision,
    /// Grid of regions (P × P).
    pub regions: Grid<BlockRegion>,
    /// Pattern of non-zero blocks, sorted by block columns.
    pub block_col_pattern: BlockSparsePattern,
}

/// Pattern of non-zero blocks in sparse matrix, sorted by block columns.
#[derive(Debug, Clone)]
pub struct BlockSparsePattern {
    /// List of indices into nonzero_blocks (len = B+1).
    pub nonzero_idx_by_block_col: Vec<usize>,

    /// Flat list of non-zero blocks.
    pub nonzero_blocks: Vec<NonZeroBlock>,
}

/// A region of the [BlockSparseMatrix].
#[derive(Debug, Clone)]
pub struct BlockRegion {
    /// Back-to-back blocks, column-major.
    pub storage: Vec<f64>,
}

/// A non-zero block of the sparse matrix.
///
/// This struct does not hold the actual data but indexes into [BlockRegion::storage].
#[derive(Clone, Copy, Debug)]
pub struct NonZeroBlock {
    /// Block-row index (0..B-1).
    pub block_row_idx: u32,
    /// Scalar offset into [BlockRegion::storage].
    pub storage_base: u32,
}

impl BlockSparseMatrix {
    /// Total number of block horizontally (or vertically) `B`.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.subdivision.block_count()
    }

    /// Number of non-zero blocks in the sparse matrix.
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
    pub fn non_zero_block(&self, global_block_col_idx: usize) -> &[NonZeroBlock] {
        &self.block_col_pattern.nonzero_blocks[self.col_range_by_block(global_block_col_idx)]
    }

    /// Return column compressed pattern of the upper-triangular blocks of the sparse matrix.
    pub fn build_pattern_upper(&self) -> ColumnCompressedPattern {
        let block_count = self.block_count();

        let mut counts = vec![0usize; block_count];
        for col_j in 0..block_count {
            for storage_idx in self.col_range_by_block(col_j) {
                let row_i =
                    self.block_col_pattern.nonzero_blocks[storage_idx].block_row_idx as usize;
                if row_i > col_j {
                    counts[row_i] += 1;
                }
            }
        }

        // Prefix sum
        let mut storage_offset_by_col = vec![0usize; block_count + 1];
        for i in 0..block_count {
            storage_offset_by_col[i + 1] = storage_offset_by_col[i] + counts[i];
        }
        let mut next = storage_offset_by_col.clone();
        let mut row_ind = vec![0usize; storage_offset_by_col[block_count]];

        for col_j in 0..block_count {
            for storage_idx in self.col_range_by_block(col_j) {
                let row_i =
                    self.block_col_pattern.nonzero_blocks[storage_idx].block_row_idx as usize;
                if row_i > col_j {
                    let dst = next[row_i];
                    row_ind[dst] = col_j;
                    next[row_i] += 1;
                }
            }
        }
        ColumnCompressedPattern::new(block_count, storage_offset_by_col, row_ind)
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

    /// Return true if a non-zero block exists at (row_idx, col_idx) — no allocation.
    #[inline]
    pub fn has_block(&self, row_idx: PartitionBlockIndex, col_idx: PartitionBlockIndex) -> bool {
        let global_row =
            self.subdivision.block_offset_by_partition[row_idx.partition] + row_idx.block;
        let global_col =
            self.subdivision.block_offset_by_partition[col_idx.partition] + col_idx.block;
        let col_slice = self.non_zero_block(global_col);
        col_slice
            .binary_search_by(|e| (e.block_row_idx as usize).cmp(&global_row))
            .is_ok()
    }

    /// Return block at requested row and column index.
    pub fn get_block(
        &self,
        row_idx: crate::matrix::PartitionBlockIndex,
        col_idx: crate::matrix::PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64> {
        let h = self.subdivision.block_dim(row_idx.partition);
        let w = self.subdivision.block_dim(col_idx.partition);

        // Convert (partition, local block) -> global block indices.
        let global_row =
            self.subdivision.block_offset_by_partition[row_idx.partition] + row_idx.block;
        let global_col =
            self.subdivision.block_offset_by_partition[col_idx.partition] + col_idx.block;

        let col_slice = self.non_zero_block(global_col);
        if let Ok(pos) = col_slice.binary_search_by(|e| (e.block_row_idx as usize).cmp(&global_row))
        {
            let entry = col_slice[pos];
            let base = entry.storage_base as usize;
            let size = h * w;

            let storage = &self
                .regions
                .get(&[row_idx.partition, col_idx.partition])
                .storage;
            debug_assert_le!(base + size, storage.len());

            return nalgebra::DMatrix::<f64>::from_column_slice(h, w, &storage[base..base + size]);
        }

        nalgebra::DMatrix::<f64>::zeros(h, w)
    }

    /// Return a zero-copy view of the block at (row_idx, col_idx), or `None` if the block is
    /// structurally zero.
    ///
    /// Unlike `get_block`, this method does not allocate — it returns a `DMatrixView` that
    /// borrows directly from the region storage.  If the block is absent, `None` is returned
    /// (rather than a freshly-allocated zero matrix).
    pub fn try_get_block_view(
        &self,
        row_idx: crate::matrix::PartitionBlockIndex,
        col_idx: crate::matrix::PartitionBlockIndex,
    ) -> Option<nalgebra::DMatrixView<'_, f64>> {
        let h = self.subdivision.block_dim(row_idx.partition);
        let w = self.subdivision.block_dim(col_idx.partition);

        let global_row =
            self.subdivision.block_offset_by_partition[row_idx.partition] + row_idx.block;
        let global_col =
            self.subdivision.block_offset_by_partition[col_idx.partition] + col_idx.block;

        let col_slice = self.non_zero_block(global_col);
        let pos = col_slice
            .binary_search_by(|e| (e.block_row_idx as usize).cmp(&global_row))
            .ok()?;

        let entry = col_slice[pos];
        let base = entry.storage_base as usize;
        let size = h * w;

        let storage = &self
            .regions
            .get(&[row_idx.partition, col_idx.partition])
            .storage;
        debug_assert_le!(base + size, storage.len());

        Some(nalgebra::DMatrixView::from_slice(
            &storage[base..base + size],
            h,
            w,
        ))
    }
}

/// Subdivision structure of block-sparse `N x N` matrix.
///
/// The matrix is partitions into regions which form an `P x P` grid. The total
/// number of matrix blocks horizontally (and vertically) is `B`.
#[derive(Debug, Clone)]
pub struct BlockMatrixSubdivision {
    // Scalar index offset per block vertically (or horizontally) (len = B+1).
    scalar_offset_by_block: Vec<usize>,

    // Maps block row index to row partition index (len = B).
    partition_idx_by_block: Vec<u16>,

    // Block index offset per partition (len = P+1).
    block_offset_by_partition: Vec<usize>,

    // Row (or column) partition set (len = P).
    partitions: PartitionSet,
}

impl BlockMatrixSubdivision {
    pub(crate) fn new(
        scalar_offset_by_block: Vec<usize>,
        partition_idx_by_block: Vec<u16>,
        block_offset_by_partition: Vec<usize>,
        partitions: PartitionSet,
    ) -> Self {
        Self {
            scalar_offset_by_block,
            partition_idx_by_block,
            block_offset_by_partition,
            partitions,
        }
    }

    /// Scalar matrix dimension `N` of the `N x N` matrix.
    #[inline]
    pub fn scalar_dim(&self) -> usize {
        self.partitions.scalar_dim()
    }

    /// Count of partitions horizontally (or vertically) `P`.
    #[inline]
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// Total number of block rows (or block columns) in this matrix: `B`.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.partition_idx_by_block.len()
    }

    /// Maximum block dimension.
    #[inline]
    pub fn max_block_dim(&self) -> usize {
        self.partitions
            .specs()
            .iter()
            .map(|s| s.block_dim)
            .max()
            .unwrap()
    }

    /// Returns scalar index offset into whole matrix given block index.
    #[inline]
    pub fn scalar_offset(&self, global_block_idx: usize) -> usize {
        self.scalar_offset_by_block[global_block_idx]
    }

    /// Return the partition set.
    #[inline]
    pub fn partitions(&self) -> &PartitionSet {
        &self.partitions
    }

    /// Return partition/block index given global block index.
    #[inline]
    pub fn idx(&self, global_block_idx: usize) -> PartitionBlockIndex {
        let row_partition_idx = self.partition_idx_by_block[global_block_idx] as usize;
        PartitionBlockIndex {
            partition: row_partition_idx,
            block: global_block_idx - self.block_offset_by_partition[row_partition_idx],
        }
    }

    /// Return matrix dimension of blocks in requested partition.
    #[inline]
    pub fn block_dim(&self, partition_idx: usize) -> usize {
        self.partitions.specs()[partition_idx].block_dim
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

/// Block entry containing a view to its data and the global block-row index.
pub struct BlockEntry<'a> {
    /// Index of the block-row.
    pub global_block_row_idx: usize,
    /// View of the matrix block.
    pub view: DMatrixView<'a, f64>,
}

impl<'a> BlockColSlice<'a> {
    /// Iterate over blocks in this column of blocks.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = BlockEntry<'_>> {
        let partition_idx_col_j =
            self.mat_subdivision.partition_idx_by_block[self.block_col_idx] as usize;
        let block_width_col_j =
            self.mat_subdivision.partitions.specs()[partition_idx_col_j].block_dim;

        self.slice.iter().map(move |nonzero_block| {
            let block_row_idx = nonzero_block.block_row_idx as usize;

            let partition_idx_row_i =
                self.mat_subdivision.partition_idx_by_block[block_row_idx] as usize;
            let block_height_row_i =
                self.mat_subdivision.partitions.specs()[partition_idx_row_i].block_dim;
            let base = nonzero_block.storage_base as usize;
            let storage = &self
                .regions
                .get(&[partition_idx_row_i, partition_idx_col_j])
                .storage;

            debug_assert!(base + block_height_row_i * block_width_col_j <= storage.len());

            BlockEntry {
                global_block_row_idx: block_row_idx,
                view: DMatrixView::from_slice(
                    &storage[base..base + block_height_row_i * block_width_col_j],
                    block_height_row_i,
                    block_width_col_j,
                ),
            }
        })
    }
}

impl BlockSparseTripletMatrix {
    /// Convert to [BlockSparseMatrix].
    pub fn to_compressed(&self) -> BlockSparseMatrix {
        let partition_count = self.partitions.len();

        let total_count = self.scalar_dimension();

        // Blocks-per-partition.
        let mut blocks_per_partition = vec![0usize; partition_count];
        for partition_idx in 0..partition_count {
            let start = self.partitions.scalar_offsets_by_partition()[partition_idx];
            let end = if partition_idx + 1 < partition_count {
                self.partitions.scalar_offsets_by_partition()[partition_idx + 1]
            } else {
                total_count
            };
            let block_height = self.partitions.specs()[partition_idx].block_dim;
            debug_assert_eq!((end - start) % block_height, 0,);
            blocks_per_partition[partition_idx] =
                (end - start).checked_div(block_height).unwrap_or(0);
        }

        // Global block offset by partition.
        let mut block_offset_by_partition = vec![0usize; partition_count + 1];
        for r in 0..partition_count {
            block_offset_by_partition[r + 1] =
                block_offset_by_partition[r] + blocks_per_partition[r];
        }

        let block_count = block_offset_by_partition[partition_count];

        // Calculate scalar index offsets into whole matrix for each block index.
        let mut scalar_offset_by_block = Vec::with_capacity(block_count + 1);
        {
            let mut offset = 0usize;
            for partition_idx in 0..partition_count {
                let block_dim = self.partitions.specs()[partition_idx].block_dim;
                for _ in 0..blocks_per_partition[partition_idx] {
                    scalar_offset_by_block.push(offset);
                    offset += block_dim;
                }
            }
            scalar_offset_by_block.push(offset);
            debug_assert_eq!(offset, total_count);
        }

        // Fast maps: block row/col → partition row/col.
        let mut partition_idx_by_block = vec![0u16; block_count];
        for partition_idx in 0..partition_count {
            for block_idx in block_offset_by_partition[partition_idx]
                ..block_offset_by_partition[partition_idx + 1]
            {
                partition_idx_by_block[block_idx] = partition_idx as u16;
            }
        }

        // Temporary per-region buffers.
        #[derive(Default, Clone)]
        struct TempRegion {
            storage: Vec<f64>,
            triplets: Vec<BlockTriplet>,
        }
        let mut dedup_storage =
            Grid::new([partition_count, partition_count], TempRegion::default());
        let mut storage_base_by_local_block: HashMap<[usize; 2], usize> = HashMap::new();

        for row_partition_idx in 0..partition_count {
            let block_height = self.partitions.specs()[row_partition_idx].block_dim;
            for col_partition_idx in 0..partition_count {
                let block_width = self.partitions.specs()[col_partition_idx].block_dim;
                let block_area = block_height * block_width;

                let src_region = self
                    .triplet_grid
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
                    let dst_base = match storage_base_by_local_block.entry(triplet.block_idx) {
                        Entry::Occupied(e) => *e.get(),
                        Entry::Vacant(vacant_entry) => {
                            // Allocate space for a fresh block (zero-initialized).
                            let storage_base = tmp_region.storage.len();
                            tmp_region.storage.resize(storage_base + block_area, 0.0);
                            vacant_entry.insert(storage_base);
                            tmp_region.triplets.push(BlockTriplet {
                                block_idx: triplet.block_idx,
                                storage_base,
                            });
                            storage_base
                        }
                    };

                    let src_base = triplet.storage_base;
                    let src = &src_region.flattened_block_storage[src_base..src_base + block_area];

                    // Column-major add
                    for j in 0..block_width {
                        let dst_col = &mut tmp_region.storage
                            [dst_base + j * block_height..dst_base + (j + 1) * block_height];
                        let src_col = &src[j * block_height..(j + 1) * block_height];
                        for i in 0..block_height {
                            dst_col[i] += src_col[i];
                        }
                    }
                }
            }
        }

        // Materialize the output regions grid.
        let mut regions = Grid::new(
            [partition_count, partition_count],
            BlockRegion {
                storage: Vec::new(),
            },
        );
        for row_partition_idx in 0..partition_count {
            for col_partition_idx in 0..partition_count {
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
        let mut block_colums: Vec<Vec<NonZeroBlock>> = vec![Vec::new(); block_count];

        for row_partition_idx in 0..partition_count {
            let block_height = self.partitions.specs()[row_partition_idx].block_dim;
            for col_partition_idx in 0..partition_count {
                let block_width = self.partitions.specs()[col_partition_idx].block_dim;
                let stride = block_height * block_width;

                let unique_region = dedup_storage.get(&[row_partition_idx, col_partition_idx]);
                if unique_region.triplets.is_empty() {
                    continue;
                }

                for triplet in unique_region.triplets.iter() {
                    let block_row_idx =
                        block_offset_by_partition[row_partition_idx] + triplet.block_idx[0];
                    let block_col_idx =
                        block_offset_by_partition[col_partition_idx] + triplet.block_idx[1];

                    debug_assert!(block_row_idx < block_count && block_col_idx < block_count);
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
        for block_col_idx in 0..block_count {
            block_colums[block_col_idx].sort_unstable_by_key(|e| e.block_row_idx);
            debug_assert!(
                block_colums[block_col_idx]
                    .windows(2)
                    .all(|w| w[0].block_row_idx < w[1].block_row_idx),
                "Duplicate block rows remained in column {block_col_idx} after dedup"
            );
        }

        // Build non-zero block storage.
        let mut nonzero_idx_by_block_col = Vec::with_capacity(block_count + 1);
        nonzero_idx_by_block_col.push(0);
        for block_col_idx in 0..block_count {
            nonzero_idx_by_block_col
                .push(nonzero_idx_by_block_col[block_col_idx] + block_colums[block_col_idx].len());
        }

        let mut nonzero_blocks = vec![
            NonZeroBlock {
                block_row_idx: 0,
                storage_base: 0
            };
            nonzero_idx_by_block_col[block_count]
        ];
        for block_col_idx in 0..block_count {
            let start = nonzero_idx_by_block_col[block_col_idx];
            nonzero_blocks[start..start + block_colums[block_col_idx].len()]
                .copy_from_slice(&block_colums[block_col_idx]);
        }

        BlockSparseMatrix {
            block_col_pattern: BlockSparsePattern {
                nonzero_idx_by_block_col,
                nonzero_blocks,
            },
            subdivision: BlockMatrixSubdivision {
                block_offset_by_partition: block_offset_by_partition.clone(),
                scalar_offset_by_block,
                partition_idx_by_block,
                partitions: self.partitions.clone(),
            },
            regions,
        }
    }
}
