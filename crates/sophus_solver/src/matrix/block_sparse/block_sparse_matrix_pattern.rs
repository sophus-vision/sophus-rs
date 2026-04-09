use crate::matrix::{
    PartitionBlockIndex,
    PartitionSet,
    block_sparse::{
        BlockMatrixSubdivision,
        BlockSparsePattern,
        NonZeroBlock,
    },
    grid::Grid,
};

/// Per-region precomputed pattern: sorted unique block indices with CSR row index.
#[derive(Clone, Debug)]
pub(crate) struct BlockRegionPattern {
    /// Sorted unique block indices `[row, col]` within this region.
    pub(crate) sorted_blocks: Vec<[usize; 2]>,
    /// Block dimensions `[h, w]`.
    pub(crate) block_shape: [usize; 2],
    /// CSR-style row-start offsets: `row_starts[r]` is the first index in `sorted_blocks`
    /// with row == r.  Length = num_block_rows + 1.
    ///
    /// Reduces `find_offset` from O(log K) over the full region to O(log C) where C is the
    /// number of non-zero columns in row r (typically ≪ K in bundle adjustment problems).
    pub(crate) row_starts: Vec<usize>,
}

impl BlockRegionPattern {
    /// Find the flat-storage offset for block `[row, col]`.
    ///
    /// Uses the CSR row index to restrict the binary search to only the entries in
    /// the requested row, reducing search length from K (total blocks) to C (columns in row).
    #[inline]
    pub(crate) fn find_offset(&self, [r, c]: [usize; 2]) -> Option<usize> {
        if r + 1 >= self.row_starts.len() {
            return None;
        }
        let start = self.row_starts[r];
        let end = self.row_starts[r + 1];
        // Within [start, end) the blocks are sorted by [row, col]; since all have row == r,
        // we binary-search by the full key (which degenerates to comparing only col).
        self.sorted_blocks[start..end]
            .binary_search(&[r, c])
            .ok()
            .map(|pos| (start + pos) * self.block_shape[0] * self.block_shape[1])
    }

    /// Total number of floats stored for this region.
    #[inline]
    pub(crate) fn total_floats(&self) -> usize {
        self.sorted_blocks.len() * self.block_shape[0] * self.block_shape[1]
    }
}

/// Precomputed sparsity pattern for a block-sparse matrix.
///
/// Built once via [`BlockSparseSymbolicBuilder`] and reused across optimizer
/// iterations.  Contains:
/// - per-region sorted block index tables for O(log K) lookup
/// - precomputed CSC structure (`nonzero_idx_by_block_col`, `nonzero_blocks` with stable
///   `storage_base` offsets)
/// - pre-allocated, zeroed value storage (one flat `Vec<f64>` per region)
///
/// Call [`BlockSparseMatrixPattern::reset`] at the start of each iteration to
/// zero the storage, then accumulate with [`BlockSparseMatrixPattern::add_block`],
/// then call [`BlockSparseMatrixPattern::build`].
#[derive(Clone, Debug)]
pub struct BlockSparseMatrixPattern {
    pub(crate) region_patterns: Grid<BlockRegionPattern>,
    /// Pre-allocated value storage per region, zeroed by `reset()`.
    pub(crate) region_storage: Grid<Vec<f64>>,
    /// Precomputed CSC column-pointer array (stable across iterations).
    pub(crate) nonzero_idx_by_block_col: Vec<usize>,
    /// Precomputed CSC non-zero block list (stable across iterations).
    pub(crate) nonzero_blocks: Vec<NonZeroBlock>,
    /// Stable subdivision (block/partition layout).
    pub(crate) subdivision: BlockMatrixSubdivision,
}

impl BlockSparseMatrixPattern {
    /// Number of partitions.
    pub fn partition_count(&self) -> usize {
        self.subdivision.partition_count()
    }

    /// Zero all value storage — call once per optimizer iteration before accumulating.
    pub fn reset(&mut self) {
        let p = self.partition_count();
        for r in 0..p {
            for c in 0..p {
                self.region_storage.get_mut(&[r, c]).fill(0.0);
            }
        }
    }

    /// Accumulate `block` into position `(row_idx, col_idx)`.
    ///
    /// Panics in debug mode if the block was not recorded during the symbolic pass.
    #[inline]
    pub fn add_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
        block: &nalgebra::DMatrixView<f64>,
    ) {
        let rp = self
            .region_patterns
            .get(&[row_idx.partition, col_idx.partition]);
        let [h, w] = rp.block_shape;
        debug_assert_eq!((block.nrows(), block.ncols()), (h, w));

        let offset = rp.find_offset([row_idx.block, col_idx.block]);
        debug_assert!(
            offset.is_some(),
            "block [{}, {}] in region [{}, {}] was not recorded in the symbolic pass",
            row_idx.block,
            col_idx.block,
            row_idx.partition,
            col_idx.partition,
        );

        if let Some(off) = offset {
            let storage = self
                .region_storage
                .get_mut(&[row_idx.partition, col_idx.partition]);
            for c in 0..w {
                let col_slice = block.column(c);
                let dst = &mut storage[off + c * h..off + (c + 1) * h];
                for (d, &v) in dst.iter_mut().zip(col_slice.iter()) {
                    *d += v;
                }
            }
        }
    }

    /// Accumulate another pattern's value storage into this one (`self += other`).
    ///
    /// The two patterns must have identical structure (same partition set, same sparsity).
    /// Panics in debug mode if the storage sizes differ.
    pub fn merge_from(&mut self, other: &Self) {
        let p = self.partition_count();
        for r in 0..p {
            for c in 0..p {
                let src = other.region_storage.get(&[r, c]);
                let dst = self.region_storage.get_mut(&[r, c]);
                debug_assert_eq!(
                    dst.len(),
                    src.len(),
                    "region storage size mismatch in merge_from"
                );
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d += s;
                }
            }
        }
    }

    /// Assemble the final [`BlockSparseMatrix`] from the current storage.
    ///
    /// The pattern (CSC structure) is cloned cheaply; only the value storage
    /// is copied.
    pub fn build(&self) -> crate::matrix::block_sparse::block_sparse_matrix::BlockSparseMatrix {
        use crate::matrix::block_sparse::block_sparse_matrix::{
            BlockRegion,
            BlockSparseMatrix,
        };

        let p = self.partition_count();
        let mut regions = Grid::new(
            [p, p],
            BlockRegion {
                storage: Vec::new(),
            },
        );
        for r in 0..p {
            for c in 0..p {
                let src = self.region_storage.get(&[r, c]);
                if !src.is_empty() {
                    regions.get_mut(&[r, c]).storage.clone_from(src);
                }
            }
        }
        BlockSparseMatrix {
            block_col_pattern: BlockSparsePattern {
                nonzero_idx_by_block_col: self.nonzero_idx_by_block_col.clone(),
                nonzero_blocks: self.nonzero_blocks.clone(),
            },
            subdivision: self.subdivision.clone(),
            regions,
        }
    }
}

/// Symbolic builder: records which blocks exist without storing any values.
///
/// Run once (or whenever the problem structure changes) over the same
/// accumulation code that the numeric pass will use, but passing zero blocks
/// or simply calling [`BlockSparseSymbolicBuilder::add_block`] with the
/// indices only.  Then call [`BlockSparseSymbolicBuilder::into_pattern`] to
/// obtain a [`BlockSparseMatrixPattern`] that can be reused for every
/// subsequent optimizer iteration.
pub struct BlockSparseSymbolicBuilder {
    /// Per-region collected block indices (unsorted, may have duplicates).
    region_indices: Grid<Vec<[usize; 2]>>,
    partitions: PartitionSet,
}

impl BlockSparseSymbolicBuilder {
    /// Create a new symbolic builder for the given partition set.
    pub fn new(partitions: PartitionSet) -> Self {
        let p = partitions.len();
        Self {
            region_indices: Grid::new([p, p], Vec::new()),
            partitions,
        }
    }

    /// Record that block `(row_idx, col_idx)` will be written during numeric passes.
    ///
    /// Values are ignored — only the index is recorded.
    #[inline]
    pub fn add_block(&mut self, row_idx: PartitionBlockIndex, col_idx: PartitionBlockIndex) {
        self.region_indices
            .get_mut(&[row_idx.partition, col_idx.partition])
            .push([row_idx.block, col_idx.block]);
    }

    /// Finalise: sort + dedup indices, compute CSC structure, allocate storage.
    ///
    /// Returns a [`BlockSparseMatrixPattern`] ready for numeric accumulation.
    pub fn into_pattern(mut self) -> BlockSparseMatrixPattern {
        let partition_count = self.partitions.len();

        // --- Per-region: sort + dedup, build BlockRegionPattern ----------------
        let mut region_patterns = Grid::new(
            [partition_count, partition_count],
            BlockRegionPattern {
                sorted_blocks: Vec::new(),
                block_shape: [0, 0],
                row_starts: Vec::new(),
            },
        );
        for r in 0..partition_count {
            for c in 0..partition_count {
                let indices = self.region_indices.get_mut(&[r, c]);
                indices.sort_unstable();
                indices.dedup();

                // Build CSR row-start index from the now-sorted block list.
                // row_starts[i] = first index in sorted_blocks whose row == i.
                let num_block_rows = self.partitions.specs()[r].block_count;
                let mut row_starts = vec![0usize; num_block_rows + 1];
                for &[row, _col] in indices.iter() {
                    row_starts[row + 1] += 1;
                }
                for i in 0..num_block_rows {
                    row_starts[i + 1] += row_starts[i];
                }

                *region_patterns.get_mut(&[r, c]) = BlockRegionPattern {
                    sorted_blocks: std::mem::take(indices),
                    block_shape: [
                        self.partitions.specs()[r].block_dim,
                        self.partitions.specs()[c].block_dim,
                    ],
                    row_starts,
                };
            }
        }

        // --- Partition / block layout ------------------------------------------
        let blocks_per_partition: Vec<usize> = self
            .partitions
            .specs()
            .iter()
            .map(|s| s.block_count)
            .collect();

        let mut block_offset_by_partition = vec![0usize; partition_count + 1];
        for r in 0..partition_count {
            block_offset_by_partition[r + 1] =
                block_offset_by_partition[r] + blocks_per_partition[r];
        }
        let block_count = block_offset_by_partition[partition_count];

        let mut scalar_offset_by_block = Vec::with_capacity(block_count + 1);
        let mut offset = 0usize;
        for partition_idx in 0..partition_count {
            let block_dim = self.partitions.specs()[partition_idx].block_dim;
            for _ in 0..blocks_per_partition[partition_idx] {
                scalar_offset_by_block.push(offset);
                offset += block_dim;
            }
        }
        scalar_offset_by_block.push(offset);

        let mut partition_idx_by_block = vec![0u16; block_count];
        for partition_idx in 0..partition_count {
            for block_idx in block_offset_by_partition[partition_idx]
                ..block_offset_by_partition[partition_idx + 1]
            {
                partition_idx_by_block[block_idx] = partition_idx as u16;
            }
        }

        // --- CSC block columns from region patterns ----------------------------
        //
        // For region (r, c), the k-th block in sorted_blocks is stored at
        // storage_base = k * h * w within that region's flat storage.
        // This offset is stable across iterations, so we precompute it here.
        let mut block_columns: Vec<Vec<NonZeroBlock>> = vec![Vec::new(); block_count];
        for row_part in 0..partition_count {
            for col_part in 0..partition_count {
                let rp = region_patterns.get(&[row_part, col_part]);
                let [h, w] = rp.block_shape;
                let block_size = h * w;
                for (k, &block_idx) in rp.sorted_blocks.iter().enumerate() {
                    let block_row_idx = block_offset_by_partition[row_part] + block_idx[0];
                    let block_col_idx = block_offset_by_partition[col_part] + block_idx[1];
                    block_columns[block_col_idx].push(NonZeroBlock {
                        block_row_idx: block_row_idx as u32,
                        storage_base: (k * block_size) as u32,
                    });
                }
            }
        }
        for col in block_columns.iter_mut() {
            col.sort_unstable_by_key(|e| e.block_row_idx);
        }

        let mut nonzero_idx_by_block_col = Vec::with_capacity(block_count + 1);
        nonzero_idx_by_block_col.push(0);
        for col_idx in 0..block_count {
            nonzero_idx_by_block_col
                .push(nonzero_idx_by_block_col[col_idx] + block_columns[col_idx].len());
        }
        let total_nz = *nonzero_idx_by_block_col.last().unwrap();
        let mut nonzero_blocks = Vec::with_capacity(total_nz);
        for col in &block_columns {
            nonzero_blocks.extend_from_slice(col);
        }

        // --- Pre-allocate value storage per region ----------------------------
        let mut region_storage = Grid::new([partition_count, partition_count], Vec::new());
        for r in 0..partition_count {
            for c in 0..partition_count {
                let n = region_patterns.get(&[r, c]).total_floats();
                *region_storage.get_mut(&[r, c]) = vec![0.0f64; n];
            }
        }

        BlockSparseMatrixPattern {
            region_patterns,
            region_storage,
            nonzero_idx_by_block_col,
            nonzero_blocks,
            subdivision: BlockMatrixSubdivision::new(
                scalar_offset_by_block,
                partition_idx_by_block,
                block_offset_by_partition,
                self.partitions,
            ),
        }
    }
}
