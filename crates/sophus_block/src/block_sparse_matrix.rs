use std::fmt::Debug;

use super::{
    PartitionSpec,
    grid::Grid,
};
use crate::{
    CompressedBlockMatrix,
    CompressedBlockRegion,
};

/// A builder for a block sparse matrix.
///
/// The target block sparse matrix has the following structure:
///
/// ```ascii
/// --------------------------------------------------------------------------
/// | M1×N1 ... M1×N1 | M1×N2 ... M1×N2 |                 |  M1×Ny ... M1×Ny |
/// |   .  .      .   |   .         .   |                 |    .         .   |
/// |   .      .  .   |   .         .   |     *  *  *     |    .         .   |
/// | M1×N1 ... M1×N1 | M1×N2 ... M1×N2 |                 |  M1×Ny ... M1×Ny |
/// --------------------------------------------------------------------------
/// | M2×N1 ... M2×N1 | M2×N2 ... M2×N2 |                 |                  |
/// |   .         .   |   .  .      .   |                 |         *        |
/// |   .         .   |   .      .  .   |                 |         *        |
/// | M2×N1 ... M2×N1 | M2×N2 ... M2×N2 |                 |                  |
/// --------------------------------------------------------------------------
/// |                 |                 |                 |                  |
/// |        *        |                 |      *          |         *        |
/// |        *        |                 |          *      |         *        |
/// |                 |                 |                 |                  |
/// --------------------------------------------------------------------------
/// | Mx×N1 ... Mx×N1 |                 |                 |  Mx×Ny ... Mx×Ny |
/// |   .  .      .   |                 |                 |    . .       .   |
/// |   .      .  .   |     *  *  *     |     *  *  *     |    .      .  .   |
/// | Mx×N1 ... Mx×N1 |                 |                 |  Mx×Ny ... Mx×Ny |
/// --------------------------------------------------------------------------
/// ```
///
/// It is split into a grid of regions and each region is split into a grid of block matrices.
/// Within each region, all block matrices have the same shape. E.g., the region (0,0) contains
/// only M1×N1-shaped blocks, the region (0,1) contains only (M1×N2) blocks, etc.
#[derive(Debug)]
pub struct BlockSparseMatrix {
    pub(crate) region_grid: Grid<BlockTripletRegion>,
    pub(crate) index_offset_per_row_partition: Vec<usize>,
    pub(crate) index_offset_per_col_partition: Vec<usize>,
    pub(crate) scalar_shape: [usize; 2],
}

/// A single block "AxB" in a region of the block sparse matrix.
#[derive(Debug, Clone)]
pub(crate) struct BlockTriplet {
    // index (row, column) of block within the region
    pub(crate) block_idx: [usize; 2],
    // index into flattened_block_storage
    pub(crate) start_data_idx: usize,
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

pub(crate) enum ToScalarTripletsImplMode {
    // input: upper triangular block sparse matrix
    // output: symmetric matrix in scalar triplet representation
    SymmetricFromUpperTriangularBlockPattern,
    // input: upper triangular block sparse matrix
    // output: upper triangular matrix in scalar triplet representation
    UpperTriangularFromUpperTriangularBlockPattern,
    // input: general block sparse matrix
    // output: general matrix in scalar triplet representation
    General,
}

pub(crate) enum ToDenseImplMode {
    // input: upper triangular block sparse matrix
    // output: symmetric dense matrix
    SymmetricFromUpperTriangularBlockPattern,
    // input: general block sparse matrix
    // output: general dense matrix
    General,
}

impl BlockSparseMatrix {
    /// Create a sparse block matrix "filled" with zeros.
    ///
    /// The shape of the block matrix is determined by the provided of row and column partitions.
    pub fn zero(row_partitions: &[PartitionSpec], col_partitions: &[PartitionSpec]) -> Self {
        let mut row_block_dims = Vec::new();
        let mut col_block_dims = Vec::new();

        let mut row_index_offsets = Vec::new();
        let mut row_index_offset = 0;
        for row_partition in row_partitions {
            row_index_offsets.push(row_index_offset);
            row_index_offset += row_partition.block_dim * row_partition.num_blocks;
            row_block_dims.push(row_partition.block_dim);
        }

        let mut col_index_offsets = Vec::new();
        let mut col_index_offset = 0;
        for col_partition in col_partitions {
            col_index_offsets.push(col_index_offset);
            col_index_offset += col_partition.block_dim * col_partition.num_blocks;
            col_block_dims.push(col_partition.block_dim);
        }

        let mut region_grid = Grid::new(
            [row_block_dims.len(), col_block_dims.len()],
            BlockTripletRegion {
                flattened_block_storage: Vec::new(),
                triplets: Vec::new(),
                block_shape: [0, 0],
            },
        );

        for r in 0..row_block_dims.len() {
            for c in 0..col_block_dims.len() {
                region_grid.get_mut(&[r, c]).block_shape = [row_block_dims[r], col_block_dims[c]];
            }
        }

        Self {
            region_grid,
            index_offset_per_row_partition: row_index_offsets,
            index_offset_per_col_partition: col_index_offsets,
            scalar_shape: [row_index_offset, col_index_offset],
        }
    }

    /// scalar dimension of the matrix.
    pub fn scalar_shape(&self) -> [usize; 2] {
        self.scalar_shape
    }

    /// region grid dimension
    pub fn region_grid_shape(&self) -> [usize; 2] {
        [
            self.index_offset_per_row_partition.len(),
            self.index_offset_per_col_partition.len(),
        ]
    }

    pub(crate) fn get_region(&self, region_idx: &[usize; 2]) -> &BlockTripletRegion {
        self.region_grid.get(region_idx)
    }

    pub(crate) fn get_region_mut(&mut self, region_idx: &[usize; 2]) -> &mut BlockTripletRegion {
        self.region_grid.get_mut(region_idx)
    }

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block. d
    pub fn add_block(
        &mut self,
        region_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &nalgebra::DMatrixView<f64>,
    ) {
        let grid_region = self.get_region_mut(region_idx);
        debug_assert_eq!(block.shape().0, grid_region.block_shape[0]);
        debug_assert_eq!(block.shape().1, grid_region.block_shape[1]);
        grid_region.triplets.push(BlockTriplet {
            block_idx: block_index,
            start_data_idx: grid_region.flattened_block_storage.len(),
        });
        for col in 0..block.ncols() {
            grid_region
                .flattened_block_storage
                .extend_from_slice(block.column(col).as_slice());
        }
    }

    pub(crate) fn to_scalar_triplets_impl(
        &self,
        mode: ToScalarTripletsImplMode,
    ) -> Vec<faer::sparse::Triplet<usize, usize, f64>> {
        let mut triplets = Vec::new();

        for region_x_idx in 0..self.region_grid_shape()[0] {
            for region_y_idx in 0..self.region_grid_shape()[1] {
                let region = self.get_region(&[region_x_idx, region_y_idx]);
                for block_triplet in &region.triplets {
                    let region_rows = region.block_shape[0];
                    let region_cols = region.block_shape[1];
                    let row_offset = self.index_offset_per_row_partition[region_x_idx]
                        + block_triplet.block_idx[0] * region_rows;
                    let col_offset = self.index_offset_per_col_partition[region_y_idx]
                        + block_triplet.block_idx[1] * region_cols;
                    let data_idx = block_triplet.start_data_idx;
                    let block = &region.flattened_block_storage
                        [data_idx..data_idx + (region_rows * region_cols)];
                    let is_block_on_diagonal = region_x_idx == region_y_idx
                        && block_triplet.block_idx[0] == block_triplet.block_idx[1];

                    for c in 0..region_cols {
                        for r in 0..region_rows {
                            let value = block[c * region.block_shape[0] + r];
                            let scalar_r = row_offset + r;
                            let scalar_c = col_offset + c;

                            match mode {
                                ToScalarTripletsImplMode::SymmetricFromUpperTriangularBlockPattern => {
                                    // Assumption: Input has an upper triangular block pattern. See
                                    // SymmetricBlockSparseMatrix for details. Hence, we need to
                                    // duplicate blocks above the diagonal and mirror them below the
                                    // diagonal to get a symmetric matrix.
                                    triplets.push(faer::sparse::Triplet::new(scalar_r, scalar_c, value));

                                    if !is_block_on_diagonal
                                    {
                                        triplets.push(faer::sparse::Triplet::new(scalar_c, scalar_r, value));
                                    }
                                }
                                ToScalarTripletsImplMode::UpperTriangularFromUpperTriangularBlockPattern => {
                                    // Assumption: Input has an upper triangular block pattern. See
                                    // SymmetricBlockSparseMatrix for details. Hence, there is no
                                    // need for logic to skip blocks below the diagonal.
                                    // However, the blocks on the diagonal are not upper triangular itself,
                                    // hence we need to skip scalar entries below the diagonal for these blocks.
                                    if is_block_on_diagonal
                                        && r > c
                                    {
                                        continue;
                                    }
                                    triplets.push(faer::sparse::Triplet::new(scalar_r, scalar_c, value));
                                }
                                ToScalarTripletsImplMode::General => {
                                    triplets.push(faer::sparse::Triplet::new(scalar_r, scalar_c, value));
                                }
                            }
                        }
                    }
                }
            }
        }
        triplets
    }

    /// Convert to scalar triplets.
    pub fn to_scalar_triplets(&self) -> Vec<faer::sparse::Triplet<usize, usize, f64>> {
        self.to_scalar_triplets_impl(ToScalarTripletsImplMode::General)
    }

    pub(crate) fn to_dense_impl(&self, mode: ToDenseImplMode) -> nalgebra::DMatrix<f64> {
        let mut full_matrix =
            nalgebra::DMatrix::from_element(self.scalar_shape()[0], self.scalar_shape()[1], 0.0);

        for region_x_idx in 0..self.region_grid_shape()[0] {
            for region_y_idx in 0..self.region_grid_shape()[1] {
                let region = self.get_region(&[region_x_idx, region_y_idx]);
                for block_triplet in &region.triplets {
                    let region_rows = region.block_shape[0];
                    let region_cols = region.block_shape[1];
                    let row_offset = self.index_offset_per_row_partition[region_x_idx]
                        + block_triplet.block_idx[0] * region_rows;
                    let col_offset = self.index_offset_per_col_partition[region_y_idx]
                        + block_triplet.block_idx[1] * region_cols;
                    let data_idx = block_triplet.start_data_idx;
                    let block = &region.flattened_block_storage
                        [data_idx..data_idx + (region_rows * region_cols)];
                    let is_block_on_diagonal = region_x_idx == region_y_idx
                        && block_triplet.block_idx[0] == block_triplet.block_idx[1];

                    for c in 0..region_cols {
                        for r in 0..region_rows {
                            let value = block[c * region.block_shape[0] + r];
                            let scalar_r = row_offset + r;
                            let scalar_c = col_offset + c;

                            full_matrix[(scalar_r, scalar_c)] += value;

                            match mode {
                                ToDenseImplMode::SymmetricFromUpperTriangularBlockPattern => {
                                    // Assumption: Input has an upper triangular block pattern. See
                                    // SymmetricBlockSparseMatrix for details. Hence, we need
                                    // to duplicate blocks above the diagonal and mirror them below
                                    // the diagonal.
                                    if !is_block_on_diagonal {
                                        full_matrix[(scalar_c, scalar_r)] += value;
                                    }
                                }
                                ToDenseImplMode::General => {}
                            }
                        }
                    }
                }
            }
        }

        full_matrix
    }

    /// Convert the block sparse matrix to dense matrix.
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        self.to_dense_impl(ToDenseImplMode::General)
    }

    /// Convert to compressed block form.
    pub fn to_compressed(&self) -> CompressedBlockMatrix {
        let rows_of_regions = self.index_offset_per_row_partition.len();
        let cols_of_regions = self.index_offset_per_col_partition.len();

        let mut out = CompressedBlockMatrix {
            region_grid: Grid::new(
                [rows_of_regions, cols_of_regions],
                CompressedBlockRegion::empty(),
            ),
            index_offset_per_row_partition: self.index_offset_per_row_partition.clone(),
            index_offset_per_col_partition: self.index_offset_per_col_partition.clone(),
            scalar_shape: self.scalar_shape,
        };

        for region_x_idx in 0..rows_of_regions {
            for region_y_idx in 0..cols_of_regions {
                let r = self.compress_region_impl(region_x_idx, region_y_idx);
                *out.region_grid.get_mut(&[region_x_idx, region_y_idx]) = r;
            }
        }
        out
    }

    fn compress_region_impl(
        &self,
        region_x_idx: usize,
        region_y_idx: usize,
    ) -> CompressedBlockRegion {
        use hashbrown::HashMap;

        let region = self.get_region(&[region_x_idx, region_y_idx]);
        if region.block_shape[0] == 0 || region.block_shape[1] == 0 {
            return CompressedBlockRegion::empty();
        }
        let num_block_elems = region.block_shape[0] * region.block_shape[1];

        // #block rows in this region’s partition grid
        let num_block_rows = {
            let start = self.index_offset_per_row_partition[region_x_idx];
            let end = self
                .index_offset_per_row_partition
                .get(region_x_idx + 1)
                .copied()
                .unwrap_or(self.scalar_shape[0]);
            (end - start) / region.block_shape[0]
        };
        // #block cols in this region’s partition grid
        let num_block_cols = {
            let start = self.index_offset_per_col_partition[region_y_idx];
            let end = self
                .index_offset_per_col_partition
                .get(region_y_idx + 1)
                .copied()
                .unwrap_or(self.scalar_shape[1]);
            (end - start) / region.block_shape[1]
        };

        // outputs
        let mut entries: Vec<(usize, usize, usize)> = Vec::new(); // (block_row_idx, block_col_idx, entry_idx)
        let mut flattened_block_storage: Vec<f64> = Vec::new(); // entries.len() * num_block_elems
        let mut count_cols = vec![0usize; num_block_cols];
        let mut count_rows = vec![0usize; num_block_rows];
        let mut diag_entry_indices: Vec<Option<usize>> = vec![None; num_block_cols];

        // (block_row_idx,block_col_idx) -> storage_idx
        let mut map: HashMap<u64, usize> = HashMap::with_capacity(region.triplets.len() * 2);

        for t in &region.triplets {
            let block_row_idx = t.block_idx[0];
            let block_col_idx = t.block_idx[1];
            debug_assert!(block_row_idx < num_block_rows && block_col_idx < num_block_cols);

            // 64-bit key packs the two usize indices into one number. The key is unique as long as
            // block_row_idx/block_col_idx fit in u32.
            let key = ((block_row_idx as u64) << 32) | (block_col_idx as u64);
            debug_assert!(block_row_idx <= u32::MAX as usize && block_col_idx <= u32::MAX as usize);

            let entry_idx = *map.entry(key).or_insert_with(|| {
                // flattened_block_storage is a single contiguous array holding all unique blocks
                // for this region. We add blocks in chunks of num_block_elems, the current len() is
                // always a multiple of num_block_elems.
                let entry_idx = flattened_block_storage.len() / num_block_elems;
                // Append space for one block (all zeros), reserving num_block_elems.
                flattened_block_storage
                    .resize(flattened_block_storage.len() + num_block_elems, 0.0);
                // Records this unique block’s coordinates in the region grid (block_row_idx,
                // block_col_idx) and the entry index idx where its values live
                entries.push((block_row_idx, block_col_idx, entry_idx));
                count_cols[block_col_idx] += 1;
                count_rows[block_row_idx] += 1;
                if region_x_idx == region_y_idx && block_row_idx == block_col_idx {
                    diag_entry_indices[block_col_idx] = Some(entry_idx);
                }
                entry_idx
            });

            let dst = &mut flattened_block_storage
                [entry_idx * num_block_elems..(entry_idx + 1) * num_block_elems];
            let src = &region.flattened_block_storage
                [t.start_data_idx..t.start_data_idx + num_block_elems];
            for i in 0..num_block_elems {
                dst[i] += src[i];
            }
        }

        let num_non_empty_blocks = entries.len();

        // CSC

        // CSC pointers with a cumulative sum (last entry equal to num_non_empty_blocks).
        let mut csc_col_ptr = vec![0usize; num_block_cols + 1];
        for block_col_idx in 0..num_block_cols {
            // csc_col_ptr[j] is the starting slot in csc_row_idx for column j.
            // csc_col_ptr[j+1] is the end (start of the next column).
            csc_col_ptr[block_col_idx + 1] = csc_col_ptr[block_col_idx] + count_cols[block_col_idx];
        }
        // stores block row index for the entry placed at pos.
        let mut csc_row_idx = vec![0u32; num_non_empty_blocks];
        // stores entry_idx for that entry.
        let mut csc_entry_of_pos = vec![0usize; num_non_empty_blocks];
        // remembers where the diagonal block ended up in column j.
        let mut diag_pos_in_csc = vec![None; num_block_cols];
        // per-column write cursor.
        let mut col_cursor = csc_col_ptr.clone();

        // writes only off-diagonal blocks into each column in increasing order of discovery.
        for &(block_row_idx, block_col_idx, entry_idx) in &entries {
            if region_x_idx == region_y_idx && block_row_idx == block_col_idx {
                continue;
            }
            let pos = col_cursor[block_col_idx];
            // advance cursor at block_col_idx.
            col_cursor[block_col_idx] += 1;
            csc_row_idx[pos] = block_row_idx as u32;
            csc_entry_of_pos[pos] = entry_idx;
        }
        if region_x_idx == region_y_idx {
            // On diagonal regions, we put the diagonal block (if it exists) at the last slot of its
            // column.
            for block_col_idx in 0..num_block_cols {
                if let Some(entry_idx) = diag_entry_indices[block_col_idx] {
                    let p = csc_col_ptr[block_col_idx + 1] - 1;
                    csc_row_idx[p] = block_col_idx as u32;
                    csc_entry_of_pos[p] = entry_idx;
                    diag_pos_in_csc[block_col_idx] = Some(p);
                }
            }
        }

        // CSR

        // CSR pointers with a cumulative sum (last entry equal to num_non_empty_blocks).
        let mut csr_row_ptr = vec![0usize; num_block_rows + 1];
        for block_row_idx in 0..num_block_rows {
            // csr_row_ptr[i] is the starting slot in csr_row_ptr for row i.
            // csr_row_ptr[i+1] is the end (start of the next row).
            csr_row_ptr[block_row_idx + 1] = csr_row_ptr[block_row_idx] + count_rows[block_row_idx];
        }
        // stores block row index for the entry placed at pos.
        let mut csr_col_idx = vec![0u32; num_non_empty_blocks];
        // stores entry_idx for that entry.
        let mut csr_entry_of_pos = vec![0usize; num_non_empty_blocks];
        // per-column write cursor.
        let mut row_cursor = csr_row_ptr.clone();

        // writes only off-diagonal blocks into each row in increasing order of discovery.
        for &(block_row_idx, block_col_idx, eidx) in &entries {
            let pos = row_cursor[block_row_idx];
            // advance cursor at block_row_idx.
            row_cursor[block_row_idx] += 1;
            csr_col_idx[pos] = block_col_idx as u32;
            csr_entry_of_pos[pos] = eidx;
        }

        CompressedBlockRegion {
            block_shape: region.block_shape,
            region_shape: [num_block_rows, num_block_cols],
            num_non_empty_blocks,
            flattened_block_storage,
            csc_col_ptr,
            csc_row_idx,
            csc_entry_of_pos,
            diag_pos_in_csc,
            csr_row_ptr,
            csr_col_idx,
            csr_entry_of_pos,
        }
    }
}
