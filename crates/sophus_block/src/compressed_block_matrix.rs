use crate::Grid;

/// Block matrix in compressed form.
#[derive(Debug)]
pub struct CompressedBlockMatrix {
    pub(crate) region_grid: Grid<CompressedBlockRegion>,
    pub(crate) index_offset_per_row_partition: Vec<usize>,
    pub(crate) index_offset_per_col_partition: Vec<usize>,
    pub(crate) scalar_shape: [usize; 2],
}

#[derive(Debug, Clone)]
pub(crate) struct CompressedBlockRegion {
    // Dimensions (rows, columns) of each block
    pub(crate) block_shape: [usize; 2],
    // The region as region_shape[0] x region_shape[1] blocks.
    pub(crate) region_shape: [usize; 2],
    // Number of non-empty blocks
    pub(crate) num_non_empty_blocks: usize,

    // Packed blocks: num_blocks() * (block_rows * block_cols), column-major per block
    pub(crate) flattened_block_storage: Vec<f64>,

    // CSC (by block-column)
    pub(crate) csc_col_ptr: Vec<usize>, // len = num_block_cols + 1
    pub(crate) csc_row_idx: Vec<u32>,   // len = num_blocks(); block row indices
    pub(crate) csc_entry_of_pos: Vec<usize>, // len = num_blocks(); maps CSC position -> entry_idx
    pub(crate) diag_pos_in_csc: Vec<Option<usize>>, /* len = num_block_cols; None for
                                         * off-diagonal regions */

    // CSR (by block-row)
    pub(crate) csr_row_ptr: Vec<usize>, // len = n_block_rows + 1
    pub(crate) csr_col_idx: Vec<u32>,   // len = num_blocks(); block col indices
    pub(crate) csr_entry_of_pos: Vec<usize>, // len = num_blocks(); maps CSR position -> entry_idx
}

impl CompressedBlockRegion {
    pub fn empty() -> Self {
        Self {
            block_shape: [0, 0],
            region_shape: [0, 0],
            num_non_empty_blocks: 0,
            flattened_block_storage: Vec::new(),
            csc_col_ptr: vec![0],
            csc_row_idx: Vec::new(),
            csc_entry_of_pos: Vec::new(),
            diag_pos_in_csc: Vec::new(),
            csr_row_ptr: vec![0],
            csr_col_idx: Vec::new(),
            csr_entry_of_pos: Vec::new(),
        }
    }

    #[inline]
    pub fn num_block_elems(&self) -> usize {
        self.block_shape[0] * self.block_shape[1]
    }

    #[inline]
    pub fn block_slice(&self, entry_idx: usize) -> &[f64] {
        let o = entry_idx * self.num_block_elems();
        &self.flattened_block_storage[o..o + self.num_block_elems()]
    }

    /// Iterate a block row (CSR): yields (block-col, entry_idx)
    #[inline]
    pub fn iter_csr_row(&self, br: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let start = self.csr_row_ptr[br];
        let end = self.csr_row_ptr[br + 1];
        (start..end).map(move |pos| (self.csr_col_idx[pos] as usize, self.csr_entry_of_pos[pos]))
    }

    /// Iterate a block col (CSC): yields (block-row, entry_idx)
    #[inline]
    pub fn iter_csc_col(&self, bc: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let start = self.csc_col_ptr[bc];
        let end = self.csc_col_ptr[bc + 1];
        (start..end).map(move |pos| (self.csc_row_idx[pos] as usize, self.csc_entry_of_pos[pos]))
    }
}
