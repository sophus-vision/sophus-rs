use crate::{
    BlockSparseMatrixBuilder,
    Grid,
    PartitionIndexOffsets,
};

/// Block matrix in compressed form.
#[derive(Debug)]
pub struct CompressedBlockMatrix {
    pub(crate) region_grid: Grid<CompressedBlockRegion>,
    pub(crate) index_offsets: PartitionIndexOffsets,
    pub(crate) sym_block_pattern: SymbolicBlockPattern,
}

#[inline]
fn num_block_rows_per_row_partition(
    index_offsets: &PartitionIndexOffsets,
    region_grid: &Grid<CompressedBlockRegion>,
) -> Vec<usize> {
    let n = index_offsets.per_row_partition.len();
    let mut out = Vec::with_capacity(n);
    for rx in 0..n {
        let reg = region_grid.get(&[rx, rx]);
        out.push(reg.region_shape[0]); // #block rows in row partition rx
    }
    out
}

#[inline]
fn num_block_cols_per_col_partition(
    index_offsets: &PartitionIndexOffsets,
    region_grid: &Grid<CompressedBlockRegion>,
) -> Vec<usize> {
    let n = index_offsets.per_col_partition.len();
    let mut out = Vec::with_capacity(n);
    for ry in 0..n {
        let reg = region_grid.get(&[ry, ry]);
        out.push(reg.region_shape[1]); // #block cols in col partition ry
    }
    out
}

/// Global block pattern for symbolic decomposition.
#[derive(Debug, Clone)]
pub struct SymbolicBlockPattern {
    /// total #block columns/rows (square)
    pub num_block_cols: usize,
    /// CSC pointers over block columns; len = n_block_cols + 1
    pub csc_col_ptr: Vec<usize>,
    /// row indices (block rows) for each structural entry; strictly i < j
    pub csc_row_idx: Vec<u32>,

    /// prefix sums of #block-rows per row partition.
    pub block_index_offset_per_row_partition: Vec<usize>,
    /// prefix sums of #block-cols per col partition.
    pub block_index_offset_per_col_partition: Vec<usize>,
}

impl SymbolicBlockPattern {
    #[inline]
    fn prefix_sum_usize(v: &[usize]) -> Vec<usize> {
        let mut ps = Vec::with_capacity(v.len() + 1);
        let mut s = 0usize;
        ps.push(0);
        for &x in v {
            s += x;
            ps.push(s);
        }
        ps
    }

    /// Build symbolic block pattern.
    pub(crate) fn from_regions(
        index_offsets: &PartitionIndexOffsets,
        region_grid: &Grid<CompressedBlockRegion>,
    ) -> SymbolicBlockPattern {
        let n_parts = index_offsets.per_row_partition.len();
        assert_eq!(n_parts, index_offsets.per_col_partition.len());

        let nb_rows_per_rx = num_block_rows_per_row_partition(index_offsets, region_grid);
        let nb_cols_per_ry = num_block_cols_per_col_partition(index_offsets, region_grid);

        let block_row_off = SymbolicBlockPattern::prefix_sum_usize(&nb_rows_per_rx); // len = n_parts + 1
        let block_col_off = SymbolicBlockPattern::prefix_sum_usize(&nb_cols_per_ry); // len = n_parts + 1
        let num_block_cols = block_col_off[n_parts];
        assert_eq!(
            num_block_cols, block_row_off[n_parts],
            "must be square at block level"
        );

        // ---- pass 1: count per block column j ----
        let mut count_per_col = vec![0usize; num_block_cols];

        for ry in 0..n_parts {
            for block_col_idx in 0..nb_cols_per_ry[ry] {
                let j = block_col_off[ry] + block_col_idx;

                // strictly above diagonal regions: rx < ry
                for rx in 0..ry {
                    let reg = region_grid.get(&[rx, ry]);
                    if reg.num_non_empty_blocks == 0 {
                        continue;
                    }
                    for (block_row_idx, _entry_idx) in reg.iter_csc_col(block_col_idx) {
                        let i = block_row_off[rx] + block_row_idx;
                        debug_assert!(i < j, "structure says above diagonal");
                        count_per_col[j] += 1;
                    }
                }

                // strictly upper blocks inside diagonal region (ry,ry): br < bc
                let reg_diag = region_grid.get(&[ry, ry]);
                if reg_diag.num_non_empty_blocks != 0 {
                    for (block_row_idx, _entry_idx) in reg_diag.iter_csc_col(block_col_idx) {
                        if block_row_idx < block_col_idx {
                            count_per_col[j] += 1;
                        }
                    }
                }
            }
        }

        // ---- cumsum → pointers ----
        let mut csc_col_ptr = vec![0usize; num_block_cols + 1];
        for j in 0..num_block_cols {
            csc_col_ptr[j + 1] = csc_col_ptr[j] + count_per_col[j];
        }
        let nnz = csc_col_ptr[num_block_cols];
        let mut csc_row_idx = vec![0u32; nnz];
        let mut w = csc_col_ptr.clone(); // write cursors

        // ---- pass 2: fill rows into each column ----
        for ry in 0..n_parts {
            let num_block_cols = nb_cols_per_ry[ry];
            for block_col_idx in 0..num_block_cols {
                let j = block_col_off[ry] + block_col_idx;

                // from regions strictly above diagonal
                for rx in 0..ry {
                    let reg = region_grid.get(&[rx, ry]);
                    if reg.num_non_empty_blocks == 0 {
                        continue;
                    }
                    for (block_row_idx, _entry_idx) in reg.iter_csc_col(block_col_idx) {
                        let i = block_row_off[rx] + block_row_idx;
                        let pos = w[j];
                        w[j] += 1;
                        csc_row_idx[pos] = i as u32;
                    }
                }

                // from diagonal region: strictly upper only
                let reg_diag = region_grid.get(&[ry, ry]);
                if reg_diag.num_non_empty_blocks != 0 {
                    for (block_row_idx, _entry_idx) in reg_diag.iter_csc_col(block_col_idx) {
                        if block_row_idx < block_col_idx {
                            let i = block_row_off[ry] + block_row_idx;
                            let pos = w[j];
                            w[j] += 1;
                            csc_row_idx[pos] = i as u32;
                        }
                    }
                }

                // optional: make order deterministic
                let start = csc_col_ptr[j];
                let end = w[j];
                csc_row_idx[start..end].sort_unstable();
                // (no dedup needed: your region compression already produced unique blocks)
            }
        }

        SymbolicBlockPattern {
            num_block_cols,
            csc_col_ptr,
            csc_row_idx,
            block_index_offset_per_row_partition: block_row_off,
            block_index_offset_per_col_partition: block_col_off,
        }
    }
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
    pub(crate) diag_pos_in_csc: Vec<Option<usize>>,
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
        }
    }

    pub fn from_block_sparse_matrix(
        mat: &BlockSparseMatrixBuilder,
        region_idx: [usize; 2],
    ) -> CompressedBlockRegion {
        use hashbrown::HashMap;

        let region_x_idx = region_idx[0];
        let region_y_idx = region_idx[1];

        let region = mat.get_region(&region_idx);
        if region.block_shape[0] == 0 || region.block_shape[1] == 0 {
            return CompressedBlockRegion::empty();
        }
        let num_block_elems = region.block_shape[0] * region.block_shape[1];

        // #block rows in this region’s partition grid
        let num_block_rows = {
            let start = mat.index_offsets.per_row_partition[region_x_idx];
            let end = mat
                .index_offsets
                .per_row_partition
                .get(region_x_idx + 1)
                .copied()
                .unwrap_or(mat.scalar_shape[0]);
            (end - start) / region.block_shape[0]
        };
        // #block cols in this region’s partition grid
        let num_block_cols = {
            let start = mat.index_offsets.per_col_partition[region_y_idx];
            let end = mat
                .index_offsets
                .per_col_partition
                .get(region_y_idx + 1)
                .copied()
                .unwrap_or(mat.scalar_shape[1]);
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

        CompressedBlockRegion {
            block_shape: region.block_shape,
            region_shape: [num_block_rows, num_block_cols],
            num_non_empty_blocks,
            flattened_block_storage,
            csc_col_ptr,
            csc_row_idx,
            csc_entry_of_pos,
            diag_pos_in_csc,
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

    // /// Iterate a block row (CSR): yields (block-col, entry_idx)
    // #[inline]
    // pub fn iter_csr_row(&self, br: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
    //     let start = self.csr_row_ptr[br];
    //     let end = self.csr_row_ptr[br + 1];
    //     (start..end).map(move |pos| (self.csr_col_idx[pos] as usize, self.csr_entry_of_pos[pos]))
    // }

    /// Iterate a block col (CSC): yields (block-row, entry_idx)
    #[inline]
    pub fn iter_csc_col(&self, bc: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let start = self.csc_col_ptr[bc];
        let end = self.csc_col_ptr[bc + 1];
        (start..end).map(move |pos| (self.csc_row_idx[pos] as usize, self.csc_entry_of_pos[pos]))
    }
}

// What we need from C to build numeric column j of A (lower part + diag):
pub(crate) struct AssembledEntry<'a> {
    pub i: usize,        // row block index (i > j)
    pub rdim: usize,     // m_i
    pub a_ij: &'a [f64], // (m_i x m_j), col-major
}
pub(crate) struct AssembledCol<'a> {
    pub m_j: usize,              // block size of column j
    pub diag: Option<&'a [f64]>, // (m_j x m_j)
    pub entries: Vec<AssembledEntry<'a>>,
}

impl CompressedBlockMatrix {
    /// Assemble LOWER part of column j (i > j) and the diagonal block A_jj.
    pub(crate) fn assemble_numeric_col_lower<'a>(&'a self, j: usize, out: &mut AssembledCol<'a>) {
        let sp = &self.sym_block_pattern;
        let col_off = &sp.block_index_offset_per_col_partition;

        // locate (rx, bc) for j
        let mut rx = 0usize;
        while rx + 1 < col_off.len() && j >= col_off[rx + 1] {
            rx += 1;
        }
        let bc = j - col_off[rx];

        let reg_diag = self.region_grid.get(&[rx, rx]);
        let m = reg_diag.block_shape[0];
        debug_assert_eq!(m, reg_diag.block_shape[1]);
        out.m_j = m;
        out.entries.clear();

        // Diagonal block A_jj
        out.diag = reg_diag
            .diag_pos_in_csc
            .get(bc)
            .and_then(|&p| p)
            .map(|pos| reg_diag.block_slice(reg_diag.csc_entry_of_pos[pos]));

        // Strictly-lower inside diagonal region: br > bc
        if reg_diag.num_non_empty_blocks != 0 {
            let rdim = reg_diag.block_shape[0]; // == m
            for (br, eidx) in reg_diag.iter_csc_col(bc) {
                if br <= bc {
                    continue;
                }
                let i = sp.block_index_offset_per_row_partition[rx] + br;
                out.entries.push(AssembledEntry {
                    i,
                    rdim,
                    a_ij: reg_diag.block_slice(eidx),
                });
            }
        }

        // Regions below the diagonal: (rxx > rx, rx)
        let n_parts = self.index_offsets.per_row_partition.len();
        for rxx in (rx + 1)..n_parts {
            let reg = self.region_grid.get(&[rxx, rx]);
            if reg.num_non_empty_blocks == 0 {
                continue;
            }
            let rdim = reg.block_shape[0];
            for (br, eidx) in reg.iter_csc_col(bc) {
                let i = sp.block_index_offset_per_row_partition[rxx] + br;
                out.entries.push(AssembledEntry {
                    i,
                    rdim,
                    a_ij: reg.block_slice(eidx),
                });
            }
        }

        // stable order by i
        out.entries.sort_by_key(|e| e.i);
    }
}
