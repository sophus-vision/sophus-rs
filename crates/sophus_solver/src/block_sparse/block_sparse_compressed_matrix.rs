use crate::{
    block_sparse::{
        BlockSparseMatrixBuilder,
        PartitionIndexOffsets,
    },
    grid::Grid,
};

/// Block matrix in compressed form.
#[derive(Debug)]
pub struct BlockSparseCompressedMatrix {
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

        // ---- pass 1: count per global block column j ----
        let mut count_per_col = vec![0usize; num_block_cols];

        for ry in 0..n_parts {
            let nbc_ry = nb_cols_per_ry[ry];
            for bc in 0..nbc_ry {
                let j = block_col_off[ry] + bc;

                // strictly above-diagonal regions: rx < ry
                for rx in 0..ry {
                    let reg = region_grid.get(&[rx, ry]);
                    if reg.num_non_empty_blocks == 0 {
                        continue;
                    }
                    for (br, _) in reg.iter_csc_col(bc) {
                        let i = block_row_off[rx] + br;
                        debug_assert!(i < j, "structure says above diagonal");
                        count_per_col[j] += 1;
                    }
                }

                // strictly upper blocks inside diagonal region (ry,ry): br < bc
                let reg_diag = region_grid.get(&[ry, ry]);
                if reg_diag.num_non_empty_blocks != 0 {
                    for (br, _) in reg_diag.iter_csc_col(bc) {
                        if br < bc {
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

        // Reuse `count_per_col` as write cursors (no new allocation):
        // make it equal to the starting offsets of each column.
        let mut w = count_per_col; // length == num_block_cols
        w.copy_from_slice(&csc_col_ptr[..num_block_cols]);

        // ---- pass 2: fill rows into each column ----
        for ry in 0..n_parts {
            let nbc_ry = nb_cols_per_ry[ry];
            for bc in 0..nbc_ry {
                let j = block_col_off[ry] + bc;

                // from regions strictly above diagonal
                for rx in 0..ry {
                    let reg = region_grid.get(&[rx, ry]);
                    if reg.num_non_empty_blocks == 0 {
                        continue;
                    }
                    for (br, _) in reg.iter_csc_col(bc) {
                        let i = block_row_off[rx] + br;
                        let pos = w[j];
                        w[j] = pos + 1;
                        csc_row_idx[pos] = i as u32;
                    }
                }

                // from diagonal region: strictly upper only
                let reg_diag = region_grid.get(&[ry, ry]);
                if reg_diag.num_non_empty_blocks != 0 {
                    for (br, _) in reg_diag.iter_csc_col(bc) {
                        if br < bc {
                            let i = block_row_off[ry] + br;
                            let pos = w[j];
                            w[j] = pos + 1;
                            csc_row_idx[pos] = i as u32;
                        }
                    }
                }

                // Keep deterministic ordering (cost-free in allocations; sort is in-place).
                let start = csc_col_ptr[j];
                let end = w[j];
                csc_row_idx[start..end].sort_unstable();

                // If you ensure region CSC columns are sorted and you iterate rx in ascending order
                // (which you do), you can remove the sort above entirely.
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
        let [region_x_idx, region_y_idx] = region_idx;

        let region = mat.get_region(&region_idx);
        let br_dim = region.block_shape[0];
        let bc_dim = region.block_shape[1];
        if br_dim == 0 || bc_dim == 0 {
            return CompressedBlockRegion::empty();
        }
        let num_block_elems = br_dim * bc_dim;

        // #block rows in this region’s partition grid
        let num_block_rows = {
            let start = mat.index_offsets.per_row_partition[region_x_idx];
            let end = mat
                .index_offsets
                .per_row_partition
                .get(region_x_idx + 1)
                .copied()
                .unwrap_or(mat.scalar_shape[0]);
            (end - start) / br_dim
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
            (end - start) / bc_dim
        };

        // Fast path: no triplets in this region (but keep shapes/ptrs consistent)
        if region.triplets.is_empty() {
            return CompressedBlockRegion {
                block_shape: [br_dim, bc_dim],
                region_shape: [num_block_rows, num_block_cols],
                num_non_empty_blocks: 0,
                flattened_block_storage: Vec::new(),
                csc_col_ptr: vec![0; num_block_cols + 1],
                csc_row_idx: Vec::new(),
                csc_entry_of_pos: Vec::new(),
                diag_pos_in_csc: vec![None; num_block_cols],
            };
        }

        // ------------- Pass 0: sort & coalesce keys (no hashing) -------------
        // Build (br, bc, start_data_idx) and sort by (bc, br) so we can group in O(n).
        let mut coords: Vec<(usize, usize, usize)> = Vec::with_capacity(region.triplets.len());
        for t in &region.triplets {
            let br = t.block_idx[0];
            let bc = t.block_idx[1];
            debug_assert!(br < num_block_rows && bc < num_block_cols);
            coords.push((br, bc, t.start_data_idx));
        }
        coords.sort_unstable_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0))); // (bc, br)

        // ------------- Pass 1: count unique blocks per column, diag flags -------------
        let mut count_cols = vec![0usize; num_block_cols]; // total uniques per col
        let mut diag_exists = vec![false; num_block_cols];
        let mut unique_blocks = 0usize;

        let mut k = 0usize;
        while k < coords.len() {
            let (br, bc, _) = coords[k];
            // advance over equal (br, bc)
            k += 1;
            while k < coords.len() && coords[k].0 == br && coords[k].1 == bc {
                k += 1;
            }
            count_cols[bc] += 1;
            if region_x_idx == region_y_idx && br == bc {
                diag_exists[bc] = true;
            }
            unique_blocks += 1;
        }

        // ------------- CSC structure: col_ptr & diag placement -------------
        let mut csc_col_ptr = vec![0usize; num_block_cols + 1];
        for j in 0..num_block_cols {
            csc_col_ptr[j + 1] = csc_col_ptr[j] + count_cols[j];
        }
        let nnz = csc_col_ptr[num_block_cols];

        let mut csc_row_idx = vec![0u32; nnz];
        let mut csc_entry_of_pos = vec![0usize; nnz];
        let mut diag_pos_in_csc = vec![None; num_block_cols];

        // Off-diagonal write cursors per column; reuse count_cols as cursors
        // by initializing it with the starts.
        count_cols[..num_block_cols].copy_from_slice(&csc_col_ptr[..num_block_cols]);

        // Pre-compute where the diagonal (if present) should go: last slot in the column.
        for j in 0..num_block_cols {
            if diag_exists[j] {
                diag_pos_in_csc[j] = Some(csc_col_ptr[j + 1] - 1);
            }
        }

        // ------------- Allocate block storage (exact) -------------
        let mut flattened_block_storage = vec![0.0f64; unique_blocks * num_block_elems];

        // ------------- Pass 2: fill storage + CSC arrays -------------
        let mut eidx = 0usize; // entry index in block storage
        let mut ptr = 0usize;
        while ptr < coords.len() {
            let (br, bc, start0) = coords[ptr];
            // Determine target CSC position for (br, bc)
            let pos = if region_x_idx == region_y_idx && br == bc {
                // diagonal goes last
                diag_pos_in_csc[bc].unwrap()
            } else {
                let p = count_cols[bc];
                count_cols[bc] = p + 1;
                p
            };
            csc_row_idx[pos] = br as u32;
            csc_entry_of_pos[pos] = eidx;

            // Sum all duplicates of this (br, bc) into the eidx-th block
            let dst =
                &mut flattened_block_storage[eidx * num_block_elems..(eidx + 1) * num_block_elems];
            {
                let mut q = ptr;
                while q < coords.len() && coords[q].0 == br && coords[q].1 == bc {
                    let start = coords[q].2;
                    let src = &region.flattened_block_storage[start..start + num_block_elems];
                    // dst += src
                    for t in 0..num_block_elems {
                        dst[t] += src[t];
                    }
                    q += 1;
                }
                ptr = q;
            }

            eidx += 1;
        }
        debug_assert_eq!(eidx, unique_blocks);

        CompressedBlockRegion {
            block_shape: [br_dim, bc_dim],
            region_shape: [num_block_rows, num_block_cols],
            num_non_empty_blocks: unique_blocks,
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

impl BlockSparseCompressedMatrix {
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
