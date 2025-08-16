use crate::{
    BlockSparseMatrixBuilder,
    grid::Grid,
};

#[derive(Debug)]
pub struct BlockCscMatrix {
    pub nrows: usize, // scalar
    pub ncols: usize, // scalar

    // Scalar prefix sums over *global block rows/cols* (BR+1 and BC+1 long).
    pub row_splits: Vec<usize>,
    pub col_splits: Vec<usize>,

    // Block-level CSC over *global* block columns (BC+1).
    pub col_ptr: Vec<usize>,  // length = BC + 1
    pub row_ind: Vec<u32>,    // length = nnzb (global block-row indices)
    pub region_loc: Vec<u32>, // length = nnzb (index into regions[ri,ci].storage)

    // Maps global block row/col → region row/col (0..R-1 / 0..C-1).
    pub row_class_of_br: Vec<u16>, // length = BR
    pub col_class_of_bc: Vec<u16>, // length = BC

    // Region payloads, sized exactly like the partition grid: R × C.
    pub regions: Grid<BlockRegion>,
}

#[derive(Debug, Clone)]
pub struct BlockRegion {
    // Back-to-back blocks for this (region_row r, region_col c) cell.
    // Each block is column-major with stride = h(r) * w(c).
    pub storage: Vec<f64>,
}

use std::collections::hash_map::{
    Entry,
    HashMap,
};

impl BlockSparseMatrixBuilder {
    /// Convert to Variable-block CSC, *deduplicating* repeated block indices
    /// inside each (region_row r, region_col c) by summing their payloads.
    pub fn to_block_csc(&self) -> BlockCscMatrix {
        // ----- Basic sizes and dims per region row/col -----
        let region_rows = self.index_offsets.per_row_partition.len(); // # region rows (row partitions)
        let region_cols = self.index_offsets.per_col_partition.len(); // # region cols (col partitions)
        let nrows = self.scalar_shape[0];
        let ncols = self.scalar_shape[1];

        // Block dims per region row/col (Mi, Nj)
        let mut row_block_dim = vec![0usize; region_rows];
        for r in 0..region_rows {
            row_block_dim[r] = self.region_grid.get(&[r, 0]).block_shape[0];
            debug_assert!(row_block_dim[r] > 0);
        }
        let mut col_block_dim = vec![0usize; region_cols];
        for c in 0..region_cols {
            col_block_dim[c] = self.region_grid.get(&[0, c]).block_shape[1];
            debug_assert!(col_block_dim[c] > 0);
        }

        // Blocks-per-partition (not scalars)
        let mut blocks_per_row_part = vec![0usize; region_rows];
        for r in 0..region_rows {
            let start = self.index_offsets.per_row_partition[r];
            let end = if r + 1 < region_rows {
                self.index_offsets.per_row_partition[r + 1]
            } else {
                nrows
            };
            let h = row_block_dim[r];
            debug_assert_eq!(
                (end - start) % h,
                0,
                "Row partition length must be multiple of its block dim"
            );
            blocks_per_row_part[r] = (end - start) / h;
        }
        let mut blocks_per_col_part = vec![0usize; region_cols];
        for c in 0..region_cols {
            let start = self.index_offsets.per_col_partition[c];
            let end = if c + 1 < region_cols {
                self.index_offsets.per_col_partition[c + 1]
            } else {
                ncols
            };
            let w = col_block_dim[c];
            debug_assert_eq!(
                (end - start) % w,
                0,
                "Col partition length must be multiple of its block dim"
            );
            blocks_per_col_part[c] = (end - start) / w;
        }

        // Global block-row/col prefix sums
        let mut br_off = vec![0usize; region_rows + 1];
        for r in 0..region_rows {
            br_off[r + 1] = br_off[r] + blocks_per_row_part[r];
        }
        let mut bc_off = vec![0usize; region_cols + 1];
        for c in 0..region_cols {
            bc_off[c + 1] = bc_off[c] + blocks_per_col_part[c];
        }
        let BR = br_off[region_rows]; // # global block rows
        let BC = bc_off[region_cols]; // # global block cols

        // Scalar splits over global block rows/cols
        let mut row_splits = Vec::with_capacity(BR + 1);
        {
            let mut acc = 0usize;
            for r in 0..region_rows {
                let h = row_block_dim[r];
                for _ in 0..blocks_per_row_part[r] {
                    row_splits.push(acc);
                    acc += h;
                }
            }
            row_splits.push(acc);
            debug_assert_eq!(acc, nrows);
        }
        let mut col_splits = Vec::with_capacity(BC + 1);
        {
            let mut acc = 0usize;
            for c in 0..region_cols {
                let w = col_block_dim[c];
                for _ in 0..blocks_per_col_part[c] {
                    col_splits.push(acc);
                    acc += w;
                }
            }
            col_splits.push(acc);
            debug_assert_eq!(acc, ncols);
        }

        // Fast maps: global block row/col → region row/col
        let mut row_class_of_br = vec![0u16; BR];
        for r in 0..region_rows {
            for gbr in br_off[r]..br_off[r + 1] {
                row_class_of_br[gbr] = r as u16;
            }
        }
        let mut col_class_of_bc = vec![0u16; BC];
        for c in 0..region_cols {
            for gbc in bc_off[c]..bc_off[c + 1] {
                col_class_of_bc[gbc] = c as u16;
            }
        }

        // ----- DEDUP per region (r,c) -----
        // We produce two temporary grids:
        //   1) dedup_storage[r,c]: Vec<f64> with unique, summed blocks (column-major)
        //   2) uniq_list[r,c]: Vec<([usize;2], usize)> of (local_block_idx, region_loc)
        #[derive(Default, Clone)]
        struct RegionUniq {
            storage: Vec<f64>,
            // (local_row_block, local_col_block, region_loc)
            uniq: Vec<([usize; 2], usize)>,
        }
        let mut dedup_storage = Grid::new([region_rows, region_cols], RegionUniq::default());

        for r in 0..region_rows {
            let h = row_block_dim[r];
            for c in 0..region_cols {
                let w = col_block_dim[c];
                let stride = h * w;

                let region = self.region_grid.get(&[r, c]);
                if region.triplets.is_empty() {
                    continue;
                }

                // Key: (local_row_block, local_col_block) -> region_loc (index in dedup storage)
                let mut where_is: HashMap<(usize, usize), usize> = HashMap::new();

                // Destination buffers
                let dst = dedup_storage.get_mut(&[r, c]);

                // Walk all triplets and sum duplicates
                for t in &region.triplets {
                    let (bi, bj) = (t.block_idx[0], t.block_idx[1]); // local block indices
                    let dst_loc = match where_is.entry((bi, bj)) {
                        Entry::Occupied(e) => *e.get(),
                        Entry::Vacant(v) => {
                            let loc = dst.storage.len() / stride;
                            // allocate space for a fresh block (zero-initialized)
                            dst.storage.resize(dst.storage.len() + stride, 0.0);
                            v.insert(loc);
                            dst.uniq.push(([bi, bj], loc));
                            loc
                        }
                    };

                    // Accumulate payload: src is column-major, length = h*w
                    let src0 = t.start_data_idx;
                    let src = &region.flattened_block_storage[src0..src0 + stride];
                    let base = dst_loc * stride;
                    // Column-major add
                    for j in 0..w {
                        let dst_col = &mut dst.storage[base + j * h..base + (j + 1) * h];
                        let src_col = &src[j * h..(j + 1) * h];
                        // elementwise +=
                        for i in 0..h {
                            dst_col[i] += src_col[i];
                        }
                    }
                }
            }
        }

        // Materialize the output regions grid (payload only)
        let mut regions = Grid::new(
            [region_rows, region_cols],
            BlockRegion {
                storage: Vec::new(),
            },
        );
        for r in 0..region_rows {
            for c in 0..region_cols {
                regions.get_mut(&[r, c]).storage =
                    std::mem::take(&mut dedup_storage.get_mut(&[r, c]).storage);
            }
        }

        // ----- Assemble global block CSC (unique & sorted) -----
        // Collect entries per *global* block column
        let mut per_gbc: Vec<
            Vec<(
                u32, /* gbr */
                u32, /* region_loc */
                u16, /* r */
            )>,
        > = vec![Vec::new(); BC];

        for r in 0..region_rows {
            let h = row_block_dim[r];
            for c in 0..region_cols {
                let w = col_block_dim[c];
                let stride = h * w;

                let ru = dedup_storage.get(&[r, c]); // has uniq list, storage already moved out
                if ru.uniq.is_empty() {
                    continue;
                }

                for &([bi, bj], loc) in &ru.uniq {
                    let gbr = br_off[r] + bi;
                    let gbc = bc_off[c] + bj;

                    // sanity checks
                    debug_assert!(gbr < BR && gbc < BC);
                    debug_assert_eq!(regions.get(&[r, c]).storage.len() % stride, 0);

                    per_gbc[gbc].push((gbr as u32, loc as u32, r as u16));
                }
            }
        }

        // Sort each column by global block-row
        for bc in 0..BC {
            per_gbc[bc].sort_unstable_by_key(|&(gbr, _, _)| gbr);
            // Optional: debug check uniqueness of gbr within column
            debug_assert!(
                per_gbc[bc].windows(2).all(|w| w[0].0 < w[1].0),
                "Duplicate global block rows remained in column {bc} after dedup"
            );
        }

        // Build col_ptr and flatten
        let mut col_ptr = Vec::with_capacity(BC + 1);
        col_ptr.push(0);
        for bc in 0..BC {
            col_ptr.push(col_ptr[bc] + per_gbc[bc].len());
        }
        let nnzb = col_ptr[BC];
        let mut row_ind = vec![0u32; nnzb];
        let mut region_loc = vec![0u32; nnzb];

        for bc in 0..BC {
            let start = col_ptr[bc];
            for (offset, (gbr, loc, _r)) in per_gbc[bc].iter().enumerate() {
                row_ind[start + offset] = *gbr;
                region_loc[start + offset] = *loc;
            }
        }

        BlockCscMatrix {
            nrows,
            ncols,
            row_splits,
            col_splits,
            col_ptr,
            row_ind,
            region_loc,
            row_class_of_br,
            col_class_of_bc,
            regions,
        }
    }
}
