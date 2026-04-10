//! Kernels for supernodal LDLᵀ factorization.
//!
//! These will be used by the upcoming supernodal block-sparse LDLᵀ rewrite.

/// Gather rows from `src` into `dst` using an index map.
///
/// `dst[li, c] = src[row_indices[li], c]` for all columns.
/// Both matrices in column-major layout.
#[inline]
#[allow(dead_code)]
pub fn gather_rows(
    dst: &mut [f64],
    src: &[f64],
    row_indices: &[usize],
    src_nrows: usize,
    ncols: usize,
) {
    let nm = row_indices.len();
    debug_assert_eq!(dst.len(), nm * ncols);
    debug_assert_eq!(src.len(), src_nrows * ncols);

    for c in 0..ncols {
        let src_base = c * src_nrows;
        let dst_base = c * nm;
        for (li, &ri) in row_indices.iter().enumerate() {
            dst[dst_base + li] = src[src_base + ri];
        }
    }
}
