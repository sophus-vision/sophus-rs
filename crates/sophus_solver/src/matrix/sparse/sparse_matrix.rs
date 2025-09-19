use std::fmt::Display;

use crate::matrix::sparse::TripletMatrix;

/// Sparse `N x N` matrix in compressed column form.
///
/// invariant: pattern.row_idx.len() == values.len()
#[derive(Clone, Debug)]
pub struct SparseMatrix {
    pattern: ColumnCompressedPattern,
    value_storage: Vec<f64>,
}

/// Read-only sparsity pattern of [SparseMatrix].
///
/// Invariants:
/// - `storage_idx_by_col.len() == scalar_dim + 1`
/// - `storage_idx_by_col[0] == 0`
/// - `storage_idx_by_col[col_count] == row_idx_storage.len()`
/// - `storage_idx_by_col` is non-decreasing
/// - every `row_idx_storage[k] < scalar_dim`
/// - within each column, `row_idx_storage` is strictly increasing (i.e. no duplicates)
#[derive(Clone, Debug)]
pub struct ColumnCompressedPattern {
    scalar_dim: usize, // N
    storage_idx_by_col: Vec<usize>,
    row_idx_storage: Vec<usize>,
}

impl Display for SparseMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\nvalues: {:?}", self.pattern, self.value_storage())
    }
}

impl SparseMatrix {
    /// Create new CSC (Compressed Sparse Column) matrix.
    pub fn new(pattern: ColumnCompressedPattern, values: Vec<f64>) -> Self {
        assert_eq!(pattern.row_idx_storage.len(), values.len(),);

        Self {
            pattern,
            value_storage: values,
        }
    }

    /// Construct CSC (Compressed Sparse Column) matrix from triplets.
    pub fn from_triplets(triplet_mat: &TripletMatrix) -> SparseMatrix {
        let scalar_dim = triplet_mat.scalar_dimension();
        let non_zero_count = triplet_mat.sorted_triplets().len();

        // count per column
        let mut col_counts = vec![0usize; scalar_dim];
        for k in 0..non_zero_count {
            col_counts[triplet_mat.sorted_triplets()[k].1] += 1;
        }

        // prefix sum -> storage_offset_by_col
        let mut storage_offset_by_col = vec![0usize; scalar_dim + 1];
        for j in 0..scalar_dim {
            storage_offset_by_col[j + 1] = storage_offset_by_col[j] + col_counts[j];
        }

        // fill with coalescing
        let mut row_idx = vec![0usize; non_zero_count];
        let mut values = vec![0f64; non_zero_count];
        let mut next = storage_offset_by_col.clone();

        for k in 0..non_zero_count {
            let (i, j, x) = triplet_mat.sorted_triplets()[k];
            let pos = next[j];
            if pos > storage_offset_by_col[j] && row_idx[pos - 1] == i {
                values[pos - 1] += x; // coalesce duplicates
            } else {
                row_idx[pos] = i;
                values[pos] = x;
                next[j] += 1;
            }
        }

        // compact columns
        let mut write_ptr = 0usize;
        for j in 0..scalar_dim {
            let start = storage_offset_by_col[j];
            let stop = next[j];
            if write_ptr != start {
                row_idx.copy_within(start..stop, write_ptr);
                values.copy_within(start..stop, write_ptr);
            }
            storage_offset_by_col[j] = write_ptr;
            write_ptr += stop - start;
        }
        storage_offset_by_col[scalar_dim] = write_ptr;
        row_idx.truncate(write_ptr);
        values.truncate(write_ptr);

        let pattern = ColumnCompressedPattern::new(
            scalar_dim,
            storage_offset_by_col,
            row_idx,
        );

        SparseMatrix {
            pattern,
            value_storage: values,
        }
    }

    /// Get CSC pattern struct.
    #[inline]
    pub fn pattern(&self) -> &ColumnCompressedPattern {
        &self.pattern
    }

    /// Get values.
    #[inline]
    pub fn value_storage(&self) -> &[f64] {
        &self.value_storage
    }

    /// Number of rows (or columns) `N`.
    #[inline]
    pub fn scalar_dim(&self) -> usize {
        self.pattern.scalar_dim
    }

    /// Number of stored non-zeros.
    #[inline]
    pub fn nonzero_count(&self) -> usize {
        self.pattern.row_idx_storage.len()
    }

    /// Column pointer slice.
    #[inline]
    pub fn storage_idx_by_col(&self) -> &[usize] {
        &self.pattern.storage_idx_by_col
    }

    /// Row indices slice.
    #[inline]
    pub fn row_idx_storage(&self) -> &[usize] {
        &self.pattern.row_idx_storage
    }

    /// Transpose matrix
    #[inline]
    pub fn transpose(&self) -> Self {
        let mut t = self.clone();
        t.pattern = self.pattern.transpose();
        t
    }

    /// Convert CSC to a dense matrix.
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let n = self.scalar_dim();
        let mut dense = nalgebra::DMatrix::<f64>::zeros(n, n);

        let storage_offset_by_col = self.storage_idx_by_col();
        let row_idx = self.row_idx_storage();
        let vals = self.value_storage();

        for j in 0..n {
            let start = storage_offset_by_col[j];
            let end = storage_offset_by_col[j + 1];
            for p in start..end {
                let i = row_idx[p];
                dense[(i, j)] = vals[p];
            }
        }
        dense
    }

    /// Under the assumption that this is an upper or lower triangular matrix, the corresponding
    /// dense symmetric matrix is returned.
    ///
    /// Preconditions:
    ///  - self.row_count() == self.col_count()
    ///  - the sparsity pattern has (upper or lower) triangular shape
    ///
    /// ```
    /// use sophus_autodiff::linalg::MatF64;
    /// use sophus_solver::{
    ///     matrix::sparse::{
    ///         SparseMatrix,
    ///         TripletMatrix,
    ///     },
    ///     prelude::*,
    /// };
    ///
    /// let lower_triplets = TripletMatrix::new(
    ///     [(0, 0, 2.0), (2, 1, 3.0), (1, 0, 1.0), (2, 2, 0.5)].to_vec(),
    ///     3,
    /// );
    /// let lower = SparseMatrix::from_triplets(&lower_triplets);
    /// let upper = lower.transpose();
    ///
    /// let symmetric = MatF64::<3, 3>::from_array2([
    ///     [2.0, 1.0, 0.0], //
    ///     [1.0, 0.0, 3.0],
    ///     [0.0, 3.0, 0.5],
    /// ]);
    ///
    /// assert_eq!(lower.triangular_to_dense_symmetric(), symmetric);
    /// assert_eq!(upper.triangular_to_dense_symmetric(), symmetric);
    /// ```
    pub fn triangular_to_dense_symmetric(&self) -> nalgebra::DMatrix<f64> {
        let n = self.scalar_dim();
        let mut dense = nalgebra::DMatrix::<f64>::zeros(n, n);

        let storage_offset_by_col = self.storage_idx_by_col();
        let row_idx_storage = self.row_idx_storage();
        let vals = self.value_storage();

        for col_j in 0..n {
            let start = storage_offset_by_col[col_j];
            let end = storage_offset_by_col[col_j + 1];
            for storage_idx in start..end {
                let row_i = row_idx_storage[storage_idx];
                dense[(col_j, row_i)] = vals[storage_idx];
                dense[(row_i, col_j)] = vals[storage_idx];
            }
        }
        dense
    }
}

impl Display for ColumnCompressedPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "#rows: {}, #cols{}\n storage_offset_by_col: {:?}\n row_idx: {:?}",
            self.scalar_dim(),
            self.scalar_dim(),
            self.storage_idx_by_col(),
            self.row_idx_storage()
        )
    }
}

impl ColumnCompressedPattern {
    /// Create a CSC pattern.
    pub fn new(
        scalar_dim: usize,
        storage_offset_by_col: Vec<usize>,
        row_idx_storage: Vec<usize>,
    ) -> Self {
        // Cheap shape checks (always on)
        assert!(
            !storage_offset_by_col.is_empty(),
            "storage_offset_by_col must have length N+1 (>= 1)"
        );
        assert_eq!(
            storage_offset_by_col.len(),
            scalar_dim + 1,
            "storage_offset_by_col length must be N+1 (got {}, expected {})",
            storage_offset_by_col.len(),
            scalar_dim + 1
        );
        assert_eq!(
            storage_offset_by_col[0], 0,
            "storage_offset_by_col[0] must be 0"
        );
        assert_eq!(
            *storage_offset_by_col.last().unwrap(),
            row_idx_storage.len(),
            "storage_offset_by_col[N] must equal row_idx.len() (got {}, expected {})",
            storage_offset_by_col[scalar_dim],
            row_idx_storage.len()
        );

        #[cfg(debug_assertions)]
        {
            // storage_offset_by_col non-decreasing
            for (i, w) in storage_offset_by_col.windows(2).enumerate() {
                debug_assert!(
                    w[0] <= w[1],
                    "storage_offset_by_col must be non-decreasing (storage_offset_by_col[{i}]={}, storage_offset_by_col[{i}+1]={})",
                    w[0],
                    w[1]
                );
            }

            // Row bounds
            for (k, &r) in row_idx_storage.iter().enumerate() {
                debug_assert!(
                    r < scalar_dim,
                    "row_idx[{k}] = {r} out of bounds (row_count = {scalar_dim})"
                );
            }

            // Per-column slice bounds + strictly increasing rows (no dups)
            let nnz = row_idx_storage.len();
            for j in 0..scalar_dim {
                let start = storage_offset_by_col[j];
                let end = storage_offset_by_col[j + 1];

                debug_assert!(
                    start <= end && end <= nnz,
                    "invalid col slice in column {j}: [{start}..{end}) with nnz={nnz}"
                );

                let mut it = row_idx_storage[start..end].iter().copied();
                if let Some(mut prev) = it.next() {
                    for r in it {
                        debug_assert!(
                            prev < r,
                            "row_idx within column {j} must be strictly increasing ({prev} !< {r}); \
                         duplicates or unsorted entries detected"
                        );
                        prev = r;
                    }
                }
            }
        }

        ColumnCompressedPattern {
            scalar_dim,
            storage_idx_by_col: storage_offset_by_col,
            row_idx_storage,
        }
    }

    /// Number of rows (or columns) `N`.
    #[inline]
    pub fn scalar_dim(&self) -> usize {
        self.scalar_dim
    }

    /// Number of non-zeros stored.
    #[inline]
    pub fn nonzero_count(&self) -> usize {
        self.row_idx_storage.len()
    }

    /// Column pointer slice.
    #[inline]
    pub fn storage_idx_by_col(&self) -> &[usize] {
        &self.storage_idx_by_col
    }

    /// Row indices slice.
    #[inline]
    pub fn row_idx_storage(&self) -> &[usize] {
        &self.row_idx_storage
    }

    /// Transpose the CSC pattern.
    pub fn transpose(&self) -> ColumnCompressedPattern {
        let scalar_dim = self.scalar_dim();
        let nnz = self.nonzero_count();

        // 1) Count entries per transposed column (== per original row).
        let mut counts = vec![0usize; scalar_dim];
        for &i in self.row_idx_storage() {
            counts[i] += 1;
        }

        // 2) Prefix-sum to get storage_offset_by_col_t (len = m + 1).
        let mut storage_offset_by_col = vec![0usize; scalar_dim + 1];
        for i in 0..scalar_dim {
            storage_offset_by_col[i + 1] = storage_offset_by_col[i] + counts[i];
        }

        // 3) Fill row_idx_t by scanning original columns.
        let mut row_idx = vec![0usize; nnz];
        let mut next = storage_offset_by_col.clone();
        for j in 0..scalar_dim {
            let start = self.storage_idx_by_col()[j];
            let end = self.storage_idx_by_col()[j + 1];
            for p in start..end {
                let i = self.row_idx_storage()[p]; // original row
                let dst = next[i];
                row_idx[dst] = j; // transposed row index = original column
                next[i] += 1;
            }
        }

        // Use the constructor to assert invariants in debug builds.
        ColumnCompressedPattern::new(scalar_dim, storage_offset_by_col, row_idx)
    }
}
