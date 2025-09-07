use std::fmt::Display;

use crate::matrix::sparse::TripletMatrix;

/// Read-only sparsity pattern of an MxN matrix in CSC (Compressed Sparse Column) format.
///
/// Invariants:
/// - `storage_idx_by_col.len() == col_count + 1`
/// - `storage_idx_by_col[0] == 0`
/// - `storage_idx_by_col[col_count] == row_idx_storage.len()`
/// - `storage_idx_by_col` is non-decreasing
/// - every `row_idx_storage[k] < row_count`
/// - within each column, `row_idx_storage` is strictly increasing (i.e. no duplicates)
#[derive(Clone, Debug)]
pub struct ColumnCompressedPattern {
    row_count: usize, // M
    col_count: usize, // N
    storage_idx_by_col: Vec<usize>,
    row_idx_storage: Vec<usize>,
}

impl Display for ColumnCompressedPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "#rows: {}, #cols{}\n storage_offset_by_col: {:?}\n row_idx: {:?}",
            self.row_count(),
            self.col_count(),
            self.storage_idx_by_col(),
            self.row_idx_storage()
        )
    }
}

impl ColumnCompressedPattern {
    /// Create a CSC pattern.
    ///
    /// Invariants enforced:
    /// - `storage_idx_by_col.len() == col_count + 1`
    /// - `storage_idx_by_col[0] == 0`
    /// - `storage_idx_by_col[col_count] == row_idx_storage.len()`
    ///
    /// Invariants additionally checked in debug builds:
    /// - `storage_idx_by_col` is non-decreasing
    /// - all `row_idx_storage[k] < row_count`
    /// - within each column, `row_idx_storage` is strictly increasing (no duplicates)
    pub fn new(
        row_count: usize,
        col_count: usize,
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
            col_count + 1,
            "storage_offset_by_col length must be N+1 (got {}, expected {})",
            storage_offset_by_col.len(),
            col_count + 1
        );
        assert_eq!(
            storage_offset_by_col[0], 0,
            "storage_offset_by_col[0] must be 0"
        );
        assert_eq!(
            *storage_offset_by_col.last().unwrap(),
            row_idx_storage.len(),
            "storage_offset_by_col[N] must equal row_idx.len() (got {}, expected {})",
            storage_offset_by_col[col_count],
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
                    r < row_count,
                    "row_idx[{k}] = {r} out of bounds (row_count = {row_count})"
                );
            }

            // Per-column slice bounds + strictly increasing rows (no dups)
            let nnz = row_idx_storage.len();
            for j in 0..col_count {
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
            row_count,
            col_count,
            storage_idx_by_col: storage_offset_by_col,
            row_idx_storage,
        }
    }

    /// Number of rows (M).
    #[inline]
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Number of columns (N).
    #[inline]
    pub fn col_count(&self) -> usize {
        self.col_count
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

    /// Transpose the CSC pattern. Result is an NxM CSC (original^T).
    pub fn transpose(&self) -> ColumnCompressedPattern {
        let original_row_count = self.row_count(); // original M
        let original_col_count = self.col_count(); // original N
        let nnz = self.nonzero_count();

        // 1) Count entries per transposed column (== per original row).
        let mut counts = vec![0usize; original_row_count];
        for &i in self.row_idx_storage() {
            counts[i] += 1;
        }

        // 2) Prefix-sum to get storage_offset_by_col_t (len = m + 1).
        let mut storage_offset_by_col = vec![0usize; original_row_count + 1];
        for i in 0..original_row_count {
            storage_offset_by_col[i + 1] = storage_offset_by_col[i] + counts[i];
        }

        // 3) Fill row_idx_t by scanning original columns.
        let mut row_idx = vec![0usize; nnz];
        let mut next = storage_offset_by_col.clone();
        for j in 0..original_col_count {
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
        ColumnCompressedPattern::new(
            original_col_count,
            original_row_count,
            storage_offset_by_col,
            row_idx,
        )
    }
}

/// Compressed sparse column (CSC) matrix.
///
/// invariant: pattern.row_idx.len() == values.len()
#[derive(Clone, Debug)]
pub struct ColumnCompressedMatrix {
    pattern: ColumnCompressedPattern,
    value_storage: Vec<f64>,
}

impl Display for ColumnCompressedMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\nvalues: {:?}", self.pattern, self.value_storage())
    }
}

impl ColumnCompressedMatrix {
    /// Create new CSC (Compressed Sparse Column) matrix.
    pub fn new(pattern: ColumnCompressedPattern, values: Vec<f64>) -> Self {
        assert_eq!(pattern.row_idx_storage.len(), values.len(),);

        Self {
            pattern,
            value_storage: values,
        }
    }

    /// Construct CSC (Compressed Sparse Column) matrix from triplets.
    pub fn from_triplets(triplet_mat: &TripletMatrix) -> ColumnCompressedMatrix {
        let row_count = triplet_mat.row_count();
        let col_count = triplet_mat.col_count();
        let non_zero_count = triplet_mat.sorted_triplets().len();

        // count per column
        let mut col_counts = vec![0usize; col_count];
        for k in 0..non_zero_count {
            col_counts[triplet_mat.sorted_triplets()[k].1] += 1;
        }

        // prefix sum -> storage_offset_by_col
        let mut storage_offset_by_col = vec![0usize; col_count + 1];
        for j in 0..col_count {
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
        for j in 0..col_count {
            let start = storage_offset_by_col[j];
            let stop = next[j];
            if write_ptr != start {
                row_idx.copy_within(start..stop, write_ptr);
                values.copy_within(start..stop, write_ptr);
            }
            storage_offset_by_col[j] = write_ptr;
            write_ptr += stop - start;
        }
        storage_offset_by_col[col_count] = write_ptr;
        row_idx.truncate(write_ptr);
        values.truncate(write_ptr);

        let pattern =
            ColumnCompressedPattern::new(row_count, col_count, storage_offset_by_col, row_idx);

        ColumnCompressedMatrix {
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

    /// Number of rows (M).
    #[inline]
    pub fn row_count(&self) -> usize {
        self.pattern.row_count
    }

    /// Number of columns (N).
    #[inline]
    pub fn col_count(&self) -> usize {
        self.pattern.col_count
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
        let m = self.row_count();
        let n = self.col_count();
        let mut dense = nalgebra::DMatrix::<f64>::zeros(m, n);

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
    ///         ColumnCompressedMatrix,
    ///         TripletMatrix,
    ///     },
    ///     prelude::*,
    /// };
    ///
    /// let lower_triplets = TripletMatrix::new(
    ///     [(0, 0, 2.0), (2, 1, 3.0), (1, 0, 1.0), (2, 2, 0.5)].to_vec(),
    ///     3,
    ///     3,
    /// );
    /// let lower = ColumnCompressedMatrix::from_triplets(&lower_triplets);
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
        let m = self.row_count();
        let n = self.col_count();
        assert_eq!(m, n);
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

#[cfg(test)]
mod tests {
    use sophus_autodiff::{
        linalg::MatF64,
        prelude::*,
    };

    use super::*;

    fn mk_trips(trips: &[(usize, usize, f64)], m: usize, n: usize) -> TripletMatrix {
        TripletMatrix::new(trips.to_vec(), m, n)
    }

    #[test]
    fn basic_2x3_no_dups() {
        // A = [[10, 0, 30],
        //      [ 0,20,  0]]
        let t = mk_trips(&[(0, 0, 10.0), (1, 1, 20.0), (0, 2, 30.0)], 2, 3);
        let c = ColumnCompressedMatrix::from_triplets(&t);

        // shape
        assert_eq!(c.row_count(), 2);
        assert_eq!(c.col_count(), 3);
        assert_eq!(c.nonzero_count(), 3);

        // CSC structure
        // Column 0: row 0
        // Column 1: row 1
        // Column 2: row 0
        assert_eq!(c.storage_idx_by_col(), &[0, 1, 2, 3]);
        assert_eq!(c.row_idx_storage(), &[0, 1, 0]);
        assert_eq!(c.value_storage(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn coalesces_duplicates_in_same_col_row() {
        // Two triplets at (0,1) should coalesce.
        // A = [[0, 5+7, 0],
        //      [0,  0 , 0]]
        let t = mk_trips(&[(0, 1, 5.0), (0, 1, 7.0)], 2, 3);
        let c = ColumnCompressedMatrix::from_triplets(&t);

        assert_eq!(c.row_count(), 2);
        assert_eq!(c.col_count(), 3);
        assert_eq!(c.nonzero_count(), 1);
        assert_eq!(c.storage_idx_by_col(), &[0, 0, 1, 1]); // nothing in col 0; one in col 1; none in col 2
        assert_eq!(c.row_idx_storage(), &[0]);
        assert_eq!(c.value_storage(), &[12.0]);
    }

    #[test]
    fn handles_empty_columns() {
        // Only column 2 has entries.
        // A = [[0,0,1],
        //      [0,0,2],
        //      [0,0,0]]
        let t = mk_trips(&[(0, 2, 1.0), (1, 2, 2.0)], 3, 3);
        let c = ColumnCompressedMatrix::from_triplets(&t);

        assert_eq!(c.storage_idx_by_col(), &[0, 0, 0, 2]);
        assert_eq!(c.row_idx_storage(), &[0, 1]);
        assert_eq!(c.value_storage(), &[1.0, 2.0]);
    }

    #[test]
    fn zero_matrix() {
        // No triplets at all: all columns empty.
        let t = mk_trips(&[], 4, 5);
        let c = ColumnCompressedMatrix::from_triplets(&t);

        assert_eq!(c.row_count(), 4);
        assert_eq!(c.col_count(), 5);
        assert_eq!(c.nonzero_count(), 0);
        assert_eq!(c.storage_idx_by_col(), &[0, 0, 0, 0, 0, 0]);
        assert!(c.row_idx_storage().is_empty());
        assert!(c.value_storage().is_empty());
    }

    #[test]
    fn row_indices_are_strictly_increasing_within_each_column() {
        // Column 0 rows: 0,2; Column 1 rows: 1,3; Column 2 rows: 0
        let t = mk_trips(
            &[
                (2, 0, 3.0),
                (0, 0, 1.0),
                (3, 1, 2.0),
                (1, 1, 4.0),
                (0, 2, 5.0),
            ],
            4,
            3,
        );
        let c = ColumnCompressedMatrix::from_triplets(&t);
        let storage_offset_by_col = c.storage_idx_by_col();
        let rows = c.row_idx_storage();

        for j in 0..c.col_count() {
            let s = storage_offset_by_col[j];
            let e = storage_offset_by_col[j + 1];
            for k in s + 1..e {
                assert!(
                    rows[k - 1] < rows[k],
                    "row_idx must be strictly increasing within a column"
                );
            }
        }
    }

    #[test]
    fn arbitrary_triplet_order_yields_same_csc() {
        // Same logical A, different triplet input order → same CSC.
        let trips1 = mk_trips(&[(0, 0, 2.0), (2, 1, 3.0), (1, 0, 1.0)], 3, 2);
        let trips2 = mk_trips(&[(2, 1, 3.0), (1, 0, 1.0), (0, 0, 2.0)], 3, 2);

        let c1 = ColumnCompressedMatrix::from_triplets(&trips1);
        let c2 = ColumnCompressedMatrix::from_triplets(&trips2);

        assert_eq!(c1.row_count(), c2.row_count());
        assert_eq!(c1.col_count(), c2.col_count());
        assert_eq!(c1.storage_idx_by_col(), c2.storage_idx_by_col());
        assert_eq!(c1.row_idx_storage(), c2.row_idx_storage());
        assert_eq!(c1.value_storage(), c2.value_storage());
    }

    #[test]
    fn dimension_propagation_is_correct() {
        // This would fail if row/col counts were accidentally swapped in from_triplets.
        let t = mk_trips(&[(0, 1, 1.0)], 2, 4); // 2x4
        let c = ColumnCompressedMatrix::from_triplets(&t);
        assert_eq!(c.row_count(), 2);
        assert_eq!(c.col_count(), 4);
        assert_eq!(c.storage_idx_by_col().len(), 5);
    }

    #[test]
    fn transpose_round_trip() {
        // Randomish small case with duplicates to be coalesced.
        let tm = TripletMatrix::new(
            vec![
                (0, 0, 1.0),
                (1, 0, 2.0),
                (1, 0, 3.0), // dup at (1,0) → sums to 5.0
                (3, 2, 1.0),
                (2, 1, 4.0),
            ],
            4,
            3,
        );
        let mat_a = ColumnCompressedMatrix::from_triplets(&tm);
        let mat_a_transpose = mat_a.pattern().transpose();
        let mat_a_roundtrip = mat_a_transpose.transpose();

        assert_eq!(mat_a_roundtrip.row_count(), mat_a.pattern().row_count());
        assert_eq!(mat_a_roundtrip.col_count(), mat_a.pattern().col_count());
        assert_eq!(
            mat_a_roundtrip.storage_idx_by_col(),
            mat_a.pattern().storage_idx_by_col()
        );
        assert_eq!(
            mat_a_roundtrip.row_idx_storage(),
            mat_a.pattern().row_idx_storage()
        );
    }

    #[test]
    fn to_dense_csc_rounds_triplets() {
        // Same example as above through CSC path.
        let t = TripletMatrix::new(vec![(0, 0, 10.0), (1, 1, 20.0), (0, 2, 30.0)], 2, 3);
        let c = ColumnCompressedMatrix::from_triplets(&t);
        let d = c.to_dense();
        let expected = nalgebra::DMatrix::from_row_slice(2, 3, &[10.0, 0.0, 30.0, 0.0, 20.0, 0.0]);
        assert_eq!(d, expected);
    }

    #[test]
    fn to_symmetric_dense() {
        let lower_trips = mk_trips(&[(0, 0, 2.0), (2, 1, 3.0), (1, 0, 1.0), (2, 2, 0.5)], 3, 3);
        let lower = ColumnCompressedMatrix::from_triplets(&lower_trips);
        let upper = lower.transpose();

        let symmetric = MatF64::<3, 3>::from_array2([
            [2.0, 1.0, 0.0], //
            [1.0, 0.0, 3.0],
            [0.0, 3.0, 0.5],
        ]);

        assert_eq!(lower.triangular_to_dense_symmetric(), symmetric);
        assert_eq!(upper.triangular_to_dense_symmetric(), symmetric);
    }
}
