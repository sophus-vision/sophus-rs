use std::fmt::Display;

use crate::sparse::TripletMatrix;

/// Read-only sparsity pattern of an MxN matrix in CSC (Compressed Sparse Column) format.
///
/// Invariants:
/// - `col_ptr.len() == col_count + 1`
/// - `col_ptr[0] == 0`
/// - `col_ptr[col_count] == row_idx.len()`
/// - `col_ptr` is non-decreasing
/// - every `row_idx[k] < row_count`
/// - within each column, `row_idx` is strictly increasing (i.e. no duplicates)
#[derive(Clone, Debug)]
pub struct CscPattern {
    row_count: usize,    // M
    col_count: usize,    // N
    col_ptr: Vec<usize>, // len = N + 1, non-decreasing, starts at 0, ends at nonzero_count()
    row_idx: Vec<usize>, // len = nnz, each < M, typically sorted (strictly) within each column
}

impl Display for CscPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "#rows: {}, #cols{}\n col_ptr: {:?}\n row_idx: {:?}",
            self.row_count(),
            self.col_count(),
            self.col_ptr(),
            self.row_idx()
        )
    }
}

impl CscPattern {
    /// Create a CSC pattern.
    ///
    /// Invariants enforced:
    /// - `col_ptr.len() == col_count + 1`
    /// - `col_ptr[0] == 0`
    /// - `col_ptr[col_count] == row_idx.len()`
    ///
    /// Invariants additionally checked in debug builds:
    /// - `col_ptr` is non-decreasing
    /// - all `row_idx[k] < row_count`
    /// - within each column, `row_idx` is strictly increasing (no duplicates)
    pub fn new(
        row_count: usize,
        col_count: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
    ) -> Self {
        // Cheap shape checks (always on)
        assert!(!col_ptr.is_empty(), "col_ptr must have length N+1 (>= 1)");
        assert_eq!(
            col_ptr.len(),
            col_count + 1,
            "col_ptr length must be N+1 (got {}, expected {})",
            col_ptr.len(),
            col_count + 1
        );
        assert_eq!(col_ptr[0], 0, "col_ptr[0] must be 0");
        assert_eq!(
            *col_ptr.last().unwrap(),
            row_idx.len(),
            "col_ptr[N] must equal row_idx.len() (got {}, expected {})",
            col_ptr[col_count],
            row_idx.len()
        );

        #[cfg(debug_assertions)]
        {
            // col_ptr non-decreasing
            for (i, w) in col_ptr.windows(2).enumerate() {
                debug_assert!(
                    w[0] <= w[1],
                    "col_ptr must be non-decreasing (col_ptr[{i}]={}, col_ptr[{i}+1]={})",
                    w[0],
                    w[1]
                );
            }

            // Row bounds
            for (k, &r) in row_idx.iter().enumerate() {
                debug_assert!(
                    r < row_count,
                    "row_idx[{k}] = {r} out of bounds (row_count = {row_count})"
                );
            }

            // Per-column slice bounds + strictly increasing rows (no dups)
            let nnz = row_idx.len();
            for j in 0..col_count {
                let start = col_ptr[j];
                let end = col_ptr[j + 1];

                debug_assert!(
                    start <= end && end <= nnz,
                    "invalid col slice in column {j}: [{start}..{end}) with nnz={nnz}"
                );

                let mut it = row_idx[start..end].iter().copied();
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

        CscPattern {
            row_count,
            col_count,
            col_ptr,
            row_idx,
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
        self.row_idx.len()
    }

    /// Column pointer slice.
    #[inline]
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    /// Row indices slice.
    #[inline]
    pub fn row_idx(&self) -> &[usize] {
        &self.row_idx
    }

    /// Transpose the CSC pattern. Result is an NxM CSC (original^T).
    pub fn transpose(&self) -> CscPattern {
        let original_row_count = self.row_count(); // original M
        let original_col_count = self.col_count(); // original N
        let nnz = self.nonzero_count();

        // 1) Count entries per transposed column (== per original row).
        let mut counts = vec![0usize; original_row_count];
        for &i in self.row_idx() {
            counts[i] += 1;
        }

        // 2) Prefix-sum to get col_ptr_t (len = m + 1).
        let mut col_ptr = vec![0usize; original_row_count + 1];
        for i in 0..original_row_count {
            col_ptr[i + 1] = col_ptr[i] + counts[i];
        }

        // 3) Fill row_idx_t by scanning original columns.
        let mut row_idx = vec![0usize; nnz];
        let mut next = col_ptr.clone();
        for j in 0..original_col_count {
            let start = self.col_ptr()[j];
            let end = self.col_ptr()[j + 1];
            for p in start..end {
                let i = self.row_idx()[p]; // original row
                let dst = next[i];
                row_idx[dst] = j; // transposed row index = original column
                next[i] += 1;
            }
        }

        // Use the constructor to assert invariants in debug builds.
        CscPattern::new(original_col_count, original_row_count, col_ptr, row_idx)
    }
}

/// Compressed sparse column (CSC) matrix.
///
/// invariant: pattern.row_idx.len() == values.len()
#[derive(Clone, Debug)]
pub struct CscMatrix {
    pattern: CscPattern,
    values: Vec<f64>,
}

impl Display for CscMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\nvalues: {:?}", self.pattern, self.values())
    }
}

impl CscMatrix {
    /// Create new CSC (Compressed Sparse Column) matrix.
    pub fn new(pattern: CscPattern, values: Vec<f64>) -> Self {
        assert_eq!(pattern.row_idx.len(), values.len(),);

        Self { pattern, values }
    }

    /// Construct CSC (Compressed Sparse Column) matrix from triplets.
    pub fn from_triplets(triplet_mat: &TripletMatrix) -> CscMatrix {
        let row_count = triplet_mat.row_count();
        let col_count = triplet_mat.col_count();
        let non_zero_count = triplet_mat.sorted_triplets().len();

        // count per column
        let mut col_counts = vec![0usize; col_count];
        for k in 0..non_zero_count {
            col_counts[triplet_mat.sorted_triplets()[k].1] += 1;
        }

        // prefix sum -> col_ptr
        let mut col_ptr = vec![0usize; col_count + 1];
        for j in 0..col_count {
            col_ptr[j + 1] = col_ptr[j] + col_counts[j];
        }

        // fill with coalescing
        let mut row_idx = vec![0usize; non_zero_count];
        let mut values = vec![0f64; non_zero_count];
        let mut next = col_ptr.clone();

        for k in 0..non_zero_count {
            let (i, j, x) = triplet_mat.sorted_triplets()[k];
            let pos = next[j];
            if pos > col_ptr[j] && row_idx[pos - 1] == i {
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
            let start = col_ptr[j];
            let stop = next[j];
            if write_ptr != start {
                row_idx.copy_within(start..stop, write_ptr);
                values.copy_within(start..stop, write_ptr);
            }
            col_ptr[j] = write_ptr;
            write_ptr += stop - start;
        }
        col_ptr[col_count] = write_ptr;
        row_idx.truncate(write_ptr);
        values.truncate(write_ptr);

        let pattern = CscPattern::new(row_count, col_count, col_ptr, row_idx);

        CscMatrix { pattern, values }
    }

    /// Get CSC pattern struct.
    #[inline]
    pub fn pattern(&self) -> &CscPattern {
        &self.pattern
    }

    /// Get values.
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
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
        self.pattern.row_idx.len()
    }

    /// Column pointer slice.
    #[inline]
    pub fn col_ptr(&self) -> &[usize] {
        &self.pattern.col_ptr
    }

    /// Row indices slice.
    #[inline]
    pub fn row_idx(&self) -> &[usize] {
        &self.pattern.row_idx
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

        let col_ptr = self.col_ptr();
        let row_idx = self.row_idx();
        let vals = self.values();

        for j in 0..n {
            let start = col_ptr[j];
            let end = col_ptr[j + 1];
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
    ///     prelude::*,
    ///     sparse::{
    ///         CscMatrix,
    ///         TripletMatrix,
    ///     },
    /// };
    ///
    /// let lower_triplets = TripletMatrix::new(
    ///     [(0, 0, 2.0), (2, 1, 3.0), (1, 0, 1.0), (2, 2, 0.5)].to_vec(),
    ///     3,
    ///     3,
    /// );
    /// let lower = CscMatrix::from_triplets(&lower_triplets);
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

        let col_ptr = self.col_ptr();
        let row_idx = self.row_idx();
        let vals = self.values();

        for j in 0..n {
            let start = col_ptr[j];
            let end = col_ptr[j + 1];
            for p in start..end {
                let i = row_idx[p];
                dense[(j, i)] = vals[p];
                dense[(i, j)] = vals[p];
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
        let c = CscMatrix::from_triplets(&t);

        // shape
        assert_eq!(c.row_count(), 2);
        assert_eq!(c.col_count(), 3);
        assert_eq!(c.nonzero_count(), 3);

        // CSC structure
        // Column 0: row 0
        // Column 1: row 1
        // Column 2: row 0
        assert_eq!(c.col_ptr(), &[0, 1, 2, 3]);
        assert_eq!(c.row_idx(), &[0, 1, 0]);
        assert_eq!(c.values(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn coalesces_duplicates_in_same_col_row() {
        // Two triplets at (0,1) should coalesce.
        // A = [[0, 5+7, 0],
        //      [0,  0 , 0]]
        let t = mk_trips(&[(0, 1, 5.0), (0, 1, 7.0)], 2, 3);
        let c = CscMatrix::from_triplets(&t);

        assert_eq!(c.row_count(), 2);
        assert_eq!(c.col_count(), 3);
        assert_eq!(c.nonzero_count(), 1);
        assert_eq!(c.col_ptr(), &[0, 0, 1, 1]); // nothing in col 0; one in col 1; none in col 2
        assert_eq!(c.row_idx(), &[0]);
        assert_eq!(c.values(), &[12.0]);
    }

    #[test]
    fn handles_empty_columns() {
        // Only column 2 has entries.
        // A = [[0,0,1],
        //      [0,0,2],
        //      [0,0,0]]
        let t = mk_trips(&[(0, 2, 1.0), (1, 2, 2.0)], 3, 3);
        let c = CscMatrix::from_triplets(&t);

        assert_eq!(c.col_ptr(), &[0, 0, 0, 2]);
        assert_eq!(c.row_idx(), &[0, 1]);
        assert_eq!(c.values(), &[1.0, 2.0]);
    }

    #[test]
    fn zero_matrix() {
        // No triplets at all: all columns empty.
        let t = mk_trips(&[], 4, 5);
        let c = CscMatrix::from_triplets(&t);

        assert_eq!(c.row_count(), 4);
        assert_eq!(c.col_count(), 5);
        assert_eq!(c.nonzero_count(), 0);
        assert_eq!(c.col_ptr(), &[0, 0, 0, 0, 0, 0]);
        assert!(c.row_idx().is_empty());
        assert!(c.values().is_empty());
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
        let c = CscMatrix::from_triplets(&t);
        let col_ptr = c.col_ptr();
        let rows = c.row_idx();

        for j in 0..c.col_count() {
            let s = col_ptr[j];
            let e = col_ptr[j + 1];
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

        let c1 = CscMatrix::from_triplets(&trips1);
        let c2 = CscMatrix::from_triplets(&trips2);

        assert_eq!(c1.row_count(), c2.row_count());
        assert_eq!(c1.col_count(), c2.col_count());
        assert_eq!(c1.col_ptr(), c2.col_ptr());
        assert_eq!(c1.row_idx(), c2.row_idx());
        assert_eq!(c1.values(), c2.values());
    }

    #[test]
    fn dimension_propagation_is_correct() {
        // This would fail if row/col counts were accidentally swapped in from_triplets.
        let t = mk_trips(&[(0, 1, 1.0)], 2, 4); // 2x4
        let c = CscMatrix::from_triplets(&t);
        assert_eq!(c.row_count(), 2);
        assert_eq!(c.col_count(), 4);
        assert_eq!(c.col_ptr().len(), 5);
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
        let mat_a = CscMatrix::from_triplets(&tm);
        let mat_a_transpose = mat_a.pattern().transpose();
        let mat_a_roundtrip = mat_a_transpose.transpose();

        assert_eq!(mat_a_roundtrip.row_count(), mat_a.pattern().row_count());
        assert_eq!(mat_a_roundtrip.col_count(), mat_a.pattern().col_count());
        assert_eq!(mat_a_roundtrip.col_ptr(), mat_a.pattern().col_ptr());
        assert_eq!(mat_a_roundtrip.row_idx(), mat_a.pattern().row_idx());
    }

    #[test]
    fn to_dense_csc_rounds_triplets() {
        // Same example as above through CSC path.
        let t = TripletMatrix::new(vec![(0, 0, 10.0), (1, 1, 20.0), (0, 2, 30.0)], 2, 3);
        let c = CscMatrix::from_triplets(&t);
        let d = c.to_dense();
        let expected = nalgebra::DMatrix::from_row_slice(2, 3, &[10.0, 0.0, 30.0, 0.0, 20.0, 0.0]);
        assert_eq!(d, expected);
    }

    #[test]
    fn to_symmetric_dense() {
        let lower_trips = mk_trips(&[(0, 0, 2.0), (2, 1, 3.0), (1, 0, 1.0), (2, 2, 0.5)], 3, 3);
        let lower = CscMatrix::from_triplets(&lower_trips);
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
