use crate::prelude::*;

/// Read-only sparse MxN matrix in triplets form.
///
/// Invariants:
///  - for all triplet t: t[0] < row_count  and  t[1] < col_count
pub struct TripletMatrix {
    /// Triplets: (row_idx, col_idx, value)
    pub triplets: Vec<(usize, usize, f64)>,
    /// number of rows (M)
    pub row_count: usize,
    /// number of columns (N)
    pub col_count: usize,
}

impl TripletMatrix {
    /// Create new triplet matrix.
    pub fn new(triplets: Vec<(usize, usize, f64)>, row_count: usize, col_count: usize) -> Self {
        TripletMatrix {
            triplets,
            row_count,
            col_count,
        }
    }
}

impl IsCompressibleMatrix for TripletMatrix {
    type Compressed = CscMatrix;

    fn compress(&self) -> Self::Compressed {
        CscMatrix::from_triplets(self)
    }
}

/// Read-only sparsity pattern of an MxN matrix in CSC (Compressed Sparse Column) format.
///
/// Invariants:
/// - col_ptr.len() == col_count + 1
/// - col_ptr[0] == 0, col_ptr[col_count] == row_idx.len()
/// - col_ptr is non-decreasing and
/// -
/// - all row_idx[k] < row_count
/// - shall not have duplicate entries
#[derive(Clone, Debug)]
pub struct CscPattern {
    row_count: usize,    // M
    col_count: usize,    // N
    col_ptr: Vec<usize>, // len = N + 1, non-decreasing, starts at 0, ends at nonzero_count()
    row_idx: Vec<usize>, // len = nnz, each < M, typically sorted (strictly) within each column
}

impl CscPattern {
    /// Create a CSC pattern.
    ///
    /// Invariants checked:
    /// - col_ptr.len() == col_count + 1
    /// - col_ptr is non-decreasing and col_ptr[0] == 0
    /// - col_ptr[col_count] == row_idx.len()
    /// - all k: row_idx[k] < row_count
    pub fn new(
        row_count: usize,
        col_count: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
    ) -> Self {
        assert!(
            !col_ptr.is_empty(),
            "col_ptr must have length N+1 (at least 1)"
        );
        assert_eq!(col_ptr.len(), col_count + 1, "col_ptr length must be N+1");
        assert_eq!(col_ptr[0], 0, "col_ptr[0] must be 0");
        assert_eq!(
            col_ptr[col_count],
            row_idx.len(),
            "col_ptr[N] must equal row_idx.len()"
        );

        #[cfg(debug_assertions)]
        {
            for w in col_ptr.windows(2) {
                use crate::assert_le;

                assert_le!(w[0], w[1], "col_ptr must be non-decreasing");
            }
            for &r in &row_idx {
                use crate::assert_lt;

                assert_lt!(r, row_count, "row index out of bounds");
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
}

/// Compressed sparse column (CSC) matrix.
///
/// invariant: pattern.row_idx.len() == values.len()
#[derive(Clone, Debug)]
pub struct CscMatrix {
    pattern: CscPattern,
    values: Vec<f64>,
}

impl CscMatrix {
    /// Create new CSC (Compressed Sparse Column) matrix.
    pub fn new(pattern: CscPattern, values: Vec<f64>) -> Self {
        assert_eq!(pattern.row_idx.len(), values.len(),);

        Self { pattern, values }
    }

    /// Construct CSC (Compressed Sparse Column) matrix from triplets.
    pub fn from_triplets(triplets: &TripletMatrix) -> CscMatrix {
        let row_count = triplets.row_count;
        let col_count = triplets.col_count;
        let nnz = triplets.triplets.len();
        let mut idx: Vec<usize> = (0..nnz).collect();

        // sort by (col, row)
        idx.sort_unstable_by(|&a, &b| {
            let ca = triplets.triplets[a].1.cmp(&triplets.triplets[b].1);
            if ca == std::cmp::Ordering::Equal {
                triplets.triplets[a].0.cmp(&triplets.triplets[b].0)
            } else {
                ca
            }
        });

        // count per column
        let mut col_counts = vec![0usize; col_count];
        for &k in &idx {
            col_counts[triplets.triplets[k].1] += 1;
        }

        // prefix sum -> col_ptr
        let mut col_ptr = vec![0usize; col_count + 1];
        for j in 0..col_count {
            col_ptr[j + 1] = col_ptr[j] + col_counts[j];
        }

        // fill with coalescing
        let mut row_idx = vec![0usize; nnz];
        let mut values = vec![0f64; nnz];
        let mut next = col_ptr.clone();

        for &k in &idx {
            let (i, j, x) = triplets.triplets[k];
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
}

#[cfg(test)]
mod tests {
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
}
