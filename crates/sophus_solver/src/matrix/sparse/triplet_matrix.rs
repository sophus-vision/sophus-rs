use crate::{
    matrix::ColumnCompressedMatrix,
    prelude::*,
};

/// Read-only sparse MxN matrix in triplets form.
///
/// Invariants:
///  - for all triplet t: `t[0]` < row_count  and  `t[1]` < col_count
///  - triplets sorted by (col, row)
pub struct TripletMatrix {
    sorted_triplets: Vec<(usize, usize, f64)>,
    row_count: usize,
    col_count: usize,
}

impl TripletMatrix {
    /// Create new triplet matrix.
    pub fn new(mut triplets: Vec<(usize, usize, f64)>, row_count: usize, col_count: usize) -> Self {
        triplets.sort_unstable_by_key(|&(i, j, _)| (j, i));

        TripletMatrix {
            sorted_triplets: triplets,
            row_count,
            col_count,
        }
    }

    /// Convert triplets to a dense matrix (coalesces duplicates by summation).
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let mut dense = nalgebra::DMatrix::<f64>::zeros(self.row_count, self.col_count);
        for &(i, j, v) in &self.sorted_triplets {
            debug_assert!(i < self.row_count && j < self.col_count, "triplet OOB");
            dense[(i, j)] += v;
        }
        dense
    }

    /// number of rows (M)
    #[inline]
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// number of columns (N)
    #[inline]
    pub fn col_count(&self) -> usize {
        self.col_count
    }

    /// Triplets: (row_idx, col_idx, value) - sorted by (col, row)
    #[inline]
    pub fn sorted_triplets(&self) -> &[(usize, usize, f64)] {
        &self.sorted_triplets
    }
}

impl IsCompressibleMatrix for TripletMatrix {
    type Compressed = ColumnCompressedMatrix;

    fn compress(&self) -> Self::Compressed {
        ColumnCompressedMatrix::from_triplets(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_dense_triplets_basic() {
        // A = [[10, 0, 30],
        //      [ 0,20,  0]]
        let t = TripletMatrix::new(vec![(0, 0, 10.0), (1, 1, 20.0), (0, 2, 30.0)], 2, 3);
        let d = t.to_dense();
        let expected = nalgebra::DMatrix::from_row_slice(2, 3, &[10.0, 0.0, 30.0, 0.0, 20.0, 0.0]);
        assert_eq!(d, expected);
    }

    #[test]
    fn to_dense_triplets_coalesce() {
        // duplicates at (0,1) → 5+7
        let t = TripletMatrix::new(vec![(0, 1, 5.0), (0, 1, 7.0)], 2, 3);
        let d = t.to_dense();
        let expected = nalgebra::DMatrix::from_row_slice(2, 3, &[0.0, 12.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(d, expected);
    }
}
