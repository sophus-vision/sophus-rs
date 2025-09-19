/// Read-only sparse `N x N` matrix in triplets form.
///
/// Invariants:
///  - for all triplet t: `t[0]` < `N`  and  `t[1]` < `N`.
///  - triplets sorted by (col, row)
pub struct TripletMatrix {
    sorted_triplets: Vec<(usize, usize, f64)>,
    scalar_dim: usize,
}

impl TripletMatrix {
    /// Create new triplet matrix given a set of triplets and dimension `N`.
    /// 
    /// The provided triplets are sorted on creation.
    pub fn new(mut triplets: Vec<(usize, usize, f64)>, scalar_dim: usize) -> Self {
        triplets.sort_unstable_by_key(|&(i, j, _)| (j, i));

        TripletMatrix {
            sorted_triplets: triplets,
            scalar_dim,
        }
    }

    /// Convert triplets to a dense matrix (coalesces duplicates by summation).
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let mut dense =
            nalgebra::DMatrix::<f64>::zeros(self.scalar_dim, self.scalar_dim);
        for &(i, j, v) in &self.sorted_triplets {
            debug_assert!(
                i < self.scalar_dim && j < self.scalar_dim,
                "triplet OOB"
            );
            dense[(i, j)] += v;
        }
        dense
    }

    /// Number of rows (or columns) `N`.
    #[inline]
    pub fn scalar_dimension(&self) -> usize {
        self.scalar_dim
    }

    /// Triplets: (row_idx, col_idx, value) - sorted by (col, row)
    #[inline]
    pub fn sorted_triplets(&self) -> &[(usize, usize, f64)] {
        &self.sorted_triplets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_dense_triplets_basic() {
        let t = TripletMatrix::new(
            vec![
                (0, 0, 1.0), //
                (1, 1, 2.0),
                (0, 2, 3.0),
            ],
            3,
        );
        let d = t.to_dense();
        let expected = nalgebra::DMatrix::from_row_slice(
            3,
            3,
            &[
                1.0, 0.0, 3.0, //
                0.0, 2.0, 0.0, //
                0.0, 0.0, 0.0,
            ],
        );
        assert_eq!(d, expected);
    }

    #[test]
    fn to_dense_triplets_coalesce() {
        let t = TripletMatrix::new(
            vec![
                (0, 1, 5.0), //
                (0, 1, 4.0),
            ],
            3,
        );
        let d = t.to_dense();
        let expected = nalgebra::DMatrix::from_row_slice(
            3,
            3,
            &[
                0.0, 9.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0,
            ],
        );
        assert_eq!(d, expected);
    }
}
