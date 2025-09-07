use faer::sparse::Triplet;

use crate::{
    matrix::TripletMatrix,
    prelude::*,
};

/// Sparse matrix in triplet form to be used with faer crate.
pub struct FaerTripletMatrix {
    /// triplets
    pub triplets: Vec<faer::sparse::Triplet<usize, usize, f64>>,
    /// scalar dimension
    pub dimension: usize,
}

impl FaerTripletMatrix {
    pub(crate) fn from_lower(t: &TripletMatrix) -> FaerTripletMatrix {
        let mut triplets = Vec::with_capacity(t.sorted_triplets().len() * 2);

        for &(row, col, val) in t.sorted_triplets() {
            // Always add the original lower entry
            triplets.push(Triplet { row, col, val });
            // Add the mirrored entry if it's off-diagonal
            if row != col {
                triplets.push(Triplet {
                    row: col,
                    col: row,
                    val,
                });
            }
        }

        FaerTripletMatrix {
            triplets,
            dimension: t.row_count(),
        }
    }
}

/// Compressed sparse matrix to be used with faer crate.
#[derive(Debug)]
pub struct FaerCompressedMatrix {
    /// column-compressed sparse matrix
    pub csc: faer::sparse::SparseColMat<usize, f64>,
}

impl IsCompressibleMatrix for FaerTripletMatrix {
    type Compressed = FaerCompressedMatrix;

    fn compress(&self) -> Self::Compressed {
        FaerCompressedMatrix {
            csc: faer::sparse::SparseColMat::try_new_from_triplets(
                self.dimension,
                self.dimension,
                &self.triplets,
            )
            .unwrap(),
        }
    }
}

/// Sparse upper triangular matrix in triplet form to be used with faer crate.
pub struct FaerUpperTripletMatrix {
    /// triplets
    pub triplets: Vec<faer::sparse::Triplet<usize, usize, f64>>,
    /// scalar dimension
    pub dimension: usize,
}

impl FaerUpperTripletMatrix {
    /// Build an upper-triangular triplets from a lower-triangular ones.
    /// Each (i, j, v) with i >= j becomes (j, i, v).
    pub fn from_lower(t: &TripletMatrix) -> FaerUpperTripletMatrix {
        let mut triplets = Vec::with_capacity(t.sorted_triplets().len());

        for &(i, j, v) in t.sorted_triplets() {
            // Lower storage usually ensures i >= j, but be robust:
            let (r, c) = if i < j { (i, j) } else { (j, i) };
            triplets.push(Triplet {
                row: r,
                col: c,
                val: v,
            });
        }

        FaerUpperTripletMatrix {
            triplets,
            dimension: t.row_count(),
        }
    }
}

/// Compressed sparse upper-triangular matrix to be used with faer crate.
#[derive(Debug)]
pub struct FaerUpperCompressedMatrix {
    /// column-compressed sparse matrix
    pub csc: faer::sparse::SparseColMat<usize, f64>,
}

impl IsCompressibleMatrix for FaerUpperTripletMatrix {
    type Compressed = FaerUpperCompressedMatrix;

    fn compress(&self) -> Self::Compressed {
        FaerUpperCompressedMatrix {
            csc: faer::sparse::SparseColMat::try_new_from_triplets(
                self.dimension,
                self.dimension,
                &self.triplets,
            )
            .unwrap(),
        }
    }
}
