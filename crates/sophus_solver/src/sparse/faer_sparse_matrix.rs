use faer::sparse::Triplet;

use crate::{
    prelude::*,
    sparse::TripletMatrix,
};

/// Sparse matrix in triplet form to be used with faer crate.
pub struct FaerTripletsMatrix {
    /// triplets
    pub triplets: Vec<faer::sparse::Triplet<usize, usize, f64>>,
    /// scalar dimension
    pub dimension: usize,
}

impl FaerTripletsMatrix {
    pub(crate) fn from_lower(t: &TripletMatrix) -> FaerTripletsMatrix {
        let mut triplets = Vec::with_capacity(t.triplets.len() * 2);

        for &(row, col, val) in &t.triplets {
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

        FaerTripletsMatrix {
            triplets,
            dimension: t.row_count,
        }
    }
}

/// Compressed sparse matrix to be used with faer crate.
#[derive(Debug)]
pub struct FaerCompressedMatrix {
    /// column-compressed sparse matrix
    pub csc: faer::sparse::SparseColMat<usize, f64>,
}

impl IsCompressibleMatrix for FaerTripletsMatrix {
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
pub struct FaerUpperTripletsMatrix {
    /// triplets
    pub triplets: Vec<faer::sparse::Triplet<usize, usize, f64>>,
    /// scalar dimension
    pub dimension: usize,
}

impl FaerUpperTripletsMatrix {
    /// Build an upper-triangular triplets from a lower-triangular ones.
    /// Each (i, j, v) with i >= j becomes (j, i, v).
    pub fn from_lower(t: &TripletMatrix) -> FaerUpperTripletsMatrix {
        let mut triplets = Vec::with_capacity(t.triplets.len());

        for &(i, j, v) in &t.triplets {
            // Lower storage usually ensures i >= j, but be robust:
            let (r, c) = if i < j { (i, j) } else { (j, i) };
            triplets.push(Triplet {
                row: r,
                col: c,
                val: v,
            });
        }

        FaerUpperTripletsMatrix {
            triplets,
            dimension: t.row_count,
        }
    }
}

/// Compressed sparse upper-triangular matrix to be used with faer crate.
#[derive(Debug)]
pub struct FaerUpperCompressedMatrix {
    /// column-compressed sparse matrix
    pub csc: faer::sparse::SparseColMat<usize, f64>,
}

impl IsCompressibleMatrix for FaerUpperTripletsMatrix {
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
