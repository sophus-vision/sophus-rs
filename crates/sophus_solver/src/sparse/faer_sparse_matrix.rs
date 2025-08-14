use faer::sparse::{
    SparseColMat,
    Triplet,
};

use crate::{
    IsSymmetricMatrix,
    sparse::LowerTripletsMatrix,
};

pub struct FaerTripletsMatrix {
    /// t
    pub triplets: Vec<faer::sparse::Triplet<usize, usize, f64>>,

    /// s
    pub dimension: usize,
}

impl FaerTripletsMatrix {
    pub(crate) fn from_lower(t: &LowerTripletsMatrix) -> FaerTripletsMatrix {
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
            dimension: t.scalar_dimension,
        }
    }
}

pub struct FaerCompressedMatrix {
    /// t
    pub csc: faer::sparse::SparseColMat<usize, f64>,
}

impl IsSymmetricMatrix for FaerTripletsMatrix {
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

pub struct FaerUpperTripletsMatrix {
    /// t
    pub triplets: Vec<faer::sparse::Triplet<usize, usize, f64>>,

    /// s
    pub dimension: usize,
}

impl FaerUpperTripletsMatrix {
    /// Build an upper-triangular triplet set from a lower-triangular one.
    /// Each (i, j, v) with i >= j becomes (j, i, v). Diagonal stays (i, i, v).
    pub fn from_lower(t: &LowerTripletsMatrix) -> FaerUpperTripletsMatrix {
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

        // Debug sanity: ensure upper-view
        #[cfg(debug_assertions)]
        {
            if let Some((k, bad)) = triplets.iter().enumerate().find(|(_, e)| e.row > e.col) {
                eprintln!(
                    "[FaerUpperTripletsMatrix::from_lower] non-upper at idx {k}: ({}, {})",
                    bad.row, bad.col
                );
            } else {
                eprintln!(
                    "[FaerUpperTripletsMatrix::from_lower] dim={}, in_lower_nnz={}, out_upper_nnz={}",
                    t.scalar_dimension,
                    t.triplets.len(),
                    triplets.len()
                );
            }
        }

        FaerUpperTripletsMatrix {
            triplets,
            dimension: t.scalar_dimension,
        }
    }
}

pub struct FaerUpperCompressedMatrix {
    /// t
    pub csc: faer::sparse::SparseColMat<usize, f64>,
}

impl IsSymmetricMatrix for FaerUpperTripletsMatrix {
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
