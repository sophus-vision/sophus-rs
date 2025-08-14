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

pub struct FaerCsCMatrix {
    /// csc
    csc_mat: faer::sparse::SparseColMat<usize, f64>,
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

impl IsSymmetricMatrix for FaerTripletsMatrix {
    type Compressed = faer::sparse::SparseColMat<usize, f64>;

    fn compress(&self) -> Self::Compressed {
        faer::sparse::SparseColMat::try_new_from_triplets(
            self.dimension,
            self.dimension,
            &self.triplets,
        )
        .unwrap()
    }
}
