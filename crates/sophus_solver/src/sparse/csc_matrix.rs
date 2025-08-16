use crate::IsSymmetricMatrix;

/// Structural CSC (no values) for block symbolics (one node per *block* column).
#[derive(Clone, Debug)]
pub struct CscPattern {
    pub(crate) n: usize,
    pub(crate) col_ptr: Vec<usize>,
    pub(crate) row_ind: Vec<usize>,
}
impl CscPattern {
    /// Transpose a CSC.
    pub(crate) fn transpose(&self) -> CscPattern {
        let n = self.n;
        let nnz = self.row_ind.len();

        // Count by future column (== current row indices)
        let mut row_counts = vec![0usize; n];
        for &r in &self.row_ind {
            row_counts[r] += 1;
        }

        // Prefix sum -> col_ptr_t
        let mut col_ptr = vec![0usize; n + 1];
        for i in 0..n {
            col_ptr[i + 1] = col_ptr[i] + row_counts[i];
        }
        let mut next = col_ptr.clone();

        let mut row_ind = vec![0usize; nnz];

        for j in 0..n {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_ind[p];
                let dst = next[i];
                row_ind[dst] = j;
                next[i] += 1;
            }
        }
        CscPattern {
            n,
            col_ptr,
            row_ind,
        }
    }
}

/// Compressed sparse column (CSC) matrix.
#[derive(Clone, Debug)]
pub struct CscMatrix {
    pub(crate) pattern: CscPattern,
    pub(crate) values: Vec<f64>, // len = nnz
}

impl CscMatrix {
    pub(crate) fn new(
        n: usize,
        col_ptr: Vec<usize>,
        row_ind: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(col_ptr.len(), n + 1);
        debug_assert_eq!(row_ind.len(), values.len());
        Self {
            pattern: CscPattern {
                n,
                col_ptr,
                row_ind,
            },
            values,
        }
    }
}

///s
pub struct LowerCscMatrix {
    pub(crate) mat: CscMatrix,
}

/// t
pub struct LowerTripletsMatrix {
    /// t
    pub triplets: Vec<(usize, usize, f64)>,

    /// s
    pub scalar_dimension: usize,
}

impl IsSymmetricMatrix for LowerTripletsMatrix {
    type Compressed = LowerCscMatrix;

    fn compress(&self) -> Self::Compressed {
        let n = self.scalar_dimension;
        let nnz = self.triplets.len();
        let mut idx: Vec<usize> = (0..nnz).collect();

        // sort by (col, row)
        idx.sort_unstable_by(|&a, &b| {
            let ca = self.triplets[a].1.cmp(&self.triplets[b].1);
            if ca == std::cmp::Ordering::Equal {
                self.triplets[a].0.cmp(&self.triplets[b].0)
            } else {
                ca
            }
        });

        // count per column
        let mut col_counts = vec![0usize; n];
        for &k in &idx {
            col_counts[self.triplets[k].1] += 1;
        }

        // prefix sum -> col_ptr
        let mut col_ptr = vec![0usize; n + 1];
        for j in 0..n {
            col_ptr[j + 1] = col_ptr[j] + col_counts[j];
        }

        // fill Ai/Ax with coalescing
        let mut mat_a_i = vec![0usize; nnz];
        let mut mat_a_x = vec![0f64; nnz];
        let mut next = col_ptr.clone();

        for &k in &idx {
            let (i, j, x) = self.triplets[k];
            let pos = next[j];
            if pos > col_ptr[j] && mat_a_i[pos - 1] == i {
                mat_a_x[pos - 1] += x; // coalesce duplicates
            } else {
                mat_a_i[pos] = i;
                mat_a_x[pos] = x;
                next[j] += 1;
            }
        }

        // compact columns
        let mut write_ptr = 0usize;
        for j in 0..n {
            let start = col_ptr[j];
            let stop = next[j];
            if write_ptr != start {
                mat_a_i.copy_within(start..stop, write_ptr);
                mat_a_x.copy_within(start..stop, write_ptr);
            }
            col_ptr[j] = write_ptr;
            write_ptr += stop - start;
        }
        col_ptr[n] = write_ptr;
        mat_a_i.truncate(write_ptr);
        mat_a_x.truncate(write_ptr);

        LowerCscMatrix {
            mat: CscMatrix::new(n, col_ptr, mat_a_i, mat_a_x),
        }
    }
}
