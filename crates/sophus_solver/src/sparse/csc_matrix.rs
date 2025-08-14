use crate::IsCompressableMatrix;

/// Compressed sparse column (CSC) matrix.
#[derive(Clone, Debug)]
pub struct CscMatrix {
    pub(crate) n: usize,            // square n x n
    pub(crate) col_ptr: Vec<usize>, // len = n+1
    pub(crate) row_ind: Vec<usize>, // len = nnz
    pub(crate) values: Vec<f64>,    // len = nnz
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
            n,
            col_ptr,
            row_ind,
            values,
        }
    }
    pub(crate) fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// t
pub struct LowerTripletsMatrix {
    /// t
    pub triplets: Vec<(usize, usize, f64)>,

    /// s
    pub scalar_dimension: usize,
}

impl IsCompressableMatrix for LowerTripletsMatrix {
    type Compressed = CscMatrix;

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
        let mut Ai = vec![0usize; nnz];
        let mut Ax = vec![0f64; nnz];
        let mut next = col_ptr.clone();

        for &k in &idx {
            let (i, j, x) = self.triplets[k];
            let pos = next[j];
            if pos > col_ptr[j] && Ai[pos - 1] == i {
                Ax[pos - 1] += x; // coalesce duplicates
            } else {
                Ai[pos] = i;
                Ax[pos] = x;
                next[j] += 1;
            }
        }

        // compact columns
        let mut write_ptr = 0usize;
        for j in 0..n {
            let start = col_ptr[j];
            let stop = next[j];
            if write_ptr != start {
                Ai.copy_within(start..stop, write_ptr);
                Ax.copy_within(start..stop, write_ptr);
            }
            col_ptr[j] = write_ptr;
            write_ptr += stop - start;
        }
        col_ptr[n] = write_ptr;
        Ai.truncate(write_ptr);
        Ax.truncate(write_ptr);

        CscMatrix::new(n, col_ptr, Ai, Ax)
    }
}
