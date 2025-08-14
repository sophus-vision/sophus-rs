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
