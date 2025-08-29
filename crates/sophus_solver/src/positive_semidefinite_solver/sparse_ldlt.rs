use nalgebra::DVector;

use crate::{
    EliminationTree,
    LinearSolverError,
    SymmetricMatrixBuilderEnum,
    prelude::*,
    sparse::{
        CscMatrix,
        CscPattern,
        SparseSymmetricMatrixBuilder,
        TripletMatrix,
    },
};

/// Sparse solver using LDLᵀ decomposition.
#[derive(Clone, Copy, Debug)]
pub struct SparseLdlt {
    /// Relative tolerance for pivot check.
    pub tol_rel: f64,
}

impl Default for SparseLdlt {
    fn default() -> Self {
        SparseLdlt { tol_rel: 1e-12_f64 }
    }
}

impl IsLinearSolver for SparseLdlt {
    const NAME: &'static str = "sparse LDLt";
    type Matrix = TripletMatrix;

    fn solve_in_place(
        &self,
        _parallelize: bool,
        a_lower: &CscMatrix,
        b: &mut DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let mut tracer = NoopTracer {};
        let fact = self
            .factorize(a_lower, &mut tracer)
            .map_err(|_| LinearSolverError::FactorizationFailed)?;

        // Solve (L D Lᵀ) x = b.
        fact.solve_in_place(b);
        Ok(())
    }

    fn matrix_builder(&self, partitions: &[crate::PartitionSpec]) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::SparseLower(SparseSymmetricMatrixBuilder::zero(partitions))
    }
}

impl SparseLdlt {
    /// Factorize `a_lower` (lower triangle of `A`) into `L` and `D`.
    pub fn factorize(
        &self,
        a_lower: &CscMatrix,
        tracer: &mut impl LdltTracer,
    ) -> Result<LdltFactor, &'static str> {
        let n = a_lower.row_count();
        debug_assert_eq!(n, a_lower.col_count());

        let mut etree = EliminationTree::new(a_lower.pattern().transpose());

        tracer.after_etree(&etree);

        // A = L D Lᵀ
        let mut d = vec![0.0; n];
        let mut max_abs_pivot: f64 = 0.0;

        // L as vector-of-columns.
        let mut l_storage = LStorage::new(n);
        let mut ws = LdltWorkspace::new(n);

        // Main loop over columns
        for j in 0..n {
            // 1. Load A(:,j) lower into y
            ws.load_column(j, a_lower);
            // and calculate reach on elimination tree for column `j`.
            let reach = etree.reach(j);

            tracer.after_load_column_and_reach(j, reach, &ws);

            // 2. Apply updates from reach (root -> leaf).
            for &k in reach {
                // Does column k contribute to column j? (i.e., is L[j,k] nonzero?)
                if let Some(l_jk) = l_storage.cols[k].find(j) {
                    let l_jk_times_dk = l_jk * d[k];
                    let col_k = &l_storage.cols[k].data;
                    let start = l_storage.cols[k].lower_bound(j); // skip all rows < j

                    for &(i, l_ik) in &col_k[start..] {
                        // L[j,k] * d[k] * L[i,k]   (with i >= j)
                        let delta = l_jk_times_dk * l_ik;
                        if !ws.was_row_touched[i] {
                            ws.y[i] = -delta;
                            ws.was_row_touched[i] = true;
                            ws.touched_rows.push(i);
                        } else {
                            ws.y[i] -= delta;
                        }
                        tracer.after_update(LdltIndices { i, j, k }, &d, l_ik, l_jk, &ws);
                    }
                }
            }

            // 3) Calculate pivot and perform semi positive definiteness check.
            let djj = {
                let djj = ws.djj(j);
                if !djj.is_finite() {
                    return Err("LDLt failed: non-finite pivot");
                }
                max_abs_pivot = max_abs_pivot.max(djj.abs());
                let tau = max_abs_pivot.max(1.0) * self.tol_rel;
                if djj <= tau {
                    return Err("LDLt failed: non-positive/too-small pivot");
                }
                djj
            };
            d[j] = djj;

            // 4) Append L(:,j) = y[i]/djj for i>j and sort rows for canonical CSC.
            l_storage.cols[j].append_and_sort(j, djj, &ws);

            tracer.after_append_and_sort(j, &l_storage, &d);

            // 5) Clear workspace after column `j`.
            ws.clear();
        }

        Ok(LdltFactor {
            mat_l: l_storage.compress(),
            d,
        })
    }
}

/// LDLᵀ workspace
#[derive(Debug)]
pub struct LdltWorkspace {
    /// Column accumulator
    pub y: Vec<f64>,
    /// Mark per row for the current column `j`
    pub was_row_touched: Vec<bool>,
    /// List of touched rows for the current `j`
    pub touched_rows: Vec<usize>,
}

impl LdltWorkspace {
    /// Create workspace for N x N matrix.
    pub fn new(n: usize) -> Self {
        Self {
            y: vec![0.0; n],
            was_row_touched: vec![false; n],
            touched_rows: Vec::with_capacity(n),
        }
    }

    /// Load lower(A(:,j)) into accumulator `y`.
    #[inline(always)]
    fn load_column(&mut self, j: usize, a_lower: &CscMatrix) {
        for p in a_lower.col_ptr()[j]..a_lower.col_ptr()[j + 1] {
            let i = a_lower.row_idx()[p];
            let v = a_lower.values()[p];
            if !self.was_row_touched[i] {
                self.y[i] = v;
                self.was_row_touched[i] = true;
                self.touched_rows.push(i);
            } else {
                debug_assert!(false, "duplicate ({i},{j}) in input lower(A)");
            }
        }
    }

    /// Return y[j] if touched in this column, else 0.
    #[inline(always)]
    fn djj(&self, j: usize) -> f64 {
        if self.was_row_touched[j] {
            self.y[j]
        } else {
            0.0
        }
    }

    /// Clear accumulator entries that were touched during column j.
    #[inline(always)]
    fn clear(&mut self) {
        for i in self.touched_rows.drain(..) {
            // accumulator back to zero for next column
            self.y[i] = 0.0;
            // unmark this row
            self.was_row_touched[i] = false;
        }
    }
}

/// L as a vector of columns.
#[derive(Debug)]
pub struct LStorage {
    pub(crate) cols: Vec<LCol>,
}

impl LStorage {
    pub(crate) fn new(n: usize) -> Self {
        LStorage {
            cols: (0..n).map(|_| LCol::new()).collect(),
        }
    }

    /// Compress to CSC matrix.
    pub fn compress(&self) -> CscMatrix {
        let n = self.cols.len();
        let mut col_ptr = vec![0usize; n + 1];
        for j in 0..n {
            col_ptr[j + 1] = col_ptr[j] + self.cols[j].data.len();
        }
        let nnz = col_ptr[n];
        let mut row_ind = vec![0usize; nnz];
        let mut values = vec![0f64; nnz];

        let mut base = 0usize;
        for j in 0..n {
            let col = &self.cols[j];
            for (k, &(i, v)) in col.data.iter().enumerate() {
                row_ind[base + k] = i;
                values[base + k] = v;
            }
            base += col.data.len();
        }
        CscMatrix::new(CscPattern::new(n, n, col_ptr, row_ind), values)
    }
}

#[derive(Debug)]
pub(crate) struct LCol {
    pub(crate) data: Vec<(usize, f64)>,
}

impl LCol {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    #[inline(always)]
    fn find(&self, row: usize) -> Option<f64> {
        match self.data.binary_search_by_key(&row, |&(i, _)| i) {
            Ok(pos) => Some(self.data[pos].1),
            Err(_) => None,
        }
    }

    #[inline(always)]
    fn lower_bound(&self, row: usize) -> usize {
        match self.data.binary_search_by_key(&row, |&(i, _)| i) {
            Ok(pos) | Err(pos) => pos,
        }
    }

    #[inline(always)]
    fn append_and_sort(&mut self, j: usize, djj: f64, ws: &LdltWorkspace) {
        for &i in ws.touched_rows.iter() {
            if i > j {
                let lij = ws.y[i] / djj;
                if lij != 0.0 {
                    self.data.push((i, lij));
                }
            }
        }

        self.data.sort_unstable_by_key(|&(i, _)| i);
    }
}

/// Factorization product `A = L D Lᵀ`.
#[derive(Debug)]
pub struct LdltFactor {
    /// unit-lower in CSC, diagonal implied `1.0`, only strict lower stored
    pub mat_l: CscMatrix,
    /// diagonal pivots
    pub d: Vec<f64>,
}

impl LdltFactor {
    /// Solve `(L D Lᵀ) x = b` in-place (overwrites `b` with `x`).
    #[inline]
    pub fn solve_in_place(&self, b: &mut DVector<f64>) {
        let l = &self.mat_l;
        let d = &self.d;
        let n = l.row_count();
        debug_assert_eq!(b.len(), n);
        debug_assert_eq!(d.len(), n);

        // Forward: L y = b (unit-lower)
        for j in 0..n {
            let t = b[j];
            for p in l.col_ptr()[j]..l.col_ptr()[j + 1] {
                let i = l.row_idx()[p]; // i > j
                b[i] -= l.values()[p] * t;
            }
        }

        // Diagonal: z = D^{-1} y
        for i in 0..n {
            b[i] /= d[i];
        }

        // Backward: Lᵀ x = z
        for j in (0..n).rev() {
            for p in l.col_ptr()[j]..l.col_ptr()[j + 1] {
                let i = l.row_idx()[p]; // i > j
                b[j] -= l.values()[p] * b[i];
            }
        }
    }
}

/// Indices used by LdltTracer
pub struct LdltIndices {
    /// the column of interest `j``
    pub j: usize,
    /// column connect to `j` through elimination tree reach.
    pub k: usize,
    /// row `i`
    pub i: usize,
}

/// Tracer - for optional debug insights.
pub trait LdltTracer {
    /// Trace to show the elimination tree.
    #[inline]
    fn after_etree(&mut self, _etree: &EliminationTree) {}

    /// Trace to show the loaded column and etree reach for column `j`.
    #[inline]
    fn after_load_column_and_reach(&mut self, _j: usize, _reach: &[usize], _ws: &LdltWorkspace) {}

    /// Update on reach for column `j`.
    #[inline]
    fn after_update(
        &mut self,
        _indices: LdltIndices,
        _d: &[f64],
        _l_ik: f64,
        _l_jk: f64,
        _ws: &LdltWorkspace,
    ) {
    }

    /// Show final L for column `j`.
    #[inline]
    fn after_append_and_sort(&mut self, _j: usize, _l_storage: &LStorage, _d: &[f64]) {}
}

/// No-op tracer
#[derive(Debug, Clone, Copy)]
pub struct NoopTracer;
impl LdltTracer for NoopTracer {}
