use nalgebra::DVector;
use snafu::ResultExt;

use crate::{
    LdltDecompositionError,
    LinearSolverError,
    SparseLdltSnafu,
    matrix::{
        ColumnCompressedMatrix,
        ColumnCompressedPattern,
        PartitionSpec,
        SparseSymmetricMatrixBuilder,
        SymmetricMatrixBuilderEnum,
        TripletMatrix,
    },
    positive_semidefinite::{
        EliminationTree,
        IsLMatBuilder,
        IsLdltTracer,
        IsLdltWorkspace,
        LdltIndices,
        NoopLdltTracer,
    },
    prelude::*,
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
        a_lower: &ColumnCompressedMatrix,
        b: &mut DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let mut tracer = NoopLdltTracer::new();
        let fact = self
            .factorize(a_lower, &mut tracer)
            .context(SparseLdltSnafu)?;

        // Solve (L D Lᵀ) x = b.
        fact.solve_in_place(b);
        Ok(())
    }

    fn matrix_builder(&self, partitions: &[PartitionSpec]) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::SparseLower(SparseSymmetricMatrixBuilder::zero(partitions))
    }
}

/// LDLᵀ workspace
#[derive(Debug)]
pub struct LdltWorkspace {
    /// Active column j.
    pub col_j: usize,
    /// Column accumulator.
    pub c: Vec<f64>,
    /// Mark per row whether it was touched.
    pub was_row_touched: Vec<bool>,
    /// List of touched rows for the current `j`.
    pub touched_rows: Vec<usize>,
}

impl IsLdltWorkspace for LdltWorkspace {
    type MatLBuilder = SparseLFactorBuilder;

    type Error = LdltDecompositionError;

    type Matrix = ColumnCompressedMatrix;
    type Diag = Vec<f64>;

    type MatrixEntry = f64;
    type DiagnalEntry = f64;

    fn calc_etree(a_lower: &Self::Matrix) -> EliminationTree {
        EliminationTree::new(a_lower.pattern().transpose())
    }

    fn activate_col(&mut self, j: usize) {
        self.col_j = j;
    }

    #[inline(always)]
    fn load_column(&mut self, a_lower: &ColumnCompressedMatrix) {
        for storage_idx in
            a_lower.storage_idx_by_col()[self.col_j]..a_lower.storage_idx_by_col()[self.col_j + 1]
        {
            let row_i = a_lower.row_idx_storage()[storage_idx];
            let v_ij = a_lower.value_storage()[storage_idx];
            if !self.was_row_touched[row_i] {
                self.c[row_i] = v_ij;
                self.was_row_touched[row_i] = true;
                self.touched_rows.push(row_i);
            }
        }
    }

    #[inline]
    fn apply_to_col_k_in_reach(
        &mut self,
        col_k: usize,
        mat_l_builder: &Self::MatLBuilder,
        diag: &Self::Diag,
        tracer: &mut impl IsLdltTracer<Self>,
    ) {
        // Does column k contribute to column j? (i.e., is L[j,k] nonzero?)
        if let Some(l_jk) = mat_l_builder.cols[col_k].find(self.col_j) {
            let d_k = diag[col_k];
            let l_jk_times_dk = l_jk * d_k;
            let col_k_data = &mat_l_builder.cols[col_k].data;
            let start = mat_l_builder.cols[col_k].lower_bound(self.col_j); // skip all rows < j

            for mat_l_ik in &col_k_data[start..] {
                // L[j,k] * d[k] * L[i,k]   (with i >= j)
                let row_i = mat_l_ik.row_idx;
                let delta = l_jk_times_dk * mat_l_ik.val;
                if !self.was_row_touched[row_i] {
                    self.c[row_i] = -delta;
                    self.was_row_touched[row_i] = true;
                    self.touched_rows.push(row_i);
                } else {
                    self.c[row_i] -= delta;
                }
                tracer.after_update(
                    LdltIndices {
                        row_i,
                        col_j: self.col_j,
                        col_k,
                    },
                    d_k,
                    mat_l_ik.val,
                    l_jk,
                    self.c[row_i],
                );
            }
        }
    }

    fn append_to_ldlt(
        &mut self,
        l_mat_builder: &mut Self::MatLBuilder,
        diag: &mut Self::Diag,
    ) -> Result<(), LdltDecompositionError> {
        let mut max_abs_pivot: f64 = 0.0;

        let d_jj = {
            let d_jj = self.djj(self.col_j);
            if !d_jj.is_finite() {
                return Err(LdltDecompositionError::NonFinitePivot {
                    j: self.col_j,
                    d_jj,
                });
            }
            max_abs_pivot = max_abs_pivot.max(d_jj.abs());
            let tau = max_abs_pivot.max(1.0) * l_mat_builder.tol_rel;
            if d_jj <= tau {
                return Err(LdltDecompositionError::NegativeFinitePivot {
                    j: self.col_j,
                    d_jj,
                });
            }
            d_jj
        };
        diag[self.col_j] = d_jj;

        let column = &mut l_mat_builder.cols[self.col_j];
        for &row_i in self.touched_rows.iter() {
            if row_i > self.col_j {
                let mat_l_ij = self.c[row_i] / d_jj;
                if mat_l_ij != 0.0 {
                    column.data.push(Entry {
                        row_idx: row_i,
                        val: mat_l_ij,
                    });
                }
            }
        }
        column.data.sort_unstable_by_key(|e| e.row_idx);

        Ok(())
    }

    #[inline(always)]
    fn clear(&mut self) {
        for row_i in self.touched_rows.drain(..) {
            self.c[row_i] = 0.0;
            self.was_row_touched[row_i] = false;
        }
    }
}

impl LdltWorkspace {
    /// Create workspace for N x N matrix.
    pub fn new(n: usize) -> Self {
        Self {
            c: vec![0.0; n],
            was_row_touched: vec![false; n],
            touched_rows: Vec::with_capacity(n),
            col_j: 0,
        }
    }

    /// Return `y[j]` if touched in this column, else 0.
    #[inline(always)]
    pub fn djj(&self, j: usize) -> f64 {
        if self.was_row_touched[j] {
            self.c[j]
        } else {
            0.0
        }
    }
}

/// L as a vector of columns.
#[derive(Debug)]
pub struct SparseLFactorBuilder {
    cols: Vec<LCol>,
    tol_rel: f64,
}

impl IsLMatBuilder for SparseLFactorBuilder {
    type Matrix = ColumnCompressedMatrix;

    fn compress(self) -> ColumnCompressedMatrix {
        let n = self.cols.len();
        let mut storage_idx_by_col = vec![0usize; n + 1];
        for col_j in 0..n {
            storage_idx_by_col[col_j + 1] = storage_idx_by_col[col_j] + self.cols[col_j].data.len();
        }
        let nonzero_count = storage_idx_by_col[n];
        let mut row_idx_storage = vec![0usize; nonzero_count];
        let mut value_storage = vec![0f64; nonzero_count];

        let mut base = 0usize;
        for col_j in 0..n {
            let col = &self.cols[col_j];
            for (k, e) in col.data.iter().enumerate() {
                row_idx_storage[base + k] = e.row_idx;
                value_storage[base + k] = e.val;
            }
            base += col.data.len();
        }
        ColumnCompressedMatrix::new(
            ColumnCompressedPattern::new(n, n, storage_idx_by_col, row_idx_storage),
            value_storage,
        )
    }
}

impl SparseLFactorBuilder {
    pub(crate) fn new(n: usize, tol_rel: f64) -> Self {
        SparseLFactorBuilder {
            cols: (0..n).map(|_| LCol::new()).collect(),
            tol_rel,
        }
    }
}

#[derive(Debug)]
struct Entry {
    row_idx: usize,
    val: f64,
}

#[derive(Debug)]
struct LCol {
    data: Vec<Entry>,
}

impl LCol {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    #[inline(always)]
    pub fn find(&self, row_i: usize) -> Option<f64> {
        match self.data.binary_search_by_key(&row_i, |e| e.row_idx) {
            Ok(pos) => Some(self.data[pos].val),
            Err(_) => None,
        }
    }

    #[inline(always)]
    pub fn lower_bound(&self, row: usize) -> usize {
        match self.data.binary_search_by_key(&row, |e| e.row_idx) {
            Ok(pos) | Err(pos) => pos,
        }
    }
}

impl SparseLdlt {
    /// Factorize `a_lower` (lower triangle of `A`) into `L` and `D`.
    pub fn factorize(
        &self,
        a_lower: &ColumnCompressedMatrix,
        tracer: &mut impl IsLdltTracer<LdltWorkspace>,
    ) -> Result<LdltFactor, LdltDecompositionError> {
        puffin::profile_scope!("ldlt fact");

        let n = a_lower.col_count();
        debug_assert_eq!(n, a_lower.row_count());

        let mut etree = LdltWorkspace::calc_etree(a_lower);

        tracer.after_etree(&etree);

        // A = L D Lᵀ
        let mut d = vec![0.0; n];

        // L as vector-of-columns.
        let mut l_storage = SparseLFactorBuilder::new(n, self.tol_rel);
        let mut ws = LdltWorkspace::new(n);

        // Main loop over columns
        for j in 0..n {
            ws.activate_col(j);
            // 1. Load A(:,j) lower into y
            ws.load_column(a_lower);
            // and calculate reach on elimination tree for column `j`.
            let reach = etree.reach(j);

            tracer.after_load_column_and_reach(j, reach, &ws);

            // 2. Apply updates from reach (root -> leaf).
            for &k in reach {
                ws.apply_to_col_k_in_reach(k, &l_storage, &d, tracer);
            }

            ws.append_to_ldlt(&mut l_storage, &mut d)?;

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

/// Factorization product `A = L D Lᵀ`.
#[derive(Debug)]
pub struct LdltFactor {
    /// unit-lower in CSC, diagonal implied `1.0`, only strict lower stored
    pub mat_l: ColumnCompressedMatrix,
    /// diagonal pivots
    pub d: Vec<f64>,
}

impl LdltFactor {
    /// Solve `(L D Lᵀ) x = b` in-place (overwrites `b` with `x`).
    #[inline]
    pub fn solve_in_place(&self, b: &mut DVector<f64>) {
        puffin::profile_scope!("ldlt solve");

        let mat_l = &self.mat_l;
        let d: &Vec<f64> = &self.d;
        let n: usize = mat_l.col_count();
        debug_assert_eq!(b.len(), n);
        debug_assert_eq!(d.len(), n);

        // Solve: L y = b.
        for col_j in 0..n {
            let b_j = b[col_j];
            for storage_idx in
                mat_l.storage_idx_by_col()[col_j]..mat_l.storage_idx_by_col()[col_j + 1]
            {
                let row_i = mat_l.row_idx_storage()[storage_idx];
                b[row_i] -= mat_l.value_storage()[storage_idx] * b_j;
            }
        }

        // Solve: z = D⁻¹ y
        for row_i in 0..n {
            b[row_i] /= d[row_i];
        }

        // Solve: Lᵀ x = z
        for col_j in (0..n).rev() {
            for storage_idx in
                mat_l.storage_idx_by_col()[col_j]..mat_l.storage_idx_by_col()[col_j + 1]
            {
                let row_i = mat_l.row_idx_storage()[storage_idx];
                b[col_j] -= mat_l.value_storage()[storage_idx] * b[row_i];
            }
        }
    }
}
