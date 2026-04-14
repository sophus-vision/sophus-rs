use snafu::ResultExt;

use crate::{
    IsFactor,
    LdltDecompositionError,
    LinearSolverEnum,
    LinearSolverError,
    SparseLdltSnafu,
    ldlt::{
        EliminationTree,
        IsLMatBuilder,
        IsLdltTracer,
        IsLdltWorkspace,
        LdltIndices,
        NoopLdltTracer,
    },
    matrix::{
        PartitionSet,
        SymmetricMatrixBuilderEnum,
        sparse::{
            ColumnCompressedPattern,
            SparseMatrix,
            SparseSymmetricMatrixBuilder,
            sparse_symmetric_matrix::SparseSymmetricMatrix,
        },
    },
    prelude::*,
};

/// Build a block-level upper CSC pattern from the scalar lower CSC + partition info.
///
/// For each scalar off-diagonal lower entry `(row_i, col_j)` with `row_i > col_j`,
/// we add a block-level edge `(gb_col_j, gb_row_i)` to the upper triangle,
/// where `gb_*` is the global block index of that scalar.
fn build_block_upper_from_scalar(
    lower: &SparseMatrix,
    partitions: &PartitionSet,
) -> ColumnCompressedPattern {
    let n = lower.scalar_dim();
    let nb: usize = partitions.specs().iter().map(|s| s.block_count).sum();

    // Build scalar → global block lookup.
    let mut scalar_to_block = vec![0usize; n];
    let mut global_block = 0usize;
    for (p_idx, spec) in partitions.specs().iter().enumerate() {
        let p_start = partitions.scalar_offsets_by_partition()[p_idx];
        for b in 0..spec.block_count {
            let b_start = p_start + b * spec.block_dim;
            for s in 0..spec.block_dim {
                scalar_to_block[b_start + s] = global_block;
            }
            global_block += 1;
        }
    }

    // Collect (col_block, row_block) edges for the upper triangle (row_block < col_block).
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for col_j in 0..n {
        let gb_j = scalar_to_block[col_j];
        for idx in lower.storage_idx_by_col()[col_j]..lower.storage_idx_by_col()[col_j + 1] {
            let row_i = lower.row_idx_storage()[idx];
            if row_i > col_j {
                let gb_i = scalar_to_block[row_i];
                if gb_i > gb_j {
                    // Lower block entry (gb_i, gb_j) → upper entry at col=gb_i, row=gb_j.
                    edges.push((gb_i, gb_j));
                }
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();

    // Build CSC from sorted, deduped edges.
    let mut counts = vec![0usize; nb];
    for &(col, _) in &edges {
        counts[col] += 1;
    }
    let mut storage_offset_by_col = vec![0usize; nb + 1];
    for i in 0..nb {
        storage_offset_by_col[i + 1] = storage_offset_by_col[i] + counts[i];
    }
    let nnz = storage_offset_by_col[nb];
    let mut row_idx = vec![0usize; nnz];
    let mut next = storage_offset_by_col.clone();
    for &(col, row) in &edges {
        row_idx[next[col]] = row;
        next[col] += 1;
    }

    ColumnCompressedPattern::new(nb, storage_offset_by_col, row_idx)
}

/// Run block-level AMD on the scalar sparse matrix using the partition structure.
///
/// Returns `(block_perm, block_perm_inv)` where `block_perm[new_gb] = old_gb`.
fn block_amd_from_scalar(
    lower: &SparseMatrix,
    partitions: &PartitionSet,
) -> (Vec<usize>, Vec<usize>) {
    let block_upper = build_block_upper_from_scalar(lower, partitions);
    let nb = block_upper.scalar_dim();

    // SAFETY: `block_upper` was built with valid CSC invariants.
    let symbolic = unsafe {
        faer::sparse::SymbolicSparseColMat::<usize>::new_unchecked(
            nb,
            nb,
            block_upper.storage_idx_by_col().to_vec(),
            None,
            block_upper.row_idx_storage().to_vec(),
        )
    };

    crate::ldlt::amd_order(symbolic.as_ref()).expect("AMD ordering allocation failed")
}

/// Build scalar permutation from a block-level AMD permutation.
///
/// Returns `(scalar_perm, scalar_perm_inv)` where `scalar_perm[new_pos] = old_pos`.
fn scalar_perm_from_block_perm(
    partitions: &PartitionSet,
    block_perm: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let n = partitions.scalar_dim();
    let mut scalar_perm = vec![0usize; n];
    let mut scalar_perm_inv = vec![0usize; n];

    // Build global_block → (partition_idx, block_within_partition) lookup.
    let nb: usize = partitions.specs().iter().map(|s| s.block_count).sum();
    let mut block_to_scalar_start = vec![0usize; nb];
    let mut block_to_dim = vec![0usize; nb];
    let mut gb = 0usize;
    for (p_idx, spec) in partitions.specs().iter().enumerate() {
        let p_start = partitions.scalar_offsets_by_partition()[p_idx];
        for b in 0..spec.block_count {
            block_to_scalar_start[gb] = p_start + b * spec.block_dim;
            block_to_dim[gb] = spec.block_dim;
            gb += 1;
        }
    }

    let mut new_scalar = 0usize;
    for &old_gb in block_perm {
        let old_start = block_to_scalar_start[old_gb];
        let dim = block_to_dim[old_gb];
        for s in 0..dim {
            scalar_perm[new_scalar + s] = old_start + s;
            scalar_perm_inv[old_start + s] = new_scalar + s;
        }
        new_scalar += dim;
    }

    (scalar_perm, scalar_perm_inv)
}

/// Permute a lower-triangular scalar sparse matrix using a scalar permutation.
///
/// For entry `(row_i, col_j, val)` in original lower:
/// `new_row = perm_inv[row_i]`, `new_col = perm_inv[col_j]`.
/// Stored in lower triangle (new_row >= new_col); if new_row < new_col, the entry
/// is reflected (symmetric matrix, value is unchanged).
fn permute_sparse_lower(lower: &SparseMatrix, scalar_perm_inv: &[usize]) -> SparseMatrix {
    let n = lower.scalar_dim();

    // Collect permuted (col, row, val) triples.
    let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(lower.row_idx_storage().len());
    for col_j in 0..n {
        for idx in lower.storage_idx_by_col()[col_j]..lower.storage_idx_by_col()[col_j + 1] {
            let row_i = lower.row_idx_storage()[idx];
            let val = lower.value_storage()[idx];
            let new_col = scalar_perm_inv[col_j];
            let new_row = scalar_perm_inv[row_i];
            // Keep in lower triangle.
            let (r, c) = if new_row >= new_col {
                (new_row, new_col)
            } else {
                (new_col, new_row)
            };
            entries.push((c, r, val));
        }
    }
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Build CSC.
    let mut storage_idx_by_col = vec![0usize; n + 1];
    for &(col, _, _) in &entries {
        storage_idx_by_col[col + 1] += 1;
    }
    for i in 0..n {
        storage_idx_by_col[i + 1] += storage_idx_by_col[i];
    }
    let nnz = entries.len();
    let mut row_idx_storage = vec![0usize; nnz];
    let mut value_storage = vec![0.0f64; nnz];
    let mut next = storage_idx_by_col.clone();
    for &(col, row, val) in &entries {
        let pos = next[col];
        row_idx_storage[pos] = row;
        value_storage[pos] = val;
        next[col] += 1;
    }

    SparseMatrix::new(
        ColumnCompressedPattern::new(n, storage_idx_by_col, row_idx_storage),
        value_storage,
    )
}

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

    fn zero(&self, partitions: PartitionSet) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::SparseLower(
            SparseSymmetricMatrixBuilder::zero(partitions),
            LinearSolverEnum::SparseLdlt(*self),
        )
    }

    type SymmetricMatrixBuilder = SparseSymmetricMatrixBuilder;

    type Factor = SparseLdltFactor;

    fn factorize(&self, mat_a: &SparseSymmetricMatrix) -> Result<Self::Factor, LinearSolverError> {
        let mut tracer = NoopLdltTracer::new();
        let fact = self
            .factorize_impl(mat_a, &mut tracer, None)
            .context(SparseLdltSnafu)?;
        Ok(fact)
    }

    /// Does not support parallel execution. This function is no-op.
    fn set_parallelize(&mut self, _parallelize: bool) {}
}

impl IsFactor for SparseLdltFactor {
    type Matrix = SparseSymmetricMatrix;

    fn solve_inplace(&self, b: &mut nalgebra::DVector<f64>) -> Result<(), LinearSolverError> {
        profile_scope!("ldlt solve");

        let mat_l = &self.mat_l;
        let d: &Vec<f64> = &self.d;
        let n: usize = mat_l.scalar_dim();
        debug_assert_eq!(b.len(), n);
        debug_assert_eq!(d.len(), n);

        // Permute b → b_perm (scalar_perm[new] = old).
        let mut b_perm = if let Some(ref sp) = self.scalar_perm {
            let mut v = nalgebra::DVector::zeros(n);
            for (new_pos, &old_pos) in sp.iter().enumerate() {
                v[new_pos] = b[old_pos];
            }
            v
        } else {
            b.clone()
        };

        // Solve: L y = b_perm.
        for col_j in 0..n {
            let b_j = b_perm[col_j];
            for storage_idx in
                mat_l.storage_idx_by_col()[col_j]..mat_l.storage_idx_by_col()[col_j + 1]
            {
                let row_i = mat_l.row_idx_storage()[storage_idx];
                b_perm[row_i] -= mat_l.value_storage()[storage_idx] * b_j;
            }
        }

        // Solve: z = D⁻¹ y
        for row_i in 0..n {
            b_perm[row_i] /= d[row_i];
        }

        // Solve: Lᵀ x = z
        for col_j in (0..n).rev() {
            for storage_idx in
                mat_l.storage_idx_by_col()[col_j]..mat_l.storage_idx_by_col()[col_j + 1]
            {
                let row_i = mat_l.row_idx_storage()[storage_idx];
                b_perm[col_j] -= mat_l.value_storage()[storage_idx] * b_perm[row_i];
            }
        }

        // Unpermute x_perm → x.
        if let Some(ref sp) = self.scalar_perm {
            for (new_pos, &old_pos) in sp.iter().enumerate() {
                b[old_pos] = b_perm[new_pos];
            }
        } else {
            b.copy_from(&b_perm);
        }

        Ok(())
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

    type Matrix = SparseMatrix;
    type Diag = Vec<f64>;

    type MatrixEntry = f64;
    type DiagonalEntry = f64;

    fn calc_etree(a_lower: &Self::Matrix) -> EliminationTree {
        EliminationTree::new(a_lower.pattern().transpose())
    }

    fn activate_col(&mut self, j: usize) {
        self.col_j = j;
    }

    #[inline(always)]
    fn load_column(&mut self, a_lower: &SparseMatrix) {
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
            if d_jj.abs() <= tau {
                // PSD zero pivot — rank-deficient; store 0 to avoid division.
                0.0
            } else {
                // PD (d_jj > 0) or indefinite (d_jj < 0).
                d_jj
            }
        };
        diag[self.col_j] = d_jj;

        let column = &mut l_mat_builder.cols[self.col_j];
        if d_jj != 0.0 {
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
    type Matrix = SparseMatrix;

    fn compress(self) -> SparseMatrix {
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
        SparseMatrix::new(
            ColumnCompressedPattern::new(n, storage_idx_by_col, row_idx_storage),
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
    /// Factorize `mat_a`, reusing a previously computed symbolic factor if provided.
    ///
    /// When `cached_symb` is `Some`, block-level AMD is skipped.
    pub(crate) fn factorize_with_cached_symb(
        &self,
        mat_a: &SparseSymmetricMatrix,
        cached_symb: Option<SparseLdltSymbolic>,
    ) -> Result<SparseLdltFactor, LinearSolverError> {
        let mut tracer = NoopLdltTracer::new();
        self.factorize_impl(mat_a, &mut tracer, cached_symb)
            .context(SparseLdltSnafu)
    }

    /// Factorize `a_lower` (lower triangle of `A`) into `L` and `D`.
    ///
    /// When `cached_symb` is `Some`, the AMD ordering is skipped and the cached permutation
    /// is reused directly, saving O(nb²) AMD work when the sparsity pattern is unchanged.
    pub fn factorize_impl(
        &self,
        mat_a: &SparseSymmetricMatrix,
        tracer: &mut impl IsLdltTracer<LdltWorkspace>,
        cached_symb: Option<SparseLdltSymbolic>,
    ) -> Result<SparseLdltFactor, LdltDecompositionError> {
        profile_scope!("ldlt fact");

        // --- Block-level AMD ordering or reuse cached permutation ---------
        let (scalar_perm, scalar_perm_inv) = if let Some(c) = cached_symb {
            (c.scalar_perm, c.scalar_perm_inv)
        } else {
            let (block_perm, _block_perm_inv) = {
                profile_scope!("block_amd");
                block_amd_from_scalar(mat_a.lower(), mat_a.partitions())
            };
            scalar_perm_from_block_perm(mat_a.partitions(), &block_perm)
        };

        // --- Permuted lower matrix (always needed: values change each iter) --
        let a_lower_perm = {
            profile_scope!("permute");
            permute_sparse_lower(mat_a.lower(), &scalar_perm_inv)
        };

        // --- Numeric LDLᵀ on permuted matrix ----------------------------
        let n = a_lower_perm.scalar_dim();
        let mut etree = LdltWorkspace::calc_etree(&a_lower_perm);

        tracer.after_etree(&etree);

        let mut d = vec![0.0; n];
        let mut l_storage = SparseLFactorBuilder::new(n, self.tol_rel);
        let mut ws = LdltWorkspace::new(n);

        for j in 0..n {
            ws.activate_col(j);
            ws.load_column(&a_lower_perm);
            let reach = etree.reach(j);

            tracer.after_load_column_and_reach(j, reach, &ws);

            for &k in reach {
                ws.apply_to_col_k_in_reach(k, &l_storage, &d, tracer);
            }

            ws.append_to_ldlt(&mut l_storage, &mut d)?;

            tracer.after_append_and_sort(j, &l_storage, &d);

            ws.clear();
        }

        Ok(SparseLdltFactor {
            mat_l: l_storage.compress(),
            d,
            partitions: mat_a.partitions().clone(),
            scalar_perm: Some(scalar_perm),
            scalar_perm_inv: Some(scalar_perm_inv),
        })
    }
}

/// Symbolic factor (AMD permutation) for scalar sparse LDLᵀ.
///
/// Reusable across iterations when the sparsity pattern does not change.
#[derive(Clone, Debug)]
pub struct SparseLdltSymbolic {
    /// Scalar permutation: `scalar_perm[new_pos] = old_pos`.
    pub(crate) scalar_perm: Vec<usize>,
    /// Inverse scalar permutation: `scalar_perm_inv[old_pos] = new_pos`.
    pub(crate) scalar_perm_inv: Vec<usize>,
}

/// Factorization product `A = L D Lᵀ`.
#[derive(Clone, Debug)]
pub struct SparseLdltFactor {
    /// unit-lower in CSC, diagonal implied `1.0`, only strict lower stored (in AMD-permuted order)
    pub mat_l: SparseMatrix,
    /// diagonal pivots (in AMD-permuted order)
    pub d: Vec<f64>,
    /// Original (pre-AMD) partition set — used for external block-range queries.
    pub partitions: PartitionSet,
    /// Scalar permutation: `scalar_perm[new_pos] = old_pos`. `None` = identity.
    pub(crate) scalar_perm: Option<Vec<usize>>,
    /// Inverse scalar permutation (new_pos = perm_inv[old_pos]). Cached for symbolic reuse.
    pub(crate) scalar_perm_inv: Option<Vec<usize>>,
}

impl SparseLdltFactor {
    /// Extract the symbolic factor (AMD permutation) for reuse in the next iteration.
    pub(crate) fn into_symbolic(self) -> Option<SparseLdltSymbolic> {
        match (self.scalar_perm, self.scalar_perm_inv) {
            (Some(sp), Some(sp_inv)) => Some(SparseLdltSymbolic {
                scalar_perm: sp,
                scalar_perm_inv: sp_inv,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::{
        IsFactor,
        IsLinearSolver,
        ldlt::SparseLdlt,
        matrix::{
            PartitionBlockIndex,
            PartitionSet,
            PartitionSpec,
            block_sparse::BlockSparseSymmetricMatrix,
        },
    };

    /// Build a block-sparse KKT matrix and convert to sparse for SparseLdlt.
    fn build_ill_conditioned_kkt() -> (super::SparseSymmetricMatrix, DVector<f64>) {
        use crate::ldlt::BlockSparseLdlt;

        // [[ε, 1], [1, -ε]] with ε = 1e-12 — ill-conditioned indefinite.
        let eps = 1e-12;
        let specs = vec![PartitionSpec {
            eliminate_last: false,
            block_count: 1,
            block_dim: 2,
        }];
        let partitions = PartitionSet::new(specs);
        let bs_solver = BlockSparseLdlt::default();
        let mut builder = bs_solver.zero(partitions);

        let idx = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };
        builder.add_lower_block(
            idx,
            idx,
            &nalgebra::DMatrix::from_row_slice(2, 2, &[eps, 1.0, 1.0, -eps]).as_view(),
        );

        let (built, _) = builder.build_with_pattern();
        let bsm: BlockSparseSymmetricMatrix = built.into_block_sparse_lower().unwrap();
        let sparse = bsm.to_sparse_symmetric();
        let b = DVector::from_row_slice(&[1.0, 2.0]);
        (sparse, b)
    }

    #[test]
    fn sparse_ldlt_ill_conditioned_indefinite_poor_accuracy() {
        // SparseLdlt has no BK fallback — on ill-conditioned indefinite matrices
        // it produces results with significant error compared to LU.
        let (sparse, b) = build_ill_conditioned_kkt();

        let solver = SparseLdlt::default();
        let factor = solver.factorize(&sparse).unwrap();
        let mut x = b.clone();
        factor.solve_inplace(&mut x).unwrap();

        // Reference via dense LU.
        let dense = nalgebra::DMatrix::from_row_slice(2, 2, &[1e-12, 1.0, 1.0, -1e-12]);
        let x_ref = dense.lu().solve(&b).unwrap();

        let error = (&x - &x_ref).norm();
        // SparseLdlt should have noticeably worse accuracy than LU on this problem.
        // We don't assert failure — just document that it's less accurate.
        // The error is large because d[0] = ε = 1e-12 causes element growth.
        eprintln!(
            "SparseLdlt error on ill-conditioned indefinite: {:.2e} (x={:?}, ref={:?})",
            error,
            x.as_slice(),
            x_ref.as_slice()
        );
        // With ε = 1e-12, the pivot condition is ~1e-24 and we expect
        // significant loss of precision (error >> 1e-6).
        assert!(
            error > 1e-6,
            "expected poor accuracy without BK fallback, but error={:.2e}",
            error,
        );
    }
}
