use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    CachedSymbolicFactor,
    FactorEnum,
    LinearSolverEnum,
    error::LinearSolverError,
    ldlt::SchurFactor,
    matrix::{
        IsSymmetricMatrix,
        PartitionBlockIndex,
        PartitionSet,
        block_sparse::block_sparse_symmetric_matrix::BlockSparseSymmetricMatrix,
    },
};

/// Schur-complement wrapper around a block-sparse symmetric matrix.
///
/// The `Debug` impl only prints structural info (not the full inner matrix) for brevity.
///
/// Partitions the matrix into free (first `num_free_partitions` partitions) and marginalized
/// (remaining) blocks. `solve` uses the Schur complement to efficiently solve
/// the reduced system; `inverse_block` extracts blocks from the full inverse
/// via the cached `SchurFactor`.
pub struct Schur<M> {
    /// The full Hessian (free + marg + optional constraint blocks).
    pub inner: M,
    /// Number of free partitions (first `num_free_partitions` partitions are free).
    pub num_free_partitions: usize,
    /// Total variable partitions: `num_free_partitions + mpc` (free + marginalized, excluding
    /// constraint rows).
    pub total_var_partitions: usize,
    /// Solver used for the reduced S = H_ff - H_fm H_mm^-1 H_mf system.
    pub(crate) solver: LinearSolverEnum,
    /// Whether to parallelise the Schur forward pass.
    pub(crate) parallelize: bool,
    /// Cached S sparsity pattern — reused across optimizer iterations to skip AMD.
    pub(crate) cached_symbolic: Option<CachedSymbolicFactor>,
    /// Cached full factorization — kept for covariance / inverse_block queries.
    pub(crate) cached_factor: Option<FactorEnum>,
}

impl<M: Clone> Clone for Schur<M> {
    fn clone(&self) -> Self {
        Schur {
            inner: self.inner.clone(),
            num_free_partitions: self.num_free_partitions,
            total_var_partitions: self.total_var_partitions,
            solver: self.solver,
            parallelize: self.parallelize,
            cached_symbolic: self.cached_symbolic.clone(),
            // Drop cached_factor: it is lazily rebuilt on next solve(), and
            // FactorEnum::FaerSparseLblt is not Clone (faer SymbolicCholesky).
            cached_factor: None,
        }
    }
}

impl<M: std::fmt::Debug> std::fmt::Debug for Schur<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Schur")
            .field("num_free_partitions", &self.num_free_partitions)
            .field("solver", &self.solver)
            .field("parallelize", &self.parallelize)
            .finish_non_exhaustive()
    }
}

impl Schur<BlockSparseSymmetricMatrix> {
    /// Create a new Schur wrapper.
    ///
    /// `num_free_partitions` — number of free-variable partitions (first `num_free_partitions`
    /// partitions). `total_var_partitions` — `num_free_partitions + mpc`, i.e. free +
    /// marginalized; any remaining partitions are equality-constraint rows and are handled as
    /// KKT rows in the reduced system.
    pub fn new(
        inner: BlockSparseSymmetricMatrix,
        num_free_partitions: usize,
        total_var_partitions: usize,
        solver: LinearSolverEnum,
        parallelize: bool,
    ) -> Self {
        Self {
            inner,
            num_free_partitions,
            total_var_partitions,
            solver,
            parallelize,
            cached_symbolic: None,
            cached_factor: None,
        }
    }
}

impl IsSymmetricMatrix for Schur<BlockSparseSymmetricMatrix> {
    fn has_block(&self, row_idx: PartitionBlockIndex, col_idx: PartitionBlockIndex) -> bool {
        self.inner.has_block(row_idx, col_idx)
    }

    fn get_block(
        &self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> DMatrix<f64> {
        self.inner.get_block(row_idx, col_idx)
    }

    fn partitions(&self) -> &PartitionSet {
        self.inner.partitions()
    }

    fn solve(&mut self, rhs: &DVector<f64>) -> Result<DVector<f64>, LinearSolverError> {
        let s_pattern = self
            .cached_symbolic
            .take()
            .and_then(|c| c.into_schur_pattern());
        let mut sf = SchurFactor::factorize(
            &self.inner,
            self.num_free_partitions,
            self.total_var_partitions,
            self.solver,
            s_pattern,
            rhs,
            self.parallelize,
        )?;
        let dx = sf.solve()?;
        // Extract S pattern before wrapping — pattern is gone from sf after this.
        if let Some(p) = sf.take_cached_s_pattern() {
            self.cached_symbolic = Some(CachedSymbolicFactor::from_schur_pattern(p));
        }
        self.cached_factor = Some(FactorEnum::Schur(Box::new(sf)));
        Ok(dx)
    }

    fn inverse_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> Result<DMatrix<f64>, LinearSolverError> {
        if self.cached_factor.is_none() {
            let zero_rhs = DVector::zeros(self.inner.partitions().scalar_dim());
            self.solve(&zero_rhs)?;
        }
        Ok(self
            .cached_factor
            .as_mut()
            .and_then(|f| f.as_schur_mut())
            .expect("Schur factor must be present after solve")
            .inverse_block(row_idx, col_idx))
    }
}
