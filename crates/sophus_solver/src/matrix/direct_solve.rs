use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    CachedSymbolicFactor,
    IsFactor,
    IsInvertible,
    LinearSolverEnum,
    error::{
        LinearSolverError,
        UnsupportedForInverseBlockSnafu,
    },
    matrix::{
        IsSymmetricMatrix,
        PartitionBlockIndex,
        PartitionSet,
        block_sparse::block_sparse_symmetric_matrix::BlockSparseSymmetricMatrix,
        dense::DenseSymmetricMatrix,
        sparse::{
            FaerSparseMatrix,
            FaerSparseSymmetricMatrix,
            sparse_symmetric_matrix::SparseSymmetricMatrix,
        },
    },
};

/// Inner matrix type for the direct-solve path (all non-Schur variants).
#[derive(Debug, Clone)]
pub enum DirectSolveMatrix {
    /// Dense symmetric matrix.
    Dense(DenseSymmetricMatrix),
    /// Sparse symmetric matrix (lower-triangular storage).
    SparseLower(SparseSymmetricMatrix),
    /// Block-sparse symmetric matrix (lower block-triangular storage).
    BlockSparseLower(BlockSparseSymmetricMatrix),
    /// Sparse matrix for the faer crate.
    FaerSparse(FaerSparseMatrix),
    /// Sparse symmetric matrix (upper-triangular storage) for the faer crate.
    FaerSparseUpper(FaerSparseSymmetricMatrix),
}

impl IsSymmetricMatrix for DirectSolveMatrix {
    fn get_block(
        &self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> DMatrix<f64> {
        match self {
            DirectSolveMatrix::Dense(m) => m.get_block(row_idx, col_idx),
            DirectSolveMatrix::SparseLower(m) => m.get_block(row_idx, col_idx),
            DirectSolveMatrix::BlockSparseLower(m) => m.get_block(row_idx, col_idx),
            DirectSolveMatrix::FaerSparse(m) => m.get_block(row_idx, col_idx),
            DirectSolveMatrix::FaerSparseUpper(m) => m.get_block(row_idx, col_idx),
        }
    }

    fn partitions(&self) -> &PartitionSet {
        match self {
            DirectSolveMatrix::Dense(m) => m.partitions(),
            DirectSolveMatrix::SparseLower(m) => m.partitions(),
            DirectSolveMatrix::BlockSparseLower(m) => m.partitions(),
            DirectSolveMatrix::FaerSparse(m) => m.partitions(),
            DirectSolveMatrix::FaerSparseUpper(m) => m.partitions(),
        }
    }

    fn has_block(&self, row_idx: PartitionBlockIndex, col_idx: PartitionBlockIndex) -> bool {
        match self {
            DirectSolveMatrix::Dense(m) => m.has_block(row_idx, col_idx),
            DirectSolveMatrix::SparseLower(m) => m.has_block(row_idx, col_idx),
            DirectSolveMatrix::BlockSparseLower(m) => m.has_block(row_idx, col_idx),
            DirectSolveMatrix::FaerSparse(m) => m.has_block(row_idx, col_idx),
            DirectSolveMatrix::FaerSparseUpper(m) => m.has_block(row_idx, col_idx),
        }
    }

    fn to_dense(&self) -> DMatrix<f64> {
        match self {
            DirectSolveMatrix::Dense(m) => m.to_dense(),
            DirectSolveMatrix::SparseLower(m) => m.to_dense(),
            DirectSolveMatrix::BlockSparseLower(m) => m.to_dense(),
            DirectSolveMatrix::FaerSparse(m) => m.to_dense(),
            DirectSolveMatrix::FaerSparseUpper(m) => m.to_dense(),
        }
    }

    fn solve(&mut self, _rhs: &DVector<f64>) -> Result<DVector<f64>, LinearSolverError> {
        panic!("solve() not supported directly on DirectSolveMatrix; use DirectSolve::solve()")
    }
}

impl DirectSolveMatrix {
    /// Subtract `nu` from every scalar diagonal entry `M[i,i]` in-place.
    ///
    /// The sparsity structure is unchanged, so no caches need invalidation.
    pub fn subtract_scalar_diagonal(&mut self, nu: f64) {
        match self {
            DirectSolveMatrix::Dense(m) => m.subtract_scalar_diagonal(nu),
            DirectSolveMatrix::SparseLower(m) => m.subtract_scalar_diagonal(nu),
            DirectSolveMatrix::BlockSparseLower(m) => m.subtract_scalar_diagonal(nu),
            DirectSolveMatrix::FaerSparse(m) => m.subtract_scalar_diagonal(nu),
            DirectSolveMatrix::FaerSparseUpper(m) => m.subtract_scalar_diagonal(nu),
        }
    }

    /// Return a reference to the inner `DenseSymmetricMatrix`, if that variant.
    pub fn as_dense(&self) -> Option<&DenseSymmetricMatrix> {
        match self {
            DirectSolveMatrix::Dense(m) => Some(m),
            _ => None,
        }
    }

    /// Return a reference to the inner `SparseSymmetricMatrix`, if that variant.
    pub fn as_sparse_lower(&self) -> Option<&SparseSymmetricMatrix> {
        match self {
            DirectSolveMatrix::SparseLower(m) => Some(m),
            _ => None,
        }
    }

    /// Return a reference to the inner `BlockSparseSymmetricMatrix`, if that variant.
    pub fn as_block_sparse_lower(&self) -> Option<&BlockSparseSymmetricMatrix> {
        match self {
            DirectSolveMatrix::BlockSparseLower(m) => Some(m),
            _ => None,
        }
    }

    /// Return a reference to the inner `FaerSparseMatrix`, if that variant.
    pub fn as_faer_sparse(&self) -> Option<&FaerSparseMatrix> {
        match self {
            DirectSolveMatrix::FaerSparse(m) => Some(m),
            _ => None,
        }
    }

    /// Return a reference to the inner `FaerSparseSymmetricMatrix`, if that variant.
    pub fn as_faer_sparse_upper(&self) -> Option<&FaerSparseSymmetricMatrix> {
        match self {
            DirectSolveMatrix::FaerSparseUpper(m) => Some(m),
            _ => None,
        }
    }
}

/// Direct-solve wrapper: bundles a concrete symmetric matrix with solver state.
///
/// Analogous to `Schur<M>` but for the non-Schur (direct factorization) path.
/// `solve()` calls `LinearSolverEnum::factorize_inner`, caching the AMD
/// symbolic factorization across optimizer iterations.
#[derive(Debug, Clone)]
pub struct DirectSolve {
    /// The underlying concrete matrix.
    pub inner: DirectSolveMatrix,
    /// Solver used for factorization.
    pub(crate) solver: LinearSolverEnum,
    /// Cached AMD permutation + symbolic factorization.  Reused across calls to
    /// avoid the expensive AMD step on subsequent optimizer iterations.
    cached_h_symbolic: Option<CachedSymbolicFactor>,
}

impl DirectSolve {
    /// Create a new `DirectSolve` wrapper.
    pub fn new(inner: DirectSolveMatrix, solver: LinearSolverEnum) -> Self {
        Self {
            inner,
            solver,
            cached_h_symbolic: None,
        }
    }
}

impl IsSymmetricMatrix for DirectSolve {
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

    fn has_block(&self, row_idx: PartitionBlockIndex, col_idx: PartitionBlockIndex) -> bool {
        self.inner.has_block(row_idx, col_idx)
    }

    fn to_dense(&self) -> DMatrix<f64> {
        self.inner.to_dense()
    }

    /// Compute a block of the (pseudo) inverse of H via min-norm LDLᵀ factorization.
    fn inverse_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> Result<DMatrix<f64>, LinearSolverError> {
        let effective_solver = if self.solver.is_schur() {
            self.solver.schur_inner_solver()
        } else {
            self.solver
        };
        let factor = effective_solver
            .factorize_inner(&self.inner, None)
            .expect("factorize_inner failed in inverse_block");
        match factor.into_invertible() {
            Some(mut invertible) => Ok(invertible.pseudo_inverse_block(row_idx, col_idx)),
            None => Err(UnsupportedForInverseBlockSnafu {
                solver: format!("{:?}", effective_solver),
            }
            .build()),
        }
    }

    fn solve(&mut self, rhs: &DVector<f64>) -> Result<DVector<f64>, LinearSolverError> {
        // For Schur-variant solvers, unwrap to the inner non-Schur solver.
        let effective_solver = if self.solver.is_schur() {
            self.solver.schur_inner_solver()
        } else {
            self.solver
        };
        let cached_symb = self.cached_h_symbolic.take();
        let factor = effective_solver.factorize_inner(&self.inner, cached_symb)?;
        let dx = factor.solve(rhs)?;
        self.cached_h_symbolic = factor.into_symbolic();
        Ok(dx)
    }
}
