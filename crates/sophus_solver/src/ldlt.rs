pub(crate) mod block_diag_ldlt;
pub(crate) mod block_sparse_ldlt;
pub(crate) mod dense_ldlt;
pub(crate) mod elimination_tree;
pub(crate) mod faer_sparse_ldlt;
/// Min-norm LDLt pseudo-inverse solver.
pub mod min_norm_ldlt;
/// Schur-complement factorization.
pub mod schur_ldlt;
pub(crate) mod sparse_ldlt;

use std::marker::PhantomData;

pub use block_diag_ldlt::*;
pub use block_sparse_ldlt::*;
pub use dense_ldlt::*;
pub use elimination_tree::*;
use faer::{
    dyn_stack::{
        MemBuffer,
        MemStack,
    },
    sparse::{
        FaerError,
        linalg::amd,
    },
};
pub use faer_sparse_ldlt::*;
pub use schur_ldlt::SchurFactor;
pub use sparse_ldlt::*;

use crate::{
    FaerSparseSolverError,
    IsFactor,
    LinearSolverError,
};

/// Run AMD fill-reducing ordering on a symbolic upper-triangular sparse matrix.
///
/// Returns `(perm, perm_inv)` where `perm[new_pos] = old_pos`.
pub(crate) fn amd_order(
    symbolic: faer::sparse::SymbolicSparseColMatRef<'_, usize>,
) -> Result<(Vec<usize>, Vec<usize>), LinearSolverError> {
    let nb = symbolic.nrows();
    let nnz = symbolic.col_ptr()[nb];
    let mut perm = vec![0usize; nb];
    let mut perm_inv = vec![0usize; nb];
    let mut mem = MemBuffer::try_new(amd::order_scratch::<usize>(nb, nnz))
        .map_err(|e| faer_amd_err(e.into()))?;
    amd::order(
        &mut perm,
        &mut perm_inv,
        symbolic,
        amd::Control::default(),
        MemStack::new(&mut mem),
    )
    .map_err(faer_amd_err)?;
    Ok((perm, perm_inv))
}

fn faer_amd_err(e: FaerError) -> LinearSolverError {
    LinearSolverError::FaerSparseLdltError {
        faer_error: match e {
            FaerError::OutOfMemory => FaerSparseSolverError::OutOfMemory,
            FaerError::IndexOverflow => FaerSparseSolverError::IndexOverflow,
            _ => FaerSparseSolverError::Unspecific,
        },
    }
}

/// Workspace for sparse LDLᵀ such as [SparseLdlt] or [BlockSparseLdlt].
pub trait IsLdltWorkspace: Sized {
    /// Decomposition error;
    type Error;

    /// Type of matrix A, we want to decompose: `A = L D Lᵀ`.
    type Matrix;

    /// Type of lower-triangular matrix `L`.
    type MatLBuilder;
    /// Type of matrix diagonal or block-diagonal `D`.
    type Diag;

    /// Type of a matrix `L[i,k]`.
    type MatrixEntry;
    /// Type of diagonal entry `d[k]`.
    type DiagnalEntry;

    /// Calculate symbolic elimination tree from matrix A.
    fn calc_etree(a_lower: &Self::Matrix) -> EliminationTree;

    /// Activate column j.
    fn activate_col(&mut self, col_j: usize);

    /// Load A(i,j) into accumulator `C(i,j)`:  `C(i,j) := A[i,j]`.
    fn load_column(&mut self, a_lower: &Self::Matrix);

    /// Apply columns in reach of column j.
    fn apply_to_col_k_in_reach(
        &mut self,
        col_k: usize,
        mat_l_builder: &Self::MatLBuilder,
        diag: &Self::Diag,
        tracer: &mut impl IsLdltTracer<Self>,
    );

    /// Append data from accumulator C to L and D.
    fn append_to_ldlt(
        &mut self,
        mat_l_builder: &mut Self::MatLBuilder,
        diag: &mut Self::Diag,
    ) -> Result<(), Self::Error>;

    /// Clear accumulator entries `C[:,j]` that were touched during column j.
    fn clear(&mut self);
}

/// Builder for L factor.
pub trait IsLMatBuilder {
    /// Matrix type to represent L.
    type Matrix;

    /// Return compressed matrix form.
    fn compress(self) -> Self::Matrix;
}

/// Indices used by LdltTracer
pub struct LdltIndices {
    /// the column of interest `j``
    pub col_j: usize,
    /// column connect to `j` through elimination tree reach.
    pub col_k: usize,
    /// row `i`
    pub row_i: usize,
}

/// Tracer - for optional debug insights.
pub trait IsLdltTracer<Workspace: IsLdltWorkspace> {
    /// Trace to show the elimination tree.
    #[inline]
    fn after_etree(&mut self, _etree: &EliminationTree) {}

    /// Trace to show the loaded column and etree reach for column `j`.
    #[inline]
    fn after_load_column_and_reach(&mut self, _j: usize, _reach: &[usize], _ws: &Workspace) {}

    /// Update on reach for column `j`.
    #[inline]
    fn after_update(
        &mut self,
        _indices: LdltIndices,
        _d: Workspace::DiagnalEntry,
        _l_ik: Workspace::MatrixEntry,
        _l_jk: Workspace::MatrixEntry,
        _c: Workspace::MatrixEntry,
    ) {
    }

    /// Show final L for column `j`.
    #[inline]
    fn after_append_and_sort(&mut self, _j: usize, _l_storage: &SparseLFactorBuilder, _d: &[f64]) {}
}

/// No-op tracer
#[derive(Debug)]
pub struct NoopLdltTracer<Workspace: IsLdltWorkspace> {
    phantom: PhantomData<Workspace>,
}
impl<Workspace: IsLdltWorkspace> Default for NoopLdltTracer<Workspace> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Workspace: IsLdltWorkspace> NoopLdltTracer<Workspace> {
    /// New no-op tracer.
    pub fn new() -> Self {
        NoopLdltTracer {
            phantom: Default::default(),
        }
    }
}
impl<Workspace: IsLdltWorkspace> IsLdltTracer<Workspace> for NoopLdltTracer<Workspace> {}
