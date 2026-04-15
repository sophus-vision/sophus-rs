//! Faer sparse LBLᵀ (Bunch-Kaufman) solver for symmetric indefinite systems.
//!
//! Wraps faer's `factorize_numeric_intranode_lblt` which uses supernodal
//! Bunch-Kaufman pivoting. Handles PD, PSD, and indefinite matrices.

#[cfg(not(target_arch = "wasm32"))]
use std::num::NonZeroUsize;

use faer::{
    Conj,
    Par,
    Side,
    Spec,
    dyn_stack::{
        MemBuffer,
        MemStack,
        StackReq,
    },
    linalg::cholesky::lblt::factor::LbltParams,
    mat::MatMut,
    reborrow::ReborrowMut,
    sparse::{
        FaerError,
        SparseColMat,
        linalg::cholesky::{
            CholeskySymbolicParams,
            SymbolicCholesky,
            SymmetricOrdering,
            factorize_symbolic_cholesky,
        },
    },
};

use crate::{
    FaerSparseSolverError,
    IsFactor,
    LinearSolverEnum,
    LinearSolverError,
    matrix::{
        PartitionSet,
        SymmetricMatrixBuilderEnum,
        sparse::{
            FaerSparseSymmetricMatrix,
            FaerSparseSymmetricMatrixBuilder,
        },
    },
    prelude::*,
};

fn faer_err(e: FaerError) -> LinearSolverError {
    LinearSolverError::FaerSparseLdltError {
        faer_error: match e {
            FaerError::OutOfMemory => FaerSparseSolverError::OutOfMemory,
            FaerError::IndexOverflow => FaerSparseSolverError::IndexOverflow,
            _ => FaerSparseSolverError::Unspecific,
        },
    }
}

/// Parameters for faer's sparse LBLᵀ (Bunch-Kaufman) solver.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FaerSparseLblt {
    /// Perform parallel execution?
    pub parallelize: bool,
}

impl Default for FaerSparseLblt {
    fn default() -> Self {
        Self { parallelize: false }
    }
}

/// Cached symbolic factor for reuse across iterations.
#[derive(Debug)]
pub(crate) struct FaerSparseLbltSymbolic {
    pub(crate) symbolic: SymbolicCholesky<usize>,
}

/// Numeric factorization result for the faer sparse LBLᵀ solver.
#[derive(Debug)]
pub struct FaerSparseLbltSystem {
    mat_a: FaerSparseSymmetricMatrix,
    symbolic: SymbolicCholesky<usize>,
    parallelize: bool,
}

impl FaerSparseLbltSystem {
    /// Extract the symbolic factor for reuse in the next iteration.
    pub(crate) fn into_symbolic(self) -> FaerSparseLbltSymbolic {
        FaerSparseLbltSymbolic {
            symbolic: self.symbolic,
        }
    }
}

impl IsLinearSolver for FaerSparseLblt {
    type SymmetricMatrixBuilder = FaerSparseSymmetricMatrixBuilder;

    const NAME: &'static str = "faer sparse LBLᵀ";

    fn zero(&self, partitions: PartitionSet) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::FaerSparseUpper(
            FaerSparseSymmetricMatrixBuilder::zero(partitions),
            LinearSolverEnum::FaerSparseLblt(*self),
        )
    }

    fn name(&self) -> String {
        Self::NAME.into()
    }

    type Factor = FaerSparseLbltSystem;

    fn factorize(
        &self,
        mat_a: &FaerSparseSymmetricMatrix,
    ) -> Result<Self::Factor, LinearSolverError> {
        self.factorize_with_cached_symb(mat_a, None)
    }

    fn set_parallelize(&mut self, parallelize: bool) {
        self.parallelize = parallelize;
    }
}

impl FaerSparseLblt {
    /// Factorize `mat_a`, reusing a previously computed symbolic factor if provided.
    pub(crate) fn factorize_with_cached_symb(
        &self,
        mat_a: &FaerSparseSymmetricMatrix,
        cached_symb: Option<FaerSparseLbltSymbolic>,
    ) -> Result<FaerSparseLbltSystem, LinearSolverError> {
        let symbolic = match cached_symb {
            Some(c) => c.symbolic,
            None => compute_symbolic(&mat_a.upper)?,
        };

        Ok(FaerSparseLbltSystem {
            parallelize: self.parallelize,
            symbolic,
            mat_a: mat_a.clone(),
        })
    }
}

/// Compute symbolic Cholesky factorization (AMD ordering + structure analysis).
fn compute_symbolic(
    upper: &SparseColMat<usize, f64>,
) -> Result<SymbolicCholesky<usize>, LinearSolverError> {
    factorize_symbolic_cholesky(
        upper.symbolic(),
        Side::Upper,
        SymmetricOrdering::Amd,
        CholeskySymbolicParams::default(),
    )
    .map_err(faer_err)
}

enum SparseLbltError {
    FaerError(FaerError),
}

impl FaerSparseLbltSystem {
    fn solve(&self, b: &nalgebra::DVector<f64>) -> Result<nalgebra::DVector<f64>, SparseLbltError> {
        let dim = self.mat_a.upper.ncols();
        let symbolic = &self.symbolic;

        #[cfg(not(target_arch = "wasm32"))]
        let par = if self.parallelize {
            Par::Rayon(NonZeroUsize::new(rayon::current_num_threads()).unwrap())
        } else {
            Par::Seq
        };
        #[cfg(target_arch = "wasm32")]
        let _ignore = self.parallelize;
        #[cfg(target_arch = "wasm32")]
        let par = Par::Seq;

        let params = Spec::<LbltParams, f64>::default();

        // Allocate storage for numeric factorization.
        let mut l_values = vec![0.0f64; symbolic.len_val()];
        let mut subdiag = vec![0.0f64; dim];
        let mut perm_fwd = vec![0usize; dim];
        let mut perm_inv = vec![0usize; dim];

        // Scratch space.
        let mut mem = MemBuffer::try_new(StackReq::all_of(&[
            symbolic.factorize_numeric_intranode_lblt_scratch::<f64>(par, params),
            symbolic.solve_in_place_scratch::<f64>(1, par),
        ]))
        .map_err(|_| SparseLbltError::FaerError(FaerError::OutOfMemory))?;
        let mut stack = MemStack::new(&mut mem);

        // Numeric LBLᵀ factorization.
        let lblt = symbolic.factorize_numeric_intranode_lblt(
            &mut l_values,
            &mut subdiag,
            &mut perm_fwd,
            &mut perm_inv,
            self.mat_a.upper.as_ref(),
            Side::Upper,
            par,
            ReborrowMut::rb_mut(&mut stack),
            params,
        );

        // Solve A x = b.
        let mut x = b.clone();
        let x_ref = MatMut::<f64>::from_column_major_slice_mut(x.as_mut_slice(), dim, 1);

        lblt.solve_in_place_with_conj(Conj::No, x_ref, par, ReborrowMut::rb_mut(&mut stack));

        Ok(x)
    }
}

impl IsFactor for FaerSparseLbltSystem {
    type Matrix = FaerSparseSymmetricMatrix;

    fn solve_inplace(&self, b: &mut nalgebra::DVector<f64>) -> Result<(), LinearSolverError> {
        match self.solve(b) {
            Ok(x) => {
                *b = x;
                Ok(())
            }
            Err(SparseLbltError::FaerError(e)) => Err(LinearSolverError::FaerSparseLdltError {
                faer_error: match e {
                    FaerError::OutOfMemory => FaerSparseSolverError::OutOfMemory,
                    FaerError::IndexOverflow => FaerSparseSolverError::IndexOverflow,
                    _ => FaerSparseSolverError::Unspecific,
                },
            }),
        }
    }
}
