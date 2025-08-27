extern crate alloc;

use alloc::vec::Vec;
#[cfg(not(target_arch = "wasm32"))]
use std::num::NonZeroUsize;

use faer::{
    Conj,
    Par,
    Side,
    dyn_stack::{
        MemBuffer,
        MemStack,
        StackReq,
    },
    linalg::cholesky::ldlt::factor::LdltRegularization,
    mat::MatMut,
    perm::PermRef,
    reborrow::ReborrowMut,
    sparse::{
        FaerError,
        SparseColMat,
        SymbolicSparseColMat,
        linalg::{
            amd,
            cholesky::simplicial,
        },
        utils,
    },
};

use crate::{
    IsLinearSolver,
    IsSymmetricMatrixBuilder,
    LinearSolverError,
    SparseSolverError,
    SymmetricMatrixBuilderEnum,
    sparse::{
        SparseSymmetricMatrixBuilder,
        faer_sparse_matrix::{
            FaerUpperCompressedMatrix,
            FaerUpperTripletsMatrix,
        },
    },
};

/// Parameters for faer's sparse LDLᵀ solver.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FaerSparseLdltParams {
    /// Numeric regularization for LDLᵀ.
    pub regularization_eps: f64,
}

impl Default for FaerSparseLdltParams {
    fn default() -> Self {
        Self {
            regularization_eps: 1e-6,
        }
    }
}

/// Sparse LDLᵀ solver using faer crate.
#[derive(Copy, Clone, Debug, Default)]
pub struct FaerSparseLdlt {
    params: FaerSparseLdltParams,
}

impl FaerSparseLdlt {
    /// Create new sparse LDLᵀ solver from params.
    pub fn new(params: FaerSparseLdltParams) -> Self {
        Self { params }
    }
}

impl IsLinearSolver for FaerSparseLdlt {
    type Matrix = FaerUpperTripletsMatrix;

    const NAME: &'static str = "faer sparse LDLᵀ";

    fn matrix_builder(&self, partitions: &[crate::PartitionSpec]) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::FaerSparseUpper(SparseSymmetricMatrixBuilder::zero(partitions))
    }

    fn solve_in_place(
        &self,
        parallelize: bool,
        upper: &FaerUpperCompressedMatrix,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        match SimplicialSparseLdlt::from_csc(upper.csc.clone(), parallelize, self.params).solve(b) {
            Ok(x) => {
                *b = x;
                Ok(())
            }
            Err(e) => Err(LinearSolverError::SparseLdltError {
                details: match e {
                    SimplicialSparseLdltError::FaerError(fe) => match fe {
                        FaerError::OutOfMemory => SparseSolverError::OutOfMemory,
                        FaerError::IndexOverflow => SparseSolverError::IndexOverflow,
                        _ => SparseSolverError::Unspecific,
                    },
                    SimplicialSparseLdltError::LdltError => SparseSolverError::LdltError,
                },
            }),
        }
    }

    fn solve(
        &self,
        parallelize: bool,
        matrix: &<Self::Matrix as crate::IsCompressibleMatrix>::Compressed,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, LinearSolverError> {
        let mut x = b.clone();
        self.solve_in_place(parallelize, matrix, &mut x)?;
        Ok(x)
    }

    fn name(&self) -> String {
        Self::NAME.into()
    }
}

struct SimplicialSparseLdlt {
    upper_ccs: SparseColMat<usize, f64>,
    params: FaerSparseLdltParams,
    parallelize: bool,
}

struct SparseLdltPerm {
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
}

impl SparseLdltPerm {
    fn perm_upper_ccs(&self, upper: &SparseColMat<usize, f64>) -> SparseColMat<usize, f64> {
        // Based on example code from the faer crate:

        let dim = upper.ncols();
        let nnz = upper.compute_nnz();

        // SAFETY: we guarantee both slices are valid permutations of 0..dim.
        let perm_ref = unsafe { PermRef::new_unchecked(&self.perm, &self.perm_inv, dim) };

        // allocate destination storage
        let mut col_ptr = vec![0usize; dim + 1];
        let mut row_idx = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];

        // scratch buffer for the permutation routine
        let mut mem =
            MemBuffer::try_new(utils::permute_self_adjoint_scratch::<usize>(dim)).unwrap();

        // Permute → result is *unsorted* upper triangle.
        utils::permute_self_adjoint_to_unsorted(
            &mut values,
            &mut col_ptr,
            &mut row_idx,
            upper.as_ref(),
            perm_ref,
            Side::Upper, // source triangle
            Side::Upper, // destination triangle
            MemStack::new(&mut mem),
        );

        // SAFETY: we just produced a valid CSC representation.
        SparseColMat::new(
            unsafe { SymbolicSparseColMat::new_unchecked(dim, dim, col_ptr, None, row_idx) },
            values,
        )
    }
}

struct SparseLdltPermSymb {
    permutation: SparseLdltPerm,
    symbolic: simplicial::SymbolicSimplicialCholesky<usize>,
}

enum SimplicialSparseLdltError {
    FaerError(FaerError),
    LdltError,
}

impl SimplicialSparseLdlt {
    fn symbolic_and_perm(&self) -> Result<SparseLdltPermSymb, FaerError> {
        // Based on example code from the faer crate:

        let dim = self.upper_ccs.ncols();
        let nnz = self.upper_ccs.compute_nnz();

        // --- AMD ordering --------------------------------------------------
        let (perm, perm_inv) = {
            let mut perm = vec![0usize; dim];
            let mut perm_inv = vec![0usize; dim];

            let mut mem = MemBuffer::try_new(amd::order_scratch::<usize>(dim, nnz))?;
            amd::order(
                &mut perm,
                &mut perm_inv,
                self.upper_ccs.symbolic(),
                amd::Control::default(),
                MemStack::new(&mut mem),
            )?;
            (perm, perm_inv)
        };

        let permutation = SparseLdltPerm { perm, perm_inv };
        let perm_upper = permutation.perm_upper_ccs(&self.upper_ccs);

        // --- symbolic analysis --------------------------------------------
        let symbolic = {
            let mut mem = MemBuffer::try_new(StackReq::any_of(&[
                simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(dim, nnz),
                simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(dim),
            ]))?;
            let mut stack = MemStack::new(&mut mem);

            let mut etree = vec![0isize; dim];
            let mut col_counts = vec![0usize; dim];

            simplicial::prefactorize_symbolic_cholesky(
                &mut etree,
                &mut col_counts,
                perm_upper.symbolic(),
                ReborrowMut::rb_mut(&mut stack),
            );

            // SAFETY: `etree` is filled by the previous call.
            simplicial::factorize_simplicial_symbolic_cholesky(
                perm_upper.symbolic(),
                unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
                &col_counts,
                ReborrowMut::rb_mut(&mut stack),
            )?
        };

        Ok(SparseLdltPermSymb {
            permutation,
            symbolic,
        })
    }

    fn solve_from_symbolic(
        &self,
        b: &nalgebra::DVector<f64>,
        symb_perm: &SparseLdltPermSymb,
    ) -> Result<nalgebra::DVector<f64>, SimplicialSparseLdltError> {
        // Based on example code from the faer crate:

        let dim = self.upper_ccs.ncols();
        // SAFETY: `perm` / `perm_inv` are valid permutations of size `dim`.
        let perm_ref = unsafe {
            PermRef::new_unchecked(
                &symb_perm.permutation.perm,
                &symb_perm.permutation.perm_inv,
                dim,
            )
        };
        let symbolic = &symb_perm.symbolic;

        // Scratch-space sizes
        let mut mem = MemBuffer::try_new(StackReq::all_of(&[
            simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, f64>(dim),
            faer::perm::permute_rows_in_place_scratch::<usize, f64>(dim, 1),
            symbolic.solve_in_place_scratch::<f64>(dim),
        ]))
        .map_err(|_| SimplicialSparseLdltError::FaerError(FaerError::OutOfMemory))?;
        let mut stack = MemStack::new(&mut mem);

        // Numeric LDLᵀ factorization
        let mut lval = vec![0.0f64; symbolic.len_val()];
        let perm_upper = symb_perm.permutation.perm_upper_ccs(&self.upper_ccs);

        simplicial::factorize_simplicial_numeric_ldlt::<usize, f64>(
            &mut lval,
            perm_upper.as_ref(),
            LdltRegularization {
                dynamic_regularization_signs: Some(&vec![1; dim]), // +1 on every diagonal
                dynamic_regularization_delta: self.params.regularization_eps,
                dynamic_regularization_epsilon: self.params.regularization_eps,
            },
            symbolic,
            ReborrowMut::rb_mut(&mut stack),
        )
        .map_err(|_| SimplicialSparseLdltError::LdltError)?;

        let ldlt = simplicial::SimplicialLdltRef::<usize, f64>::new(symbolic, &lval);

        // Solve Pᵀ A P x = Pᵀ b
        let mut x = b.clone();
        let mut x_ref = MatMut::<f64>::from_column_major_slice_mut(x.as_mut_slice(), dim, 1);

        // Pᵀ b
        faer::perm::permute_rows_in_place(
            ReborrowMut::rb_mut(&mut x_ref),
            perm_ref,
            ReborrowMut::rb_mut(&mut stack),
        );

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

        // (LDLᵀ)⁻¹
        ldlt.solve_in_place_with_conj(
            Conj::No,
            ReborrowMut::rb_mut(&mut x_ref),
            par,
            ReborrowMut::rb_mut(&mut stack),
        );
        // P x
        faer::perm::permute_rows_in_place(
            ReborrowMut::rb_mut(&mut x_ref),
            perm_ref.inverse(),
            ReborrowMut::rb_mut(&mut stack),
        );

        Ok(x)
    }

    fn from_csc(
        upper_ccs: faer::sparse::SparseColMat<usize, f64>,
        parallelize: bool,
        params: FaerSparseLdltParams,
    ) -> Self {
        // Triplets are assumed valid (caller generated them).
        // `try_new_from_triplets` sorts / deduplicates if needed.
        Self {
            upper_ccs,
            params,
            parallelize,
        }
    }

    fn solve(
        &self,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, SimplicialSparseLdltError> {
        let symb_perm = self
            .symbolic_and_perm()
            .map_err(SimplicialSparseLdltError::FaerError)?;
        self.solve_from_symbolic(b, &symb_perm)
    }
}
