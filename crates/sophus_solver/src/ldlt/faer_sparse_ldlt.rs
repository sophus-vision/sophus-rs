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
    FearSparseSolverError,
    IsFactor,
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

/// Parameters for faer's sparse LDLᵀ solver.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FaerSparseLdlt {
    /// Numeric regularization for LDLᵀ.
    pub regularization_eps: f64,
    /// Perform parallel execution?
    pub parallelize: bool,
}

impl Default for FaerSparseLdlt {
    fn default() -> Self {
        Self {
            regularization_eps: 1e-6,
            parallelize: false,
        }
    }
}

/// f
#[derive(Clone, Debug)]
pub struct FaerSparseLdltSystem {
    mat_a: FaerSparseSymmetricMatrix,
    params: FaerSparseLdlt,
    symb: SparseLdltPermSymb,
    parallelize: bool,
}

impl IsLinearSolver for FaerSparseLdlt {
    type SymmetricMatrixBuilder = FaerSparseSymmetricMatrixBuilder;

    const NAME: &'static str = "faer sparse LDLᵀ";

    fn zero(&self, partitions: PartitionSet) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::FaerSparseUpper(FaerSparseSymmetricMatrixBuilder::zero(
            partitions,
        ))
    }

    fn name(&self) -> String {
        Self::NAME.into()
    }

    type Factor = FaerSparseLdltSystem;

    fn factorize(
        &self,
        mat_a: &FaerSparseSymmetricMatrix,
    ) -> Result<Self::Factor, LinearSolverError> {
        puffin::profile_scope!("symbolic_and_perm");

        // Based on example code from the faer crate:

        let dim = mat_a.upper.ncols();
        let nnz = mat_a.upper.compute_nnz();

        // TODO: fix error handling!!!!!!

        // --- AMD ordering --------------------------------------------------
        let (perm, perm_inv) = {
            let mut perm = vec![0usize; dim];
            let mut perm_inv = vec![0usize; dim];

            let mut mem = MemBuffer::try_new(amd::order_scratch::<usize>(dim, nnz)).unwrap();
            amd::order(
                &mut perm,
                &mut perm_inv,
                mat_a.upper.symbolic(),
                amd::Control::default(),
                MemStack::new(&mut mem),
            )
            .unwrap();
            (perm, perm_inv)
        };

        let permutation = SparseLdltPerm { perm, perm_inv };
        let perm_upper = permutation.perm_upper_ccs(&mat_a.upper);

        // --- symbolic analysis --------------------------------------------
        let symbolic = {
            let mut mem = MemBuffer::try_new(StackReq::any_of(&[
                simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(dim, nnz),
                simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(dim),
            ]))
            .unwrap();
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
            )
            .unwrap()
        };

        Ok(FaerSparseLdltSystem {
            params: *self,
            parallelize: self.parallelize,
            symb: SparseLdltPermSymb {
                permutation,
                symbolic,
            },
            mat_a: mat_a.clone(),
        })
    }

    /// Set the `parallelize`` option.
    fn set_parallelize(&mut self, parallelize: bool) {
        self.parallelize = parallelize;
    }
}

impl IsFactor for FaerSparseLdltSystem {
    type Matrix = FaerSparseSymmetricMatrix;

    fn solve_inplace(&self, b: &mut nalgebra::DVector<f64>) -> Result<(), LinearSolverError> {
        match self.solve(b) {
            Ok(x) => {
                *b = x;
                Ok(())
            }
            Err(e) => Err(LinearSolverError::FaerSparseLdltError {
                faer_error: match e {
                    SimplicialSparseLdltError::FaerError(fe) => match fe {
                        FaerError::OutOfMemory => FearSparseSolverError::OutOfMemory,
                        FaerError::IndexOverflow => FearSparseSolverError::IndexOverflow,
                        _ => FearSparseSolverError::Unspecific,
                    },
                    SimplicialSparseLdltError::LdltError => FearSparseSolverError::LdltError,
                },
            }),
        }
    }
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
struct SparseLdltPermSymb {
    permutation: SparseLdltPerm,
    symbolic: simplicial::SymbolicSimplicialCholesky<usize>,
}

enum SimplicialSparseLdltError {
    FaerError(FaerError),
    LdltError,
}

impl FaerSparseLdltSystem {
    fn solve_from_symbolic(
        &self,
        b: &nalgebra::DVector<f64>,
        symb_perm: &SparseLdltPermSymb,
    ) -> Result<nalgebra::DVector<f64>, SimplicialSparseLdltError> {
        puffin::profile_scope!("solve_from_symbolic");

        // Based on example code from the faer crate:

        let dim = self.mat_a.upper.ncols();
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
        let perm_upper = symb_perm.permutation.perm_upper_ccs(&self.mat_a.upper);

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

    fn solve(
        &self,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, SimplicialSparseLdltError> {
        self.solve_from_symbolic(b, &self.symb)
    }
}
