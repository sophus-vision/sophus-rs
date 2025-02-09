use faer::sparse::FaerError;

use super::{
    IsSparseSymmetricLinearSystem,
    NllsError,
};
use crate::block::symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder;

extern crate alloc;

/// sparse LDLT factorization parameters
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SparseLdltParams {
    /// Regularization for LDLT factorization
    pub regularization_eps: f64,
}

impl Default for SparseLdltParams {
    fn default() -> Self {
        Self {
            regularization_eps: 1e-6,
        }
    }
}

/// Sparse LDLt solver
///
/// Sparse LDLt decomposition - based on an example from the faer crate.
#[derive(Default, Debug)]
pub struct SparseLdlt {
    params: SparseLdltParams,
}

impl SparseLdlt {
    /// Create a new sparse LDLt solver
    pub fn new(params: SparseLdltParams) -> Self {
        Self { params }
    }
}

impl IsSparseSymmetricLinearSystem for SparseLdlt {
    fn solve(
        &self,
        upper_triangular: &SymmetricBlockSparseMatrixBuilder,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, NllsError> {
        Ok(
            match SimplicalSparseLdlt::from_triplets(
                &upper_triangular.to_upper_triangular_scalar_triplets(),
                upper_triangular.scalar_dimension(),
                self.params,
            )
            .solve(b)
            {
                Ok(x) => x,
                Err(e) => {
                    return Err(NllsError::SparseLdltError {
                        details: match e {
                            FaerError::IndexOverflow => super::SparseSolverError::IndexOverflow,
                            FaerError::OutOfMemory => super::SparseSolverError::OutOfMemory,
                            _ => super::SparseSolverError::Unspecific,
                        },
                    })
                }
            },
        )
    }
}

struct SimplicalSparseLdlt {
    upper_ccs: faer::sparse::SparseColMat<usize, f64>,
    params: SparseLdltParams,
}

struct SparseLdltPerm {
    perm: alloc::vec::Vec<usize>,
    perm_inv: alloc::vec::Vec<usize>,
}

impl SparseLdltPerm {
    fn perm_upper_ccs(
        &self,
        upper_ccs: &faer::sparse::SparseColMat<usize, f64>,
    ) -> faer::sparse::SparseColMat<usize, f64> {
        let dim = upper_ccs.ncols();
        let nnz = upper_ccs.compute_nnz();
        let perm_ref =
            unsafe { faer::perm::PermRef::new_unchecked(&self.perm, &self.perm_inv, dim) };

        let mut ccs_perm_col_ptrs = alloc::vec::Vec::new();
        let mut ccs_perm_row_indices = alloc::vec::Vec::new();
        let mut ccs_perm_values = alloc::vec::Vec::new();

        ccs_perm_col_ptrs.try_reserve_exact(dim + 1).unwrap();
        ccs_perm_col_ptrs.resize(dim + 1, 0usize);
        ccs_perm_row_indices.try_reserve_exact(nnz).unwrap();
        ccs_perm_row_indices.resize(nnz, 0usize);
        ccs_perm_values.try_reserve_exact(nnz).unwrap();
        ccs_perm_values.resize(nnz, 0.0f64);

        let mut mem = faer::dyn_stack::GlobalPodBuffer::try_new(
            faer::sparse::utils::permute_hermitian_req::<usize>(dim).unwrap(),
        )
        .unwrap();
        faer::sparse::utils::permute_hermitian::<usize, f64>(
            &mut ccs_perm_values,
            &mut ccs_perm_col_ptrs,
            &mut ccs_perm_row_indices,
            upper_ccs.as_ref(),
            perm_ref,
            faer::Side::Upper,
            faer::Side::Upper,
            faer::dyn_stack::PodStack::new(&mut mem),
        );

        faer::sparse::SparseColMat::<usize, f64>::new(
            unsafe {
                faer::sparse::SymbolicSparseColMat::new_unchecked(
                    dim,
                    dim,
                    ccs_perm_col_ptrs,
                    None,
                    ccs_perm_row_indices,
                )
            },
            ccs_perm_values,
        )
    }
}

struct SparseLdltPermSymb {
    permutation: SparseLdltPerm,
    symbolic: faer::sparse::linalg::cholesky::simplicial::SymbolicSimplicialCholesky<usize>,
}

impl SimplicalSparseLdlt {
    fn symbolic_and_perm(&self) -> Result<SparseLdltPermSymb, FaerError> {
        // Following low-level example from faer crate:
        // https://github.com/sarah-quinones/faer-rs/blob/main/src/sparse/linalg/cholesky.rs#L11
        let dim = self.upper_ccs.ncols();

        let nnz = self.upper_ccs.compute_nnz();

        let (perm, perm_inv) = {
            let mut perm = alloc::vec::Vec::new();
            let mut perm_inv = alloc::vec::Vec::new();
            perm.try_reserve_exact(dim)?;
            perm_inv.try_reserve_exact(dim)?;
            perm.resize(dim, 0usize);
            perm_inv.resize(dim, 0usize);

            let mut mem =
                faer::dyn_stack::GlobalPodBuffer::try_new(faer::sparse::linalg::amd::order_req::<
                    usize,
                >(dim, nnz)?)?;
            faer::sparse::linalg::amd::order(
                &mut perm,
                &mut perm_inv,
                self.upper_ccs.symbolic(),
                faer::sparse::linalg::amd::Control::default(),
                faer::dyn_stack::PodStack::new(&mut mem),
            )?;

            (perm, perm_inv)
        };
        let permutation = SparseLdltPerm { perm, perm_inv };

        let ccs_perm_upper = permutation.perm_upper_ccs(&self.upper_ccs);

        let symbolic = {
            let mut mem = faer::dyn_stack::GlobalPodBuffer::try_new(
                faer::dyn_stack::StackReq::try_any_of([
                    faer::sparse::linalg::cholesky::simplicial::prefactorize_symbolic_cholesky_req::<
                        usize,
                    >(dim, self.upper_ccs.compute_nnz())?,
                    faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_symbolic_req::<
                        usize,
                    >(dim)?,
                ])?,
            )?;
            let mut stack = faer::dyn_stack::PodStack::new(&mut mem);

            let mut etree = alloc::vec::Vec::new();
            let mut col_counts = alloc::vec::Vec::new();
            etree.try_reserve_exact(dim)?;
            etree.resize(dim, 0isize);
            col_counts.try_reserve_exact(dim)?;
            col_counts.resize(dim, 0usize);

            faer::sparse::linalg::cholesky::simplicial::prefactorize_symbolic_cholesky(
                &mut etree,
                &mut col_counts,
                ccs_perm_upper.symbolic(),
                faer::reborrow::ReborrowMut::rb_mut(&mut stack),
            );
            faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_symbolic(
                ccs_perm_upper.symbolic(),
                // SAFETY: `etree` was filled correctly by
                // `simplicial::prefactorize_symbolic_cholesky`.
                unsafe {
                    faer::sparse::linalg::cholesky::simplicial::EliminationTreeRef::from_inner(
                        &etree,
                    )
                },
                &col_counts,
                faer::reborrow::ReborrowMut::rb_mut(&mut stack),
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
        symbolic_perm: &SparseLdltPermSymb,
    ) -> Result<nalgebra::DVector<f64>, FaerError> {
        let dim = self.upper_ccs.ncols();
        let perm_ref = unsafe {
            faer::perm::PermRef::new_unchecked(
                &symbolic_perm.permutation.perm,
                &symbolic_perm.permutation.perm_inv,
                dim,
            )
        };
        let symbolic = &symbolic_perm.symbolic;

        let mut mem =
            faer::dyn_stack::GlobalPodBuffer::try_new(faer::dyn_stack::StackReq::try_all_of([
                faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_numeric_ldlt_req::<
                    usize,
                    f64,
                >(dim)?,
                faer::perm::permute_rows_in_place_req::<usize, f64>(dim, 1)?,
                symbolic.solve_in_place_req::<f64>(dim)?,
            ])?)?;
        let mut stack = faer::dyn_stack::PodStack::new(&mut mem);

        let mut l_values = alloc::vec::Vec::new();
        l_values.try_reserve_exact(symbolic.len_values())?;
        l_values.resize(symbolic.len_values(), 0.0f64);
        let ccs_perm_upper = symbolic_perm.permutation.perm_upper_ccs(&self.upper_ccs);

        faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_numeric_ldlt::<usize, f64>(
            &mut l_values,
            ccs_perm_upper.as_ref(),
            faer::sparse::linalg::cholesky::LdltRegularization {
                dynamic_regularization_signs: Some(&vec![1; dim]),
                dynamic_regularization_delta: self.params.regularization_eps,
                dynamic_regularization_epsilon: self.params.regularization_eps,
            },
            symbolic,
            faer::reborrow::ReborrowMut::rb_mut(&mut stack),
        );

        let ldlt =
            faer::sparse::linalg::cholesky::simplicial::SimplicialLdltRef::<'_, usize, f64>::new(
                symbolic, &l_values,
            );

        let mut x = b.clone();
        let x_mut_slice = x.as_mut_slice();
        let mut x_ref = faer::mat::from_column_major_slice_mut(x_mut_slice, dim, 1);
        faer::perm::permute_rows_in_place(
            faer::reborrow::ReborrowMut::rb_mut(&mut x_ref),
            perm_ref,
            faer::reborrow::ReborrowMut::rb_mut(&mut stack),
        );
        ldlt.solve_in_place_with_conj(
            faer::Conj::No,
            faer::reborrow::ReborrowMut::rb_mut(&mut x_ref),
            faer::Parallelism::None,
            faer::reborrow::ReborrowMut::rb_mut(&mut stack),
        );
        faer::perm::permute_rows_in_place(
            faer::reborrow::ReborrowMut::rb_mut(&mut x_ref),
            perm_ref.inverse(),
            faer::reborrow::ReborrowMut::rb_mut(&mut stack),
        );

        Ok(x)
    }

    fn from_triplets(
        sym_tri_mat: &[(usize, usize, f64)],
        size: usize,
        params: SparseLdltParams,
    ) -> Self {
        SimplicalSparseLdlt {
            upper_ccs: faer::sparse::SparseColMat::try_new_from_triplets(size, size, sym_tri_mat)
                .unwrap(),
            params,
        }
    }

    fn solve(&self, b: &nalgebra::DVector<f64>) -> Result<nalgebra::DVector<f64>, FaerError> {
        let symbolic_perm = self.symbolic_and_perm()?;
        self.solve_from_symbolic(b, &symbolic_perm)
    }
}
