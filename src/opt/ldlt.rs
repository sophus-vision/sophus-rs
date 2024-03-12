use nalgebra::DMatrix;

/// A matrix in triplet format for sparse LDLT factorization / using faer crate
pub struct SparseLdlt {
    upper_ccs: faer::sparse::SparseColMat<usize, f64>,
}

/// A matrix in triplet format
pub struct SymmetricTripletMatrix {
    /// upper diagonal triplets
    pub upper_triplets: Vec<(usize, usize, f64)>,
    /// row count (== column count)
    pub size: usize,
}

impl SymmetricTripletMatrix {
    /// Create an example matrix
    pub fn example() -> Self {
        Self {
            upper_triplets: vec![
                (0, 0, 3.05631771),
                (1, 1, 60.05631771),
                (2, 2, 6.05631771),
                (3, 3, 5.05631771),
                (4, 4, 8.05631771),
                (5, 5, 5.05631771),
                (6, 6, 0.05631771),
                (7, 7, 10.005631771),
                (0, 1, 2.41883573),
                (0, 3, 2.41883573),
                (0, 5, 1.88585946),
                (1, 3, 1.73897015),
                (1, 5, 2.12387697),
                (1, 7, 1.47609157),
                (2, 7, 1.4541327),
                (3, 4, 2.35666066),
                (3, 7, 0.94642903),
            ],
            size: 8,
        }
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let mut full_matrix = nalgebra::DMatrix::from_element(self.size, self.size, 0.0);
        for &(row, col, value) in self.upper_triplets.iter() {
            full_matrix[(row, col)] = value;
            if row != col {
                full_matrix[(col, row)] = value;
            }
        }
        full_matrix
    }

    /// Returns true if the matrix is semi-positive definite
    ///
    /// TODO: This is a very inefficient implementation
    pub fn is_semi_positive_definite(&self) -> bool {
        let full_matrix = self.to_dense();
        let eigen_comp = full_matrix.symmetric_eigen();
        eigen_comp.eigenvalues.iter().all(|&x| x >= 0.0)
    }
}

struct SparseLdltPerm {
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
}

impl SparseLdltPerm {
    fn perm_upper_ccs(
        &self,
        upper_ccs: &faer::sparse::SparseColMat<usize, f64>,
    ) -> faer::sparse::SparseColMat<usize, f64> {
        let dim = upper_ccs.ncols();
        let nnz = upper_ccs.compute_nnz();
        let perm_ref = unsafe { faer::perm::PermRef::new_unchecked(&self.perm, &self.perm_inv) };

        let mut ccs_perm_col_ptrs = Vec::new();
        let mut ccs_perm_row_indices = Vec::new();
        let mut ccs_perm_values = Vec::new();

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

/// Symbolic LDLT factorization and permutation
struct SparseLdltPermSymb {
    permutation: SparseLdltPerm,
    symbolic: faer::sparse::linalg::cholesky::simplicial::SymbolicSimplicialCholesky<usize>,
}

impl SparseLdlt {
    fn symbolic_and_perm(&self) -> SparseLdltPermSymb {
        let dim = self.upper_ccs.ncols();

        let nnz = self.upper_ccs.compute_nnz();

        let (perm, perm_inv) = {
            let mut perm = Vec::new();
            let mut perm_inv = Vec::new();
            perm.try_reserve_exact(dim).unwrap();
            perm_inv.try_reserve_exact(dim).unwrap();
            perm.resize(dim, 0usize);
            perm_inv.resize(dim, 0usize);

            let mut mem = faer::dyn_stack::GlobalPodBuffer::try_new(
                faer::sparse::linalg::amd::order_req::<usize>(dim, nnz).unwrap(),
            )
            .unwrap();
            faer::sparse::linalg::amd::order(
                &mut perm,
                &mut perm_inv,
                self.upper_ccs.symbolic(),
                faer::sparse::linalg::amd::Control::default(),
                faer::dyn_stack::PodStack::new(&mut mem),
            )
            .unwrap();

            (perm, perm_inv)
        };
        let permutation = SparseLdltPerm { perm, perm_inv };

        let ccs_perm_upper = permutation.perm_upper_ccs(&self.upper_ccs);

        let symbolic = {
            let mut mem = faer::dyn_stack::GlobalPodBuffer::try_new(
                faer::dyn_stack::StackReq::try_any_of([
                    faer::sparse::linalg::cholesky::simplicial::prefactorize_symbolic_cholesky_req::<
                        usize,
                    >(dim, self.upper_ccs.compute_nnz()).unwrap(),
                    faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_symbolic_req::<
                        usize,
                    >(dim).unwrap(),
                ]).unwrap(),
            ).unwrap();
            let mut stack = faer::dyn_stack::PodStack::new(&mut mem);

            let mut etree = Vec::new();
            let mut col_counts = Vec::new();
            etree.try_reserve_exact(dim).unwrap();
            etree.resize(dim, 0isize);
            col_counts.try_reserve_exact(dim).unwrap();
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
            )
            .unwrap()
        };
        SparseLdltPermSymb {
            permutation,
            symbolic,
        }
    }

    fn solve_from_symbolic(
        &self,
        b: &nalgebra::DVector<f64>,
        symbolic_perm: &SparseLdltPermSymb,
    ) -> nalgebra::DVector<f64> {
        let dim = self.upper_ccs.ncols();
        let perm_ref = unsafe {
            faer::perm::PermRef::new_unchecked(
                &symbolic_perm.permutation.perm,
                &symbolic_perm.permutation.perm_inv,
            )
        };
        let symbolic = &symbolic_perm.symbolic;

        let mut mem = faer::dyn_stack::GlobalPodBuffer::try_new(faer::dyn_stack::StackReq::try_all_of([
            faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_numeric_ldlt_req::<usize, f64>(dim).unwrap(),
            faer::perm::permute_rows_in_place_req::<usize, f64>(dim, 1).unwrap(),
            symbolic.solve_in_place_req::<f64>(dim).unwrap(),
        ]).unwrap()).unwrap();
        let mut stack = faer::dyn_stack::PodStack::new(&mut mem);

        let mut l_values = Vec::new();
        l_values.try_reserve_exact(symbolic.len_values()).unwrap();
        l_values.resize(symbolic.len_values(), 0.0f64);
        let ccs_perm_upper = symbolic_perm.permutation.perm_upper_ccs(&self.upper_ccs);

        faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_numeric_ldlt::<usize, f64>(
            &mut l_values,
            ccs_perm_upper.as_ref(),
            faer::sparse::linalg::cholesky::LdltRegularization::default(),
            &symbolic,
            faer::reborrow::ReborrowMut::rb_mut(&mut stack),
        );

        let ldlt =
            faer::sparse::linalg::cholesky::simplicial::SimplicialLdltRef::<'_, usize, f64>::new(
                &symbolic, &l_values,
            );

        let mut x = b.clone();
        let mut x_mut_slice = x.as_mut_slice();
        let mut x_ref = faer::mat::from_column_major_slice_mut::<f64>(&mut x_mut_slice, dim, 1);
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

        x
    }

    /// Create a sparse LDLT factorization from a symmetric triplet matrix
    pub fn from_triplets(sym_tri_mat: &SymmetricTripletMatrix) -> Self {
        SparseLdlt {
            upper_ccs: faer::sparse::SparseColMat::try_new_from_triplets(
                sym_tri_mat.size,
                sym_tri_mat.size,
                &sym_tri_mat.upper_triplets,
            )
            .unwrap(),
        }
    }

    /// convert to dense matrix (for testing purposes)
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let upper_dense = self.upper_ccs.to_dense();
        let mut nalgebra_mat = DMatrix::<f64>::zeros(upper_dense.nrows(), upper_dense.ncols());
        for row in 0..upper_dense.nrows() {
            for col in row..upper_dense.ncols() {
                let value = *upper_dense.get(row, col);
                *nalgebra_mat.get_mut((row, col)).unwrap() = value;
                *nalgebra_mat.get_mut((col, row)).unwrap() = value;
            }
        }
        nalgebra_mat
    }

    /// Solve the linear system `Ax = b`
    ///
    /// TODO: Consider a more efficient API where the symbolic factorization is
    ///       computed once and then reused for multiple solves.
    pub fn solve(&self, b: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        let symbolic_perm = self.symbolic_and_perm();
        self.solve_from_symbolic(b, &symbolic_perm)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use super::*;

    #[test]
    fn ldlt() {
        let sym_tri_mat = SymmetricTripletMatrix::example();
        assert!(sym_tri_mat.is_semi_positive_definite());
        let dense_mat = sym_tri_mat.to_dense();

        let sparse_ldlt = SparseLdlt::from_triplets(&sym_tri_mat);
        assert_eq!(sparse_ldlt.to_dense(), dense_mat);

        let b = DVector::from_element(8, 1.0);
        let x = sparse_ldlt.solve(&b);

        approx::assert_abs_diff_eq!(dense_mat * x, b);
    }
}
