use super::sparse_ldlt::SimplicalSparseLdlt;
use crate::nlls::LinearSolverType;
use crate::quadratic_cost::evaluated_cost::IsEvaluatedCost;
use crate::variables::VarKind;
use crate::variables::VarPool;
use faer::prelude::SpSolver;
use snafu::Snafu;

extern crate alloc;

#[derive(Copy, Clone, Debug, PartialEq)]
/// Triplet type
pub enum TripletType {
    /// Upper diagonal matrix
    Upper,
    /// Full matrix
    Full,
}

#[derive(Clone, Debug)]
/// A matrix in triplet format
pub struct SymmetricTripletMatrix {
    /// diagonal triplets (either upper diagonal or full matrix, depending on the linear solver type)
    pub triplets: alloc::vec::Vec<(usize, usize, f64)>,
    /// trplet type
    pub triplet_type: TripletType,
    /// row count (== column count)
    pub size: usize,
}

impl SymmetricTripletMatrix {
    /// Create an example matrix
    pub fn upper_diagonal_example() -> Self {
        Self {
            triplets: alloc::vec![
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
            triplet_type: TripletType::Upper,
            size: 8,
        }
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let mut full_matrix = nalgebra::DMatrix::from_element(self.size, self.size, 0.0);
        for &(row, col, value) in self.triplets.iter() {
            full_matrix[(row, col)] += value;
            if self.triplet_type == TripletType::Upper && row != col {
                full_matrix[(col, row)] += value;
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

/// Normal equation
pub struct NormalEquation {
    sparse_hessian: SymmetricTripletMatrix,
    neg_gradient: nalgebra::DVector<f64>,
    linear_solver: LinearSolverType,
}

/// Linear solver error
#[derive(Snafu, Debug)]
pub enum SolveError {
    /// Sparse LDLt error
    #[snafu(display("sparse LDLt error {}", details))]
    SparseLdltError {
        /// details
        details: faer::sparse::FaerError,
    },
    /// Sparse LU error
    #[snafu(display("sparse LU error {}", details))]
    SparseLuError {
        /// details
        details: faer::sparse::LuError,
    },
    /// Sparse QR error
    #[snafu(display("sparse QR error {}", details))]
    SparseQrError {
        /// details
        details: faer::sparse::FaerError,
    },
    /// Dense cholesky error
    #[snafu(display("dense cholesky factorization failed"))]
    DenseCholeskyError,
    /// Dense LU error
    #[snafu(display("dense LU solve failed"))]
    DenseLuError,
}

impl NormalEquation {
    fn from_families_and_cost(
        variables: &VarPool,
        costs: alloc::vec::Vec<alloc::boxed::Box<dyn IsEvaluatedCost>>,
        nu: f64,
        linear_solver: LinearSolverType,
    ) -> NormalEquation {
        assert!(variables.num_of_kind(VarKind::Marginalized) == 0);
        assert!(variables.num_of_kind(VarKind::Free) >= 1);

        // Note let's first focus on these special cases, before attempting a
        // general version covering all cases holistically. Also, it might not be trivial
        // to implement VarKind::Marginalized > 1.
        //  -  Example, the the arrow-head sparsity uses a recursive application of the Schur-Complement.
        let num_var_params = variables.num_free_params();
        let mut hessian_triplet = alloc::vec::Vec::new();
        let mut neg_grad = nalgebra::DVector::<f64>::zeros(num_var_params);
        let triplet_type = linear_solver.triplet_type();

        for cost in costs.iter() {
            cost.populate_normal_equation(
                variables,
                nu,
                &mut hessian_triplet,
                &mut neg_grad,
                triplet_type == TripletType::Upper,
            );
        }

        Self {
            sparse_hessian: SymmetricTripletMatrix {
                triplets: hessian_triplet,
                triplet_type,
                size: num_var_params,
            },
            neg_gradient: neg_grad,
            linear_solver,
        }
    }

    fn solve(&mut self) -> Result<nalgebra::DVector<f64>, SolveError> {
        match self.linear_solver {
            LinearSolverType::FaerSparseLdlt(ldlt_params) => Ok(
                match SimplicalSparseLdlt::from_triplets(&self.sparse_hessian, ldlt_params)
                    .solve(&self.neg_gradient)
                {
                    Ok(x) => x,
                    Err(e) => return Err(SolveError::SparseLdltError { details: e }),
                },
            ),
            LinearSolverType::FaerSparsePartialPivLu => {
                let csr = faer::sparse::SparseColMat::try_new_from_triplets(
                    self.sparse_hessian.size,
                    self.sparse_hessian.size,
                    &self.sparse_hessian.triplets,
                )
                .unwrap();
                let x = self.neg_gradient.clone();
                let x_slice_mut = self.neg_gradient.as_mut_slice();
                let mut x_ref = faer::mat::from_column_major_slice_mut(
                    x_slice_mut,
                    self.sparse_hessian.size,
                    1,
                );
                match csr.sp_lu() {
                    Ok(lu) => {
                        lu.solve_in_place(faer::reborrow::ReborrowMut::rb_mut(&mut x_ref));
                        Ok(x)
                    }
                    Err(e) => Err(SolveError::SparseLuError { details: e }),
                }
            }
            LinearSolverType::FaerSparseQR => {
                let csr = faer::sparse::SparseColMat::try_new_from_triplets(
                    self.sparse_hessian.size,
                    self.sparse_hessian.size,
                    &self.sparse_hessian.triplets,
                )
                .unwrap();
                let x = self.neg_gradient.clone();
                let x_slice_mut = self.neg_gradient.as_mut_slice();
                let mut x_ref = faer::mat::from_column_major_slice_mut(
                    x_slice_mut,
                    self.sparse_hessian.size,
                    1,
                );
                match csr.sp_qr() {
                    Ok(qr) => {
                        qr.solve_in_place(faer::reborrow::ReborrowMut::rb_mut(&mut x_ref));
                        Ok(x)
                    }
                    Err(e) => Err(SolveError::SparseQrError { details: e }),
                }
            }
            LinearSolverType::NalgebraDenseCholesky => {
                let dense_matrix = self.sparse_hessian.to_dense();
                match dense_matrix.cholesky() {
                    Some(cholesky) => Ok(cholesky.solve(&self.neg_gradient)),
                    None => Err(SolveError::DenseCholeskyError),
                }
            }
            LinearSolverType::NalgebraDenseFullPiVLu => {
                let dense_matrix = self.sparse_hessian.to_dense();
                match dense_matrix.full_piv_lu().solve(&self.neg_gradient) {
                    Some(x) => Ok(x),
                    None => Err(SolveError::DenseLuError),
                }
            }
        }
    }
}

/// Solve the normal equation
pub fn solve(
    variables: &VarPool,
    costs: alloc::vec::Vec<alloc::boxed::Box<dyn IsEvaluatedCost>>,
    nu: f64,
    linear_solver: LinearSolverType,
) -> Result<VarPool, SolveError> {
    assert!(variables.num_of_kind(VarKind::Marginalized) <= 1);
    assert!(variables.num_of_kind(VarKind::Free) >= 1);

    if variables.num_of_kind(VarKind::Marginalized) == 0 {
        let mut sne = NormalEquation::from_families_and_cost(variables, costs, nu, linear_solver);
        let delta = sne.solve();
        Ok(variables.update(delta?))
    } else {
        todo!()
    }
}
