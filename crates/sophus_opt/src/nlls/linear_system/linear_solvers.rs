use snafu::Snafu;

use crate::block::symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder;

/// dense lu
pub mod dense_lu;
/// sparse ldlt
pub mod sparse_ldlt;
/// sparse lu
pub mod sparse_lu;
/// sparse qr
pub mod sparse_qr;

/// Sparse solver error - forwarded from faer error enums.
#[derive(Snafu, Debug)]
pub enum SparseSolverError {
    /// An index exceeding the maximum value
    IndexOverflow,
    /// Memory allocation failed.
    OutOfMemory,
    /// LU decomposition specific error
    SymbolicSingular,
    /// unspecific - to be forward compatible
    Unspecific,
}

/// Linear solver error
#[derive(Snafu, Debug)]
pub enum SolveError {
    /// Sparse LDLt error
    #[snafu(display("sparse LDLt error {}", details))]
    SparseLdltError {
        /// details
        details: SparseSolverError,
    },
    /// Sparse LU error
    #[snafu(display("sparse LU error {}", details))]
    SparseLuError {
        /// details
        details: SparseSolverError,
    },
    /// Sparse QR error
    #[snafu(display("sparse QR error {}", details))]
    SparseQrError {
        /// details
        details: SparseSolverError,
    },
    /// Dense LU error
    #[snafu(display("dense LU solve failed"))]
    DenseLuError,
}

/// Interface for linear sparse symmetric system
pub trait IsSparseSymmetricLinearSystem {
    /// Solve the linear system
    fn solve(
        &self,
        triplets: &SymmetricBlockSparseMatrixBuilder,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, SolveError>;
}

/// Interface for solving a dense linear system
pub trait IsDenseLinearSystem {
    /// Solve
    fn solve_dense(
        &self,
        mat_a: nalgebra::DMatrix<f64>,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, SolveError>;
}

#[test]
fn linear_solver_tests() {
    use log::info;
    use nalgebra::DMatrix;
    use sophus_autodiff::{
        linalg::MatF64,
        prelude::*,
    };

    use crate::nlls::linear_system::{
        DenseLu,
        PartitionSpec,
        SparseLdlt,
        SparseLu,
        SparseQr,
    };

    pub fn from_triplets_nxn(n: usize, triplets: &[(usize, usize, f64)]) -> DMatrix<f64> {
        let mut mat = DMatrix::zeros(n, n);
        for &(i, j, val) in triplets {
            assert!(
                i < n && j < n,
                "Triplet index ({}, {}) out of bounds for an {}x{} matrix",
                i,
                j,
                n,
                n
            );
            mat[(i, j)] += val;
        }
        mat
    }

    let partitions = vec![PartitionSpec {
        num_blocks: 2,
        block_dim: 3,
    }];
    let mut symmetric_matrix_builder = SymmetricBlockSparseMatrixBuilder::zero(&partitions);

    let block_a00 = MatF64::<3, 3>::from_array2([
        [2.0, 1.0, 0.0], //
        [1.0, 2.0, 1.0],
        [0.0, 1.0, 2.0],
    ]);
    let block_a01 = MatF64::<3, 3>::from_array2([
        [0.0, 0.0, 0.0], //
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]);
    let block_a11 = MatF64::<3, 3>::from_array2([
        [1.0, 0.0, 0.0], //
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);

    symmetric_matrix_builder.add_block(&[0, 0], [0, 0], &block_a00.as_view());
    symmetric_matrix_builder.add_block(&[0, 0], [0, 1], &block_a01.as_view());
    symmetric_matrix_builder.add_block(&[0, 0], [1, 1], &block_a11.as_view());
    let mat_a = symmetric_matrix_builder.to_symmetric_dense();
    let mat_a_from_triplets = from_triplets_nxn(
        symmetric_matrix_builder.scalar_dimension(),
        &symmetric_matrix_builder.to_symmetric_scalar_triplets(),
    );
    assert_eq!(mat_a, mat_a_from_triplets);

    let b = nalgebra::DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    info!(
        "symmetric_matrix_builder. {:?}",
        symmetric_matrix_builder.to_symmetric_scalar_triplets()
    );
    let x_dense_lu = DenseLu {}.solve(&symmetric_matrix_builder, &b).unwrap();
    let x_sparse_qr = SparseQr {}.solve(&symmetric_matrix_builder, &b).unwrap();
    let x_sparse_lu = SparseLu {}.solve(&symmetric_matrix_builder, &b).unwrap();
    let x_sparse_ldlt = SparseLdlt::default()
        .solve(&symmetric_matrix_builder, &b)
        .unwrap();

    info!("x_dense_lu {}", x_dense_lu);
    info!("x_sparse_qr {}", x_sparse_qr);
    info!("x_sparse_lu {}", x_sparse_lu);
    info!("x_sparse_ldlt {}", x_sparse_ldlt);

    approx::assert_abs_diff_eq!(mat_a.clone() * x_dense_lu, b.clone(), epsilon = 1e-6);
    approx::assert_abs_diff_eq!(mat_a.clone() * x_sparse_qr, b.clone(), epsilon = 1e-6);
    approx::assert_abs_diff_eq!(mat_a.clone() * x_sparse_lu, b.clone(), epsilon = 1e-6);
    approx::assert_abs_diff_eq!(mat_a.clone() * x_sparse_ldlt, b.clone(), epsilon = 1e-6);
}
