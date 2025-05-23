pub(crate) mod dense_lu;
pub(crate) mod sparse_ldlt;
pub(crate) mod sparse_lu;
pub(crate) mod sparse_qr;

use snafu::Snafu;

use crate::{
    block::symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder,
    nlls::NllsError,
};

/// Sparse solver error - forwarded from faer error enums.
#[derive(Snafu, Debug)]
pub enum SparseSolverError {
    /// An index exceeding the maximum value
    IndexOverflow,
    /// Memory allocation failed.
    OutOfMemory,
    /// LU decomposition specific error
    SymbolicSingular,
    /// LDLt Error
    LdltError,
    /// unspecific - to be forward compatible
    Unspecific,
}

pub(crate) trait IsSparseSymmetricLinearSystem {
    fn solve(
        &self,
        triplets: &SymmetricBlockSparseMatrixBuilder,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, NllsError>;
}

pub(crate) trait IsDenseLinearSystem {
    fn solve_dense(
        &self,
        mat_a: nalgebra::DMatrix<f64>,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, NllsError>;
}

#[test]
fn solver_tests() {
    use faer::sparse::Triplet;
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

    pub fn from_triplets_nxn(n: usize, triplets: &[Triplet<usize, usize, f64>]) -> DMatrix<f64> {
        let mut mat = DMatrix::zeros(n, n);
        for t in triplets {
            assert!(
                t.row < n && t.col < n,
                "Triplet index ({}, {}) out of bounds for an {n}x{n} matrix",
                t.row,
                t.row,
            );
            mat[(t.row, t.col)] += t.val;
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

    info!("x_dense_lu {x_dense_lu}");
    info!("x_sparse_qr {x_sparse_qr}");
    info!("x_sparse_lu {x_sparse_lu}");
    info!("x_sparse_ldlt {x_sparse_ldlt}");

    approx::assert_abs_diff_eq!(mat_a.clone() * x_dense_lu, b.clone(), epsilon = 1e-6);
    approx::assert_abs_diff_eq!(mat_a.clone() * x_sparse_qr, b.clone(), epsilon = 1e-6);
    approx::assert_abs_diff_eq!(mat_a.clone() * x_sparse_lu, b.clone(), epsilon = 1e-6);
    approx::assert_abs_diff_eq!(mat_a.clone() * x_sparse_ldlt, b.clone(), epsilon = 1e-6);
}
