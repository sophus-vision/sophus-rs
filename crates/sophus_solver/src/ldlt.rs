pub(crate) mod block_diag_ldlt;
pub(crate) mod block_sparse_ldlt;
pub(crate) mod dense_ldlt;
pub(crate) mod elimination_tree;
pub(crate) mod faer_sparse_ldlt;
/// Solver errors.
pub mod min_norm_ldlt;
pub(crate) mod sparse_ldlt;

use std::marker::PhantomData;

pub use block_diag_ldlt::*;
pub use block_sparse_ldlt::*;
pub use dense_ldlt::*;
pub use elimination_tree::*;
pub use faer_sparse_ldlt::*;
pub use sparse_ldlt::*;

use crate::{
    IsFactor,
    IsMinNormFactor,
};

/// g
pub trait IntoMinNormPsd: IsFactor {
    /// g
    type MinNormPsd: IsMinNormFactor;

    ///g
    fn into_min_norm_ldlt(self) -> Self::MinNormPsd;
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
//             // --- All scalar blocks ---
//             {
//                 use nalgebra as na;
//                 let parts = vec![PartitionSpec {
//                     block_dimension: 1,
//                     block_count: 8,
//                 }];
//                 let mut builder = Builder::zero(&parts);

//                 let n = parts[0].block_dimension * parts[0].block_count;
//                 let mut mat_l = na::DMatrix::<f64>::zeros(n, n);
//                 for i in 0..n {
//                     mat_l[(i, i)] = 1.0;
//                     if i > 0 {
//                         mat_l[(i, i - 1)] = 0.2;
//                     }
//                     if i > 1 {
//                         mat_l[(i, i - 2)] = 0.05;
//                     }
//                 }
//                 let mat_a = &mat_l * mat_l.transpose();
//                 for i in 0..parts[0].block_count {
//                     let idx_i = PartitionBlockIndex {
//                         partition: 0,
//                         block: i,
//                     };
//                     for j in 0..=i {
//                         let idx_j = PartitionBlockIndex {
//                             partition: 0,
//                             block: j,
//                         };
//                         builder.add_lower_block(idx_i, idx_j, &mat_a.view((i, j), (1, 1)));
//                     }
//                 }

//                 let b = na::DVector::<f64>::from_row_slice(&[
//                     1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0,
//                 ]);
//                 examples.push((builder.compress(), b));
//             }

//             examples
//         }

//         let dense_ex = all_examples::<DenseSymmetricMatrixBuilder>();
//         let sparse_ex = all_examples::<SparseSymmetricMatrixBuilder>();
//         let block_sparse_ex = all_examples::<LowerBlockSparseMatrixBuilder>();

//         let parallelize = false;

//         for i in 0..dense_ex.len() {
//             println!("example {i}\n");
//             let (dense_mat_a, b) = &dense_ex[i];
//             let (sparse_mat_a, _b) = &sparse_ex[i];
//             let (block_sparse_mat_a, _b) = &block_sparse_ex[i];

//             let faer_mat_a = FaerTripletMatrix::from_lower(sparse_mat_a);
//             let faer_upper_mat_a = FaerUpperTripletMatrix::from_lower(sparse_mat_a);

//             let x_dense_lu = DenseLu {}.solve(dense_mat_a, b).unwrap();

//             let x_faer_sparse_lu = FaerSparseLu {}.solve(&faer_mat_a.compress(), b).unwrap();
//             let x_faer_sparse_qr = FaerSparseQr {}.solve(&faer_mat_a.compress(), b).unwrap();

//             let x_dense_ldlt = DenseLdlt::default().solve(dense_mat_a, b).unwrap();

//             let x_sparse_ldlt = SparseLdlt::default()
//                 .solve(&sparse_mat_a.compress(), b)
//                 .unwrap();
//             let x_block_sparse_ldlt = BlockSparseLdlt::default()
//                 .solve(&block_sparse_mat_a.compress(), b)
//                 .unwrap();
//             let x_faer_sparse_ldlt = FaerSparseLdlt::default()
//                 .solve(&faer_upper_mat_a.compress(), b)
//                 .unwrap();

//             let dense_mat_a = dense_mat_a.view().to_owned();

//             print!("dense LU: x = {x_dense_lu}");
//             print!("faer sparse LU: x = {x_faer_sparse_lu}");

//             print!("faer sparse QR: x = {x_faer_sparse_qr}");

//             print!("dense LDLᵀ x = {x_dense_ldlt}");
//             print!("sparse LDLᵀ x = {x_sparse_ldlt}");
//             print!("block-sparse LDLᵀ x = {x_block_sparse_ldlt}");
//             print!("faer sparse LDLᵀ x = {x_faer_sparse_ldlt}");

//             approx::assert_abs_diff_eq!(
//                 dense_mat_a.clone() * x_dense_lu,
//                 b.clone(),
//                 epsilon = 1e-6
//             );
//             approx::assert_abs_diff_eq!(
//                 dense_mat_a.clone() * x_faer_sparse_lu,
//                 b.clone(),
//                 epsilon = 1e-6
//             );

//             approx::assert_abs_diff_eq!(
//                 dense_mat_a.clone() * x_faer_sparse_qr,
//                 b.clone(),
//                 epsilon = 1e-6
//             );

//             approx::assert_abs_diff_eq!(
//                 dense_mat_a.clone() * x_dense_ldlt,
//                 b.clone(),
//                 epsilon = 1e-6
//             );
//             approx::assert_abs_diff_eq!(
//                 dense_mat_a.clone() * x_sparse_ldlt,
//                 b.clone(),
//                 epsilon = 1e-6
//             );
//             approx::assert_abs_diff_eq!(
//                 dense_mat_a.clone() * x_block_sparse_ldlt,
//                 b.clone(),
//                 epsilon = 1e-6
//             );
//             approx::assert_abs_diff_eq!(
//                 dense_mat_a.clone() * x_faer_sparse_ldlt,
//                 b.clone(),
//                 epsilon = 1e-6
//             );
//         }
//     }
// }
