use nalgebra::DMatrix;

use crate::{
    BlockSparseLowerMatrixBuilder,
    CompressedMatrixEnum,
    LinearSolverEnum,
    PartitionSpec,
    dense::DenseSymmetricMatrixBuilder,
    psd_solver::block_sparse_ldlt::phase,
    sparse::{
        LowerTripletsMatrix,
        SparseSymmetricMatrixBuilder,
        faer_sparse_matrix::{
            FaerTripletsMatrix,
            FaerUpperTripletsMatrix,
        },
    },
};

/// sym mat
pub trait IsSymmetricMatrixBuilder {
    /// mat
    type Matrix: IsSymmetricMatrix;

    /// Create a symmetric matrix "filled" with zeros.
    ///
    /// The shape of the matrix is determined by the partition specs.
    fn zero(partitions: &[PartitionSpec]) -> Self;

    /// scalar dimension of the matrix.
    fn scalar_dimension(&self) -> usize;

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block.
    ///
    /// Only lower triangular blocks are accepted.
    ///
    /// In release mode, upper triangular blocks are ignored. In debug mode,
    /// this function will panic if the block is upper triangular.
    fn add_lower_block(
        &mut self,
        region_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &nalgebra::DMatrixView<f64>,
    );

    /// Export UPPER triangular scalar triplets (view) from lower storage.
    fn build(self) -> Self::Matrix;
}

/// d
pub enum SymmetricMatrixBuilderEnum {
    ///d
    Dense(DenseSymmetricMatrixBuilder),
    /// s
    SparseLower(SparseSymmetricMatrixBuilder),
    /// s
    BlockSparseUpper(BlockSparseLowerMatrixBuilder),
    /// s
    FaerSparse(SparseSymmetricMatrixBuilder),
    /// s
    FaerSparseUpper(SparseSymmetricMatrixBuilder),
}

impl SymmetricMatrixBuilderEnum {
    /// z
    pub fn zero(solver: LinearSolverEnum, partitions: &[PartitionSpec]) -> Self {
        let z = match solver {
            LinearSolverEnum::DenseLdlt(_) | LinearSolverEnum::DenseLu(_) => {
                SymmetricMatrixBuilderEnum::Dense(DenseSymmetricMatrixBuilder::zero(partitions))
            }
            LinearSolverEnum::SparseLdlt(_) => SymmetricMatrixBuilderEnum::SparseLower(
                SparseSymmetricMatrixBuilder::zero(partitions),
            ),
            LinearSolverEnum::BlockSparseLdlt(_) => SymmetricMatrixBuilderEnum::BlockSparseUpper(
                BlockSparseLowerMatrixBuilder::zero(partitions),
            ),
            LinearSolverEnum::FaerSparseQr(_) => SymmetricMatrixBuilderEnum::FaerSparse(
                SparseSymmetricMatrixBuilder::zero(partitions),
            ),
            LinearSolverEnum::FaerSparseLu(_) => SymmetricMatrixBuilderEnum::FaerSparse(
                SparseSymmetricMatrixBuilder::zero(partitions),
            ),
            LinearSolverEnum::FaerSparseLdlt(_) => SymmetricMatrixBuilderEnum::FaerSparseUpper(
                SparseSymmetricMatrixBuilder::zero(partitions),
            ),
        };
        //("zero");
        z
    }

    /// a
    pub fn add_lower_block(
        &mut self,
        region_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &nalgebra::DMatrixView<f64>,
    ) {
        match self {
            SymmetricMatrixBuilderEnum::Dense(dense_symmetric_matrix_builder) => {
                dense_symmetric_matrix_builder.add_lower_block(region_idx, block_index, block)
            }
            SymmetricMatrixBuilderEnum::SparseLower(sparse_symmetric_matrix_builder) => {
                sparse_symmetric_matrix_builder.add_lower_block(region_idx, block_index, block)
            }
            SymmetricMatrixBuilderEnum::BlockSparseUpper(block_sparse_lower_matrix_builder) => {
                block_sparse_lower_matrix_builder.add_lower_block(region_idx, block_index, block);
            }
            SymmetricMatrixBuilderEnum::FaerSparse(sparse_symmetric_matrix_builder) => {
                sparse_symmetric_matrix_builder.add_lower_block(region_idx, block_index, block);
            }
            SymmetricMatrixBuilderEnum::FaerSparseUpper(sparse_symmetric_matrix_builder) => {
                sparse_symmetric_matrix_builder.add_lower_block(region_idx, block_index, block);
            }
        }
        // phase("add_lower_block");
    }

    /// b
    pub fn build(self) -> SymmetricMatrixEnum {
        let m = match self {
            SymmetricMatrixBuilderEnum::Dense(dense_symmetric_matrix_builder) => {
                SymmetricMatrixEnum::Dense(dense_symmetric_matrix_builder.build())
            }
            SymmetricMatrixBuilderEnum::SparseLower(sparse_symmetric_matrix_builder) => {
                SymmetricMatrixEnum::SparseLower(sparse_symmetric_matrix_builder.build())
            }
            SymmetricMatrixBuilderEnum::BlockSparseUpper(block_sparse_lower_matrix_builder) => {
                SymmetricMatrixEnum::BlockSparseLower(block_sparse_lower_matrix_builder.build())
            }
            SymmetricMatrixBuilderEnum::FaerSparse(sparse_symmetric_matrix_builder) => {
                SymmetricMatrixEnum::FaerSparse(FaerTripletsMatrix::from_lower(
                    &sparse_symmetric_matrix_builder.build(),
                ))
            }
            SymmetricMatrixBuilderEnum::FaerSparseUpper(sparse_symmetric_matrix_builder) => {
                SymmetricMatrixEnum::FaerSparseUpper(FaerUpperTripletsMatrix::from_lower(
                    &sparse_symmetric_matrix_builder.build(),
                ))
            }
        };
        //phase("build");
        m
    }
}

/// f
pub trait IsSymmetricMatrix {
    /// c
    type Compressed;

    /// c
    fn compress(&self) -> Self::Compressed;
}

/// c
pub enum SymmetricMatrixEnum {
    ///d
    Dense(DMatrix<f64>),
    /// s
    SparseLower(LowerTripletsMatrix),
    /// s
    BlockSparseLower(BlockSparseLowerMatrixBuilder),
    /// hh
    FaerSparse(FaerTripletsMatrix),
    /// hh
    FaerSparseUpper(FaerUpperTripletsMatrix),
}

impl IsSymmetricMatrix for SymmetricMatrixEnum {
    type Compressed = CompressedMatrixEnum;

    fn compress(&self) -> Self::Compressed {
        let c = match self {
            SymmetricMatrixEnum::Dense(matrix) => CompressedMatrixEnum::Dense(matrix.compress()),
            SymmetricMatrixEnum::BlockSparseLower(block_sparse_compressed_matrix) => {
                CompressedMatrixEnum::BlockSparseLower(block_sparse_compressed_matrix.compress())
            }
            SymmetricMatrixEnum::SparseLower(lower_triplets_matrix) => {
                CompressedMatrixEnum::SparseLower(lower_triplets_matrix.compress())
            }
            SymmetricMatrixEnum::FaerSparse(faer_triplets_matrix) => {
                CompressedMatrixEnum::FaerSparse(faer_triplets_matrix.compress())
            }
            SymmetricMatrixEnum::FaerSparseUpper(faer_upper_triplets_matrix) => {
                CompressedMatrixEnum::FaerSparseUpper(faer_upper_triplets_matrix.compress())
            }
        };
        phase("compress");
        c
    }
}
