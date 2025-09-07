use crate::{
    LinearSolverEnum,
    matrix::{
        CompressibleMatrixEnum,
        DenseSymmetricMatrixBuilder,
        FaerTripletMatrix,
        FaerUpperTripletMatrix,
        IsCompressibleMatrix,
        LowerBlockSparseMatrixBuilder,
        PartitionSpec,
        SparseSymmetricMatrixBuilder,
    },
};

/// Symmetric matrix builder trait.
pub trait IsSymmetricMatrixBuilder {
    /// mat
    type Matrix: IsCompressibleMatrix;

    /// Create a symmetric matrix "filled" with zeros.
    ///
    /// The shape of the matrix is determined by the partition specs.
    fn zero(partitions: &[PartitionSpec]) -> Self;

    /// Scalar dimension of the matrix.
    fn scalar_dimension(&self) -> usize;

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block.
    ///
    /// Only lower triangular blocks are accepted.
    ///
    /// In release mode, upper triangular blocks are ignored. In debug mode,
    /// this function will panic if the block is added to the upper triangular part.
    fn add_lower_block(
        &mut self,
        region_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &nalgebra::DMatrixView<f64>,
    );

    /// Return built matrix.
    fn build(self) -> Self::Matrix;
}

#[derive(Debug, Clone)]
/// Symmetric matrix builder enum.
pub enum SymmetricMatrixBuilderEnum {
    /// Builder for dense symmetric matrix.
    Dense(DenseSymmetricMatrixBuilder),
    /// Builder for sparse lower matrix.
    SparseLower(SparseSymmetricMatrixBuilder),
    /// Builder for sparse lower matrix.
    BlockSparseLower(LowerBlockSparseMatrixBuilder),
    /// Builder for sparse symmetric matrix to interact with the faer crate.
    FaerSparse(SparseSymmetricMatrixBuilder),
    /// Builder for sparse upper triangular matrix to interact with the faer crate.
    FaerSparseUpper(SparseSymmetricMatrixBuilder),
}

impl SymmetricMatrixBuilderEnum {
    /// Create a symmetric matrix "filled" with zeros - to be used with given solver.
    ///
    /// The shape of the matrix is determined by the partition specs.
    pub fn zero(solver: LinearSolverEnum, partitions: &[PartitionSpec]) -> Self {
        match solver {
            LinearSolverEnum::DenseLdlt(_) | LinearSolverEnum::DenseLu(_) => {
                SymmetricMatrixBuilderEnum::Dense(DenseSymmetricMatrixBuilder::zero(partitions))
            }
            LinearSolverEnum::FaerSparseQr(_) => SymmetricMatrixBuilderEnum::FaerSparse(
                SparseSymmetricMatrixBuilder::zero(partitions),
            ),
            LinearSolverEnum::FaerSparseLu(_) => SymmetricMatrixBuilderEnum::FaerSparse(
                SparseSymmetricMatrixBuilder::zero(partitions),
            ),
            LinearSolverEnum::FaerSparseLdlt(_) => SymmetricMatrixBuilderEnum::FaerSparseUpper(
                SparseSymmetricMatrixBuilder::zero(partitions),
            ),
            LinearSolverEnum::SparseLdlt(_) => SymmetricMatrixBuilderEnum::SparseLower(
                SparseSymmetricMatrixBuilder::zero(partitions),
            ),
            LinearSolverEnum::BlockSparseLdlt(_) => SymmetricMatrixBuilderEnum::BlockSparseLower(
                LowerBlockSparseMatrixBuilder::zero(partitions),
            ),
        }
    }

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block.
    ///
    /// Only lower triangular blocks are accepted.
    ///
    /// In release mode, upper triangular blocks are ignored. In debug mode,
    /// this function will panic if the block is added to the upper triangular part.
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
            SymmetricMatrixBuilderEnum::FaerSparse(sparse_symmetric_matrix_builder) => {
                sparse_symmetric_matrix_builder.add_lower_block(region_idx, block_index, block);
            }
            SymmetricMatrixBuilderEnum::FaerSparseUpper(sparse_symmetric_matrix_builder) => {
                sparse_symmetric_matrix_builder.add_lower_block(region_idx, block_index, block);
            }
            SymmetricMatrixBuilderEnum::SparseLower(sparse_symmetric_matrix_builder) => {
                sparse_symmetric_matrix_builder.add_lower_block(region_idx, block_index, block)
            }
            SymmetricMatrixBuilderEnum::BlockSparseLower(sparse_symmetric_matrix_builder) => {
                sparse_symmetric_matrix_builder.add_lower_block(region_idx, block_index, block)
            }
        }
    }

    /// Return built matrix.
    pub fn build(self) -> CompressibleMatrixEnum {
        match self {
            SymmetricMatrixBuilderEnum::Dense(dense_symmetric_matrix_builder) => {
                CompressibleMatrixEnum::Dense(dense_symmetric_matrix_builder.build())
            }
            SymmetricMatrixBuilderEnum::SparseLower(sparse_symmetric_matrix_builder) => {
                CompressibleMatrixEnum::SparseLower(sparse_symmetric_matrix_builder.build())
            }
            SymmetricMatrixBuilderEnum::FaerSparse(sparse_symmetric_matrix_builder) => {
                CompressibleMatrixEnum::FaerSparse(FaerTripletMatrix::from_lower(
                    &sparse_symmetric_matrix_builder.build(),
                ))
            }
            SymmetricMatrixBuilderEnum::FaerSparseUpper(sparse_symmetric_matrix_builder) => {
                CompressibleMatrixEnum::FaerSparseUpper(FaerUpperTripletMatrix::from_lower(
                    &sparse_symmetric_matrix_builder.build(),
                ))
            }
            SymmetricMatrixBuilderEnum::BlockSparseLower(sparse_symmetric_matrix_builder) => {
                CompressibleMatrixEnum::BlockSparseLower(sparse_symmetric_matrix_builder.build())
            }
        }
    }
}
