use crate::matrix::{
    PartitionBlockIndex,
    PartitionSpec,
};

/// A region of a block vector.
#[derive(Debug, Clone)]
struct BlockVectorRegion {
    scalar_offset: usize,
    block_dim: usize,
}

/// Block vector.
///
/// ```ascii
/// -----
/// | A |
/// | . |
/// | . |
/// | A |
/// -----
/// | B |
/// | . |
/// | . |
/// | B |
/// -----
/// |   |
/// | * |
/// | * |
/// |   |
/// -----
/// ```
///
/// It is split into partitions and each partition is split into a sequence of sub-vectors.
/// Within each partition, all sub-vectors have the same dimension. E.g., partition 0 contains
/// only A-dimensional sub-vectors, partition 1 contains only B-dimensional sub-vectors, etc.
#[derive(Debug, Clone)]
pub struct BlockVector {
    partitions: Vec<BlockVectorRegion>,
    vec: nalgebra::DVector<f64>,
}

impl BlockVector {
    /// Create a block vector filled with zeros.
    ///
    /// The size of the block vector is determined by the provided partition specs.
    pub fn zero(partition_specs: &[PartitionSpec]) -> Self {
        let mut scalar_offsets = Vec::new();
        let mut offset = 0;

        let mut partitions: Vec<BlockVectorRegion> = Vec::with_capacity(partition_specs.len());

        for partition_specs in partition_specs {
            scalar_offsets.push(offset);
            partitions.push(BlockVectorRegion {
                scalar_offset: offset,
                block_dim: partition_specs.block_dim,
            });
            offset += partition_specs.block_dim * partition_specs.block_count;
        }

        Self {
            partitions,
            vec: nalgebra::DVector::zeros(offset),
        }
    }

    /// Number of partitions.
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// Fill vector with a constant value.
    pub fn fill(&mut self, value: f64) {
        self.vec.fill(value);
    }

    /// Move out of self and return scalar vector representation.
    pub fn into_scalar_vector(self) -> nalgebra::DVector<f64> {
        self.vec
    }

    /// Get the scalar vector representation.
    pub fn scalar_vector(&self) -> &nalgebra::DVector<f64> {
        &self.vec
    }

    /// Get the scalar vector representation.
    pub fn scalar_vector_mut(&mut self) -> &mut nalgebra::DVector<f64> {
        &mut self.vec
    }

    /// Add a block to the vector. This is a `+=` operation.
    pub fn add_block(&mut self, idx: PartitionBlockIndex, block: &nalgebra::DVectorView<f64>) {
        let partition = &self.partitions[idx.partition];
        assert_eq!(block.shape().0, partition.block_dim);
        let scalar_offset = partition.scalar_offset + idx.block * partition.block_dim;
        use std::ops::AddAssign;

        self.vec
            .rows_mut(scalar_offset, partition.block_dim)
            .add_assign(block);
    }

    /// Add `scale * block` to this vector at index `idx`. No temporary allocation.
    #[inline]
    pub fn axpy_block(
        &mut self,
        idx: PartitionBlockIndex,
        block: &nalgebra::DVectorView<f64>,
        scale: f64,
    ) {
        let partition = &self.partitions[idx.partition];
        let scalar_offset = partition.scalar_offset + idx.block * partition.block_dim;
        self.vec
            .rows_mut(scalar_offset, partition.block_dim)
            .axpy(scale, block, 1.0);
    }

    /// Get a view of a specific block.
    pub fn get_block(&'_ self, idx: PartitionBlockIndex) -> nalgebra::DVectorView<'_, f64> {
        let partition = &self.partitions[idx.partition];
        let scalar_offset = partition.scalar_offset + idx.block * partition.block_dim;
        self.vec.rows(scalar_offset, partition.block_dim)
    }

    /// Get a mutable view of a specific block.
    pub fn get_block_mut(
        &'_ mut self,
        idx: PartitionBlockIndex,
    ) -> nalgebra::DVectorViewMut<'_, f64> {
        let partition = &self.partitions[idx.partition];
        let scalar_offset = partition.scalar_offset + idx.block * partition.block_dim;
        self.vec.rows_mut(scalar_offset, partition.block_dim)
    }
}
