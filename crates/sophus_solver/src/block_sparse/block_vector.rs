use super::PartitionSpec;

/// A region of a block vector.
#[derive(Debug, Clone)]
struct BlockVectorRegion {
    scalar_offset: usize,
    block_dim: usize,
}

/// Block vector.
///
/// ``ascii
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
/// | Z |
/// | . |
/// | . |
/// | Z |
/// -----
///
/// It ia split into regions and each region is split into a sequence of sub-vectors.
/// Within each region, all sub-vectors have the same dimension. E.g., region 0 contains
/// only A-dimensional sub-vectors, region 1 contains only B-dimensional sub-vectors, etc.
#[derive(Debug, Clone)]
pub struct BlockVector {
    partitions: Vec<BlockVectorRegion>,
    vec: nalgebra::DVector<f64>,
}

impl BlockVector {
    /// Create a block vector filled with zeros.
    ///
    /// The shape of the block vector is determined by the provided partition specs.
    pub fn zero(partition_specs: &[PartitionSpec]) -> Self {
        let mut scalar_offsets = Vec::new();
        let mut offset = 0;

        let mut partitions: Vec<BlockVectorRegion> = Vec::with_capacity(partition_specs.len());

        for partition_specs in partition_specs {
            scalar_offsets.push(offset);
            partitions.push(BlockVectorRegion {
                scalar_offset: offset,
                block_dim: partition_specs.block_dimension,
            });
            offset += partition_specs.block_dimension * partition_specs.block_count;
        }

        Self {
            partitions,
            vec: nalgebra::DVector::zeros(offset),
        }
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

    /// Add a block to the vector.
    pub fn add_block(
        &mut self,
        partition_idx: usize,
        block_index: usize,
        block: &nalgebra::DVectorView<f64>,
    ) {
        let partition = &self.partitions[partition_idx];
        assert_eq!(block.shape().0, partition.block_dim);
        let scalar_offset = partition.scalar_offset + block_index * partition.block_dim;
        use std::ops::AddAssign;

        self.vec
            .rows_mut(scalar_offset, partition.block_dim)
            .add_assign(block);
    }

    /// Get a block
    pub fn get_block(
        &'_ self,
        partition_idx: usize,
        block_index: usize,
    ) -> nalgebra::DVectorView<'_, f64> {
        let partition = &self.partitions[partition_idx];
        let scalar_offset = partition.scalar_offset + block_index * partition.block_dim;
        self.vec.rows(scalar_offset, partition.block_dim)
    }
}
