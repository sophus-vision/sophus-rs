use nalgebra::{
    DMatrixView,
    DMatrixViewMut,
};

use crate::{
    assert_gt,
    assert_le,
    matrix::PartitionSpec,
};

#[derive(Debug, Clone)]
struct BlockColumnRegion {
    scalar_offset: usize,
    block_dim: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct BlockColumn {
    partitions: Vec<BlockColumnRegion>,
    storage: Vec<f64>,
    max_width: usize,
    active_width: usize,
}

impl BlockColumn {
    pub fn zero(partition_specs: &[PartitionSpec], max_width: usize) -> Self {
        assert_gt!(max_width, 0);

        let mut total_elems = 0usize;
        let mut partitions: Vec<BlockColumnRegion> = Vec::with_capacity(partition_specs.len());

        for ps in partition_specs {
            partitions.push(BlockColumnRegion {
                scalar_offset: total_elems,
                block_dim: ps.block_dimension,
            });
            total_elems += ps.block_dimension * ps.block_count * max_width;
        }

        Self {
            partitions,
            storage: vec![0.0; total_elems],
            max_width,
            active_width: max_width,
        }
    }

    #[inline]
    pub(crate) fn set_active_width(&mut self, active_width: usize) {
        assert_le!(active_width, self.max_width);
        self.active_width = active_width;
    }

    #[inline]
    pub(crate) fn add_block(
        &mut self,
        partition_idx: usize,
        block_index: usize,
        block: DMatrixView<'_, f64>,
    ) {
        debug_assert_eq!(block.ncols(), self.active_width, "row dim mismatch");
        use std::ops::AddAssign;
        self.get_block_mut(partition_idx, block_index)
            .add_assign(&block)
    }

    #[inline]
    pub fn get_block(&self, partition_idx: usize, block_index: usize) -> DMatrixView<'_, f64> {
        let partition = &self.partitions[partition_idx];
        let max_block_area = partition.block_dim * self.max_width;
        let base = partition.scalar_offset + block_index * max_block_area;
        let active_block_area = partition.block_dim * self.active_width;
        let storage = &self.storage[base..base + active_block_area];
        DMatrixView::from_slice(storage, partition.block_dim, self.active_width)
    }

    #[inline]
    pub fn get_block_mut(
        &mut self,
        partition_idx: usize,
        block_index: usize,
    ) -> DMatrixViewMut<'_, f64> {
        let partition = &self.partitions[partition_idx];
        let max_block_area = partition.block_dim * self.max_width;
        let base = partition.scalar_offset + block_index * max_block_area;
        let active_block_area = partition.block_dim * self.active_width;
        let storage = &mut self.storage[base..base + active_block_area];
        DMatrixViewMut::from_slice(storage, partition.block_dim, self.active_width)
    }

    #[inline]
    pub fn zero_block(&mut self, partition_idx: usize, block_index: usize) {
        self.get_block_mut(partition_idx, block_index).fill(0.0);
    }
}
