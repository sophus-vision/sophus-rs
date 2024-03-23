use std::ops::Range;

use crate::sophus_calculus::types::M;

use crate::element::{IsScalar, IsStaticTensor};

pub trait HasShape<const RANK: usize> {
    fn dims(&self) -> TensorShape<RANK>;
    fn num_elements(&self) -> usize {
        self.dims().iter().product()
    }
}

impl HasShape<1> for TensorShape<1> {
    fn dims(&self) -> TensorShape<1> {
        *self
    }
}

impl HasShape<2> for TensorShape<2> {
    fn dims(&self) -> TensorShape<2> {
        *self
    }
}

impl HasShape<3> for TensorShape<3> {
    fn dims(&self) -> TensorShape<3> {
        *self
    }
}

impl HasShape<4> for TensorShape<4> {
    fn dims(&self) -> TensorShape<4> {
        *self
    }
}

pub type TensorShape<const RANK: usize> = [usize; RANK];

///
///
/// Memory layout:     monotonically decreasing
///                    [D1xD2xD3x.., D1xD2x.., ..., 1]
///
/// Data presentation: ROW-first indexing
///                    [D0= , ..., D{i}=ROW, D{i+1}=COL,...]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TensorLayout<const RANK: usize> {
    pub dims: TensorShape<RANK>,
    row_major_strides: [usize; RANK],
}

pub type TensorLayout1 = TensorLayout<1>;
pub type TensorLayout2 = TensorLayout<2>;
pub type TensorLayout3 = TensorLayout<3>;
pub type TensorLayout4 = TensorLayout<4>;

impl TensorLayout<1> {
    pub fn rank(dims: [usize; 1]) -> Self {
        Self {
            dims,
            row_major_strides: [1],
        }
    }
}

impl TensorLayout<2> {
    pub fn rank(dims: [usize; 2]) -> Self {
        Self {
            dims,
            row_major_strides: [dims[1], 1],
        }
    }
}

impl TensorLayout<3> {
    pub fn rank(dims: [usize; 3]) -> Self {
        Self {
            dims,
            row_major_strides: [dims[2] * dims[1], dims[2], 1],
        }
    }
}

impl TensorLayout<4> {
    pub fn rank(dims: [usize; 4]) -> Self {
        Self {
            dims,
            row_major_strides: [dims[3] * dims[2] * dims[1], dims[3] * dims[2], dims[3], 1],
        }
    }
}

impl<const RANK: usize> TensorLayout<RANK> {
    fn null() -> Self {
        let mut strides = [0; RANK];
        strides[RANK - 1] = 1;
        Self {
            dims: [0; RANK],
            row_major_strides: strides,
        }
    }
}

impl<const RANK: usize> Default for TensorLayout<RANK> {
    fn default() -> Self {
        TensorLayout::null()
    }
}

pub trait HasTensorLayout<const RANK: usize>: HasShape<RANK> {
    fn strides(&self) -> [usize; RANK];

    fn padded_area(&self) -> usize {
        self.dims()[0] * self.strides()[0]
    }

    fn num_bytes_of_padded_area<
        Scalar: IsScalar + 'static,
        const BATCH_SIZE: usize,
        const ROWS: usize,
        const COLS: usize,
    >(
        &self,
    ) -> usize {
        Self::padded_area(self) * std::mem::size_of::<Scalar>() * BATCH_SIZE * ROWS * COLS
    }

    fn layout(&self) -> TensorLayout<RANK>;

    fn index(&self, idx_tuple: [usize; RANK]) -> usize;

    fn dim0_index(&self, idx_tuple: [usize; RANK]) -> usize;
    fn dim0_range(&self, idx_tuple: [usize; RANK]) -> std::ops::Range<usize>;

    fn is_empty(&self) -> bool {
        *self.dims().iter().min().unwrap_or(&0) == 0
    }
}

macro_rules! tensor_shape {
    ($drank:literal) => {
        impl HasShape<$drank> for TensorLayout<$drank> {
            fn dims(&self) -> TensorShape<$drank> {
                self.dims
            }
            fn num_elements(&self) -> usize {
                self.dims.num_elements()
            }
        }

        impl HasTensorLayout<$drank> for TensorLayout<$drank> {
            fn layout(&self) -> TensorLayout<$drank> {
                *self
            }

            fn strides(&self) -> [usize; $drank] {
                debug_assert_eq!(self.row_major_strides[$drank - 1], 1);
                self.row_major_strides
            }

            fn index(&self, idx_tuple: [usize; $drank]) -> usize {
                self.dim0_index(idx_tuple) + idx_tuple[$drank - 1]
            }

            fn dim0_index(&self, idx_tuple: [usize; $drank]) -> usize {
                let all_idx_but_last = *array_ref![idx_tuple, 0, $drank - 1];
                let strides = self.strides();
                let all_strides_but_last = *array_ref![strides, 0, $drank - 1];

                all_idx_but_last
                    .iter()
                    .zip(all_strides_but_last.iter())
                    .map(|(x, y)| x * y)
                    .sum()
            }

            fn dim0_range(&self, idx_tuple: [usize; $drank]) -> Range<usize> {
                let idx0 = self.dim0_index(idx_tuple);
                std::ops::Range {
                    start: idx0,
                    end: idx0 + self.dims[0],
                }
            }
        }
    };
}
tensor_shape!(1);
tensor_shape!(2);
tensor_shape!(3);
tensor_shape!(4);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_shape() {
        let rank1_shape = [8];
        assert_eq!(rank1_shape.dims()[0], 8);

        let rank2_shape = [8, 7];
        assert_eq!(rank2_shape.dims()[0], 8);
        assert_eq!(rank2_shape.dims()[1], 7);

        let rank3_shape = [8, 7, 6];
        assert_eq!(rank3_shape.dims()[0], 8);
        assert_eq!(rank3_shape.dims()[1], 7);
        assert_eq!(rank3_shape.dims()[2], 6);

        let rank3_shape = [8, 7, 6, 8];
        assert_eq!(rank3_shape.dims()[0], 8);
        assert_eq!(rank3_shape.dims()[1], 7);
        assert_eq!(rank3_shape.dims()[2], 6);
        assert_eq!(rank3_shape.dims()[3], 8);
    }

    #[test]
    fn tensor_layout() {
        {
            let rank1_shape = [3];

            let rank1_layout = TensorLayout1::rank(rank1_shape);

            let arr = [4, 6, 7];

            assert_eq!(rank1_layout.num_elements(), 3);
            assert_eq!(rank1_layout.num_elements(), rank1_layout.padded_area());
            assert_eq!(arr.len(), rank1_layout.padded_area());

            for i in 0..rank1_shape.dims()[0] {
                assert_eq!(arr[i], arr[rank1_layout.index([i])]);
            }
        }
        {
            let rank2_shape = [2, 3];
            let rank2_layout = TensorLayout2::rank(rank2_shape);
            let arr = [
                4, 6, 7, //
                8, 9, 10,
            ];

            assert_eq!(rank2_layout.num_elements(), 6);
            assert_eq!(rank2_layout.num_elements(), rank2_layout.padded_area());
            assert_eq!(arr.len(), rank2_layout.padded_area());

            let row_arr = [
                [4, 6, 7], //
                [8, 9, 10],
            ];

            for d1 in 0..rank2_shape.dims()[1] {
                for d0 in 0..rank2_shape.dims()[0] {
                    assert_eq!(row_arr[d0][d1], arr[rank2_layout.index([d0, d1])]);
                }
            }
        }

        {
            let rank3_shape = [4, 2, 3];
            let rank3_layout = TensorLayout3::rank(rank3_shape);
            let arr = [
                4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                27, 28,
            ];

            assert_eq!(rank3_layout.num_elements(), 24);
            assert_eq!(rank3_layout.num_elements(), rank3_layout.padded_area());
            assert_eq!(arr.len(), rank3_layout.padded_area());

            let row_col_arr = [
                [
                    [4, 6, 7], //
                    [8, 9, 10],
                ],
                [
                    [11, 12, 13], //
                    [14, 15, 16],
                ],
                [
                    [17, 18, 19], //
                    [20, 21, 22],
                ],
                [
                    [23, 24, 25], //
                    [26, 27, 28],
                ],
            ];

            for d2 in 0..rank3_shape.dims()[2] {
                for d1 in 0..rank3_shape.dims()[1] {
                    for d0 in 0..rank3_shape.dims()[0] {
                        assert_eq!(
                            row_col_arr[d0][d1][d2],
                            arr[rank3_layout.index([d0, d1, d2])]
                        );
                    }
                }
            }
        }
    }
}
