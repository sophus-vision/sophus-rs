use dfdx_core::{shapes::*, tensor::*, tensor_ops::*};

use super::batch_types::*;

pub fn make_blockvec2<
    const BATCH: usize,
    const ROWS: usize,
    const TOP_ROWS: usize,
    const BOTTOM_ROWS: usize,
    GenTape: SophusTape + Clone,
>(
    top_rows: GenV<BATCH, TOP_ROWS, GenTape>,
    bottom_rows: GenV<BATCH, BOTTOM_ROWS, GenTape>,
) -> GenV<BATCH, ROWS, GenTape> {
    assert_eq!(ROWS, TOP_ROWS + BOTTOM_ROWS);
    let bottom_rows: Tensor<(Const<BATCH>, usize), _, _, _> = bottom_rows.realize();
    (top_rows, bottom_rows).concat_along(Axis::<1>).realize()
}

pub fn make_blockvec3<
    const BATCH: usize,
    const ROWS: usize,
    const ROWS0: usize,
    const ROWS1: usize,
    const ROWS2: usize,
    GenTape: SophusTape + Clone,
>(
    vec0: GenV<BATCH, ROWS0, GenTape>,
    vec1: GenV<BATCH, ROWS1, GenTape>,
    vec2: GenV<BATCH, ROWS2, GenTape>,
) -> GenV<BATCH, ROWS, GenTape> {
    assert_eq!(ROWS, ROWS0 + ROWS1 + ROWS2);
    let vec1: Tensor<(Const<BATCH>, usize), _, _, _> = vec1.realize();
    let vec2: Tensor<(Const<BATCH>, usize), _, _, _> = vec2.realize();

    ((vec0, vec1).concat_along(Axis::<1>), vec2)
        .concat_along(Axis::<1>)
        .realize()
}

pub fn make_blockvec4<
    const BATCH: usize,
    const ROWS: usize,
    const ROWS0: usize,
    const ROWS1: usize,
    const ROWS2: usize,
    const ROWS3: usize,
    GenTape: SophusTape + Clone,
>(
    vec0: GenV<BATCH, ROWS0, GenTape>,
    vec1: GenV<BATCH, ROWS1, GenTape>,
    vec2: GenV<BATCH, ROWS2, GenTape>,
    vec3: GenV<BATCH, ROWS3, GenTape>,
) -> GenV<BATCH, ROWS, GenTape> {
    assert_eq!(ROWS, ROWS0 + ROWS1 + ROWS2 + ROWS3);
    let vec1: Tensor<(Const<BATCH>, usize), _, _, _> = vec1.realize();
    let vec2: Tensor<(Const<BATCH>, usize), _, _, _> = vec2.realize();
    let vec3: Tensor<(Const<BATCH>, usize), _, _, _> = vec3.realize();

    (
        ((vec0, vec1).concat_along(Axis::<1>), vec2).concat_along(Axis::<1>),
        vec3,
    )
        .concat_along(Axis::<1>)
        .realize()
}

pub fn make_vec2<const BATCH: usize, GenTape: SophusTape + Clone>(
    top: GenS<BATCH, GenTape>,
    bottom: GenS<BATCH, GenTape>,
) -> GenV<BATCH, 2, GenTape> {
    make_blockvec2::<BATCH, 2, 1, 1, GenTape>(top.reshape(), bottom.reshape())
}

pub fn make_vec3<const BATCH: usize, GenTape: SophusTape + Clone>(
    top: GenS<BATCH, GenTape>,
    middle: GenS<BATCH, GenTape>,
    bottom: GenS<BATCH, GenTape>,
) -> GenV<BATCH, 3, GenTape> {
    make_blockvec3::<BATCH, 3, 1, 1, 1, GenTape>(top.reshape(), middle.reshape(), bottom.reshape())
}

// Create a matrix from a left matrix and a right matrix
pub fn make_2blockcol_mat<
    const BATCH: usize,
    const ROWS: usize,
    const COLS: usize,
    const LEFT_COLS: usize,
    const RIGHT_COLS: usize,
    LeftTape: SophusTape + Merge<RightTape>,
    RightTape: SophusTape,
>(
    left_mat: GenM<BATCH, ROWS, LEFT_COLS, LeftTape>,
    right_mat: GenM<BATCH, ROWS, RIGHT_COLS, RightTape>,
) -> GenM<BATCH, ROWS, COLS, LeftTape> {
    assert_eq!(COLS, LEFT_COLS + RIGHT_COLS);
    let right_mat: Tensor<(Const<BATCH>, Const<ROWS>, usize), _, _, _> = right_mat.realize();
    (left_mat, right_mat).concat_along(Axis::<2>).realize()
}

// Create a matrix from a mat0, mat1, mat2 (left to right)
pub fn make_3blockcol_mat<
    const BATCH: usize,
    const ROWS: usize,
    const COLS: usize,
    const COLS0: usize,
    const COLS1: usize,
    const COLS2: usize,
    GenTape: SophusTape + Clone,
>(
    mat0: GenM<BATCH, ROWS, COLS0, GenTape>,
    mat1: GenM<BATCH, ROWS, COLS1, GenTape>,
    mat2: GenM<BATCH, ROWS, COLS2, GenTape>,
) -> GenM<BATCH, ROWS, COLS, GenTape> {
    assert_eq!(COLS, COLS0 + COLS1 + COLS2);
    let mat1: Tensor<(Const<BATCH>, Const<ROWS>, usize), _, _, _> = mat1.realize();
    let mat2: Tensor<(Const<BATCH>, Const<ROWS>, usize), _, _, _> = mat2.realize();

    ((mat0, mat1).concat_along(Axis::<2>), mat2)
        .concat_along(Axis::<2>)
        .realize()
}

// Create a matrix from a mat0, mat1, mat2, mat3 (left to right)
pub fn make_4blockcol_mat<
    const BATCH: usize,
    const ROWS: usize,
    const COLS: usize,
    const COLS0: usize,
    const COLS1: usize,
    const COLS2: usize,
    const COLS3: usize,
    GenTape: SophusTape + Clone,
>(
    mat0: GenM<BATCH, ROWS, COLS0, GenTape>,
    mat1: GenM<BATCH, ROWS, COLS1, GenTape>,
    mat2: GenM<BATCH, ROWS, COLS2, GenTape>,
    mat3: GenM<BATCH, ROWS, COLS3, GenTape>,
) -> GenM<BATCH, ROWS, COLS, GenTape> {
    assert_eq!(COLS, COLS0 + COLS1 + COLS2 + COLS3);

    let mat1: Tensor<(Const<BATCH>, Const<ROWS>, usize), _, _, _> = mat1.realize();
    let mat2: Tensor<(Const<BATCH>, Const<ROWS>, usize), _, _, _> = mat2.realize();
    let mat3: Tensor<(Const<BATCH>, Const<ROWS>, usize), _, _, _> = mat3.realize();

    (
        ((mat0, mat1).concat_along(Axis::<2>), mat2).concat_along(Axis::<2>),
        mat3,
    )
        .concat_along(Axis::<2>)
        .realize()
}

// Create a matrix from two (column) vectors stacked left to right
pub fn make_2colvec_mat<
    const BATCH: usize,
    const ROWS: usize,
    LeftTape: SophusTape + Merge<RightTape>,
    RightTape: SophusTape,
>(
    left_vec: GenV<BATCH, ROWS, LeftTape>,
    right_vec: GenV<BATCH, ROWS, RightTape>,
) -> GenM<BATCH, ROWS, 2, LeftTape> {
    let left_vec: GenM<BATCH, ROWS, 1, LeftTape> = left_vec.reshape();
    let right_vec: GenM<BATCH, ROWS, 1, RightTape> = right_vec.reshape();
    make_2blockcol_mat(left_vec, right_vec)
}

// Create a matrix from two (column) vectors stacked left to right
pub fn make_3colvec_mat<const BATCH: usize, const ROWS: usize, Tape: SophusTape>(
    vec0: GenV<BATCH, ROWS, Tape>,
    vec1: GenV<BATCH, ROWS, Tape>,
    vec2: GenV<BATCH, ROWS, Tape>,
) -> GenM<BATCH, ROWS, 3, Tape> {
    let vec0: GenM<BATCH, ROWS, 1, Tape> = vec0.reshape();
    let vec1: GenM<BATCH, ROWS, 1, Tape> = vec1.reshape();
    let vec2: GenM<BATCH, ROWS, 1, Tape> = vec2.reshape();

    make_3blockcol_mat(vec0, vec1, vec2)
}

// Creates a 1x2 matrix / row vector from two scalars
pub fn make_2col_mat<
    const BATCH: usize,
    LeftTape: SophusTape + Merge<RightTape>,
    RightTape: SophusTape,
>(
    left_mat: GenS<BATCH, LeftTape>,
    right_mat: GenS<BATCH, RightTape>,
) -> GenM<BATCH, 1, 2, LeftTape> {
    make_2blockcol_mat::<BATCH, 1, 2, 1, 1, LeftTape, RightTape>(
        left_mat.reshape(),
        right_mat.reshape(),
    )
}

// Creates a 1x3 matrix / row vector from three scalars
pub fn make_3col_mat<const BATCH: usize, GenTape: SophusTape + Clone>(
    left_mat: GenS<BATCH, GenTape>,
    middle_cols: GenS<BATCH, GenTape>,
    right_mat: GenS<BATCH, GenTape>,
) -> GenM<BATCH, 1, 3, GenTape> {
    make_3blockcol_mat::<BATCH, 1, 3, 1, 1, 1, GenTape>(
        left_mat.reshape(),
        middle_cols.reshape(),
        right_mat.reshape(),
    )
}

// used
pub fn make_2rowblock_mat<
    const BATCH: usize,
    const ROWS: usize,
    const COLS: usize,
    const TOP_ROWS: usize,
    const BOTTOM_ROWS: usize,
    TopTape: SophusTape + Merge<BottomTape>,
    BottomTape: SophusTape + Clone,
>(
    top_rows: GenM<BATCH, TOP_ROWS, COLS, TopTape>,
    bottom_rows: GenM<BATCH, BOTTOM_ROWS, COLS, BottomTape>,
) -> GenM<BATCH, ROWS, COLS, TopTape> {
    assert_eq!(
        ROWS,
        TOP_ROWS + BOTTOM_ROWS,
        "ROWS ({}) != TOP_ROWS ({}) + BOTTOM_ROWS ({})",
        ROWS,
        TOP_ROWS,
        BOTTOM_ROWS
    );
    let bottom_rows: Tensor<(Const<BATCH>, usize, Const<COLS>), _, _, _> = bottom_rows.realize();
    (top_rows, bottom_rows).concat_along(Axis::<1>).realize()
}

pub fn make_3rowblock_mat<
    const BATCH: usize,
    const ROWS: usize,
    const COLS: usize,
    const ROWS0: usize,
    const ROWS1: usize,
    const ROWS2: usize,
    GenTape: SophusTape + Clone,
>(
    row0: GenM<BATCH, ROWS0, COLS, GenTape>,
    row1: GenM<BATCH, ROWS1, COLS, GenTape>,
    row2: GenM<BATCH, ROWS2, COLS, GenTape>,
) -> GenM<BATCH, ROWS, COLS, GenTape> {
    assert_eq!(ROWS, ROWS0 + ROWS1 + ROWS2);
    let row1: Tensor<(Const<BATCH>, usize, Const<COLS>), _, _, _> = row1.realize();
    let row2: Tensor<(Const<BATCH>, usize, Const<COLS>), _, _, _> = row2.realize();
    ((row0, row1).concat_along(Axis::<1>), row2)
        .concat_along(Axis::<1>)
        .realize()
}

pub fn make_4rowblock_mat<
    const BATCH: usize,
    const ROWS: usize,
    const COLS: usize,
    const ROWS0: usize,
    const ROWS1: usize,
    const ROWS2: usize,
    const ROWS3: usize,
    GenTape: SophusTape + Clone,
>(
    row0: GenM<BATCH, ROWS0, COLS, GenTape>,
    row1: GenM<BATCH, ROWS1, COLS, GenTape>,
    row2: GenM<BATCH, ROWS2, COLS, GenTape>,
    row3: GenM<BATCH, ROWS3, COLS, GenTape>,
) -> GenM<BATCH, ROWS, COLS, GenTape> {
    assert_eq!(ROWS, ROWS0 + ROWS1 + ROWS2 + ROWS3);
    let row0: Tensor<(Const<BATCH>, usize, Const<COLS>), _, _, _> = row0.realize();
    let row1: Tensor<(Const<BATCH>, usize, Const<COLS>), _, _, _> = row1.realize();
    let row2: Tensor<(Const<BATCH>, usize, Const<COLS>), _, _, _> = row2.realize();
    let row3: Tensor<(Const<BATCH>, usize, Const<COLS>), _, _, _> = row3.realize();

    (
        ((row0, row1).concat_along(Axis::<1>), row2).concat_along(Axis::<1>),
        row3,
    )
        .concat_along(Axis::<1>)
        .realize()
}
