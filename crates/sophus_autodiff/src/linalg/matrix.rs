use core::{
    borrow::Borrow,
    fmt::Debug,
    ops::{
        Add,
        Index,
        IndexMut,
        Mul,
        Neg,
        Sub,
    },
};

use approx::{
    AbsDiffEq,
    RelativeEq,
};

use crate::{
    dual::DualMatrix,
    linalg::{
        MatF64,
        VecF64,
    },
    prelude::*,
};

/// A trait for matrix types whose elements can be either real scalars (`f64`) or dual numbers,
/// in single (BATCH=1) or batch (`portable_simd`) form. It provides core matrix operations,
/// including block composition, element access, multiplication, and transformations.
///
/// # Generic Parameters
/// - `S`: The scalar type implementing [`IsScalar`].
/// - `ROWS`, `COLS`: Dimensions of the matrix.
/// - `BATCH`: Number of lanes if using batch scalars; otherwise 1.
/// - `DM`, `DN`: Dimensions for dual-number derivatives (0 if purely real).
///
/// # Implementations
/// For instance, `MatF64<ROWS, COLS>` implements `IsMatrix<f64, ROWS, COLS, 1, 0, 0>`,
/// while dual or batch variations implement their specialized forms.
pub trait IsMatrix<
    S: IsScalar<BATCH, DM, DN>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>:
    Debug
    + Clone
    + Copy
    + Sized
    + Mul<S::Vector<COLS>, Output = S::Vector<ROWS>>
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + AbsDiffEq<Epsilon = f64>
    + RelativeEq<Epsilon = f64>
{
    /// Combines two matrices side-by-side along columns into one.
    ///
    /// ```text
    /// [left_col | righ_col]
    /// ```
    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: S::Matrix<ROWS, C0>,
        righ_col: S::Matrix<ROWS, C1>,
    ) -> Self;

    /// Stacks two matrices vertically into one.
    ///
    /// ```text
    /// [top_row]
    /// [-------]
    /// [bot_row]
    /// ```
    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: S::Matrix<R0, COLS>,
        bot_row: S::Matrix<R1, COLS>,
    ) -> Self;

    /// Creates a new `[ROWS, COLS]` matrix by stitching together a 2×2 grid of blocks.
    ///
    /// ```text
    /// [top_left | top_right]
    /// [---------|----------]
    /// [bot_left | bot_right]
    /// ```
    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (S::Matrix<R0, C0>, S::Matrix<R0, C1>),
        bot_row: (S::Matrix<R1, C0>, S::Matrix<R1, C1>),
    ) -> Self;

    /// Constructs a new matrix from a 2D array of scalar elements.
    fn from_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[S; COLS]; ROWS]>;

    /// Constructs a new matrix with all elements set to the given `f64`.
    ///
    /// If `S` is a dual scalar, the dual part is zeroed out.
    /// If `S` is a batch scalar, all lanes are set to `val`.
    fn from_f64(val: f64) -> Self;

    /// Constructs a matrix from a 2D array of `f64` values, zeroing out any dual part.
    fn from_f64_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[f64; COLS]; ROWS]>;

    /// Constructs a matrix from a 2D array of real scalars `S::RealScalar`,
    /// zeroing out derivative as needed.
    fn from_real_scalar_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[S::RealScalar; COLS]; ROWS]>;

    /// Constructs a matrix from a real matrix type (e.g. `MatF64<ROWS, COLS>`).
    fn from_real_matrix<A>(val: A) -> Self
    where
        A: Borrow<S::RealMatrix<ROWS, COLS>>;

    /// Constructs a new matrix by replicating a single scalar value `val` across all elements.
    ///
    /// If `val` is dual, it preserves its dual parts. If it's real, the matrix is purely real.
    fn from_scalar<Q>(val: Q) -> Self
    where
        Q: Borrow<S>;

    /// Returns the `c`-th column of the matrix as a `[ROWS]` vector.
    fn get_col_vec(&self, c: usize) -> S::Vector<ROWS>;

    /// Returns the `r`-th row of the matrix as a `[COLS]` vector.
    fn get_row_vec(&self, r: usize) -> S::Vector<COLS>;

    /// Accesses the element in row `idx[0]`, column `idx[1]`.
    fn elem(&self, idx: [usize; 2]) -> S;

    /// Returns a mutable reference to the element in row `idx[0]`, column `idx[1]`.
    fn elem_mut(&mut self, idx: [usize; 2]) -> &mut S;

    /// Extracts a fixed submatrix of size `[R, C]`, starting at `(start_r, start_c)`.
    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> S::Matrix<R, C>;

    /// Returns the identity matrix of size `[ROWS, COLS]`. Usually only valid if `ROWS == COLS`.
    fn identity() -> Self;

    /// Matrix multiplication with another `[COLS, C2]` matrix, returning `[ROWS, C2]`.
    fn mat_mul<const C2: usize, M>(&self, other: M) -> S::Matrix<ROWS, C2>
    where
        M: Borrow<S::Matrix<COLS, C2>>;

    /// Creates a matrix of all ones, i.e., with numeric value 1.0 in every element.
    fn ones() -> Self {
        Self::from_f64(1.0)
    }

    /// Returns the underlying real matrix form, discarding derivative or batch info if present.
    fn real_matrix(&self) -> S::RealMatrix<ROWS, COLS>;

    /// Scales the matrix by a scalar `v` (elementwise `*v`).
    fn scaled<Q>(&self, v: Q) -> Self
    where
        Q: Borrow<S>;

    /// Lane-wise select operation: picks from `self` where `mask` is true, or from `other`
    /// otherwise.
    fn select<Q>(&self, mask: &S::Mask, other: Q) -> Self
    where
        Q: Borrow<Self>;

    /// Sets the entire column `c` to the vector `v`.
    fn set_col_vec(&mut self, c: usize, v: S::Vector<ROWS>);

    /// Returns the transpose of this matrix, flipping rows and columns.
    fn transposed(&self) -> S::Matrix<COLS, ROWS>;

    /// Converts this matrix into a dual matrix type with zero derivative if originally real;
    /// if already dual, merges or preserves derivative structure as appropriate.
    fn to_dual_const<const M: usize, const N: usize>(&self) -> S::DualMatrix<ROWS, COLS, M, N>;

    /// Returns a matrix of all zeros.
    fn zeros() -> Self {
        Self::from_f64(0.0)
    }
}

/// A specialized trait for real (non-dual) matrices, typically parameterized by `f64` or a batch
/// real type. Enables direct indexing `(r, c)` in addition to the other matrix operations.
///
/// # Const Parameters
/// - `S`: The real scalar type implementing [`IsRealScalar`] and [`IsScalar<..., 0, 0>`].
/// - `ROWS`, `COLS`: Dimensions of the matrix.
/// - `BATCH`: Number of lanes (1 if single-lane).
pub trait IsRealMatrix<
    S: IsRealScalar<BATCH> + IsScalar<BATCH, 0, 0>,
    const ROWS: usize,
    const COLS: usize,
    const BATCH: usize,
>:
    IsMatrix<S, ROWS, COLS, BATCH, 0, 0>
    + Index<(usize, usize), Output = S>
    + IndexMut<(usize, usize), Output = S>
    + Copy
{
}

/// A trait for single-lane matrices (not batch) whose scalar is single-lane as well.
/// This typically means either pure real matrices (`f64`) or dual matrices with `BATCH=1`.
pub trait IsSingleMatrix<
    S: IsSingleScalar<DM, DN>,
    const ROWS: usize,
    const COLS: usize,
    const DM: usize,
    const DN: usize,
>:
    IsMatrix<S, ROWS, COLS, 1, DM, DN> + Mul<S::SingleVector<COLS>, Output = S::SingleVector<ROWS>>
{
    /// Extracts the real (f64-based) version of this matrix, ignoring dual derivative info if
    /// present.
    fn single_real_matrix(&self) -> MatF64<ROWS, COLS>;
}

impl<const ROWS: usize, const COLS: usize> IsSingleMatrix<f64, ROWS, COLS, 0, 0>
    for MatF64<ROWS, COLS>
{
    fn single_real_matrix(&self) -> MatF64<ROWS, COLS> {
        *self
    }
}

impl<const ROWS: usize, const COLS: usize> IsRealMatrix<f64, ROWS, COLS, 1> for MatF64<ROWS, COLS> {}

impl<const ROWS: usize, const COLS: usize> IsMatrix<f64, ROWS, COLS, 1, 0, 0>
    for MatF64<ROWS, COLS>
{
    fn from_real_matrix<A>(val: A) -> Self
    where
        A: Borrow<MatF64<ROWS, COLS>>,
    {
        *val.borrow()
    }

    fn from_scalar<S>(val: S) -> Self
    where
        S: Borrow<f64>,
    {
        Self::from_f64(*val.borrow())
    }

    fn from_array2<A>(vals: A) -> MatF64<ROWS, COLS>
    where
        A: Borrow<[[f64; COLS]; ROWS]>,
    {
        let vals = vals.borrow();
        let mut m = MatF64::<ROWS, COLS>::zeros();

        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c];
            }
        }
        m
    }

    fn from_real_scalar_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[f64; COLS]; ROWS]>,
    {
        let vals = vals.borrow();
        let mut m = MatF64::<ROWS, COLS>::zeros();
        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c];
            }
        }
        m
    }

    fn elem(&self, idx: [usize; 2]) -> f64 {
        self[(idx[0], idx[1])]
    }

    fn elem_mut(&mut self, idx: [usize; 2]) -> &mut f64 {
        &mut self[(idx[0], idx[1])]
    }

    fn identity() -> Self {
        Self::identity()
    }

    fn real_matrix(&self) -> Self {
        *self
    }

    fn mat_mul<const C2: usize, M>(&self, other: M) -> MatF64<ROWS, C2>
    where
        M: Borrow<MatF64<COLS, C2>>,
    {
        self * other.borrow()
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: MatF64<R0, COLS>,
        bot_row: MatF64<R1, COLS>,
    ) -> Self {
        assert_eq!(ROWS, R0 + R1);
        let mut m = Self::zeros();

        m.fixed_view_mut::<R0, COLS>(0, 0).copy_from(&top_row);
        m.fixed_view_mut::<R1, COLS>(R0, 0).copy_from(&bot_row);
        m
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (MatF64<R0, C0>, MatF64<R0, C1>),
        bot_row: (MatF64<R1, C0>, MatF64<R1, C1>),
    ) -> Self {
        assert_eq!(ROWS, R0 + R1);
        assert_eq!(COLS, C0 + C1);
        let mut m = Self::zeros();

        m.fixed_view_mut::<R0, C0>(0, 0).copy_from(&top_row.0);
        m.fixed_view_mut::<R0, C1>(0, C0).copy_from(&top_row.1);

        m.fixed_view_mut::<R1, C0>(R0, 0).copy_from(&bot_row.0);
        m.fixed_view_mut::<R1, C1>(R0, C0).copy_from(&bot_row.1);
        m
    }

    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: MatF64<ROWS, C0>,
        righ_col: MatF64<ROWS, C1>,
    ) -> Self {
        assert_eq!(COLS, C0 + C1);
        let mut m = Self::zeros();

        m.fixed_view_mut::<ROWS, C0>(0, 0).copy_from(&left_col);
        m.fixed_view_mut::<ROWS, C1>(0, C0).copy_from(&righ_col);

        m
    }

    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> MatF64<R, C> {
        self.fixed_view::<R, C>(start_r, start_c).into()
    }

    fn get_col_vec(&self, c: usize) -> VecF64<ROWS> {
        self.fixed_view::<ROWS, 1>(0, c).into()
    }

    fn get_row_vec(&self, r: usize) -> VecF64<COLS> {
        self.fixed_view::<1, COLS>(r, 0).transpose()
    }

    fn scaled<Q>(&self, v: Q) -> Self
    where
        Q: Borrow<f64>,
    {
        *self * *v.borrow()
    }

    fn from_f64_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[f64; COLS]; ROWS]>,
    {
        let vals = vals.borrow();
        let mut m = MatF64::<ROWS, COLS>::zeros();
        for c in 0..COLS {
            for r in 0..ROWS {
                m[(r, c)] = vals[r][c];
            }
        }
        m
    }

    fn from_f64(val: f64) -> Self {
        MatF64::<ROWS, COLS>::from_element(val)
    }

    fn set_col_vec(&mut self, c: usize, v: VecF64<ROWS>) {
        self.fixed_columns_mut::<1>(c).copy_from(&v);
    }

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <f64 as IsScalar<1, 0, 0>>::DualMatrix<ROWS, COLS, M, N> {
        DualMatrix::from_real_matrix(self)
    }

    fn select<Q>(&self, mask: &bool, other: Q) -> Self
    where
        Q: Borrow<Self>,
    {
        if *mask {
            *self
        } else {
            *other.borrow()
        }
    }

    fn transposed(&self) -> MatF64<COLS, ROWS> {
        self.transpose()
    }
}
