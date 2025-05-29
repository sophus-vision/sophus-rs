use core::{
    borrow::Borrow,
    fmt::Debug,
    ops::{Add, Index, IndexMut, Neg, Sub},
};

use approx::{AbsDiffEq, RelativeEq};

use crate::{
    dual::DualVector,
    linalg::{MatF64, VecF64},
    prelude::*,
};

/// A trait representing a fixed-size vector whose elements can be real scalars (`f64`)
/// or dual numbers. This trait provides core vector operations such as dot products,
/// subvector extraction, scaling, normalization, and more. It also supports both
/// real-only and dual/simd-based implementations via generics.
///
/// # Generic parameters
/// - `S`: The scalar type, which might be `f64` or a dual number implementing [`IsScalar`].
/// - `ROWS`: The number of rows (dimension) of the vector.
/// - `BATCH`: If using batch/simd, the number of lanes; otherwise typically 1.
/// - `DM`, `DN`: Shape parameters for dual-number derivatives (0 for real scalars).
pub trait IsVector<
    S: IsScalar<BATCH, DM, DN>,
    const ROWS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>:
    Clone
    + Copy
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Debug
    + AbsDiffEq<Epsilon = f64>
    + RelativeEq<Epsilon = f64>
{
    /// Creates a new vector by concatenating two smaller vectors `top_row` and `bot_row`.
    /// The resulting vector has dimension `R0 + R1`.
    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: S::Vector<R0>,
        bot_row: S::Vector<R1>,
    ) -> Self;

    /// Creates a new vector by concatenating three vectors `top_row`, 'mid_row' and `bot_row`.
    /// The resulting vector has dimension `R0 + R1 + R2`.
    fn block_vec3<const R0: usize, const R1: usize, const R2: usize>(
        top_row: S::Vector<R0>,
        mid_row: S::Vector<R1>,
        bot_row: S::Vector<R2>,
    ) -> Self;

    /// Computes the dot product (inner product) with another vector.
    fn dot<V>(&self, rhs: V) -> S
    where
        V: Borrow<Self>;

    /// Creates a vector from an array of elements.
    fn from_array<A>(vals: A) -> Self
    where
        A: Borrow<[S; ROWS]>;

    /// Creates a constant (non-dual) vector from an array of real scalars.
    fn from_real_array<A>(vals: A) -> Self
    where
        A: Borrow<[S::RealScalar; ROWS]>;

    /// Creates a constant vector from a real vector type (e.g. `VecF64<ROWS>`).
    fn from_real_vector<A>(val: A) -> Self
    where
        A: Borrow<S::RealVector<ROWS>>;

    /// Creates a vector where all elements are set to the given `f64`.
    fn from_f64(val: f64) -> Self;

    /// Creates a vector from an array of `f64`.
    ///
    /// Similar to [`from_real_array`](Self::from_real_array) but may serve a
    /// different purpose in some dual or batch contexts.
    fn from_f64_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>;

    /// Returns the `idx`-th element of the vector.
    fn elem(&self, idx: usize) -> S;

    /// Returns a mutable reference to the `idx`-th element of the vector.
    fn elem_mut(&mut self, idx: usize) -> &mut S;

    /// Extracts a subvector of length `R`, starting from row index `start_r`.
    ///
    /// # Panics
    /// May panic if `start_r + R` exceeds the vector’s total size `ROWS`.
    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> S::Vector<R>;

    /// Creates a vector from an array of `S` scalars.
    fn from_scalar_array<A>(vals: A) -> Self
    where
        A: Borrow<[S; ROWS]>;

    /// Computes the Euclidean norm (length) of this vector.
    fn norm(&self) -> S;

    /// Returns a normalized version of this vector (`self / self.norm()`).
    ///
    /// # Note
    /// May return a NaN or invalid value if the norm is zero.
    fn normalized(&self) -> Self;

    /// Computes the outer product between `self` (treated as ROWS×1) and `rhs`
    /// (treated as 1×R2), yielding a ROWS×R2 matrix.
    fn outer<const R2: usize, V>(&self, rhs: V) -> S::Matrix<ROWS, R2>
    where
        V: Borrow<S::Vector<R2>>;

    /// Returns the underlying real vector (e.g., `VecF64<ROWS>`) if `S` is a real type.
    ///
    /// If `S` is dual or simd, this method returns the "real part" of the vector’s elements.
    fn real_vector(&self) -> S::RealVector<ROWS>;

    /// Lane-wise select operation: for each element or lane, picks from `self` if
    /// `mask` is true, otherwise picks from `other`.
    ///
    /// For non-batch usage (BATCH=1), `mask` is just a simple boolean. For batch/simd usage,
    /// `mask` can represent per-lane boolean flags.
    fn select<O>(&self, mask: &S::Mask, other: O) -> Self
    where
        O: Borrow<Self>;

    /// Scales the vector by a scalar `v` (e.g., `self * v`).
    fn scaled(&self, v: S) -> Self;

    /// Returns the squared Euclidean norm of this vector.
    fn squared_norm(&self) -> S;

    /// Converts this vector into a dual vector with zero infinitesimal part, if
    /// `S` is a real type. If it is already a dual type, it preserves or adapts
    /// the dual content based on generics `M` and `N`.
    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <<S as IsScalar<BATCH, DM, DN>>::DualScalar<M, N> as IsScalar<BATCH, M, N>>::Vector<ROWS>;

    /// Interprets `self` as a column vector and returns it in matrix form (ROWS×1).
    fn to_mat(&self) -> S::Matrix<ROWS, 1>;

    /// Creates a vector of all ones (1.0).
    fn ones() -> Self {
        Self::from_f64(1.0)
    }

    /// Creates a vector of all zeros (0.0).
    fn zeros() -> Self {
        Self::from_f64(0.0)
    }
}

/// A trait representing a real (non-dual) vector, typically [`VecF64<ROWS>`].
///
/// These vectors can still use batch types for SIMD, but they do not carry
/// derivative information. This trait also requires indexing support.
pub trait IsRealVector<
    S: IsRealScalar<BATCH> + IsScalar<BATCH, 0, 0>,
    const ROWS: usize,
    const BATCH: usize,
>:
    IsVector<S, ROWS, BATCH, 0, 0> + Index<usize, Output = S> + IndexMut<usize, Output = S> + Copy
{
}

/// A trait for batch vectors, where each scalar may hold multiple lanes of data
/// (e.g., via `portable_simd`), or multiple partial derivatives for autodiff.
///
/// # Type Parameters
/// - `ROWS`: Dimension of the vector.
/// - `BATCH`: Number of lanes (e.g., 4, 8, etc.).
/// - `DM`, `DN`: Shape of each dual’s Jacobian (0 if real-only).
#[cfg(feature = "simd")]
pub trait IsBatchVector<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize>:
    IsScalar<BATCH, DM, DN>
{
    /// Extracts a single lane’s scalar from a batched dual or real scalar.
    ///
    /// # Parameters
    /// - `i`: The lane index, typically in `0..BATCH`.
    fn extract_single(&self, i: usize) -> Self::SingleScalar;
}

/// A trait describing "single scalar" vectors, i.e., no batch dimension (`BATCH=1`),
/// commonly used for real or basic dual scalars. This trait provides a method
/// to set one element’s real-part directly.
///
/// # Example
/// Implemented on `VecF64<ROWS>` for real usage.
pub trait IsSingleVector<
    S: IsSingleScalar<DM, DN>,
    const ROWS: usize,
    const DM: usize,
    const DN: usize,
>: IsVector<S, ROWS, 1, DM, DN>
{
    /// Sets the real part of the element at `idx` to `v`.
    fn set_real_scalar(&mut self, idx: usize, v: f64);
}

impl<const BATCH: usize> IsSingleVector<f64, BATCH, 0, 0> for VecF64<BATCH> {
    fn set_real_scalar(&mut self, idx: usize, v: f64) {
        self[idx] = v;
    }
}

impl<const ROWS: usize> IsRealVector<f64, ROWS, 1> for VecF64<ROWS> {}

impl<const ROWS: usize> IsVector<f64, ROWS, 1, 0, 0> for VecF64<ROWS> {
    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: VecF64<R0>,
        bot_row: VecF64<R1>,
    ) -> Self {
        assert_eq!(ROWS, R0 + R1);
        let mut m = Self::zeros();

        m.fixed_view_mut::<R0, 1>(0, 0).copy_from(&top_row);
        m.fixed_view_mut::<R1, 1>(R0, 0).copy_from(&bot_row);
        m
    }

    fn block_vec3<const R0: usize, const R1: usize, const R2: usize>(
        top_row: VecF64<R0>,
        middle_row: VecF64<R1>,
        bot_row: VecF64<R2>,
    ) -> Self {
        assert_eq!(ROWS, R0 + R1 + R2);
        let mut m = Self::zeros();

        m.fixed_view_mut::<R0, 1>(0, 0).copy_from(&top_row);
        m.fixed_view_mut::<R1, 1>(R0, 0).copy_from(&middle_row);
        m.fixed_view_mut::<R2, 1>(R1, 0).copy_from(&bot_row);

        m
    }

    fn from_array<A>(vals: A) -> VecF64<ROWS>
    where
        A: Borrow<[f64; ROWS]>,
    {
        VecF64::<ROWS>::from_row_slice(&vals.borrow()[..])
    }

    fn from_real_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>,
    {
        VecF64::<ROWS>::from_row_slice(&vals.borrow()[..])
    }

    fn from_real_vector<A>(val: A) -> Self
    where
        A: Borrow<VecF64<ROWS>>,
    {
        *val.borrow()
    }

    fn from_f64_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>,
    {
        VecF64::<ROWS>::from_row_slice(&vals.borrow()[..])
    }

    fn from_scalar_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>,
    {
        VecF64::<ROWS>::from_row_slice(&vals.borrow()[..])
    }

    fn elem(&self, idx: usize) -> f64 {
        self[idx]
    }

    fn elem_mut(&mut self, idx: usize) -> &mut f64 {
        &mut self[idx]
    }

    fn norm(&self) -> f64 {
        self.norm()
    }

    fn real_vector(&self) -> Self {
        *self
    }

    fn squared_norm(&self) -> f64 {
        self.norm_squared()
    }

    fn to_mat(&self) -> MatF64<ROWS, 1> {
        *self
    }

    fn scaled(&self, v: f64) -> Self
    {
        self * v
    }

    fn dot<V>(&self, rhs: V) -> f64
    where
        V: Borrow<Self>,
    {
        VecF64::dot(self, rhs.borrow())
    }

    fn normalized(&self) -> Self {
        self.normalize()
    }

    fn from_f64(val: f64) -> Self {
        VecF64::<ROWS>::from_element(val)
    }

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <f64 as IsScalar<1, 0, 0>>::DualVector<ROWS, M, N> {
        DualVector::from_real_vector(*self)
    }

    fn outer<const R2: usize, V>(&self, rhs: V) -> MatF64<ROWS, R2>
    where
        V: Borrow<VecF64<R2>>,
    {
        self * rhs.borrow().transpose()
    }

    fn select<Q>(&self, mask: &bool, other: Q) -> Self
    where
        Q: Borrow<Self>,
    {
        if *mask { *self } else { *other.borrow() }
    }

    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> VecF64<R> {
        self.fixed_rows::<R>(start_r).into()
    }
}

/// Computes the cross product of two 3D vectors `lhs` × `rhs`.
///
/// # Generic Parameters
/// - `S`: The scalar type (real or dual).
/// - `BATCH`, `DM`, `DN`: For batch or dual usage.
///
/// # Examples
/// ```rust
/// use sophus_autodiff::linalg::{
///     VecF64,
///     cross,
/// };
///
/// let v1 = VecF64::<3>::new(1.0, 0.0, 0.0);
/// let v2 = VecF64::<3>::new(0.0, 1.0, 0.0);
/// let result = cross::<f64, 1, 0, 0>(v1, v2);
/// assert_eq!(result, VecF64::<3>::new(0.0, 0.0, 1.0));
/// ```
pub fn cross<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>(
    lhs: S::Vector<3>,
    rhs: S::Vector<3>,
) -> S::Vector<3> {
    let l0 = lhs.elem(0);
    let l1 = lhs.elem(1);
    let l2 = lhs.elem(2);

    let r0 = rhs.elem(0);
    let r1 = rhs.elem(1);
    let r2 = rhs.elem(2);

    S::Vector::from_array([l1 * r2 - l2 * r1, l2 * r0 - l0 * r2, l0 * r1 - l1 * r0])
}
