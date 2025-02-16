use core::fmt::Debug;

/// A generic trait for boolean masks, supporting both single-lane (regular `bool`) and
/// multi-lane (SIMD) usage.
///
/// A "mask" in this context indicates which elements (or "lanes") of a batch/array
/// are active (`true`) vs. inactive (`false`). This trait provides methods to:
///
/// - Create fully true/fully false masks.
/// - Check if all or any lanes are true.
/// - Count how many lanes are true.
/// - Get the total number of lanes (usually 1 for a single bool, or more for SIMD).
///
/// # Implementations
/// - For single-lane booleans, `bool` itself implements `IsBoolMask` trivially.
/// - For SIMD-based multi-lane booleans, see [`BatchMask`](crate::linalg::batch_mask::BatchMask)
///   (when the `"simd"` feature is enabled).
pub trait IsBoolMask: Debug {
    /// Creates a mask where all lanes are set to `true`.
    fn all_true() -> Self;

    /// Creates a mask where all lanes are set to `false`.
    fn all_false() -> Self;

    /// Returns `true` if *all* lanes in the mask are `true`.
    fn all(&self) -> bool;

    /// Returns `true` if *any* lane in the mask is `true`.
    fn any(&self) -> bool;

    /// Returns the number of lanes that are `true`.
    fn count(&self) -> usize;

    /// Returns the total number of lanes in the mask.
    fn lanes(&self) -> usize;
}

// A `bool`-based implementation of [`IsBoolMask`], suitable for single-lane usage.
//
// Since there is only one "lane," the methods reflect simple boolean logic:
// - `all_true()` => `true`
// - `all_false()` => `false`
// - `all()` => returns the boolean itself
// - `any()` => same as `all()` in a single-lane context
// - `count()` => returns `1` if `true`, else `0`
// - `lanes()` => returns `1`
//
// This effectively acts as the trivial, single-lane version of a boolean mask.
impl IsBoolMask for bool {
    fn all_true() -> bool {
        true
    }

    fn all_false() -> bool {
        false
    }

    fn all(&self) -> bool {
        *self
    }

    fn any(&self) -> bool {
        *self
    }

    fn count(&self) -> usize {
        if *self {
            1
        } else {
            0
        }
    }

    fn lanes(&self) -> usize {
        1
    }
}
