## Automatic differentiation with optional SIMD acceleration

This crate provides traits, structs, and helper functions to enable automatic
differentiation (AD) through the use of dual numbers. In particular, it defines
*dual scalar*, *dual vector*, and *dual matrix* types, along with their SIMD
"batch" counterparts, for forward-mode AD in both scalar and vector/matrix
contexts.

## Overview

The primary interfaces are defined by traits in [crate::linalg]:

| **Trait**                  | **Real types**      | **Dual types**            | **Batch real types**           | **Batch dual types**              |
|----------------------------|---------------------|---------------------------|--------------------------------|-----------------------------------|
| [linalg::IsScalar]         | [f64]               | [dual::DualScalar]        | [linalg::BatchScalarF64]       | [dual::DualBatchScalar]           |
| [linalg::IsVector]         | [linalg::VecF64]    | [dual::DualVector]        | [linalg::BatchVecF64]          | [dual::DualBatchVector]           |
| [linalg::IsMatrix]         | [linalg::MatF64]    | [dual::DualMatrix]        | [linalg::BatchMatF64]          | [dual::DualBatchMatrix]           |

- **Real types** are your basic floating-point types ([f64], etc.).
- **Dual types** extend these scalars, vectors, or matrices to store partial
  derivatives (the “infinitesimal” part) in addition to the real value.
- **Batch real types** and **batch dual types** leverage Rust’s `portable_simd`
  to process multiple derivatives or computations in parallel.

**Note:** Batch types such as [linalg::BatchScalarF64], [linalg::BatchVecF64],
and [linalg::BatchMatF64] require enabling the `"simd"` feature in your
`Cargo.toml`.

## Numerical differentiation
In addition to forward-mode AD via dual numbers, this module also contains
functionality to compute numerical derivatives (finite differences) of your
functions in various shapes:

 * **Curves,** functions which take a scalar input:
    - `f: ℝ -> ℝ`, [maps::ScalarValuedCurve]
    - `f: ℝ -> ℝʳ`, [maps::VectorValuedCurve]
    - `f: ℝ -> ℝʳˣᶜ`, [maps::MatrixValuedCurve]
 * **Scalar-valued maps**, functions which return a scalar:
    - `f: ℝᵐ -> ℝ`, [maps::ScalarValuedVectorMap]
    - `f: ℝᵐˣⁿ -> ℝ`, [maps::ScalarValuedMatrixMap]
 * **Vector-valued maps**, functions which return a vector:
    - `f: ℝᵐ -> ℝᵖ`, [maps::VectorValuedVectorMap]
    - `f: ℝᵐˣⁿ -> ℝᵖ`, [maps::VectorValuedMatrixMap]
 * **Matrix-valued maps**, functions which return a matrix:
    - `f: ℝᵐ -> ℝʳˣᶜ`, [maps::MatrixValuedVectorMap]
    - `f: ℝᵐˣⁿ -> ℝʳˣᶜ`, [maps::MatrixValuedMatrixMap]

You’ll find the associated functions for finite-difference computations in the
[maps submodule][crate::maps].


## Example: Differentiating a Vector Function

```rust
use sophus_autodiff::prelude::*;
use sophus_autodiff::dual::{DualScalar, DualVector};
use sophus_autodiff::maps::VectorValuedVectorMap;
use sophus_autodiff::linalg::VecF64;

// Suppose we have a function `f: ℝ³ → ℝ²`
//
//    [[ x ]]   [[ x / z ]]
//  f [[ y ]] = [[       ]]
//    [[ z ]]   [[ y / z ]]
fn proj_fn<S: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
    v: S::Vector<3>,
) -> S::Vector<2> {
    let x = v.elem(0);
    let y = v.elem(1);
    let z = v.elem(2);
    S::Vector::<2>::from_array([x / z, y / z])
}

let input = VecF64::<3>::new(1.0, 2.0, 3.0);

// (1) Finite difference approximation:
let fd_jacobian = VectorValuedVectorMap::<f64, 1>::sym_diff_quotient_jacobian(
    proj_fn::<f64, 0, 0>,
    input,
    1e-5,
);

// (2) Forward-mode autodiff using dual numbers:
let auto_jacobian =
    proj_fn::<DualScalar<3, 1>, 3, 1>(DualVector::var(input)).jacobian();

// Compare the two results:
approx::assert_abs_diff_eq!(fd_jacobian, auto_jacobian, epsilon = 1e-5);
```

By using the [dual::DualScalar] (or [dual::DualVector], etc.) approach, we get
the Jacobian via forward-mode autodiff. Alternatively, we can use finite
differences for functions that might not be easily retrofitted with dual
numbers.

## See Also

- [**linalg** module][crate::linalg] for linear algebra types and operations.
- [**manifold** module][crate::manifold] for manifold operations and tangent spaces.
- [**params** module][crate::params] for parameter-related traits.
- [**points** module][crate::points] for point types.

## Feature Flags

- `"simd"`: Enables batch types like [dual::DualBatchScalar],
  [linalg::BatchVecF64], etc. which require Rust’s nightly `portable_simd`
  feature.

## Integration with sophus-rs

This crate is part of the [sophus umbrella crate](https://crates.io/crates/sophus).
It re-exports the relevant prelude types under [prelude], so you can
seamlessly interoperate with the rest of the sophus-rs types.
