## Solvers for linear systems

This crate contains solvers for linear systems: ``A x = b``.

The main focus are solvers for linear systems  where `A` is positive semi-definite.
That means `A` is symmetric and `xᵀ A x ≥ 0` is true for all `x`. These matrices
are most efficiently solved using [`LDLᵀ` solvers](crate::ldlt). There are also
[`LU`-decomposition](crate::lu) and [`QR`-decomposition](crate::qr)-based solvers.

The [matrix](crate::matrix) module contains [dense](crate::matrix::dense),
[sparse](crate::matrix::sparse) and [block-sparse](crate::matrix::block_sparse)
matrix representations - to be used with corresponding solvers.

## Integration with sophus-rs

This crate is part of the [sophus umbrella crate](https://crates.io/crates/sophus).
It re-exports the relevant prelude types under [prelude], so you can
seamlessly interoperate with the rest of the sophus-rs types.
