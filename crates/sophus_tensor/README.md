## Tensor as a thin wrapper around [`ndarray`](https://docs.rs/ndarray) and [`nalgebra`](https://docs.rs/nalgebra)

In general, we have dynamic-size tensor (ndarrays's) of static-size
tensors (nalgebra's SMat, SVec, etc.). Note this crate is merely an
implementation detail of the `sophus_image` crate and might be absorbed into
that crate in the future.

## Integration with sophus-rs

This crate is part of the [sophus umbrella crate](https://crates.io/crates/sophus).
It re-exports the relevant prelude types under [prelude], so you can
seamlessly interoperate with the rest of the sophus-rs types.
