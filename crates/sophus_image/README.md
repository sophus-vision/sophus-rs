## Static and dynamic image types

This crates contains a number of statically and dynamically typed images.
Internally, images are represented as ndarray's of scalars such as [f32] for
single channel images or nalgebra vectors for multi-channel images.


## Integration with sophus-rs

This crate is part of the [sophus umbrella crate](https://crates.io/crates/sophus).
It re-exports the relevant prelude types under [prelude], so you can
seamlessly interoperate with the rest of the sophus-rs types.
