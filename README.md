# sophus-rs

sophus-rs is a Rust library for 2d and 3d geometry for Computer Vision and Robotics applications.
It is a spin-off of the [Sophus](https://github.com/strasdat/Sophus) C++ library which
focuses on **Lie groups** (e.g. rotations and transformations in 2d and 3d).

In addition to Lie groups, sophus-rs also includes other geometric/maths concepts such unit vector,
splines, image classes, camera models as well as a other utilities such as a non-linear least
squares optimization.

## Status

This library is in an early development stage - hence API is highly unstable. It is likely that
existing features will be removed or changed in the future.

However, the intend is to stride for correctness, facilitated using a comprehensive test suite.

## Building

sophus-rs builds on stable.

```toml
[dependencies]
sophus = "0.7.0"
```

To allow for batch types, such as BatchScalarF64, the 'simd' feature is required. This feature
depends on [`portable-simd`](https://doc.rust-lang.org/std/simd/index.html), which is currently
only available on [nightly](https://doc.rust-lang.org/book/appendix-07-nightly-rust.html). There
are plans to use any other nightly features.

```toml
[dependencies]
sophus = { version = "0.7.0", features = ["simd"] }
```
