# CLAUDE.md

> **Maintenance:** Keep this file up to date. Each commit that adds crates, changes APIs,
> or modifies build steps should include corresponding CLAUDE.md updates.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This repo uses `just` as a task runner. Prefer `just` commands over raw `cargo` where available.

```sh
just build          # cargo build --release --all-targets
just build-simd     # cargo +nightly build --release --all-targets --features simd
just check-wasm     # cargo check --lib --target wasm32-unknown-unknown
just test           # cargo test --release --features std
just test-simd      # cargo +nightly test --release --features simd
just clippy         # cargo clippy --tests --features std
just format         # pre-commit run -a && cargo +nightly fmt
just doc            # cargo +nightly doc --no-deps --all-features + doctests

# Benchmarks
just solver-bench          # sparse solver benchmarks
just ba-bench              # bundle adjustment benchmark (standard vs Schur)
just kb-projection-bench   # KB projection SIMD benchmark (requires nightly)
just render-bench          # offscreen rendering benchmark (needs a GPU)

# SIMD (requires nightly)
just build-simd     # cargo +nightly build --release --all-targets --features simd
just test-simd      # cargo +nightly test --release --features simd
```

To run a single test:
```sh
cargo test --release --features std <test_name> -- --nocapture
```

`crates/sophus_renderer/tests/offscreen_regression.rs` renders small scenes head-lessly and
asserts on the resulting pixels. Those tests skip themselves, with a message, when the host has no
GPU — so a green run there does not by itself mean they executed.

To run the interactive demo app (bundle adjustment, optimization visualizations):
```sh
cargo run --release --features std --bin demo
```

## Workspace Architecture

17 crates under `crates/`, organized in layers:

**Foundation:**
- `sophus_autodiff` — Forward-mode AD via dual numbers (`DualScalar<S, M, N>`); `no_std`-compatible
- `sophus_tensor` — Dynamic-outer / static-inner tensor types bridging `ndarray` and fixed-size math
- `sophus_assert` — Custom assertion macros (`assert_lt!`, etc.)
- `sophus_bench` — Benchmarking utilities

**Geometry:**
- `sophus_lie` — Lie groups: SO(2), SO(3), SE(2), SE(3), quaternions; exp/log/adjoint/hat/vee
- `sophus_geo` — Primitives: unit vectors, rays, hyperplanes, hyperspheres, regions/intervals
- `sophus_image` — Image types with coordinate-aware tensors

**Domain:**
- `sophus_sensor` — Camera models and projection/distortion abstractions
- `sophus_spline` — Cubic B-splines
- `sophus_timeseries` — Temporal data structures
- `sophus_solver` — Block-sparse matrices, LDLᵀ factorization, LU, QR, SVD solvers, DirectSolve dispatch
- `sophus_opt` — Unified Optimizer (NLLS), inequality constraints (IPM, SQP), phase-1 feasibility, robust kernels, BA problem

**Graphics:**
- `sophus_renderer` — `wgpu`-based rendering; see **Rendering** below
- `sophus_viewer` — Interactive viewer with `egui` + `wgpu`
- `sophus_sim` — Camera simulator

**Umbrella:**
- `sophus` — Re-exports all sub-crates; use `sophus::prelude::*` for traits

## Rendering

A frame is drawn in three stages. The scene is **rasterized** through an undistorted
*intermediate* - one plane fitted to the visible region, or five 90° frustum faces when the field
of view is too wide for a plane - a compute pass then **warps** that into the distorted image the
camera model describes, and a last pass draws over the finished image what is measured in pixels
rather than in metres. `Intermediate::choose` picks between plane and frusta, and is the only
place that decision is made.

**Traced primitives** are not rasterized at all. The warp already computes the exact ray of every
output pixel, so ellipsoids, planes, capsules and cones are intersected with it directly
(`shaders/traced.wgsl`) and composited against the rasterized scene by distance. They are therefore
exact under the real camera model - no intermediate, no resampling, no upper bound on the field of
view - and are how a sphere, a disk, a ground plane, an arrow or a set of axes is drawn. Constructors
such as `make_sphere3`, `make_arrow3` and `make_axes_arrows3` build on them, as does `axes3`,
which draws a whole field of poses as one cloud of each primitive rather than an entity per pose.

**Lines and points are the exception**: their width is in *view-port pixels*, so they are drawn in
the last pass, over the warped image (`pixel_renderer/scene_overlay.rs`), where a pixel is a pixel
of the image rather than of the intermediate. A straight segment is a curve there, so it is drawn
as a strip whose joints are each projected through the camera model. They read the inverse
distance the warp left behind to be occluded by the scene, and write their own into it, so that
what is under the pointer still has a distance in a scene which is nothing but a point cloud.

Three conventions worth knowing before touching any of it:

- **The depth buffer holds inverse *distance* along the ray**, not `z` along the optical axis: `z`
  is degenerate at 90° off axis, where a 180° camera has to work. Zero means nothing there.
  `InverseDistanceImage::metric_z` converts for anything wanting the rgb-d convention. The name
  "inverse depth" is kept for the *parameterisation* - a bearing and a range - which is what
  `sophus_geo` and the demos of that name mean by it.
- **Discriminants cancel.** A quadratic's discriminant written the textbook way subtracts two large
  numbers to reach a small one, and a small primitive far away then vanishes outright. Every
  intersection here is written to avoid that, usually via a cross product - see the comments.
- **There are no derivatives in the compute pass.** Anything needing the size of a pixel -
  silhouette antialiasing, the ground's pattern - takes it from the rays of the neighbouring
  pixels, and needs *both* neighbours: a surface raking away moves far further down the image than
  across it.

## Key Design Patterns

**Trait-based generics** — `IsScalar`, `IsVector`, `IsMatrix` allow the same code to operate over `f64`, dual numbers, and batch types. Lie groups follow `IsLieGroupImpl` / `IsRealLieGroupImpl` / `IsLieFactorGroupImpl`.

**Const generics everywhere** — All matrix/vector dimensions are compile-time: `MatF64<3, 3>`, `VecF64<6>`. No runtime size checks needed.

**Dual numbers for AD** — Jacobians flow automatically through Lie group operations. `DualScalar<S, M, N>` encodes M-output, N-input Jacobian shape.

**Prelude pattern** — Each crate exposes a `prelude` module; import `sophus::prelude::*` to get all traits in scope.

**`no_std` + feature gating:**
- Core crates are `no_std`-compatible; `std` feature gates allocations and file I/O
- `simd` feature enables `portable_simd` (nightly-only) + `sleef` for vectorized batch scalars
- `build.rs` in each crate detects nightly and sets `cfg(nightly)` for `doc_cfg` attributes

## Features

- `std` — Enable standard library support (required for most dev workflows)
- `simd` — Enable SIMD batch scalars; **requires nightly Rust**

## Rust Version

- MSRV: 1.94.0 (stable)
- Nightly required only for `simd` feature and `just doc`
