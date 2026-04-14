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

# SIMD (requires nightly)
just build-simd     # cargo +nightly build --release --all-targets --features simd
just test-simd      # cargo +nightly test --release --features simd
```

To run a single test:
```sh
cargo test --release --features std <test_name> -- --nocapture
```

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
- `sophus_renderer` — `wgpu`-based rendering
- `sophus_viewer` — Interactive viewer with `egui` + `wgpu`
- `sophus_sim` — Camera simulator

**Umbrella:**
- `sophus` — Re-exports all sub-crates; use `sophus::prelude::*` for traits

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
