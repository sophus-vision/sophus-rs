# sophus-rs

[![Latest version](https://img.shields.io/crates/v/sophus.svg)](https://crates.io/crates/sophus)
[![Documentation](https://docs.rs/strasdat/badge.svg)](https://docs.rs/sophus)

2d and 3d geometry for Computer Vision and Robotics

## Overview

sophus-rs is a Rust library for 2d and 3d geometry for Computer Vision and Robotics applications.
It is a spin-off of the [Sophus](https://github.com/strasdat/Sophus) C++ library which
focuses on *Lie groups* (e.g. rotations and transformations in 2d and 3d).

In addition to Lie groups, sophus-rs also includes other geometric/maths concepts.

## Automatic differentiation

sophus-rs provides an automatic differentiation using dual numbers such as
[autodiff::dual::DualScalar] and [autodiff::dual::DualVector].

```
use sophus::prelude::*;
use sophus::autodiff::dual::{DualScalar, DualVector};
use sophus::autodiff::linalg::VecF64;
use sophus::autodiff::maps::VectorValuedVectorMap;

//       [[ x ]]   [[ x / z ]]
//  proj [[ y ]] = [[       ]]
//       [[ z ]]   [[ y / z ]]
fn proj_fn<S: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
    v: S::Vector<3>,
) -> S::Vector<2> {
    let x = v.elem(0);
    let y = v.elem(1);
    let z = v.elem(2);
    S::Vector::<2>::from_array([x / z, y / z])
}

let a = VecF64::<3>::new(1.0, 2.0, 3.0);
// Finite difference Jacobian
let finite_diff = VectorValuedVectorMap::<f64, 1>::sym_diff_quotient_jacobian(
    proj_fn::<f64, 0, 0>,
    a,
    0.0001,
);
// Automatic differentiation Jacobian
let auto_diff = proj_fn::<DualScalar<3, 1>, 3, 1>(DualVector::var(a)).jacobian();

approx::assert_abs_diff_eq!(finite_diff, auto_diff, epsilon = 0.0001);
```

Note that proj_fn is a function that takes a 3D vector and returns a 2D vector. The Jacobian of
proj_fn is 2x3 matrix. When a (three dimensional) dual vector is passed to proj_fn, then
a 2d dual vector is returned. Since we are expecting a 2x3 Jacobian, each element of the 2d dual
vector must represent 3x1 Jacobian. This is why we use DualScalar<3, 1> as the scalar type.

## Lie Groups

sophus-rs provides a number of Lie groups, including:

 * The group of 2D rotations, [lie::Rotation2], also known as the special orthogonal group SO(2),
 * the group of 3D rotations, [lie::Rotation3], also known as the special orthogonal group SO(3),
 * the group of 2d isometries, [lie::Isometry2], also known as the Euclidean group SE(2), and
 * the group of 3d isometries, [lie::Isometry3], also known as the Euclidean group SE(3).


```
use sophus::autodiff::linalg::VecF64;
use sophus::lie::{Rotation3F64, Isometry3F64};
use std::f64::consts::FRAC_PI_4;

// Create a rotation around the z-axis by 45 degrees.
let world_from_foo_rotation = Rotation3F64::rot_z(FRAC_PI_4);

// Create a translation in 3D.
let foo_in_world = VecF64::<3>::new(1.0, 2.0, 3.0);

// Combine them into an SE(3) transform.
let world_from_foo_isometry
    = Isometry3F64::from_translation_and_rotation(foo_in_world, world_from_foo_rotation);

// Apply world_from_foo_isometry to a 3D point in the foo reference frame.
let point_in_foo = VecF64::<3>::new(10.0, 0.0, 0.0);
let point_in_world = world_from_foo_isometry.transform(&point_in_foo);

// Manually compute the expected transformation:
//  - rotate (10, 0, 0) around z by 45°
//  - then translate by (1, 2, 3)
let angle = FRAC_PI_4;
let cos = angle.cos();
let sin = angle.sin();
let expected_point_in_world = VecF64::<3>::new(1.0 + 10.0 * cos, 2.0 + 10.0 * sin, 3.0);

approx::assert_abs_diff_eq!(point_in_world[0], expected_point_in_world[0], epsilon = 1e-9);
approx::assert_abs_diff_eq!(point_in_world[1], expected_point_in_world[1], epsilon = 1e-9);
approx::assert_abs_diff_eq!(point_in_world[2], expected_point_in_world[2], epsilon = 1e-9);

// Map isometry to 6-dimensional tangent space.
let omega = world_from_foo_isometry.log();
// Map tangent space element back to the manifold.
let roundtrip_world_from_foo_isometry = Isometry3F64::exp(&omega);
approx::assert_abs_diff_eq!(roundtrip_world_from_foo_isometry.matrix(),
                            world_from_foo_isometry.matrix(),
                            epsilon = 1e-9);

// Compose with another isometry.
let world_from_bar_isometry = Isometry3F64::rot_y(std::f64::consts::FRAC_PI_6);
let bar_from_foo_isometry = world_from_bar_isometry.inverse() * world_from_foo_isometry;
```

## And more...

such unit vector, splines, image classes, camera models, non-linear least squares optimization and
some visualization tools. Check out the [documentation](https://docs.rs/sophus) for more information.


## Building

sophus-rs builds on stable.

```toml
[dependencies]
sophus = "0.13.0"
```

```toml
[dependencies]
sophus = { version = "0.13.0", features = ["simd"] }
```

To allow for batch types, such as BatchScalarF64, the 'simd' feature is required. This feature
depends on [`portable-simd`](https://doc.rust-lang.org/std/simd/index.html), which is currently
only available on [nightly](https://doc.rust-lang.org/book/appendix-07-nightly-rust.html). There
are no plans to rely on any other nightly features.

## Crate Structure and Usage

sophus-rs is an **umbrella crate** that provides a single entry point to multiple
sub-crates (modules) under the `sophus::` namespace. For example, the automatic differentiation
sub-crate can be accessed via `use sophus::autodiff`, and the lie group sub-crate via
`use sophus::lie`, etc.

- If you want all of sophus’s functionalities at once (geometry, AD, manifolds, etc.),
  simply add `sophus` in your `Cargo.toml`, and then in your Rust code:

  ```rust
  use sophus::prelude::*;
  use sophus::autodiff::dual::DualScalar;
  // ...
  ```

- If you only need the autodiff functionalities in isolation, you can also depend on the
  standalone crate underlying `sophus_autodiff`.

   ```rust
  use sophus_autodiff::prelude::*;
  use sophus_autodiff::dual::DualScalar;
  // ...
  ```

## Status

This library is in an early development stage - hence API is highly unstable. It is likely that
existing features will be removed or changed in the future.
