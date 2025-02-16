 This crate provides a collection of geometric primitives in n-dimensional Euclidean space:
 *Unit vectors* ([UnitVector]), *rays* ([Ray]), *hyperplanes* ([HyperPlane]), *Hyperspheres*
 ([HyperSphere]), *intervals* ([region::Interval]) and *box regions* ([region::BoxRegion]).

## Unit Vectors & Rays in ℝⁿ
 - A [UnitVector] is a direction with a fixed norm of 1. In code, it often appears
   to ensure a ray's direction is normalized.
 - A [Ray] extends this idea by pairing an origin point with a unit vector direction. Rays are
   helpful in ray casting or intersection tests, e.g. in computer graphics or collision detection.

```rust
use sophus_autodiff::linalg::VecF64;
use sophus_geo::prelude::*;
use sophus_geo::Ray;
use sophus_geo::UnitVector3;

// 3D ray from (1,2,3) along direction ~ (0,1,0), i.e. pointing "up" on y-axis
let origin = VecF64::<3>::new(1.0, 2.0, 3.0);
// Create a ~ (0,1,0) direction, but must be a UnitVector
let direction = UnitVector3::from_vector_and_normalize(&VecF64::<3>::new(0.0, 2.0, 0.0));
// Ray is now all points: origin + t * direction, t >= 0
let ray_3d = Ray::<f64, 2, 3, 1, 0, 0> {
    origin,
    dir: direction,
};

let p = ray_3d.at(10.0);
assert_eq!(p, VecF64::<3>::new(1.0, 12.0, 3.0));
```

## Hyperplanes and Hyperspheres
 - A [HyperPlane] in ℝⁿ is an (n-1)-dimensional subspace splitting ℝⁿ into two half-spaces. It is
   a set of points satisfying 'n·(x-o) = 0', where 'n' is the normal and 'o' is a point on the
   hyperplane. We provide geometry routines like projecting points onto a hyperplane or computing
   the plane's distance from a point.
 - A [HyperSphere] in ℝⁿ is defined by its center and radius. In 2D, it's a circle; in 3D, a sphere,
   etc. Intersection tests (line-sphere or circle-circle) are included as convenience methods.

```rust
use sophus_autodiff::linalg::VecF64;
use sophus_geo::prelude::*;
use sophus_geo::{HyperSphere, Ray};
use sophus_geo::UnitVector3;

// Suppose we have a 3D sphere (radius=5) centered at (10, 0, 0):
let sphere_3d = HyperSphere {
    center: VecF64::<3>::new(10.0, 0.0, 0.0),
    radius: 5.0,
};

// And a ray from (0,0,0) pointing along (1,0,0):
let ray_3d = Ray::<f64, 2, 3, 1, 0, 0> {
    origin: VecF64::<3>::new(0.0, 0.0, 0.0),
    dir: UnitVector3::from_vector_and_normalize(&VecF64::<3>::new(1.0, 0.0, 0.0)),
};

// We can find the nearest intersection point in front of the ray:
if let Some(hit_pt) = sphere_3d.ray_intersect(&ray_3d) {
    assert_eq!(hit_pt, VecF64::<3>::new(5.0, 0.0, 0.0));
    // The ray hits the sphere at x=5.
} else {
    panic!("No intersection (unexpected)!");
}
```

## Intervals and Axis-Aligned Bounding Boxes (AABB)
 - The 1D interval [region::Interval] is a possibly-empty closed set '[lower..upper]' in ℝ.
 - Extending intervals to n dimensions yields [region::BoxRegion], i.e. axis-aligned bounding
   boxes. Each dimension is a separate interval, so a 2D box region is '[xₗ..xᵣ] × [yₗ..yᵣ]'.
 - We provide traits [region::IsRegion], [region::IsNonEmptyRegion], etc., that unify the notion of
   an (optionally) empty region with various operations like containment, intersection,
   extension, or clamping points to remain inside the region.

```rust
use sophus_autodiff::linalg::VecF64;
use sophus_geo::region::{BoxRegion, IsRegion};
use sophus_geo::prelude::*;

// Define a 2D axis-aligned bounding box from (0,0) to (10,20).
let region_2d = BoxRegion::<2>::from_bounds(
    VecF64::<2>::new(0.0, 0.0),
    VecF64::<2>::new(10.0, 20.0),
);

// Suppose we have a point outside this box:
let point = VecF64::<2>::new(12.0, -5.0);

// We can clamp that point to the box boundary:
let clamped = region_2d.clamp_point(point);
assert_eq!(clamped, VecF64::<2>::new(10.0, 0.0));
// Clamped to the box corner: top-right is (10,0) in this coordinate layout.
```


## Integration with sophus-rs

Many of these geometric entities (e.g. planes, circles, or rays) can incorporate
dual-number-based auto-differentiation from ['sophus_autodiff'], or transformations from
['sophus_lie']. This crate re-exports the relevant prelude types under ['prelude'], so you
can seamlessly interoperate with the rest of the sophus ecosystem.
