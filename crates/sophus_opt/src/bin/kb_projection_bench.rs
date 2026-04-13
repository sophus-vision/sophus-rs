#![feature(portable_simd)]
//! Kannala-Brandt fisheye projection SIMD benchmark.
//!
//! Compares scalar (libm) vs batched (SLEEF) projection throughput for f32 and f64,
//! across batch sizes 2, 4, 8.
//!
//! Run: `just kb-projection-bench`
//!
//! Projection model:
//!
//!   `π_KB(p) = θ(1 + k₀θ² + k₁θ⁴ + k₂θ⁶ + k₃θ⁸) · (x/r, y/r)`
//!   where `r = sqrt(x²+y²)`,  `θ = atan2(r, 1)`
//!
//! Each projection requires `sqrt` + `atan2` — both dispatched to SLEEF in the batch path.
//!
//! Key characteristics:
//!  - Only 3 values gathered per SIMD lane, so gather is cheap.
//!  - Camera params are splatted (broadcast), not gathered.
//!  - `atan2` is more expensive than `sin`/`cos` on most platforms.

use sophus_autodiff::{
    linalg::{
        BatchScalarF32,
        BatchScalarF64,
        VecF32,
        VecF64,
    },
    prelude::*,
};
use sophus_image::ImageSize;
use sophus_sensor::{
    KannalaBrandtCamera,
    KannalaBrandtCameraF32,
    KannalaBrandtCameraF64,
};

fn time_us_n(n: u32, f: impl Fn()) -> f64 {
    f(); // warm-up
    let t = std::time::Instant::now();
    for _ in 0..n {
        f();
    }
    t.elapsed().as_secs_f64() * 1e6 / n as f64
}

/// Project N 3D points through a Kannala-Brandt camera — one point at a time (scalar libm, f64).
fn project_scalar(cam: &KannalaBrandtCameraF64, points: &[VecF64<3>]) -> Vec<VecF64<2>> {
    points.iter().map(|p| cam.cam_proj(*p)).collect()
}

/// Project N 3D points through a Kannala-Brandt camera — one point at a time (scalar libm, f32).
fn project_scalar_f32(cam: &KannalaBrandtCameraF32, points: &[VecF32<3>]) -> Vec<VecF32<2>> {
    points.iter().map(|p| cam.cam_proj(*p)).collect()
}

/// Project N 3D points using SIMD batch scalars (f64) — BATCH points per SLEEF call.
///
/// Camera params are the same for every point, so they are *splatted* (broadcast)
/// across SIMD lanes — no gather cost for params.
/// Only the 3D point coordinates (3 values per lane) are gathered.
fn project_batched<const BATCH: usize>(
    params: VecF64<8>,
    image_size: ImageSize,
    points: &[VecF64<3>],
) -> Vec<VecF64<2>>
where
    std::simd::Simd<f64, BATCH>: std::simd::num::SimdFloat,
{
    let n = points.len();
    let mut result = Vec::with_capacity(n);

    // Splat camera params once — same intrinsics for all lanes.
    let batch_params = nalgebra::SVector::<BatchScalarF64<BATCH>, 8>::from_fn(|row, _| {
        BatchScalarF64::<BATCH>::from_f64(params[row])
    });
    let batch_cam = KannalaBrandtCamera::<BatchScalarF64<BATCH>, BATCH, 0, 0>::from_params_and_size(
        batch_params,
        image_size,
    );

    let full = n / BATCH;
    for i in 0..full {
        let base = i * BATCH;

        // Gather 3D coordinates for BATCH consecutive points.
        let batch_pt = nalgebra::SVector::<BatchScalarF64<BATCH>, 3>::from_fn(|row, _| {
            BatchScalarF64::<BATCH>::from_real_array(core::array::from_fn(|lane| {
                points[base + lane][row]
            }))
        });

        // SIMD: compute KB projection — sqrt + atan2 via SLEEF for all BATCH points.
        let batch_uv = batch_cam.cam_proj(batch_pt);

        // Scatter: unpack BATCH projected pixels.
        let uv0 = batch_uv[0].to_real_array();
        let uv1 = batch_uv[1].to_real_array();
        for lane in 0..BATCH {
            result.push(VecF64::<2>::new(uv0[lane], uv1[lane]));
        }
    }

    // Scalar tail.
    let scalar_cam = KannalaBrandtCameraF64::from_params_and_size(params, image_size);
    for i in (full * BATCH)..n {
        result.push(scalar_cam.cam_proj(points[i]));
    }

    result
}

/// Project N 3D points using SIMD batch scalars (f32) — BATCH points per SLEEF call.
///
/// On Apple Silicon (128-bit NEON): f32-batch-4 uses 4 native lanes (4×32-bit = 128-bit),
/// while f64-batch-4 emulates 4 lanes using 2×2 (each f64 lane needs 2×64-bit ops).
fn project_batched_f32<const BATCH: usize>(
    params: VecF32<8>,
    image_size: ImageSize,
    points: &[VecF32<3>],
) -> Vec<VecF32<2>>
where
    std::simd::Simd<f32, BATCH>: std::simd::num::SimdFloat,
{
    let n = points.len();
    let mut result = Vec::with_capacity(n);

    // Splat camera params once — same intrinsics for all lanes.
    let batch_params = nalgebra::SVector::<BatchScalarF32<BATCH>, 8>::from_fn(|row, _| {
        BatchScalarF32::<BATCH>::from_f64(params[row] as f64)
    });
    let batch_cam = KannalaBrandtCamera::<BatchScalarF32<BATCH>, BATCH, 0, 0>::from_params_and_size(
        batch_params,
        image_size,
    );

    let full = n / BATCH;
    for i in 0..full {
        let base = i * BATCH;

        // Gather 3D coordinates for BATCH consecutive points.
        let batch_pt = nalgebra::SVector::<BatchScalarF32<BATCH>, 3>::from_fn(|row, _| {
            BatchScalarF32::<BATCH>::from_real_array(core::array::from_fn(|lane| {
                points[base + lane][row]
            }))
        });

        // SIMD: compute KB projection — sqrt + atan2 via SLEEF for all BATCH points.
        let batch_uv = batch_cam.cam_proj(batch_pt);

        // Scatter: unpack BATCH projected pixels.
        let uv0 = batch_uv[0].to_real_array();
        let uv1 = batch_uv[1].to_real_array();
        for lane in 0..BATCH {
            result.push(VecF32::<2>::new(uv0[lane], uv1[lane]));
        }
    }

    // Scalar tail.
    let scalar_cam = KannalaBrandtCameraF32::from_params_and_size(params, image_size);
    for i in (full * BATCH)..n {
        result.push(scalar_cam.cam_proj(points[i]));
    }

    result
}

fn main() {
    use rand::prelude::*;
    use rand_chacha::ChaCha12Rng;

    let n: usize = 100_000;
    let reps: u32 = 200;

    eprintln!("Generating {n} random 3D points in camera frame...");
    let mut rng = ChaCha12Rng::from_seed(Default::default());

    let image_size = ImageSize {
        width: 1280,
        height: 1024,
    };

    // Kannala-Brandt params: [fu, fv, cu, cv, k0, k1, k2, k3]
    // Use realistic fisheye distortion coefficients.
    let params = VecF64::<8>::from_f64_array([
        300.0, 300.0, 640.0, 512.0, // fu, fv, cu, cv
        0.01, 1e-4, 1e-6, 1e-8, // k0..k3
    ]);

    let params_f32 = VecF32::<8>::from_f64_array([
        300.0, 300.0, 640.0, 512.0, // fu, fv, cu, cv
        0.01, 1e-4, 1e-6, 1e-8, // k0..k3
    ]);

    let scalar_cam = KannalaBrandtCameraF64::from_params_and_size(params, image_size);
    let scalar_cam_f32 = KannalaBrandtCameraF32::from_params_and_size(params_f32, image_size);

    // Random 3D points in front of the camera (z in [1, 10]).
    let points: Vec<VecF64<3>> = (0..n)
        .map(|_| {
            VecF64::<3>::new(
                (rng.random::<f64>() - 0.5) * 4.0,
                (rng.random::<f64>() - 0.5) * 4.0,
                1.0 + rng.random::<f64>() * 9.0,
            )
        })
        .collect();

    // f32 version of the same points.
    let points_f32: Vec<VecF32<3>> = points
        .iter()
        .map(|p| VecF32::<3>::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect();

    // ── Timing ──────────────────────────────────────────────────────────────────
    let sc_us = time_us_n(reps, || {
        std::hint::black_box(project_scalar(&scalar_cam, &points));
    });
    let b2_us = time_us_n(reps, || {
        std::hint::black_box(project_batched::<2>(params, image_size, &points));
    });
    let b4_us = time_us_n(reps, || {
        std::hint::black_box(project_batched::<4>(params, image_size, &points));
    });
    let b8_us = time_us_n(reps, || {
        std::hint::black_box(project_batched::<8>(params, image_size, &points));
    });
    let sc_f32_us = time_us_n(reps, || {
        std::hint::black_box(project_scalar_f32(&scalar_cam_f32, &points_f32));
    });
    let b2_f32_us = time_us_n(reps, || {
        std::hint::black_box(project_batched_f32::<2>(
            params_f32,
            image_size,
            &points_f32,
        ));
    });
    let b4_f32_us = time_us_n(reps, || {
        std::hint::black_box(project_batched_f32::<4>(
            params_f32,
            image_size,
            &points_f32,
        ));
    });
    let b8_f32_us = time_us_n(reps, || {
        std::hint::black_box(project_batched_f32::<8>(
            params_f32,
            image_size,
            &points_f32,
        ));
    });

    // ── Correctness ──────────────────────────────────────────────────────────────
    let ref_out = project_scalar(&scalar_cam, &points);
    let b2_out = project_batched::<2>(params, image_size, &points);
    let b4_out = project_batched::<4>(params, image_size, &points);
    let b8_out = project_batched::<8>(params, image_size, &points);

    let ref_out_f32 = project_scalar_f32(&scalar_cam_f32, &points_f32);
    let b2_f32_out = project_batched_f32::<2>(params_f32, image_size, &points_f32);
    let b4_f32_out = project_batched_f32::<4>(params_f32, image_size, &points_f32);
    let b8_f32_out = project_batched_f32::<8>(params_f32, image_size, &points_f32);

    let max_err_f64 = |a: &[VecF64<2>], b: &[VecF64<2>]| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0_f64, f64::max)
    };
    let max_err_f32 = |a: &[VecF32<2>], b: &[VecF32<2>]| -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0_f32, f32::max)
    };

    let err2 = max_err_f64(&ref_out, &b2_out);
    let err4 = max_err_f64(&ref_out, &b4_out);
    let err8 = max_err_f64(&ref_out, &b8_out);
    let err2_f32 = max_err_f32(&ref_out_f32, &b2_f32_out);
    let err4_f32 = max_err_f32(&ref_out_f32, &b4_f32_out);
    let err8_f32 = max_err_f32(&ref_out_f32, &b8_f32_out);

    // ── Output ────────────────────────────────────────────────────────────────────
    println!("\n{}", "=".repeat(88));
    println!("  Batched Kannala-Brandt fisheye projection:  uv = π_KB(p)");
    println!("  N={n} points, {reps} reps — each call uses sqrt + atan2 (SLEEF when batched)");
    println!("{}", "=".repeat(88));
    println!("  {:<38}  {:>10}  {:>8}", "Method", "avg ms", "speedup");
    println!("{}", "-".repeat(88));
    println!(
        "  {:<38}  {:>10.3}  {:>7.2}x",
        "scalar (f64, libm)",
        sc_us / 1000.0,
        1.0_f64
    );
    println!(
        "  {:<38}  {:>10.3}  {:>7.2}x  [{}]  max_err={err2:.2e}",
        "batch-2 (BatchScalarF64<2>)",
        b2_us / 1000.0,
        sc_us / b2_us,
        if err2 < 1e-10 { "PASS" } else { "FAIL" },
    );
    println!(
        "  {:<38}  {:>10.3}  {:>7.2}x  [{}]  max_err={err4:.2e}",
        "batch-4 (BatchScalarF64<4>)",
        b4_us / 1000.0,
        sc_us / b4_us,
        if err4 < 1e-10 { "PASS" } else { "FAIL" },
    );
    println!(
        "  {:<38}  {:>10.3}  {:>7.2}x  [{}]  max_err={err8:.2e}",
        "batch-8 (BatchScalarF64<8>)",
        b8_us / 1000.0,
        sc_us / b8_us,
        if err8 < 1e-10 { "PASS" } else { "FAIL" },
    );
    println!("{}", "-".repeat(88));
    println!(
        "  {:<38}  {:>10.3}  {:>7.2}x",
        "scalar (f32, libm)",
        sc_f32_us / 1000.0,
        sc_us / sc_f32_us,
    );
    println!(
        "  {:<38}  {:>10.3}  {:>7.2}x  [{}]  max_err={err2_f32:.2e}",
        "batch-2 (BatchScalarF32<2>)",
        b2_f32_us / 1000.0,
        sc_us / b2_f32_us,
        if err2_f32 < 1e-2 { "PASS" } else { "FAIL" },
    );
    println!(
        "  {:<38}  {:>10.3}  {:>7.2}x  [{}]  max_err={err4_f32:.2e}",
        "batch-4 (BatchScalarF32<4>, native NEON)",
        b4_f32_us / 1000.0,
        sc_us / b4_f32_us,
        if err4_f32 < 1e-2 { "PASS" } else { "FAIL" },
    );
    println!(
        "  {:<38}  {:>10.3}  {:>7.2}x  [{}]  max_err={err8_f32:.2e}",
        "batch-8 (BatchScalarF32<8>)",
        b8_f32_us / 1000.0,
        sc_us / b8_f32_us,
        if err8_f32 < 1e-2 { "PASS" } else { "FAIL" },
    );
    println!("{}", "=".repeat(88));
}
