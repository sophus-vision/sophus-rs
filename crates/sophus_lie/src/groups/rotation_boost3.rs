use core::marker::PhantomData;

use sophus_autodiff::{
    linalg::{
        EPS_F64,
        IsScalar,
    },
    manifold::IsTangent,
    params::{
        HasParams,
        IsParamsImpl,
    },
    prelude::*,
};

use crate::{
    HasDisambiguate,
    IsLieFactorGroupImpl,
    IsLieGroupImpl,
    Rotation3,
    Rotation3Impl,
    lie_group::LieGroup,
};

extern crate alloc;

/// 3-d **rotation and boost** – element of the Homogeneous Galilei group HG(3)**
///
/// ## Generic parameters
/// * **BATCH** – batch dimension ( = 1 for plain `f64` or `DualScalar`).
/// * **DM, DN** – static Jacobian shape when `S = DualScalar<DM,DN>` (both 0 when `S = f64`).
///
/// ## Overview
///
/// * **Tangent space:** 6 DoF – **[ ω , ν ]**, with `ω` the 3-d **angular** rate and `α` the 3-d
///   **boost** rate.
/// * **Internal parameters:** 7 – **[ q , β ]**, quaternion `q` ( |q| = 1 ) and boost `β ∈ ℝ³`.
/// * **Action space:** 4 (HG(3) acts on 4-d space-time points)
/// * **Matrix size:** 4 (represented as 4 × 4 matrices)
///
/// ### Group structure
///
/// *Matrix representation*
/// ```ascii
/// ---------
/// | 1 | O |
/// ---------
/// | β | R |
/// ---------
/// ```
/// `R ∈ SO(3)`, `β ∈ ℝ³`.
///
/// It acts on 4d space-time points as follows:
///
/// ```ascii
/// (R, β) ⊗ (t, p) = ( t, R·p + tβ )
/// ```
///
/// *Group operation*
/// ```ascii
/// (Rₗ, βₗ) ⊗ (Rᵣ, βᵣ) = ( Rₗ·Rᵣ,  Rₗ·βᵣ + βₗ )
/// ```
/// *Inverse*
/// ```ascii
/// (R, β)⁻¹ = ( Rᵀ,  -Rᵀ·β )
/// ```
///
/// ### Lie-group properties
///
/// **Hat operator**
/// ```ascii
///            -------------
///            | 0  |  O   |
/// (β;ω)^  =  -------------
///            | β  | [ω]ₓ |
///            -------------
/// ```
///
/// **Exponential map** `exp : ℝ⁶ → HG(3)`
/// ```ascii
/// exp(ω,ν) = ( exp_so3(ω),  V(ω) · ν )
/// ```
/// where `V(ω)` is `Rotation3::mat_v`.
///
/// **Group adjoint** `Adj : HG(3) → ℝ⁶ˣ⁶`
/// ```ascii
///              |---------------|
///              |  R     | O₃ₓ₃ |
/// Adj(R,β)  =  |---------------|
///              | [β]ₓ·R |  R   |
///              |---------------|
/// ```
///
/// **Lie-algebra adjoint** `ad : ℝ⁶ → ℝ⁶ˣ⁶`
/// ```ascii
///             |-------------|
///             | [ω]ₓ | O₃ₓ₃ |
/// ad(ω; ν) =  |-------------|
///             | [ν]ₓ | [ω]ₓ |
///             |-------------|
/// ```
pub type RotationBoost3<S, const BATCH: usize, const DM: usize, const DN: usize> =
    LieGroup<S, 6, 7, 4, 4, BATCH, DM, DN, RotationBoost3Impl<S, BATCH, DM, DN>>;

/// 3d rotation-boost with f64 scalar type - element of the Homogeneous Galilei group HG(3)
///
/// See [RotationBoost3] for details.
pub type RotationBoost3F64 = RotationBoost3<f64, 1, 0, 0>;

/// 3d rotation-boost implementation.
///
/// See [RotationBoost3] for details.
#[derive(Debug, Copy, Clone, Default)]
pub struct RotationBoost3Impl<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    HasDisambiguate<S, 7, BATCH, DM, DN> for RotationBoost3Impl<S, BATCH, DM, DN>
{
    fn disambiguate(params: S::Vector<7>) -> S::Vector<7> {
        let q = params.get_fixed_subvec::<4>(0);
        let beta = params.get_fixed_subvec::<3>(4);

        let is_positive = S::from_f64(0.0).less_equal(&q.elem(0));
        let q_fixed = q.select(&is_positive, -q);
        S::Vector::block_vec2(q_fixed, beta)
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsParamsImpl<S, 7, BATCH, DM, DN> for RotationBoost3Impl<S, BATCH, DM, DN>
{
    fn params_examples() -> alloc::vec::Vec<S::Vector<7>> {
        let id = Self::identity_params();

        let rx = Rotation3::<S, BATCH, DM, DN>::rot_x(S::from_f64(0.10));
        let ry = Rotation3::<S, BATCH, DM, DN>::rot_y(S::from_f64(-0.12));
        let rz = Rotation3::<S, BATCH, DM, DN>::rot_z(S::from_f64(0.08));

        alloc::vec![
            S::Vector::<7>::block_vec2(
                *(rx * ry).params(),
                S::Vector::<3>::from_f64_array([0.02, 0.01, -0.01]),
            ),
            id,
            S::Vector::<7>::block_vec2(*ry.params(), S::Vector::<3>::zeros()),
            S::Vector::<7>::block_vec2(*rz.params(), S::Vector::<3>::zeros()),
            {
                let beta = S::Vector::<3>::from_f64_array([0.05, 0.00, 0.00]);
                S::Vector::<7>::block_vec2(
                    *Rotation3::<S, BATCH, DM, DN>::identity().params(),
                    beta,
                )
            },
            S::Vector::<7>::block_vec2(
                *ry.params(),
                S::Vector::<3>::from_f64_array([0.00, 0.03, 0.01]),
            ),
            S::Vector::<7>::block_vec2(
                *rz.params(),
                S::Vector::<3>::from_f64_array([0.01, -0.04, 0.00]),
            ),
            {
                let q = Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                    0.25, -0.18, 0.12,
                ]));
                S::Vector::<7>::block_vec2(
                    *q.params(),
                    S::Vector::<3>::from_f64_array([0.06, 0.02, -0.05]),
                )
            },
        ]
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<7>> {
        alloc::vec![S::Vector::<7>::zeros()]
    }

    fn are_params_valid(params: S::Vector<7>) -> S::Mask {
        let q = params.get_fixed_subvec::<4>(0);
        (q.norm() - S::from_f64(1.0))
            .abs()
            .less_equal(&S::from_f64(EPS_F64))
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsTangent<S, 6, BATCH, DM, DN> for RotationBoost3Impl<S, BATCH, DM, DN>
{
    fn tangent_examples() -> alloc::vec::Vec<S::Vector<6>> {
        alloc::vec![
            S::Vector::<6>::from_f64_array([0.35, -0.30, 0.10, 0.10, -0.08, 0.02]),
            S::Vector::<6>::zeros(),
            S::Vector::<6>::from_f64_array([0.10, 0.00, 0.00, 0.00, 0.00, 0.00]),
            S::Vector::<6>::from_f64_array([0.00, 0.15, 0.00, 0.00, 0.00, 0.00]),
            S::Vector::<6>::from_f64_array([0.00, 0.00, -0.12, 0.00, 0.00, 0.00]),
            S::Vector::<6>::from_f64_array([0.00, 0.00, 0.00, 0.05, 0.00, 0.00]),
            S::Vector::<6>::from_f64_array([0.00, 0.00, 0.00, 0.00, -0.07, 0.00]),
            S::Vector::<6>::from_f64_array([0.00, 0.00, 0.00, 0.00, 0.00, 0.06]),
            S::Vector::<6>::from_f64_array([0.20, 0.10, 0.00, 0.03, 0.02, 0.00]),
            S::Vector::<6>::from_f64_array([-0.05, 0.00, 0.25, -0.01, 0.00, 0.04]),
        ]
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsLieGroupImpl<S, 6, 7, 4, 4, BATCH, DM, DN> for RotationBoost3Impl<S, BATCH, DM, DN>
{
    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = false;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<7> {
        S::Vector::<7>::block_vec2(
            *Rotation3::<S, BATCH, DM, DN>::identity().params(),
            S::Vector::<3>::zeros(),
        )
    }

    fn adj(params: &S::Vector<7>) -> S::Matrix<6, 6> {
        let q = params.get_fixed_subvec::<4>(0);
        let beta = params.get_fixed_subvec::<3>(4);

        let r: S::Matrix<3, 3> = Rotation3Impl::<S, BATCH, DM, DN>::matrix(&q);
        let bhat_r: S::Matrix<3, 3> = Rotation3Impl::<S, BATCH, DM, DN>::hat(&beta).mat_mul(r);

        S::Matrix::<6, 6>::block_mat2x2::<3, 3, 3, 3>((r, S::Matrix::zeros()), (bhat_r, r))
    }

    fn group_mul(lhs: &S::Vector<7>, rhs: &S::Vector<7>) -> S::Vector<7> {
        // rotations
        let q1 = lhs.get_fixed_subvec::<4>(0);
        let q2 = rhs.get_fixed_subvec::<4>(0);
        let q = Rotation3Impl::<S, BATCH, DM, DN>::group_mul(&q1, &q2);

        // boosts
        let beta1 = lhs.get_fixed_subvec::<3>(4);
        let beta2 = rhs.get_fixed_subvec::<3>(4);
        let r1 = Rotation3Impl::<S, BATCH, DM, DN>::matrix(&q1);
        let beta = beta1 + r1 * beta2;

        S::Vector::block_vec2(q, beta)
    }

    fn inverse(p: &S::Vector<7>) -> S::Vector<7> {
        let q = p.get_fixed_subvec::<4>(0);
        let beta = p.get_fixed_subvec::<3>(4);

        let q_inv = Rotation3Impl::<S, BATCH, DM, DN>::inverse(&q);
        let r_t = Rotation3Impl::<S, BATCH, DM, DN>::matrix(&q).transposed();
        let beta_inv = -(r_t * beta);

        S::Vector::block_vec2(q_inv, beta_inv)
    }

    fn exp(xi: &S::Vector<6>) -> S::Vector<7> {
        let omega = xi.get_fixed_subvec::<3>(0);
        let v = xi.get_fixed_subvec::<3>(3);

        let q = Rotation3::<S, BATCH, DM, DN>::exp(omega);
        let beta = Rotation3Impl::<S, BATCH, DM, DN>::mat_v(&omega) * v;

        S::Vector::block_vec2(*q.params(), beta)
    }

    fn log(p: &S::Vector<7>) -> S::Vector<6> {
        let q = p.get_fixed_subvec::<4>(0);
        let beta = p.get_fixed_subvec::<3>(4);

        let ω = Rotation3Impl::<S, BATCH, DM, DN>::log(&q);
        let v = Rotation3Impl::<S, BATCH, DM, DN>::mat_v_inverse(&ω) * beta;

        S::Vector::block_vec2(ω, v)
    }

    fn hat(xi: &S::Vector<6>) -> S::Matrix<4, 4> {
        let omega_hat = Rotation3Impl::<S, BATCH, DM, DN>::hat(&xi.get_fixed_subvec::<3>(0));
        let beta = xi.get_fixed_subvec::<3>(3);

        S::Matrix::block_mat2x2::<1, 3, 1, 3>(
            (S::Matrix::zeros(), S::Matrix::zeros()),
            (beta.to_mat(), omega_hat),
        )
    }

    fn vee(m: &S::Matrix<4, 4>) -> S::Vector<6> {
        let mut omega = S::Vector::<3>::zeros();
        *omega.elem_mut(0) = m.elem([3, 2]);
        *omega.elem_mut(1) = m.elem([1, 3]);
        *omega.elem_mut(2) = m.elem([2, 1]);

        let beta = S::Vector::<3>::from_array([m.elem([1, 0]), m.elem([2, 0]), m.elem([3, 0])]);
        S::Vector::block_vec2(omega, beta)
    }

    fn transform(p: &S::Vector<7>, event: &S::Vector<4>) -> S::Vector<4> {
        Self::matrix(p) * *event
    }

    fn to_ambient(point: &S::Vector<4>) -> S::Vector<4> {
        *point
    }

    fn compact(params: &S::Vector<7>) -> S::Matrix<4, 4> {
        Self::matrix(params)
    }

    fn ad(xi: &S::Vector<6>) -> S::Matrix<6, 6> {
        let omega_hat = Rotation3Impl::<S, BATCH, DM, DN>::hat(&xi.get_fixed_subvec::<3>(0));
        let v_hat = Rotation3Impl::<S, BATCH, DM, DN>::hat(&xi.get_fixed_subvec::<3>(3));
        S::Matrix::<6, 6>::block_mat2x2::<3, 3, 3, 3>(
            (omega_hat, S::Matrix::zeros()),
            (v_hat, omega_hat),
        )
    }

    type GenG<S2: IsScalar<BATCH, DM, DN>> = RotationBoost3Impl<S2, BATCH, DM, DN>;
    type RealG = RotationBoost3Impl<S::RealScalar, BATCH, 0, 0>;
    type DualG<const M: usize, const N: usize> =
        RotationBoost3Impl<S::DualScalar<M, N>, BATCH, M, N>;

    fn matrix(p: &S::Vector<7>) -> S::Matrix<4, 4> {
        let q = p.get_fixed_subvec::<4>(0);
        let beta = p.get_fixed_subvec::<3>(4);

        let mat_r = Rotation3Impl::<S, BATCH, DM, DN>::matrix(&q);

        S::Matrix::block_mat2x2::<1, 3, 1, 3>(
            (S::Matrix::ones(), S::Matrix::zeros()),
            (beta.to_mat(), mat_r),
        )
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsLieFactorGroupImpl<S, 6, 7, 4, BATCH, DM, DN> for RotationBoost3Impl<S, BATCH, DM, DN>
{
    type GenFactorG<S2: IsScalar<BATCH, M, N>, const M: usize, const N: usize> =
        RotationBoost3Impl<S2, BATCH, M, N>;
    type RealFactorG = RotationBoost3Impl<S::RealScalar, BATCH, 0, 0>;
    type DualFactorG<const M: usize, const N: usize> =
        RotationBoost3Impl<S::DualScalar<M, N>, BATCH, M, N>;

    fn mat_v(tan: &S::Vector<6>) -> S::Matrix<4, 4> {
        let omega = tan.get_fixed_subvec::<3>(0);
        let v = tan.get_fixed_subvec::<3>(3);
        let mat_v = Rotation3Impl::<S, BATCH, DM, DN>::mat_v(&omega);
        let col0 = mat_v * v.scaled(S::from_f64(0.5));

        S::Matrix::block_mat2x2::<1, 3, 1, 3>(
            (S::Matrix::ones(), S::Matrix::zeros()),
            (col0.to_mat(), mat_v),
        )
    }

    fn mat_v_inverse(tan: &S::Vector<6>) -> S::Matrix<4, 4> {
        let omega = tan.get_fixed_subvec::<3>(0);
        let v = tan.get_fixed_subvec::<3>(3);
        let v3_inv = Rotation3Impl::<S, BATCH, DM, DN>::mat_v_inverse(&omega);
        let half_v = v.scaled(S::from_f64(-0.5));

        S::Matrix::block_mat2x2::<1, 3, 1, 3>(
            (S::Matrix::ones(), S::Matrix::zeros()),
            (half_v.to_mat(), v3_inv),
        )
    }

    fn ad_of_translation(p: &S::Vector<4>) -> S::Matrix<4, 6> {
        let t = p.elem(0);
        let x = p.get_fixed_subvec::<3>(1);
        let x_hat = Rotation3Impl::<S, BATCH, DM, DN>::hat(&x);

        let mut mat_m = S::Matrix::<4, 6>::zeros();
        for r in 0..3 {
            for c in 0..3 {
                *mat_m.elem_mut([r + 1, c]) = x_hat.elem([r, c]);
            }
        }
        for i in 0..3 {
            *mat_m.elem_mut([i + 1, i + 3]) = -t;
        }
        mat_m
    }

    fn adj_of_translation(params: &S::Vector<7>, p: &S::Vector<4>) -> S::Matrix<4, 6> {
        let t = p.elem(0);
        let x = p.get_fixed_subvec::<3>(1);
        let q = params.get_fixed_subvec::<4>(0);
        let beta = params.get_fixed_subvec::<3>(4);
        let mat_r = Rotation3Impl::<S, BATCH, DM, DN>::matrix(&q);

        let omega_block =
            Rotation3Impl::<S, BATCH, DM, DN>::hat(&(x - beta.scaled(t))).mat_mul(mat_r);
        let v_block = -(mat_r.scaled(t));

        S::Matrix::block_mat2x2::<1, 3, 3, 3>(
            (S::Matrix::zeros(), S::Matrix::zeros()),
            (omega_block, v_block),
        )
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use sophus_autodiff::linalg::VecF64;

    use super::*;

    #[test]
    fn compose_id() {
        let id = RotationBoost3F64::identity();
        let p = RotationBoost3F64::exp(VecF64::from_array([0.1, 0.2, -0.1, 0.05, 0.0, 0.0]));
        let q = id * p;
        assert_relative_eq!(p.params(), q.params(), epsilon = 1e-12);
    }

    #[test]
    fn rotation_boost3_prop_tests() {
        #[cfg(feature = "simd")]
        use sophus_autodiff::dual::DualBatchScalar;
        use sophus_autodiff::dual::DualScalar;
        #[cfg(feature = "simd")]
        use sophus_autodiff::linalg::BatchScalarF64;

        RotationBoost3F64::test_suite();
        #[cfg(feature = "simd")]
        RotationBoost3::<BatchScalarF64<8>, 8, 0, 0>::test_suite();
        RotationBoost3::<DualScalar<3, 1>, 1, 3, 1>::test_suite();
        #[cfg(feature = "simd")]
        RotationBoost3::<DualBatchScalar<8, 1, 1>, 8, 1, 1>::test_suite();
    }
}
