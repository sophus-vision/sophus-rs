use core::{
    borrow::Borrow,
    f64,
    marker::PhantomData,
};

use log::warn;
use sophus_autodiff::{
    linalg::{
        EPS_F64,
        MatF64,
        cross,
    },
    manifold::IsTangent,
    params::{
        HasParams,
        IsParamsImpl,
    },
};

use crate::{
    EmptySliceError,
    HasAverage,
    HasDisambiguate,
    IsLieGroupImpl,
    IsRealLieFactorGroupImpl,
    IsRealLieGroupImpl,
    lie_group::{
        LieGroup,
        average::{
            IterativeAverageError,
            iterative_average,
        },
    },
    prelude::*,
};

extern crate alloc;

/// 3d rotations - element of the Special Orthogonal group SO(3)
///
///  * BATCH
///     - batch dimension. If S is f64 or [sophus_autodiff::dual::DualScalar] then BATCH=1.
///  * DM, DN
///     - DM x DN is the static shape of the Jacobian to be computed if S == DualScalar<DM,DN>. If S
///       == f64, then DM==0, DN==0.
pub type Rotation3<S, const BATCH: usize, const DM: usize, const DN: usize> =
    LieGroup<S, 3, 4, 3, 3, BATCH, DM, DN, Rotation3Impl<S, BATCH, DM, DN>>;

/// 3d rotation with f64 scalar type a - element of the Special Orthogonal group SO(3)
///
/// See [Rotation3] for details.
pub type Rotation3F64 = Rotation3<f64, 1, 0, 0>;

/// 3d rotation implementation details
#[derive(Debug, Copy, Clone, Default)]
pub struct Rotation3Impl<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    HasDisambiguate<S, 4, BATCH, DM, DN> for Rotation3Impl<S, BATCH, DM, DN>
{
    fn disambiguate(params: S::Vector<4>) -> S::Vector<4> {
        // make sure real component is always positive
        let is_positive = S::from_f64(0.0).less_equal(&params.elem(0));

        params.select(&is_positive, -params)
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsParamsImpl<S, 4, BATCH, DM, DN> for Rotation3Impl<S, BATCH, DM, DN>
{
    fn params_examples() -> alloc::vec::Vec<S::Vector<4>> {
        const NEAR_PI: f64 = f64::consts::PI - 1e-6;
        const NEAR_MINUS_PI: f64 = f64::consts::PI - 1e-6;

        alloc::vec![
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([0.0, 0.0, 0.0]))
                .params(),
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([0.1, 0.5, -0.1]))
                .params(),
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([0.1, 2.0, -0.1]))
                .params(),
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([0.0, 0.2, 1.0]))
                .params(),
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([-0.2, 0.0, 0.8]))
                .params(),
            // Test cases around +π and -π
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                NEAR_PI, 0.0, 0.0
            ]))
            .params(), // +π rotation about the x-axis
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                NEAR_MINUS_PI,
                0.0,
                0.0
            ]))
            .params(), // -π rotation about the x-axis
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                0.0, NEAR_PI, 0.0
            ]))
            .params(), // +π rotation about the y-axis
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                0.0,
                NEAR_MINUS_PI,
                0.0
            ]))
            .params(), // -π rotation about the y-axis
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                0.0, 0.0, NEAR_PI
            ]))
            .params(), // +π rotation about the z-axis
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                0.0,
                0.0,
                NEAR_MINUS_PI
            ]))
            .params(), // -π rotation about the z-axis
            // Close to +π and -π, but not exactly, to test boundary behavior
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([3.0, 0.0, 0.0]))
                .params(),
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([-3.0, 0.0, 0.0]))
                .params(),
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([0.0, 3.0, 0.0]))
                .params(),
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([0.0, -3.0, 0.0]))
                .params(),
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([0.0, 0.0, 3.0]))
                .params(),
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([0.0, 0.0, -3.0]))
                .params(),
            // Halfway to π rotations, to cover intermediate rotations
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                f64::consts::FRAC_PI_2,
                0.0,
                0.0,
            ]))
            .params(), // +π/2 about x-axis
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                0.0,
                f64::consts::FRAC_PI_2,
                0.0,
            ]))
            .params(), // +π/2 about y-axis
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                0.0,
                0.0,
                f64::consts::FRAC_PI_2,
            ]))
            .params(), // +π/2 about z-axis
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                -f64::consts::FRAC_PI_2,
                0.0,
                0.0,
            ]))
            .params(), // -π/2 about x-axis
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                0.0,
                -f64::consts::FRAC_PI_2,
                0.0,
            ]))
            .params(), // -π/2 about y-axis
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                0.0,
                0.0,
                -f64::consts::FRAC_PI_2,
            ]))
            .params(), // -π/2 about z-axis
            // Complex combination rotations around the boundary
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                NEAR_PI / 2.0,
                NEAR_PI / 2.0,
                0.0,
            ]))
            .params(),
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                NEAR_MINUS_PI / 2.0,
                0.0,
                NEAR_PI / 2.0,
            ]))
            .params(),
            *Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([
                0.0,
                NEAR_PI / 2.0,
                NEAR_PI / 2.0,
            ]))
            .params(),
        ]
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<4>> {
        alloc::vec![
            S::Vector::<4>::from_f64_array([0.0, 0.0, 0.0, 0.0]),
            S::Vector::<4>::from_f64_array([0.5, 0.5, 0.5, 0.0]),
            S::Vector::<4>::from_f64_array([0.5, -0.5, 0.5, 1.0]),
        ]
    }

    fn are_params_valid<P>(params: P) -> S::Mask
    where
        P: Borrow<S::Vector<4>>,
    {
        let norm = params.borrow().norm();
        (norm - S::from_f64(1.0))
            .abs()
            .less_equal(&S::from_f64(EPS_F64))
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsTangent<S, 3, BATCH, DM, DN> for Rotation3Impl<S, BATCH, DM, DN>
{
    fn tangent_examples() -> alloc::vec::Vec<S::Vector<3>> {
        alloc::vec![
            S::Vector::<3>::from_f64_array([0.0, 0.0, 0.0]),
            S::Vector::<3>::from_f64_array([1.0, 0.0, 0.0]),
            S::Vector::<3>::from_f64_array([0.0, 1.0, 0.0]),
            S::Vector::<3>::from_f64_array([0.0, 0.0, 1.0]),
            S::Vector::<3>::from_f64_array([0.5, 0.5, 0.1]),
            S::Vector::<3>::from_f64_array([-0.1, -0.5, -0.5]),
            S::Vector::<3>::from_f64_array([2.5, 0.5, 0.1]),
            S::Vector::<3>::from_f64_array([0.5, 2.5, 0.1]),
            S::Vector::<3>::from_f64_array([0.5, 0.1, -2.5]),
        ]
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsLieGroupImpl<S, 3, 4, 3, 3, BATCH, DM, DN> for Rotation3Impl<S, BATCH, DM, DN>
{
    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = true;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<4> {
        S::Vector::<4>::from_f64_array([1.0, 0.0, 0.0, 0.0])
    }

    fn adj(params: &S::Vector<4>) -> S::Matrix<3, 3> {
        Self::matrix(params)
    }

    fn exp(omega: &S::Vector<3>) -> S::Vector<4> {
        const EPS: f64 = EPS_F64;
        let theta_sq = omega.squared_norm();

        let theta_po4 = theta_sq * theta_sq;
        let theta = theta_sq.sqrt();
        let half_theta: S = S::from_f64(0.5) * theta;

        let near_zero = theta_sq.less_equal(&S::from_f64(EPS * EPS));

        let imag_factor = (S::from_f64(0.5) - S::from_f64(1.0 / 48.0) * theta_sq
            + S::from_f64(1.0 / 3840.0) * theta_po4)
            .select(&near_zero, half_theta.sin() / theta);

        let real_factor = (S::from_f64(1.0) - S::from_f64(1.0 / 8.0) * theta_sq
            + S::from_f64(1.0 / 384.0) * theta_po4)
            .select(&near_zero, half_theta.cos());

        S::Vector::<4>::from_array([
            real_factor,
            imag_factor * omega.elem(0),
            imag_factor * omega.elem(1),
            imag_factor * omega.elem(2),
        ])
    }

    fn log(params: &S::Vector<4>) -> S::Vector<3> {
        const EPS: f64 = EPS_F64;
        let ivec: S::Vector<3> = params.get_fixed_subvec::<3>(1);

        let squared_n = ivec.squared_norm();
        let w = params.elem(0);

        let near_zero = squared_n.less_equal(&S::from_f64(EPS * EPS));

        let w_sq = w * w;
        let t0 = S::from_f64(2.0) / w - S::from_f64(2.0 / 3.0) * squared_n / (w_sq * w);

        let n = squared_n.sqrt();

        let sign = S::from_f64(-1.0).select(&w.less_equal(&S::from_f64(0.0)), S::from_f64(1.0));
        let atan_nbyw = sign * n.atan2(sign * w);

        let t = S::from_f64(2.0) * atan_nbyw / n;

        let two_atan_nbyd_by_n = t0.select(&near_zero, t);

        ivec.scaled(two_atan_nbyd_by_n)
    }

    fn hat(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        let o0 = omega.elem(0);
        let o1 = omega.elem(1);
        let o2 = omega.elem(2);

        S::Matrix::from_array2([
            [S::zero(), -o2, o1],
            [o2, S::zero(), -o0],
            [-o1, o0, S::zero()],
        ])
    }

    fn vee(omega_hat: &S::Matrix<3, 3>) -> S::Vector<3> {
        S::Vector::<3>::from_array([
            omega_hat.elem([2, 1]),
            omega_hat.elem([0, 2]),
            omega_hat.elem([1, 0]),
        ])
    }

    fn inverse(params: &S::Vector<4>) -> S::Vector<4> {
        S::Vector::from_array([
            params.elem(0),
            -params.elem(1),
            -params.elem(2),
            -params.elem(3),
        ])
    }

    fn transform(params: &S::Vector<4>, point: &S::Vector<3>) -> S::Vector<3> {
        Self::matrix(params) * *point
    }

    fn to_ambient(point: &S::Vector<3>) -> S::Vector<3> {
        *point
    }

    fn compact(params: &S::Vector<4>) -> S::Matrix<3, 3> {
        Self::matrix(params)
    }

    fn matrix(params: &S::Vector<4>) -> S::Matrix<3, 3> {
        let ivec = params.get_fixed_subvec::<3>(1);
        let re = &params.elem(0);

        let unit_x = S::Vector::from_f64_array([1.0, 0.0, 0.0]);
        let unit_y = S::Vector::from_f64_array([0.0, 1.0, 0.0]);
        let unit_z = S::Vector::from_f64_array([0.0, 0.0, 1.0]);

        let two = &S::from_f64(2.0);

        let uv_x: S::Vector<3> = cross::<S, BATCH, DM, DN>(ivec, unit_x).scaled(two);
        let uv_y: S::Vector<3> = cross::<S, BATCH, DM, DN>(ivec, unit_y).scaled(two);
        let uv_z: S::Vector<3> = cross::<S, BATCH, DM, DN>(ivec, unit_z).scaled(two);

        let col_x = unit_x + cross::<S, BATCH, DM, DN>(ivec, uv_x) + uv_x.scaled(re);
        let col_y = unit_y + cross::<S, BATCH, DM, DN>(ivec, uv_y) + uv_y.scaled(re);
        let col_z = unit_z + cross::<S, BATCH, DM, DN>(ivec, uv_z) + uv_z.scaled(re);

        S::Matrix::block_mat1x2::<1, 2>(
            col_x.to_mat(),
            S::Matrix::block_mat1x2(col_y.to_mat(), col_z.to_mat()),
        )
    }

    fn ad(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        Self::hat(omega)
    }

    type GenG<S2: IsScalar<BATCH, DM, DN>> = Rotation3Impl<S2, BATCH, DM, DN>;
    type RealG = Rotation3Impl<S::RealScalar, BATCH, 0, 0>;
    type DualG<const M: usize, const N: usize> = Rotation3Impl<S::DualScalar<M, N>, BATCH, M, N>;

    fn group_mul(lhs_params: &S::Vector<4>, rhs_params: &S::Vector<4>) -> S::Vector<4> {
        let lhs_re = lhs_params.elem(0);
        let rhs_re = rhs_params.elem(0);

        let lhs_ivec = lhs_params.get_fixed_subvec::<3>(1);
        let rhs_ivec = rhs_params.get_fixed_subvec::<3>(1);

        let re = lhs_re * rhs_re - lhs_ivec.dot(rhs_ivec);
        let ivec = rhs_ivec.scaled(lhs_re)
            + lhs_ivec.scaled(rhs_re)
            + cross::<S, BATCH, DM, DN>(lhs_ivec, rhs_ivec);

        let mut params = S::Vector::block_vec2(re.to_vec(), ivec);

        if ((params.norm() - S::from_f64(1.0))
            .abs()
            .greater_equal(&S::from_f64(EPS_F64)))
        .any()
        {
            // todo: use tailor approximation for norm close to 1
            params = params.normalized();
        }
        params
    }
}

impl<S: IsRealScalar<BATCH>, const BATCH: usize> IsRealLieGroupImpl<S, 3, 4, 3, 3, BATCH>
    for Rotation3Impl<S, BATCH, 0, 0>
{
    fn dx_exp_x_at_0() -> S::Matrix<4, 3> {
        S::Matrix::from_f64_array2([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ])
    }

    fn da_a_mul_b(a: &S::Vector<4>, b: &S::Vector<4>) -> S::Matrix<4, 4> {
        let lhs_re = a.elem(0);
        let rhs_re = b.elem(0);

        let lhs_ivec = a.get_fixed_subvec::<3>(1);
        let rhs_ivec = b.get_fixed_subvec::<3>(1);

        let re = lhs_re * rhs_re - lhs_ivec.dot(rhs_ivec);

        let is_positive = S::from_f64(0.0).less_equal(&re);

        let b_real = b.elem(0);
        let b_imag0 = b.elem(1);
        let b_imag1 = b.elem(2);
        let b_imag2 = b.elem(3);

        S::Matrix::<4, 4>::from_array2([
            [b_real, -b_imag0, -b_imag1, -b_imag2],
            [b_imag0, b_real, b_imag2, -b_imag1],
            [b_imag1, -b_imag2, b_real, b_imag0],
            [b_imag2, b_imag1, -b_imag0, b_real],
        ])
        .select(
            &is_positive,
            S::Matrix::<4, 4>::from_array2([
                [-b_real, b_imag0, b_imag1, b_imag2],
                [-b_imag0, -b_real, -b_imag2, b_imag1],
                [-b_imag1, b_imag2, -b_real, -b_imag0],
                [-b_imag2, -b_imag1, b_imag0, -b_real],
            ]),
        )
    }

    fn db_a_mul_b(a: &S::Vector<4>, b: &S::Vector<4>) -> S::Matrix<4, 4> {
        let lhs_re = a.elem(0);
        let rhs_re = b.elem(0);
        let lhs_ivec = a.get_fixed_subvec::<3>(1);
        let rhs_ivec = b.get_fixed_subvec::<3>(1);
        let re = lhs_re * rhs_re - lhs_ivec.dot(rhs_ivec);
        let is_positive = S::from_f64(0.0).less_equal(&re);

        let a_real = a.elem(0);
        let a_imag0 = a.elem(1);
        let a_imag1 = a.elem(2);
        let a_imag2 = a.elem(3);

        S::Matrix::<4, 4>::from_array2([
            [a_real, -a_imag0, -a_imag1, -a_imag2],
            [a_imag0, a_real, -a_imag2, a_imag1],
            [a_imag1, a_imag2, a_real, -a_imag0],
            [a_imag2, -a_imag1, a_imag0, a_real],
        ])
        .select(
            &is_positive,
            S::Matrix::<4, 4>::from_array2([
                [-a_real, a_imag0, a_imag1, a_imag2],
                [-a_imag0, -a_real, a_imag2, -a_imag1],
                [-a_imag1, -a_imag2, -a_real, a_imag0],
                [-a_imag2, a_imag1, -a_imag0, -a_real],
            ]),
        )
    }

    fn dx_exp_x_times_point_at_0(point: &S::Vector<3>) -> S::Matrix<3, 3> {
        Self::hat(&-*point)
    }

    fn dx_exp(omega: &S::Vector<3>) -> S::Matrix<4, 3> {
        let theta_sq = omega.squared_norm();

        let near_zero = theta_sq.less_equal(&S::from_f64(EPS_F64));

        let dx0 = Self::dx_exp_x_at_0();

        let omega_0 = omega.elem(0);
        let omega_1 = omega.elem(1);
        let omega_2 = omega.elem(2);
        let theta = theta_sq.sqrt();
        let a = (S::from_f64(0.5) * theta).sin() / theta;
        let b = (S::from_f64(0.5) * theta).cos() / (theta_sq)
            - S::from_f64(2.0) * (S::from_f64(0.5) * theta).sin() / (theta_sq * theta);

        let dx = S::Matrix::from_array2([
            [-omega_0 * a, -omega_1 * a, -omega_2 * a],
            [
                omega_0 * omega_0 * b + S::from_f64(2.0) * a,
                omega_0 * omega_1 * b,
                omega_0 * omega_2 * b,
            ],
            [
                omega_0 * omega_1 * b,
                omega_1 * omega_1 * b + S::from_f64(2.0) * a,
                omega_1 * omega_2 * b,
            ],
            [
                omega_0 * omega_2 * b,
                omega_1 * omega_2 * b,
                omega_2 * omega_2 * b + S::from_f64(2.0) * a,
            ],
        ])
        .scaled(S::from_f64(0.5));
        dx0.select(&near_zero, dx)
    }

    fn dx_log_x(params: &S::Vector<4>) -> S::Matrix<3, 4> {
        let ivec: S::Vector<3> = params.get_fixed_subvec::<3>(1);
        let w = params.elem(0);
        let squared_n = ivec.squared_norm();

        let near_zero = squared_n.less_equal(&S::from_f64(EPS_F64));

        let m0 = S::Matrix::<3, 4>::block_mat1x2(
            S::Matrix::<3, 1>::zeros(),
            S::Matrix::<3, 3>::identity().scaled(S::from_f64(2.0)),
        );

        let n = squared_n.sqrt();
        let theta = S::from_f64(2.0) * n.atan2(w);

        let dw_ivec_theta: S::Vector<3> = ivec.scaled(S::from_f64(-2.0) / (squared_n + w * w));
        let factor =
            S::from_f64(2.0) * w / (squared_n * (squared_n + w * w)) - theta / (squared_n * n);

        let mm = ivec.outer(ivec).scaled(factor);

        m0.select(
            &near_zero,
            S::Matrix::block_mat1x2(
                dw_ivec_theta.to_mat(),
                S::Matrix::<3, 3>::identity().scaled(theta / n) + mm,
            ),
        )
    }

    fn has_shortest_path_ambiguity(params: &S::Vector<4>) -> S::Mask {
        let theta = Self::log(params).norm();
        (theta - S::from_f64(core::f64::consts::PI))
            .abs()
            .less_equal(&S::from_f64(EPS_F64.sqrt()))
    }

    fn dparams_matrix(params: &<S>::Vector<4>, col_idx: usize) -> <S>::Matrix<3, 4> {
        let re = params.elem(0);
        let i = params.elem(1);
        let j = params.elem(2);
        let k = params.elem(3);

        //  helper lambda:
        let scaled = |val: S, factor: f64| -> S { val * S::from_f64(factor) };

        match col_idx {
            // --------------------------------------------------
            // col_x
            //
            // partial wrt re => (0,       2k,    -2j)
            // partial wrt i  => (0,       2j,     2k)
            // partial wrt j  => (-4j,     2i,     -2re)
            // partial wrt k  => (-4k,     2re,    2i)
            0 => S::Matrix::from_array2([
                [S::zero(), S::zero(), scaled(j, -4.0), scaled(k, -4.0)],
                [
                    scaled(k, 2.0),
                    scaled(j, 2.0),
                    scaled(i, 2.0),
                    scaled(re, 2.0),
                ],
                [
                    scaled(j, -2.0),
                    scaled(k, 2.0),
                    scaled(re, -2.0),
                    scaled(i, 2.0),
                ],
            ]),

            // --------------------------------------------------
            // col_y
            //
            // partial wrt re => (-2k,        0,       2i)
            // partial wrt i  => (2j,        -4i,      2re)
            // partial wrt j  => (2i,         0,       2k)
            // partial wrt k  => (-2re,      -4k,      2j)
            1 => S::Matrix::from_array2([
                [
                    scaled(k, -2.0),  // row0, partial wrt re
                    scaled(j, 2.0),   // row0, partial wrt i
                    scaled(i, 2.0),   // row0, partial wrt j
                    scaled(re, -2.0), // row0, partial wrt k
                ],
                [
                    S::zero(),       // row1, partial wrt re
                    scaled(i, -4.0), // row1, partial wrt i
                    S::zero(),       // row1, partial wrt j
                    scaled(k, -4.0), // row1, partial wrt k
                ],
                [
                    scaled(i, 2.0),  // row2, partial wrt re
                    scaled(re, 2.0), // row2, partial wrt i
                    scaled(k, 2.0),  // row2, partial wrt j
                    scaled(j, 2.0),  // row2, partial wrt k
                ],
            ]),

            // --------------------------------------------------
            // col_z
            //
            // partial wrt re => ( 2j,      -2i,      0 )
            // partial wrt i  => ( 2k,      -2re,   -4i )
            // partial wrt j  => ( 2re,      2k,    -4j )
            // partial wrt k  => ( 2i,       2j,     0 )
            2 => S::Matrix::from_array2([
                [
                    scaled(j, 2.0),  // row0 wrt re
                    scaled(k, 2.0),  // row0 wrt i
                    scaled(re, 2.0), // row0 wrt j
                    scaled(i, 2.0),  // row0 wrt k
                ],
                [
                    scaled(i, -2.0),  // row1 wrt re
                    scaled(re, -2.0), // row1 wrt i
                    scaled(k, 2.0),   // row1 wrt j
                    scaled(j, 2.0),   // row1 wrt k
                ],
                [
                    S::zero(),       // row2 wrt re
                    scaled(i, -4.0), // row2 wrt i
                    scaled(j, -4.0), // row2 wrt j
                    S::zero(),       // row2 wrt k
                ],
            ]),

            _ => panic!("Invalid column index: {}", col_idx),
        }
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    crate::IsLieFactorGroupImpl<S, 3, 4, 3, BATCH, DM, DN> for Rotation3Impl<S, BATCH, DM, DN>
{
    type GenFactorG<S2: IsScalar<BATCH, M, N>, const M: usize, const N: usize> =
        Rotation3Impl<S2, BATCH, M, N>;
    type RealFactorG = Rotation3Impl<S::RealScalar, BATCH, 0, 0>;
    type DualFactorG<const M: usize, const N: usize> =
        Rotation3Impl<S::DualScalar<M, N>, BATCH, M, N>;

    fn mat_v(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        let theta_sq = omega.squared_norm();
        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S, BATCH, DM, DN>::hat(omega);
        let mat_omega_sq = mat_omega.mat_mul(mat_omega);

        let near_zero = theta_sq.less_equal(&S::from_f64(EPS_F64));

        let mat_v0 = S::Matrix::<3, 3>::identity() + mat_omega.scaled(S::from_f64(0.5));

        let theta = theta_sq.sqrt();
        let mat_v = S::Matrix::<3, 3>::identity()
            + mat_omega.scaled((S::from_f64(1.0) - theta.cos()) / theta_sq)
            + mat_omega_sq.scaled((theta - theta.sin()) / (theta_sq * theta));

        mat_v0.select(&near_zero, mat_v)
    }

    fn mat_v_inverse(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        let theta_sq = omega.dot(omega);
        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S, BATCH, DM, DN>::hat(omega);
        let mat_omega_sq = mat_omega.mat_mul(mat_omega);

        let near_zero = theta_sq.less_equal(&S::from_f64(EPS_F64));

        let mat_v_inv0 = S::Matrix::<3, 3>::identity() - mat_omega.scaled(S::from_f64(0.5))
            + mat_omega_sq.scaled(S::from_f64(1. / 12.));

        let theta = theta_sq.sqrt();
        let half_theta = S::from_f64(0.5) * theta;

        let mat_v_inv = S::Matrix::<3, 3>::identity() - mat_omega.scaled(S::from_f64(0.5))
            + mat_omega_sq.scaled(
                (S::from_f64(1.0)
                    - (S::from_f64(0.5) * theta * half_theta.cos()) / half_theta.sin())
                    / (theta * theta),
            );

        mat_v_inv0.select(&near_zero, mat_v_inv)
    }

    fn adj_of_translation(params: &S::Vector<4>, point: &S::Vector<3>) -> S::Matrix<3, 3> {
        Rotation3Impl::<S, BATCH, DM, DN>::hat(point)
            .mat_mul(Rotation3Impl::<S, BATCH, DM, DN>::matrix(params))
    }

    fn ad_of_translation(point: &S::Vector<3>) -> S::Matrix<3, 3> {
        Rotation3Impl::<S, BATCH, DM, DN>::hat(point)
    }
}

impl<S: IsRealScalar<BATCH>, const BATCH: usize> IsRealLieFactorGroupImpl<S, 3, 4, 3, BATCH>
    for Rotation3Impl<S, BATCH, 0, 0>
{
    fn dx_mat_v(omega: &S::Vector<3>) -> [S::Matrix<3, 3>; 3] {
        let theta_sq = omega.squared_norm();
        let theta_p4 = theta_sq * theta_sq;
        let dt_mat_omega_pos_idx = [[2, 1], [0, 2], [1, 0]];
        let dt_mat_omega_neg_idx = [[1, 2], [2, 0], [0, 1]];

        let near_zero = theta_sq.less_equal(&S::from_f64(EPS_F64));

        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S, BATCH, 0, 0>::hat(omega);
        let mat_omega_sq = mat_omega.mat_mul(mat_omega);

        let omega_x = omega.elem(0);
        let omega_y = omega.elem(1);
        let omega_z = omega.elem(2);

        let theta = theta_sq.sqrt();
        let domega_theta =
            S::Vector::from_array([omega_x / theta, omega_y / theta, omega_z / theta]);

        let a = (S::ones() - theta.cos()) / theta_sq;
        let dt_a = (S::from_f64(-2.0) + S::from_f64(2.0) * theta.cos() + theta * theta.sin())
            / (theta * theta_sq);

        let b = (theta - theta.sin()) / (theta_sq * theta);
        let dt_b = -(S::from_f64(2.0) * theta + theta * theta.cos()
            - S::from_f64(3.0) * theta.sin())
            / (theta_p4);

        let dt_mat_omega_sq = [
            S::Matrix::from_array2([
                [S::zeros(), omega_y, omega_z],
                [omega_y, S::from_f64(-2.0) * omega_x, S::zeros()],
                [omega_z, S::zeros(), S::from_f64(-2.0) * omega_x],
            ]),
            S::Matrix::from_array2([
                [S::from_f64(-2.0) * omega_y, omega_x, S::zeros()],
                [omega_x, S::zeros(), omega_z],
                [S::zeros(), omega_z, S::from_f64(-2.0) * omega_y],
            ]),
            S::Matrix::from_array2([
                [S::from_f64(-2.0) * omega_z, S::zeros(), omega_x],
                [S::zeros(), S::from_f64(-2.0) * omega_z, omega_y],
                [omega_x, omega_y, S::zeros()],
            ]),
        ];

        let a = S::from_f64(0.5).select(&near_zero, a);

        let set = |i| {
            let tmp0 = mat_omega.scaled(dt_a * domega_theta.elem(i));
            let tmp1 = dt_mat_omega_sq[i].scaled(b);
            let tmp2 = mat_omega_sq.scaled(dt_b * domega_theta.elem(i));

            let mut l_i: S::Matrix<3, 3> =
                S::Matrix::zeros().select(&near_zero, tmp0 + tmp1 + tmp2);
            let pos_idx = dt_mat_omega_pos_idx[i];
            *l_i.elem_mut(pos_idx) = a + l_i.elem(pos_idx);

            let neg_idx = dt_mat_omega_neg_idx[i];
            *l_i.elem_mut(neg_idx) = -a + l_i.elem(neg_idx);
            l_i
        };

        let l: [S::Matrix<3, 3>; 3] = [set(0), set(1), set(2)];

        l
    }

    fn dparams_matrix_times_point(params: &S::Vector<4>, point: &S::Vector<3>) -> S::Matrix<3, 4> {
        let r = params.elem(0);
        let ivec0 = params.elem(1);
        let ivec1 = params.elem(2);
        let ivec2 = params.elem(3);

        let p0 = point.elem(0);
        let p1 = point.elem(1);
        let p2 = point.elem(2);

        S::Matrix::from_array2([
            [
                S::from_f64(2.0) * ivec1 * p2 - S::from_f64(2.0) * ivec2 * p1,
                S::from_f64(2.0) * ivec1 * p1 + S::from_f64(2.0) * ivec2 * p2,
                S::from_f64(2.0) * r * p2 + S::from_f64(2.0) * ivec0 * p1
                    - S::from_f64(4.0) * ivec1 * p0,
                S::from_f64(-2.0) * r * p1 + S::from_f64(2.0) * ivec0 * p2
                    - S::from_f64(4.0) * ivec2 * p0,
            ],
            [
                S::from_f64(-2.0) * ivec0 * p2 + S::from_f64(2.0) * ivec2 * p0,
                S::from_f64(-2.0) * r * p2 - S::from_f64(4.0) * ivec0 * p1
                    + S::from_f64(2.0) * ivec1 * p0,
                S::from_f64(2.0) * ivec0 * p0 + S::from_f64(2.0) * ivec2 * p2,
                S::from_f64(2.0) * r * p0 + S::from_f64(2.0) * ivec1 * p2
                    - S::from_f64(4.0) * ivec2 * p1,
            ],
            [
                S::from_f64(2.0) * ivec0 * p1 - S::from_f64(2.0) * ivec1 * p0,
                S::from_f64(2.0) * r * p1 - S::from_f64(4.0) * ivec0 * p2
                    + S::from_f64(2.0) * ivec2 * p0,
                S::from_f64(-2.0) * r * p0 - S::from_f64(4.0) * ivec1 * p2
                    + S::from_f64(2.0) * ivec2 * p1,
                S::from_f64(2.0) * ivec0 * p0 + S::from_f64(2.0) * ivec1 * p1,
            ],
        ])
    }

    fn dx_mat_v_inverse(omega: &S::Vector<3>) -> [S::Matrix<3, 3>; 3] {
        let theta_sq = omega.squared_norm();
        let theta = theta_sq.sqrt();
        let half_theta = S::from_f64(0.5) * theta;
        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S, BATCH, 0, 0>::hat(omega);
        let mat_omega_sq = mat_omega.mat_mul(mat_omega);

        let dt_mat_omega_pos_idx = [[2, 1], [0, 2], [1, 0]];
        let dt_mat_omega_neg_idx = [[1, 2], [2, 0], [0, 1]];

        let omega_x = omega.elem(0);
        let omega_y = omega.elem(1);
        let omega_z = omega.elem(2);

        let near_zero = theta_sq.less_equal(&S::from_f64(EPS_F64));

        let domega_theta =
            S::Vector::from_array([omega_x / theta, omega_y / theta, omega_z / theta]);

        let c = (S::from_f64(1.0)
            - (S::from_f64(0.5) * theta * half_theta.cos()) / (half_theta.sin()))
            / theta_sq;

        let dt_c = (S::from_f64(-2.0)
            + (S::from_f64(0.25) * theta_sq) / (half_theta.sin() * half_theta.sin())
            + (half_theta * half_theta.cos()) / half_theta.sin())
            / (theta * theta_sq);

        let dt_mat_omega_sq: &[S::Matrix<3, 3>; 3] = &[
            S::Matrix::from_array2([
                [S::from_f64(0.0), omega_y, omega_z],
                [omega_y, S::from_f64(-2.0) * omega_x, S::from_f64(0.0)],
                [omega_z, S::from_f64(0.0), S::from_f64(-2.0) * omega_x],
            ]),
            S::Matrix::from_array2([
                [S::from_f64(-2.0) * omega_y, omega_x, S::from_f64(0.0)],
                [omega_x, S::from_f64(0.0), omega_z],
                [S::from_f64(0.0), omega_z, S::from_f64(-2.0) * omega_y],
            ]),
            S::Matrix::from_array2([
                [S::from_f64(-2.0) * omega_z, S::from_f64(0.0), omega_x],
                [S::from_f64(0.0), S::from_f64(-2.0) * omega_z, omega_y],
                [omega_x, omega_y, S::from_f64(0.0)],
            ]),
        ];

        let set = |i| -> S::Matrix<3, 3> {
            let dt_mat_omega_sq_i: &S::Matrix<3, 3> = &dt_mat_omega_sq[i];
            let mut l_i: S::Matrix<3, 3> = S::Matrix::zeros().select(
                &near_zero,
                dt_mat_omega_sq_i.scaled(c) + mat_omega_sq.scaled(domega_theta.elem(i) * dt_c),
            );

            let pos_idx = dt_mat_omega_pos_idx[i];
            *l_i.elem_mut(pos_idx) = S::from_f64(-0.5) + l_i.elem(pos_idx);

            let neg_idx = dt_mat_omega_neg_idx[i];
            *l_i.elem_mut(neg_idx) = S::from_f64(0.5) + l_i.elem(neg_idx);
            l_i
        };

        let l: [S::Matrix<3, 3>; 3] = [set(0), set(1), set(2)];

        l
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Rotation3<S, BATCH, DM, DN>
{
    /// Rotation around the x-axis.
    pub fn rot_x<U>(theta: U) -> Self
    where
        U: Borrow<S>,
    {
        let theta: &S = theta.borrow();
        Rotation3::exp(S::Vector::<3>::from_array([*theta, S::zero(), S::zero()]))
    }

    /// Rotation around the y-axis.
    pub fn rot_y<U>(theta: U) -> Self
    where
        U: Borrow<S>,
    {
        let theta: &S = theta.borrow();
        Rotation3::exp(S::Vector::<3>::from_array([S::zero(), *theta, S::zero()]))
    }

    /// Rotation around the z-axis.
    pub fn rot_z<U>(theta: U) -> Self
    where
        U: Borrow<S>,
    {
        let theta: &S = theta.borrow();
        Rotation3::exp(S::Vector::<3>::from_array([S::zero(), S::zero(), *theta]))
    }
}

impl<S: IsSingleScalar<DM, DN> + PartialOrd, const DM: usize, const DN: usize>
    Rotation3<S, 1, DM, DN>
{
    /// Is orthogonal with determinant 1.
    pub fn is_orthogonal_with_positive_det(mat: &MatF64<3, 3>, thr: f64) -> bool {
        // We expect: R * R^T = I   and   det(R) > 0
        //
        // (If R is orthogonal, then det(R) = +/-1.)
        let max_abs = ((mat * mat.transpose()) - MatF64::identity()).abs().max();
        max_abs < thr && mat.determinant() > 0.0
    }

    /// From a 3x3 rotation matrix. The matrix must be a valid rotation matrix,
    /// i.e., it must be orthogonal with determinant 1, otherwise None is returned.
    pub fn try_from_mat<M>(mat_r: M) -> Option<Rotation3<S, 1, DM, DN>>
    where
        M: Borrow<S::SingleMatrix<3, 3>>,
    {
        let mat_r = mat_r.borrow();
        if !Self::is_orthogonal_with_positive_det(&mat_r.single_real_matrix(), EPS_F64) {
            return None;
        }
        // Quaternions, Ken Shoemake
        // https://campar.in.tum.de/twiki/pub/Chair/DwarfTutorial/quatut.pdf
        //

        //     | 1 - 2*y^2 - 2*z^2  2*x*y - 2*z*w      2*x*z + 2*y*w |
        // R = | 2*x*y + 2*z*w      1 - 2*x^2 - 2*z^2  2*y*z - 2*x*w |
        //     | 2*x*z - 2*y*w      2*y*z + 2*x*w      1 - 2*x^2 - 2*y^2 |

        let trace = mat_r.elem([0, 0]) + mat_r.elem([1, 1]) + mat_r.elem([2, 2]);

        let q_params = if trace > S::from_f64(0.0) {
            // Calculate w first:
            //
            //   tr = trace(R)
            //      = 3 - 4*x^2 - 4*y^2 - 4*z^2
            //      = 3 - 4*(w^2 - w^2 + x^2 + y^2 + z^2)
            //      = 3 - 4*(1 - w^2)            [ since w^2 + x^2 + y^2 + z^2 = 1]
            //      = 1 + 2*w^2
            //
            //   w = sqrt(tr + 1) / 2
            //
            // (Note: We are in the tr>0 case, "tr + 1" is always positive,
            //        hence we can safely take the square root, and |w| > 0.5.)
            let sqrt_trace_plus_one = (S::from_f64(1.0) + trace).sqrt();
            let w = S::from_f64(0.5) * sqrt_trace_plus_one;

            // Now x:
            //
            //    R12 = 2*y*z - 2*x*w
            //    R21 = 2*y*z + 2*x*w
            //
            //    "2*y*z" = R12 + 2*x*w
            //    "2*y*z" = R21 - 2*x*w
            //
            //  Hence,
            //
            //    R12 + 2*x*w = R21 - 2*x*w
            //          4*x*w = R21 - R12
            //              x = (R21 - R12) / 4*w
            //
            // Similarly, for y and z.

            // f := 1 / (4*w) = 0.5 / sqrt(tr + 1)
            //
            // (Note: Since |w| > 0.5, the division is never close to zero.)
            let factor = S::from_f64(0.5) / sqrt_trace_plus_one;

            S::Vector::<4>::from_array([
                w,
                factor * (mat_r.elem([2, 1]) - mat_r.elem([1, 2])),
                factor * (mat_r.elem([0, 2]) - mat_r.elem([2, 0])),
                factor * (mat_r.elem([1, 0]) - mat_r.elem([0, 1])),
            ])
        } else {
            // Let us assume that R00 is the largest diagonal entry.
            // If not, we will change the order of the indices accordingly.
            let mut i = 0;
            if mat_r.elem([1, 1]) > mat_r.elem([0, 0]) {
                i = 1;
            }
            if mat_r.elem([2, 2]) > mat_r.elem([i, i]) {
                i = 2;
            }
            let j = (i + 1) % 3;
            let k = (j + 1) % 3;

            // Now we calculate q.x as follows (for i = 0, j = 1, k = 2):
            //
            // R00 - R11 - R22
            //  = 1 - 2*y^2 - 2*z^2 - 1 + 2*x^2 + 2*z^2 - 1 + 2*x^2 + 2*y^2
            //  = 4*x^2 - 1
            //
            // <=>
            //
            // x = sqrt((R00 - R11 - R22 + 1) / 4)
            //   = sqrt(R00 - R11 - R22 + 1) / 2
            //
            // (Note: Since the trace is negative, and R00 is the largest diagonal entry,
            //        R00 - R11 - R22 + 1 is always positive.)
            let sqrt = (mat_r.elem([i, i]) - mat_r.elem([j, j]) - mat_r.elem([k, k])
                + S::from_f64(1.0))
            .sqrt();
            let mut q = S::Vector::<4>::zeros();
            *q.elem_mut(i + 1) = S::from_f64(0.5) * sqrt;

            // For w:
            //
            // R21 - R12 = 2*y*z + 2*x*w - 2*y*z + 2*x*w
            //           = 4*x*w
            //
            // <=>
            //
            // w = (R21 - R12) / 4*x
            //   = (R21 - R12) / (2*sqrt(R00 - R11 - R22 + 1))

            let one_over_two_s = S::from_f64(0.5) / sqrt;
            *q.elem_mut(0) = (mat_r.elem([k, j]) - mat_r.elem([j, k])) * one_over_two_s;

            // For y:
            //
            // R01 + R10 = 2*x*y + 2*z*w + 2*x*y - 2*z*w
            //           = 4*x*y
            //
            // <=>
            //
            // y = (R01 + R10) / 4*x
            //   = (R01 + R10) / (2*sqrt(R00 - R11 - R22 + 1))
            *q.elem_mut(j + 1) = (mat_r.elem([j, i]) + mat_r.elem([i, j])) * one_over_two_s;

            // For z ...
            *q.elem_mut(k + 1) = (mat_r.elem([k, i]) + mat_r.elem([i, k])) * one_over_two_s;
            q
        };

        Some(Rotation3::from_params(q_params))
    }
}

#[test]
fn rotation3_prop_tests() {
    #[cfg(feature = "simd")]
    use sophus_autodiff::dual::DualBatchScalar;
    use sophus_autodiff::dual::DualScalar;
    #[cfg(feature = "simd")]
    use sophus_autodiff::linalg::BatchScalarF64;

    use crate::lie_group::{
        factor_lie_group::RealFactorLieGroupTest,
        real_lie_group::RealLieGroupTest,
    };

    Rotation3F64::test_suite();
    #[cfg(feature = "simd")]
    Rotation3::<BatchScalarF64<8>, 8, 0, 0>::test_suite();
    Rotation3::<DualScalar<3, 1>, 1, 3, 1>::test_suite();
    #[cfg(feature = "simd")]
    Rotation3::<DualBatchScalar<8, 1, 1>, 8, 1, 1>::test_suite();

    Rotation3F64::run_real_tests();
    #[cfg(feature = "simd")]
    Rotation3::<BatchScalarF64<8>, 8, 0, 0>::run_real_tests();

    Rotation3F64::run_real_factor_tests();
    #[cfg(feature = "simd")]
    Rotation3::<BatchScalarF64<8>, 8, 0, 0>::run_real_factor_tests();
}

impl<S: IsSingleScalar<DM, DN> + PartialOrd, const DM: usize, const DN: usize>
    HasAverage<S, 3, 4, 3, 3, DM, DN> for Rotation3<S, 1, DM, DN>
{
    fn average(
        parent_from_body_transforms: &[Rotation3<S, 1, DM, DN>],
    ) -> Result<Self, EmptySliceError> {
        // todo: Implement close form solution.

        match iterative_average(parent_from_body_transforms, 50) {
            Ok(parent_from_body_average) => Ok(parent_from_body_average),
            Err(err) => match err {
                IterativeAverageError::EmptySlice => Err(EmptySliceError),
                IterativeAverageError::NotConverged {
                    max_iteration_count,
                    parent_from_body_estimate,
                } => {
                    warn!(
                        "iterative_average did not converge (iters={max_iteration_count}), returning best guess."
                    );
                    Ok(parent_from_body_estimate)
                }
            },
        }
    }
}

#[test]
fn from_matrix_test() {
    use approx::assert_relative_eq;
    use log::info;

    for q in Rotation3F64::element_examples() {
        let mat: MatF64<3, 3> = q.matrix();

        info!("mat = {mat:?}");
        let q2: Rotation3F64 = Rotation3::try_from_mat(mat).unwrap();
        let mat2 = q2.matrix();

        info!("mat2 = {mat2:?}");
        assert_relative_eq!(mat, mat2, epsilon = EPS_F64);
    }

    // Iterate over all tangent too, just to get more examples.
    for t in Rotation3F64::tangent_examples() {
        let mat: MatF64<3, 3> = Rotation3F64::exp(t).matrix();
        info!("mat = {mat:?}");
        let t2: Rotation3F64 = Rotation3::try_from_mat(mat).unwrap();
        let mat2 = t2.matrix();
        info!("mat2 = {mat2:?}");
        assert_relative_eq!(mat, mat2, epsilon = EPS_F64);
    }
}
