use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::traits::IsLieGroupImpl;
use crate::traits::IsRealLieFactorGroupImpl;
use crate::traits::IsRealLieGroupImpl;
use sophus_core::linalg::vector::cross;
use sophus_core::manifold::traits::TangentImpl;
use sophus_core::params::ParamsImpl;
use std::marker::PhantomData;

/// 3d rotation implementation - SO(3)
#[derive(Debug, Copy, Clone, Default)]
pub struct Rotation3Impl<S: IsScalar<BATCH>, const BATCH: usize> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH>, const BATCH: usize> ParamsImpl<S, 4, BATCH> for Rotation3Impl<S, BATCH> {
    fn params_examples() -> Vec<S::Vector<4>> {
        vec![
            Rotation3::<S, BATCH>::exp(&S::Vector::<3>::from_f64_array([0.0, 0.0, 0.0]))
                .params()
                .clone(),
            Rotation3::<S, BATCH>::exp(&S::Vector::<3>::from_f64_array([0.1, 0.5, -0.1]))
                .params()
                .clone(),
            Rotation3::<S, BATCH>::exp(&S::Vector::<3>::from_f64_array([0.0, 0.2, 1.0]))
                .params()
                .clone(),
            Rotation3::<S, BATCH>::exp(&S::Vector::<3>::from_f64_array([-0.2, 0.0, 0.8]))
                .params()
                .clone(),
        ]
    }

    fn invalid_params_examples() -> Vec<S::Vector<4>> {
        vec![
            S::Vector::<4>::from_f64_array([0.0, 0.0, 0.0, 0.0]),
            S::Vector::<4>::from_f64_array([0.5, 0.5, 0.5, 0.0]),
            S::Vector::<4>::from_f64_array([0.5, (-0.5), 0.5, 1.0]),
        ]
    }

    fn are_params_valid(params: &S::Vector<4>) -> S::Mask {
        let norm = params.norm();
        (norm - S::from_f64(1.0))
            .abs()
            .less_equal(&S::from_f64(1e-6))
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> TangentImpl<S, 3, BATCH> for Rotation3Impl<S, BATCH> {
    fn tangent_examples() -> Vec<S::Vector<3>> {
        vec![
            S::Vector::<3>::from_f64_array([0.0, 0.0, 0.0]),
            S::Vector::<3>::from_f64_array([1.0, 0.0, 0.0]),
            S::Vector::<3>::from_f64_array([0.0, 1.0, 0.0]),
            S::Vector::<3>::from_f64_array([0.0, 0.0, 1.0]),
            S::Vector::<3>::from_f64_array([0.5, 0.5, 0.1]),
            S::Vector::<3>::from_f64_array([-0.1, -0.5, -0.5]),
        ]
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> IsLieGroupImpl<S, 3, 4, 3, 3, BATCH>
    for Rotation3Impl<S, BATCH>
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
        const EPS: f64 = 1e-8;
        let theta_sq = omega.squared_norm();

        let theta_po4 = theta_sq.clone() * theta_sq.clone();
        let theta = theta_sq.clone().sqrt();
        let half_theta: S = S::from_f64(0.5) * theta.clone();

        let near_zero = theta_sq.less_equal(&S::from_f64(EPS * EPS));

        let imag_factor = (S::from_f64(0.5) - S::from_f64(1.0 / 48.0) * theta_sq.clone()
            + S::from_f64(1.0 / 3840.0) * theta_po4.clone())
        .select(&near_zero, half_theta.clone().sin() / theta);

        let real_factor = (S::from_f64(1.0) - S::from_f64(1.0 / 8.0) * theta_sq
            + S::from_f64(1.0 / 384.0) * theta_po4)
            .select(&near_zero, half_theta.cos());

        S::Vector::<4>::from_array([
            real_factor,
            imag_factor.clone() * omega.get_elem(0),
            imag_factor.clone() * omega.get_elem(1),
            imag_factor * omega.get_elem(2),
        ])
    }

    fn log(params: &S::Vector<4>) -> S::Vector<3> {
        const EPS: f64 = 1e-8;
        let ivec: S::Vector<3> = params.get_fixed_subvec::<3>(1);

        let squared_n = ivec.squared_norm();
        let w = params.get_elem(0);

        let near_zero = squared_n.less_equal(&S::from_f64(EPS * EPS));

        let w_sq = w.clone() * w.clone();
        let t0 = S::from_f64(2.0) / w.clone()
            - S::from_f64(2.0 / 3.0) * squared_n.clone() / (w_sq * w.clone());

        let n = squared_n.sqrt();

        let sign = S::from_f64(-1.0).select(&w.less_equal(&S::from_f64(0.0)), S::from_f64(1.0));
        let atan_nbyw = sign.clone() * n.clone().atan2(sign * w);

        let t = S::from_f64(2.0) * atan_nbyw / n;

        let two_atan_nbyd_by_n = t0.select(&near_zero, t);

        ivec.scaled(two_atan_nbyd_by_n)
    }

    fn hat(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        let o0 = omega.get_elem(0);
        let o1 = omega.get_elem(1);
        let o2 = omega.get_elem(2);

        S::Matrix::from_array2([
            [S::zero(), -o2.clone(), o1.clone()],
            [o2, S::zero(), -o0.clone()],
            [-o1, o0, S::zero()],
        ])
    }

    fn vee(omega_hat: &S::Matrix<3, 3>) -> S::Vector<3> {
        S::Vector::<3>::from_array([
            omega_hat.get_elem([2, 1]),
            omega_hat.get_elem([0, 2]),
            omega_hat.get_elem([1, 0]),
        ])
    }

    fn inverse(params: &S::Vector<4>) -> S::Vector<4> {
        S::Vector::from_array([
            params.get_elem(0),
            -params.get_elem(1),
            -params.get_elem(2),
            -params.get_elem(3),
        ])
    }

    fn transform(params: &S::Vector<4>, point: &S::Vector<3>) -> S::Vector<3> {
        Self::matrix(params) * point.clone()
    }

    fn to_ambient(point: &S::Vector<3>) -> S::Vector<3> {
        point.clone()
    }

    fn compact(params: &S::Vector<4>) -> S::Matrix<3, 3> {
        Self::matrix(params)
    }

    fn matrix(params: &S::Vector<4>) -> S::Matrix<3, 3> {
        let ivec = params.get_fixed_subvec::<3>(1);
        let re = params.get_elem(0);

        let unit_x = S::Vector::from_f64_array([1.0, 0.0, 0.0]);
        let unit_y = S::Vector::from_f64_array([0.0, 1.0, 0.0]);
        let unit_z = S::Vector::from_f64_array([0.0, 0.0, 1.0]);

        let two = S::from_f64(2.0);

        let uv_x: S::Vector<3> =
            cross::<S, BATCH>(ivec.clone(), unit_x.clone()).scaled(two.clone());
        let uv_y: S::Vector<3> =
            cross::<S, BATCH>(ivec.clone(), unit_y.clone()).scaled(two.clone());
        let uv_z: S::Vector<3> = cross::<S, BATCH>(ivec.clone(), unit_z.clone()).scaled(two);

        let col_x =
            unit_x + cross::<S, BATCH>(ivec.clone(), uv_x.clone()) + uv_x.scaled(re.clone());
        let col_y =
            unit_y + cross::<S, BATCH>(ivec.clone(), uv_y.clone()) + uv_y.scaled(re.clone());
        let col_z =
            unit_z + cross::<S, BATCH>(ivec.clone(), uv_z.clone()) + uv_z.scaled(re.clone());

        S::Matrix::block_mat1x2::<1, 2>(
            col_x.to_mat(),
            S::Matrix::block_mat1x2(col_y.to_mat(), col_z.to_mat()),
        )
    }

    fn ad(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        Self::hat(omega)
    }

    type GenG<S2: IsScalar<BATCH>> = Rotation3Impl<S2, BATCH>;
    type RealG = Rotation3Impl<S::RealScalar, BATCH>;
    type DualG = Rotation3Impl<S::DualScalar, BATCH>;

    fn group_mul(lhs_params: &S::Vector<4>, rhs_params: &S::Vector<4>) -> S::Vector<4> {
        let lhs_re = lhs_params.get_elem(0);
        let rhs_re = rhs_params.get_elem(0);

        let lhs_ivec = lhs_params.get_fixed_subvec::<3>(1);
        let rhs_ivec = rhs_params.get_fixed_subvec::<3>(1);

        let re = lhs_re.clone() * rhs_re.clone() - lhs_ivec.clone().dot(rhs_ivec.clone());
        let ivec = rhs_ivec.scaled(lhs_re)
            + lhs_ivec.scaled(rhs_re)
            + cross::<S, BATCH>(lhs_ivec, rhs_ivec);

        let mut params = S::Vector::block_vec2(re.to_vec(), ivec);

        if ((params.norm() - S::from_f64(1.0))
            .abs()
            .greater_equal(&S::from_f64(1e-7)))
        .any()
        {
            // todo: use tailor approximation for norm close to 1
            params = params.normalized();
        }
        params
    }
}

impl<S: IsRealScalar<BATCH>, const BATCH: usize> IsRealLieGroupImpl<S, 3, 4, 3, 3, BATCH>
    for Rotation3Impl<S, BATCH>
{
    fn dx_exp_x_at_0() -> S::Matrix<4, 3> {
        S::Matrix::from_f64_array2([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ])
    }

    fn da_a_mul_b(_a: &S::Vector<4>, b: &S::Vector<4>) -> S::Matrix<4, 4> {
        let b_real = b.get_elem(0);
        let b_imag0 = b.get_elem(1);
        let b_imag1 = b.get_elem(2);
        let b_imag2 = b.get_elem(3);

        S::Matrix::<4, 4>::from_array2([
            [b_real, -b_imag0, -b_imag1, -b_imag2],
            [b_imag0, b_real, b_imag2, -b_imag1],
            [b_imag1, -b_imag2, b_real, b_imag0],
            [b_imag2, b_imag1, -b_imag0, b_real],
        ])
    }

    fn db_a_mul_b(a: &S::Vector<4>, _b: &S::Vector<4>) -> S::Matrix<4, 4> {
        let a_real = a.get_elem(0);
        let a_imag0 = a.get_elem(1);
        let a_imag1 = a.get_elem(2);
        let a_imag2 = a.get_elem(3);

        S::Matrix::<4, 4>::from_array2([
            [a_real, -a_imag0, -a_imag1, -a_imag2],
            [a_imag0, a_real, -a_imag2, a_imag1],
            [a_imag1, a_imag2, a_real, -a_imag0],
            [a_imag2, -a_imag1, a_imag0, a_real],
        ])
    }

    fn dx_exp_x_times_point_at_0(point: S::Vector<3>) -> S::Matrix<3, 3> {
        Self::hat(&-point)
    }

    fn dx_exp(omega: &S::Vector<3>) -> S::Matrix<4, 3> {
        let theta_sq = omega.squared_norm();

        let near_zero = theta_sq.less_equal(&S::from_f64(1e-6));

        let dx0 = Self::dx_exp_x_at_0();

        println!("dx0\n{:?}", dx0);

        let omega_0 = omega.get_elem(0);
        let omega_1 = omega.get_elem(1);
        let omega_2 = omega.get_elem(2);
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
        let w = params.get_elem(0);
        let squared_n = ivec.squared_norm();

        let near_zero = squared_n.less_equal(&S::from_f64(1e-6));

        let m0 = S::Matrix::<3, 4>::block_mat1x2(
            S::Matrix::<3, 1>::zeros(),
            S::Matrix::<3, 3>::identity().scaled(S::from_f64(2.0)),
        );

        let n = squared_n.sqrt();
        let theta = S::from_f64(2.0) * n.atan2(w);

        let dw_ivec_theta: S::Vector<3> = ivec.scaled(S::from_f64(-2.0) / (squared_n + w * w));
        let factor =
            S::from_f64(2.0) * w / (squared_n * (squared_n + w * w)) - theta / (squared_n * n);

        let mm = ivec.clone().outer(ivec).scaled(factor);

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
        (theta - S::from_f64(std::f64::consts::PI))
            .abs()
            .less_equal(&S::from_f64(1e-6))
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> crate::traits::IsLieFactorGroupImpl<S, 3, 4, 3, BATCH>
    for Rotation3Impl<S, BATCH>
{
    type GenFactorG<S2: IsScalar<BATCH>> = Rotation3Impl<S2, BATCH>;
    type RealFactorG = Rotation3Impl<S::RealScalar, BATCH>;
    type DualFactorG = Rotation3Impl<S::DualScalar, BATCH>;

    fn mat_v(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        let theta_sq = omega.squared_norm();
        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S, BATCH>::hat(omega);
        let mat_omega_sq = mat_omega.clone().mat_mul(mat_omega.clone());

        let near_zero = theta_sq.less_equal(&S::from_f64(1e-6));

        let mat_v0 = S::Matrix::<3, 3>::identity() + mat_omega.scaled(S::from_f64(0.5));

        let theta = theta_sq.clone().sqrt();
        let mat_v = S::Matrix::<3, 3>::identity()
            + mat_omega.scaled((S::from_f64(1.0) - theta.clone().cos()) / theta_sq.clone())
            + mat_omega_sq.scaled((theta.clone() - theta.clone().sin()) / (theta_sq * theta));

        mat_v0.select(&near_zero, mat_v)
    }

    fn mat_v_inverse(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        let theta_sq = omega.clone().dot(omega.clone());
        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S, BATCH>::hat(omega);
        let mat_omega_sq = mat_omega.clone().mat_mul(mat_omega.clone());

        let near_zero = theta_sq.less_equal(&S::from_f64(1e-6));

        let mat_v_inv0 = S::Matrix::<3, 3>::identity() - mat_omega.scaled(S::from_f64(0.5))
            + mat_omega_sq.scaled(S::from_f64(1. / 12.));

        let theta = theta_sq.clone().sqrt();
        let half_theta = S::from_f64(0.5) * theta.clone();

        let mat_v_inv = S::Matrix::<3, 3>::identity() - mat_omega.scaled(S::from_f64(0.5))
            + mat_omega_sq.scaled(
                (S::from_f64(1.0)
                    - (S::from_f64(0.5) * theta.clone() * half_theta.clone().cos())
                        / half_theta.sin())
                    / (theta.clone() * theta),
            );

        mat_v_inv0.select(&near_zero, mat_v_inv)
    }

    fn adj_of_translation(params: &S::Vector<4>, point: &S::Vector<3>) -> S::Matrix<3, 3> {
        Rotation3Impl::<S, BATCH>::hat(point).mat_mul(Rotation3Impl::<S, BATCH>::matrix(params))
    }

    fn ad_of_translation(point: &S::Vector<3>) -> S::Matrix<3, 3> {
        Rotation3Impl::<S, BATCH>::hat(point)
    }
}

impl<S: IsRealScalar<BATCH>, const BATCH: usize> IsRealLieFactorGroupImpl<S, 3, 4, 3, BATCH>
    for Rotation3Impl<S, BATCH>
{
    fn dx_mat_v(omega: &S::Vector<3>) -> [S::Matrix<3, 3>; 3] {
        let theta_sq = omega.squared_norm();
        let theta_p4 = theta_sq * theta_sq;
        let dt_mat_omega_pos_idx = [[2, 1], [0, 2], [1, 0]];
        let dt_mat_omega_neg_idx = [[1, 2], [2, 0], [0, 1]];

        let near_zero = theta_sq.less_equal(&S::from_f64(1e-6));

        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S, BATCH>::hat(omega);
        let mat_omega_sq = mat_omega.clone().mat_mul(mat_omega.clone());

        let omega_x = omega.get_elem(0);
        let omega_y = omega.get_elem(1);
        let omega_z = omega.get_elem(2);

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
        println!("a = {:?}", a);

        println!("omega = {:?}", omega);
        println!("b = {:?}", b);
        println!("dt_b = {:?}", dt_b);

        println!("dt_mat_omega_sq = {:?}", dt_mat_omega_sq);

        let set = |i| {
            let tmp0 = mat_omega.clone().scaled(dt_a * domega_theta.get_elem(i));
            let tmp1 = dt_mat_omega_sq[i].scaled(b);
            let tmp2 = mat_omega_sq.scaled(dt_b * domega_theta.get_elem(i));

            println!("tmp2 = {:?}", tmp2);
            let mut l_i: S::Matrix<3, 3> =
                S::Matrix::zeros().select(&near_zero, tmp0 + tmp1 + tmp2);
            let pos_idx = dt_mat_omega_pos_idx[i];
            l_i.set_elem(pos_idx, a + l_i.get_elem(pos_idx));

            let neg_idx = dt_mat_omega_neg_idx[i];
            l_i.set_elem(neg_idx, -a + l_i.get_elem(neg_idx));
            l_i
        };

        let l: [S::Matrix<3, 3>; 3] = [set(0), set(1), set(2)];

        l
    }

    fn dparams_matrix_times_point(params: &S::Vector<4>, point: &S::Vector<3>) -> S::Matrix<3, 4> {
        let r = params.get_elem(0);
        let ivec0 = params.get_elem(1);
        let ivec1 = params.get_elem(2);
        let ivec2 = params.get_elem(3);

        let p0 = point.get_elem(0);
        let p1 = point.get_elem(1);
        let p2 = point.get_elem(2);

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
        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S, BATCH>::hat(omega);
        let mat_omega_sq = mat_omega.clone().mat_mul(mat_omega);

        let dt_mat_omega_pos_idx = [[2, 1], [0, 2], [1, 0]];
        let dt_mat_omega_neg_idx = [[1, 2], [2, 0], [0, 1]];

        let omega_x = omega.get_elem(0);
        let omega_y = omega.get_elem(1);
        let omega_z = omega.get_elem(2);

        let near_zero = theta_sq.less_equal(&S::from_f64(1e-6));

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
                dt_mat_omega_sq_i.scaled(c) + mat_omega_sq.scaled(domega_theta.get_elem(i) * dt_c),
            );

            let pos_idx = dt_mat_omega_pos_idx[i];
            l_i.set_elem(pos_idx, S::from_f64(-0.5) + l_i.get_elem(pos_idx));

            let neg_idx = dt_mat_omega_neg_idx[i];
            l_i.set_elem(neg_idx, S::from_f64(0.5) + l_i.get_elem(neg_idx));
            l_i
        };

        let l: [S::Matrix<3, 3>; 3] = [set(0), set(1), set(2)];

        l
    }
}

/// 3d rotation group - SO(3)
pub type Rotation3<S, const BATCH: usize> = LieGroup<S, 3, 4, 3, 3, BATCH, Rotation3Impl<S, BATCH>>;

#[test]
fn rotation3_prop_tests() {
    use crate::factor_lie_group::RealFactorLieGroupTest;
    use crate::real_lie_group::RealLieGroupTest;
    use sophus_core::calculus::dual::dual_scalar::DualBatchScalar;
    use sophus_core::calculus::dual::dual_scalar::DualScalar;
    use sophus_core::linalg::BatchScalarF64;

    Rotation3::<f64, 1>::test_suite();
    Rotation3::<BatchScalarF64<8>, 8>::test_suite();
    Rotation3::<DualScalar, 1>::test_suite();
    Rotation3::<DualBatchScalar<8>, 8>::test_suite();

    Rotation3::<f64, 1>::run_real_tests();
    Rotation3::<BatchScalarF64<8>, 8>::run_real_tests();

    Rotation3::<f64, 1>::run_real_factor_tests();
    Rotation3::<BatchScalarF64<8>, 8>::run_real_factor_tests();
}
