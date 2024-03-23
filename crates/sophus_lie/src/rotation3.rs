use std::marker::PhantomData;

use nalgebra::ComplexField;

use sophus_calculus::dual::dual_scalar::Dual;
use sophus_calculus::manifold::{self};
use sophus_calculus::types::matrix::IsMatrix;
use sophus_calculus::types::params::HasParams;
use sophus_calculus::types::params::ParamsImpl;
use sophus_calculus::types::scalar::IsScalar;
use sophus_calculus::types::vector::cross;
use sophus_calculus::types::vector::IsVector;
use sophus_calculus::types::MatF64;
use sophus_calculus::types::VecF64;

use super::lie_group::LieGroup;
use super::traits::IsLieGroupImpl;

/// 3d rotation implementation - SO(3)
#[derive(Debug, Copy, Clone)]
pub struct Rotation3Impl<S: IsScalar<1>> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<1>> ParamsImpl<S, 4, 1> for Rotation3Impl<S> {
    fn params_examples() -> Vec<S::Vector<4>> {
        let mut params = vec![];

        params.push(
            Rotation3::<S>::exp(&S::Vector::<3>::from_c_array([0.0, 0.0, 0.0]))
                .params()
                .clone(),
        );
        params.push(
            Rotation3::<S>::exp(&S::Vector::<3>::from_c_array([0.1, 0.5, -0.1]))
                .params()
                .clone(),
        );
        params.push(
            Rotation3::<S>::exp(&S::Vector::<3>::from_c_array([0.0, 0.2, 1.0]))
                .params()
                .clone(),
        );
        params.push(
            Rotation3::<S>::exp(&S::Vector::<3>::from_c_array([-0.2, 0.0, 0.8]))
                .params()
                .clone(),
        );
        params
    }

    fn invalid_params_examples() -> Vec<S::Vector<4>> {
        vec![
            S::Vector::<4>::from_array([0.0.into(), 0.0.into(), 0.0.into(), 0.0.into()]),
            S::Vector::<4>::from_array([0.5.into(), 0.5.into(), 0.5.into(), 0.0.into()]),
            S::Vector::<4>::from_array([0.5.into(), (-0.5).into(), 0.5.into(), 1.0.into()]),
        ]
    }

    fn are_params_valid(params: &S::Vector<4>) -> bool {
        let norm = params.norm().real();
        (norm - 1.0).abs() < 1e-6
    }
}

impl<S: IsScalar<1>> manifold::traits::TangentImpl<S, 3, 1> for Rotation3Impl<S> {
    fn tangent_examples() -> Vec<S::Vector<3>> {
        vec![
            S::Vector::<3>::from_c_array([0.0, 0.0, 0.0]),
            S::Vector::<3>::from_c_array([1.0, 0.0, 0.0]),
            S::Vector::<3>::from_c_array([0.0, 1.0, 0.0]),
            S::Vector::<3>::from_c_array([0.0, 0.0, 1.0]),
            S::Vector::<3>::from_c_array([0.5, 0.5, 0.1]),
            S::Vector::<3>::from_c_array([-0.1, -0.5, -0.5]),
        ]
    }
}

impl<S: IsScalar<1>> IsLieGroupImpl<S, 3, 4, 3, 3, 1> for Rotation3Impl<S> {
    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = true;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<4> {
        S::Vector::<4>::from_c_array([1.0, 0.0, 0.0, 0.0])
    }

    fn adj(params: &S::Vector<4>) -> S::Matrix<3, 3> {
        Self::matrix(params)
    }

    fn exp(omega: &S::Vector<3>) -> S::Vector<4> {
        const EPS: f64 = 1e-8;
        let theta_sq = omega.squared_norm();

        let (imag_factor, real_factor) = if theta_sq.real() < EPS * EPS {
            let theta_po4 = theta_sq.clone() * theta_sq.clone();
            (
                S::c(0.5) - S::c(1.0 / 48.0) * theta_sq.clone()
                    + S::c(1.0 / 3840.0) * theta_po4.clone(),
                S::c(1.0) - S::c(1.0 / 8.0) * theta_sq + S::c(1.0 / 384.0) * theta_po4,
            )
        } else {
            let theta = theta_sq.sqrt();
            let half_theta: S = S::c(0.5) * theta.clone();
            (half_theta.clone().sin() / theta, half_theta.cos())
        };
        S::Vector::<4>::from_array([
            real_factor,
            imag_factor.clone() * omega.get(0),
            imag_factor.clone() * omega.get(1),
            imag_factor * omega.get(2),
        ])
    }

    fn log(params: &S::Vector<4>) -> S::Vector<3> {
        const EPS: f64 = 1e-8;
        let ivec: S::Vector<3> = params.get_fixed_rows::<3>(1);

        let squared_n = ivec.squared_norm();
        let w = params.get(0);
        let w_real = w.real();

        let two_atan_nbyd_by_n: S = if squared_n.real() < EPS * EPS {
            assert!(
                w_real.abs() > EPS,
                "|params| should be close to 1. (w = {})",
                w_real
            );
            let w_sq = w.clone() * w.clone();
            S::c(2.0) / w.clone() - S::c(2.0 / 3.0) * squared_n / (w_sq * w)
        } else {
            let n = squared_n.sqrt();
            let atan_nbyw = if w_real < 0.0 {
                -n.clone().atan2(-w)
            } else {
                n.clone().atan2(w)
            };
            S::c(2.0) * atan_nbyw / n
        };
        ivec.scaled(two_atan_nbyd_by_n)
    }

    fn hat(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        let o0 = omega.get(0);
        let o1 = omega.get(1);
        let o2 = omega.get(2);

        S::Matrix::from_array2([
            [S::zero(), -o2.clone(), o1.clone()],
            [o2, S::zero(), -o0.clone()],
            [-o1, o0, S::zero()],
        ])
    }

    fn vee(omega_hat: &S::Matrix<3, 3>) -> S::Vector<3> {
        S::Vector::<3>::from_array([
            omega_hat.get((2, 1)),
            omega_hat.get((0, 2)),
            omega_hat.get((1, 0)),
        ])
    }

    fn inverse(params: &S::Vector<4>) -> S::Vector<4> {
        S::Vector::from_array([
            params.get(0),
            -params.get(1),
            -params.get(2),
            -params.get(3),
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
        let ivec = params.get_fixed_rows::<3>(1);
        let re = params.get(0);

        let unit_x = S::Vector::from_c_array([1.0, 0.0, 0.0]);
        let unit_y = S::Vector::from_c_array([0.0, 1.0, 0.0]);
        let unit_z = S::Vector::from_c_array([0.0, 0.0, 1.0]);

        let two = S::c(2.0);

        let uv_x: S::Vector<3> = cross::<S>(ivec.clone(), unit_x.clone()).scaled(two.clone());
        let uv_y: S::Vector<3> = cross::<S>(ivec.clone(), unit_y.clone()).scaled(two.clone());
        let uv_z: S::Vector<3> = cross::<S>(ivec.clone(), unit_z.clone()).scaled(two);

        let col_x = unit_x + cross::<S>(ivec.clone(), uv_x.clone()) + uv_x.scaled(re.clone());
        let col_y = unit_y + cross::<S>(ivec.clone(), uv_y.clone()) + uv_y.scaled(re.clone());
        let col_z = unit_z + cross::<S>(ivec.clone(), uv_z.clone()) + uv_z.scaled(re.clone());

        S::Matrix::block_mat1x2::<1, 2>(
            col_x.to_mat(),
            S::Matrix::block_mat1x2(col_y.to_mat(), col_z.to_mat()),
        )
    }

    fn ad(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        Self::hat(omega)
    }

    type GenG<S2: IsScalar<1>> = Rotation3Impl<S2>;
    type RealG = Rotation3Impl<f64>;
    type DualG = Rotation3Impl<Dual>;

    fn group_mul(lhs_params: &S::Vector<4>, rhs_params: &S::Vector<4>) -> S::Vector<4> {
        let lhs_re = lhs_params.get(0);
        let rhs_re = rhs_params.get(0);

        let lhs_ivec = lhs_params.get_fixed_rows::<3>(1);
        let rhs_ivec = rhs_params.get_fixed_rows::<3>(1);

        let re = lhs_re.clone() * rhs_re.clone() - lhs_ivec.clone().dot(rhs_ivec.clone());
        let ivec =
            rhs_ivec.scaled(lhs_re) + lhs_ivec.scaled(rhs_re) + cross::<S>(lhs_ivec, rhs_ivec);

        let mut params = S::Vector::block_vec2(re.to_vec(), ivec);

        if (params.norm().real() - 1.0).abs() > 1e-7 {
            // todo: use tailor approximation for norm close to 1
            params = params.normalized();
        }
        params
    }

    fn has_shortest_path_ambiguity(params: &<S as IsScalar<1>>::Vector<4>) -> bool {
        let theta = Self::log(params).real().norm();
        (theta - std::f64::consts::PI).abs() < 1e-5
    }
}

impl crate::traits::IsF64LieGroupImpl<3, 4, 3, 3> for Rotation3Impl<f64> {
    fn dx_exp_x_at_0() -> MatF64<4, 3> {
        MatF64::from_c_array2([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ])
    }

    fn da_a_mul_b(_a: &VecF64<4>, b: &VecF64<4>) -> MatF64<4, 4> {
        let b_real = b[0];
        let b_imag0 = b[1];
        let b_imag1 = b[2];
        let b_imag2 = b[3];

        MatF64::<4, 4>::from_array2([
            [b_real, -b_imag0, -b_imag1, -b_imag2],
            [b_imag0, b_real, b_imag2, -b_imag1],
            [b_imag1, -b_imag2, b_real, b_imag0],
            [b_imag2, b_imag1, -b_imag0, b_real],
        ])
    }

    fn db_a_mul_b(a: &VecF64<4>, _b: &VecF64<4>) -> MatF64<4, 4> {
        let a_real = a[0];
        let a_imag0 = a[1];
        let a_imag1 = a[2];
        let a_imag2 = a[3];

        MatF64::<4, 4>::from_array2([
            [a_real, -a_imag0, -a_imag1, -a_imag2],
            [a_imag0, a_real, -a_imag2, a_imag1],
            [a_imag1, a_imag2, a_real, -a_imag0],
            [a_imag2, -a_imag1, a_imag0, a_real],
        ])
    }

    fn dx_exp_x_times_point_at_0(point: sophus_calculus::types::VecF64<3>) -> MatF64<3, 3> {
        Self::hat(&-point)
    }

    fn dx_exp(omega: &VecF64<3>) -> MatF64<4, 3> {
        let theta_sq = omega.squared_norm();

        if theta_sq < 1e-6 {
            return Self::dx_exp_x_at_0();
        }

        let omega_0 = omega[0];
        let omega_1 = omega[1];
        let omega_2 = omega[2];
        let theta = theta_sq.sqrt();
        let a = (0.5 * theta).sin() / theta;
        let b = (0.5 * theta).cos() / (theta_sq) - 2.0 * (0.5 * theta).sin() / (theta_sq * theta);

        0.5 * MatF64::from_array2([
            [-omega_0 * a, -omega_1 * a, -omega_2 * a],
            [
                omega_0 * omega_0 * b + 2.0 * a,
                omega_0 * omega_1 * b,
                omega_0 * omega_2 * b,
            ],
            [
                omega_0 * omega_1 * b,
                omega_1 * omega_1 * b + 2.0 * a,
                omega_1 * omega_2 * b,
            ],
            [
                omega_0 * omega_2 * b,
                omega_1 * omega_2 * b,
                omega_2 * omega_2 * b + 2.0 * a,
            ],
        ])
    }

    fn dx_log_x(params: &VecF64<4>) -> MatF64<3, 4> {
        let ivec: VecF64<3> = params.get_fixed_rows::<3>(1);
        let w: f64 = params[0];
        let squared_n: f64 = ivec.squared_norm();

        if squared_n < 1e-6 {
            let mut m = MatF64::<3, 4>::zeros();
            m.fixed_columns_mut::<3>(1)
                .copy_from(&(2.0 * MatF64::<3, 3>::identity()));
            return m;
        }

        let n: f64 = squared_n.sqrt();
        let theta = 2.0 * n.atan2(w);

        let dw_ivec_theta: VecF64<3> = ivec * (-2.0 / (squared_n + w * w));
        let factor = 2.0 * w / (squared_n * (squared_n + w * w)) - theta / (squared_n * n);

        let mut m = MatF64::<3, 4>::zeros();

        m.set_column(0, &dw_ivec_theta);
        m.fixed_columns_mut::<3>(1).copy_from(
            &(MatF64::<3, 3>::identity() * theta / n + ivec * ivec.transpose() * factor),
        );
        m
    }
}

impl<S: IsScalar<1>> crate::traits::IsLieFactorGroupImpl<S, 3, 4, 3, 1> for Rotation3Impl<S> {
    type GenFactorG<S2: IsScalar<1>> = Rotation3Impl<S2>;
    type RealFactorG = Rotation3Impl<f64>;
    type DualFactorG = Rotation3Impl<Dual>;

    fn mat_v(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        let theta_sq = omega.squared_norm();
        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S>::hat(omega);
        let mat_omega_sq = mat_omega.clone().mat_mul(mat_omega.clone());
        if theta_sq.real() < 1e-6 {
            S::Matrix::<3, 3>::identity() + mat_omega.scaled(S::c(0.5))
        } else {
            let theta = theta_sq.clone().sqrt();
            S::Matrix::<3, 3>::identity()
                + mat_omega.scaled((S::c(1.0) - theta.clone().cos()) / theta_sq.clone())
                + mat_omega_sq.scaled((theta.clone() - theta.clone().sin()) / (theta_sq * theta))
        }
    }

    fn mat_v_inverse(omega: &S::Vector<3>) -> S::Matrix<3, 3> {
        let theta_sq = omega.clone().dot(omega.clone());
        let mat_omega: S::Matrix<3, 3> = Rotation3Impl::<S>::hat(omega);
        let mat_omega_sq = mat_omega.clone().mat_mul(mat_omega.clone());

        if theta_sq.real() < 1e-6 {
            S::Matrix::<3, 3>::identity() - mat_omega.scaled(S::c(0.5))
                + mat_omega_sq.scaled(S::c(1. / 12.))
        } else {
            let theta = theta_sq.clone().sqrt();
            let half_theta = S::c(0.5) * theta.clone();

            S::Matrix::<3, 3>::identity() - mat_omega.scaled(S::c(0.5))
                + mat_omega_sq.scaled(
                    (S::c(1.0)
                        - (S::c(0.5) * theta.clone() * half_theta.clone().cos())
                            / half_theta.sin())
                        / (theta.clone() * theta),
                )
        }
    }

    fn adj_of_translation(params: &S::Vector<4>, point: &S::Vector<3>) -> S::Matrix<3, 3> {
        Rotation3Impl::<S>::hat(point).mat_mul(Rotation3Impl::<S>::matrix(params))
    }

    fn ad_of_translation(point: &S::Vector<3>) -> S::Matrix<3, 3> {
        Rotation3Impl::<S>::hat(point)
    }
}

impl crate::traits::IsF64LieFactorGroupImpl<3, 4, 3> for Rotation3Impl<f64> {
    fn dx_mat_v(omega: &sophus_calculus::types::VecF64<3>) -> [MatF64<3, 3>; 3] {
        let theta_sq = omega.squared_norm();
        let dt_mat_omega_pos_idx = [(2, 1), (0, 2), (1, 0)];
        let dt_mat_omega_neg_idx = [(1, 2), (2, 0), (0, 1)];
        if theta_sq.real() < 1e-6 {
            let mut l = [MatF64::<3, 3>::zeros(); 3];

            for i in 0..3 {
                *l[i].get_mut(dt_mat_omega_pos_idx[i]).unwrap() += 0.5;
                *l[i].get_mut(dt_mat_omega_neg_idx[i]).unwrap() -= 0.5;

                println!("l[i] = {:?}", l[i])
            }

            println!("l = {:?}", l);

            return l;
        }

        let mat_omega: MatF64<3, 3> = Rotation3Impl::<f64>::hat(omega);
        let mat_omega_sq = mat_omega.clone().mat_mul(mat_omega);

        let theta = theta_sq.sqrt();
        let domega_theta =
            VecF64::from_array([omega[0] / theta, omega[1] / theta, omega[2] / theta]);

        let a = (1.0 - theta.cos()) / theta_sq;
        let dt_a = (-2.0 + 2.0 * theta.cos() + theta * theta.sin()) / (theta * theta_sq);

        let b = (theta - theta.sin()) / (theta_sq * theta);
        let dt_b = -(2.0 * theta + theta * theta.cos() - 3.0 * theta.sin()) / theta.powi(4);

        let dt_mat_omega_sq = [
            MatF64::from_array2([
                [0.0, omega[1], omega[2]],
                [omega[1], -2.0 * omega[0], 0.0],
                [omega[2], 0.0, -2.0 * omega[0]],
            ]),
            MatF64::from_array2([
                [-2.0 * omega[1], omega[0], 0.0],
                [omega[0], 0.0, omega[2]],
                [0.0, omega[2], -2.0 * omega[1]],
            ]),
            MatF64::from_array2([
                [-2.0 * omega[2], 0.0, omega[0]],
                [0.0, -2.0 * omega[2], omega[1]],
                [omega[0], omega[1], 0.0],
            ]),
        ];

        let mut l = [MatF64::<3, 3>::zeros(); 3];

        for i in 0..3 {
            l[i] = domega_theta[i] * dt_a * mat_omega;
            println!("l[i] = {:?}", l[i]);
            *l[i].get_mut(dt_mat_omega_pos_idx[i]).unwrap() += a;
            *l[i].get_mut(dt_mat_omega_neg_idx[i]).unwrap() -= a;
            println!("pl[i] = {:?}", l[i]);
            l[i] += b * dt_mat_omega_sq[i] + domega_theta[i] * dt_b * mat_omega_sq;
        }

        l
    }

    fn dparams_matrix_times_point(params: &VecF64<4>, point: &VecF64<3>) -> MatF64<3, 4> {
        let r = params[0];
        let ivec0 = params[1];
        let ivec1 = params[2];
        let ivec2 = params[3];

        let p0 = point[0];
        let p1 = point[1];
        let p2 = point[2];

        MatF64::from_array2([
            [
                2.0 * ivec1 * p2 - 2.0 * ivec2 * p1,
                2.0 * ivec1 * p1 + 2.0 * ivec2 * p2,
                2.0 * r * p2 + 2.0 * ivec0 * p1 - 4.0 * ivec1 * p0,
                -2.0 * r * p1 + 2.0 * ivec0 * p2 - 4.0 * ivec2 * p0,
            ],
            [
                -2.0 * ivec0 * p2 + 2.0 * ivec2 * p0,
                -2.0 * r * p2 - 4.0 * ivec0 * p1 + 2.0 * ivec1 * p0,
                2.0 * ivec0 * p0 + 2.0 * ivec2 * p2,
                2.0 * r * p0 + 2.0 * ivec1 * p2 - 4.0 * ivec2 * p1,
            ],
            [
                2.0 * ivec0 * p1 - 2.0 * ivec1 * p0,
                2.0 * r * p1 - 4.0 * ivec0 * p2 + 2.0 * ivec2 * p0,
                -2.0 * r * p0 - 4.0 * ivec1 * p2 + 2.0 * ivec2 * p1,
                2.0 * ivec0 * p0 + 2.0 * ivec1 * p1,
            ],
        ])
    }

    fn dx_mat_v_inverse(omega: &sophus_calculus::types::VecF64<3>) -> [MatF64<3, 3>; 3] {
        let theta_sq = omega.squared_norm();
        let theta = theta_sq.sqrt();
        let half_theta = 0.5 * theta;
        let mat_omega: MatF64<3, 3> = Rotation3Impl::<f64>::hat(omega);
        let mat_omega_sq = mat_omega.clone().mat_mul(mat_omega);

        let dt_mat_omega_pos_idx = [(2, 1), (0, 2), (1, 0)];
        let dt_mat_omega_neg_idx = [(1, 2), (2, 0), (0, 1)];

        if theta_sq.real() < 1e-6 {
            let mut l = [MatF64::<3, 3>::zeros(); 3];

            for i in 0..3 {
                *l[i].get_mut(dt_mat_omega_pos_idx[i]).unwrap() -= 0.5;
                *l[i].get_mut(dt_mat_omega_neg_idx[i]).unwrap() += 0.5;

                println!("l[i] = {:?}", l[i])
            }

            println!("l = {:?}", l);

            return l;
        }

        let domega_theta =
            VecF64::from_array([omega[0] / theta, omega[1] / theta, omega[2] / theta]);

        let c = (1.0 - (0.5 * theta * half_theta.cos()) / (half_theta.sin())) / theta_sq;

        let dt_c = (-2.0
            + (0.25 * theta_sq) / (half_theta.sin() * half_theta.sin())
            + (half_theta * half_theta.cos()) / half_theta.sin())
            / theta.powi(3);

        let dt_mat_omega_sq = [
            MatF64::from_array2([
                [0.0, omega[1], omega[2]],
                [omega[1], -2.0 * omega[0], 0.0],
                [omega[2], 0.0, -2.0 * omega[0]],
            ]),
            MatF64::from_array2([
                [-2.0 * omega[1], omega[0], 0.0],
                [omega[0], 0.0, omega[2]],
                [0.0, omega[2], -2.0 * omega[1]],
            ]),
            MatF64::from_array2([
                [-2.0 * omega[2], 0.0, omega[0]],
                [0.0, -2.0 * omega[2], omega[1]],
                [omega[0], omega[1], 0.0],
            ]),
        ];

        let mut l = [MatF64::<3, 3>::zeros(); 3];

        for i in 0..3 {
            l[i][dt_mat_omega_pos_idx[i]] += -0.5;
            l[i][dt_mat_omega_neg_idx[i]] -= -0.5;
            l[i] += dt_mat_omega_sq[i].scaled(c) + domega_theta[i] * mat_omega_sq.scaled(dt_c);
        }

        l
    }
}

/// 3d rotation group - SO(3)
pub type Rotation3<S> = LieGroup<S, 3, 4, 3, 3, 1, Rotation3Impl<S>>;

/// 3d isometry implementation - SE(3)
pub type Isometry3Impl<S> = crate::translation_product_product::TranslationProductGroupImpl<
    S,
    6,
    7,
    3,
    4,
    3,
    4,
    Rotation3Impl<S>,
>;

mod tests {

    #[test]
    fn rotation3_prop_tests() {
        use super::Rotation3;
        use sophus_calculus::dual::dual_scalar::Dual;

        Rotation3::<f64>::test_suite();
        Rotation3::<Dual>::test_suite();
        Rotation3::<f64>::real_test_suite();
        Rotation3::<f64>::real_factor_test_suite();
    }
}
