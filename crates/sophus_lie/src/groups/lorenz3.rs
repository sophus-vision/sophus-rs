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
    RotationBoost3Impl,
    HasDisambiguate,
    IsLieFactorGroupImpl,
    IsLieGroupImpl,
    Rotation3,
    Rotation3Impl,
    Sl2cImpl,
    lie_group::LieGroup,
};

extern crate alloc;

/// 3-d Lorenz transformations – element of the Homogeneous Lorenz group.
///
/// This mirrors [`RotationBoost3`] but multiplies rotation quaternions using
/// the [`Sl2c`] group.
pub type Lorenz3<S, const BATCH: usize, const DM: usize, const DN: usize> =
    LieGroup<S, 6, 7, 4, 4, BATCH, DM, DN, Lorenz3Impl<S, BATCH, DM, DN>>;

/// Lorenz3 with f64 scalar type.
pub type Lorenz3F64 = Lorenz3<f64, 1, 0, 0>;

/// Implementation of [`Lorenz3`].
#[derive(Debug, Copy, Clone, Default)]
pub struct Lorenz3Impl<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: PhantomData<S>,
}

fn quat_to_sl2c<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>(
    q: S::Vector<4>,
) -> S::Vector<8> {
    let a = q.elem(0);
    let b = q.elem(1);
    let c = q.elem(2);
    let d = q.elem(3);
    S::Vector::<8>::from_array([a, b, c, d, -c, d, a, -b])
}

fn sl2c_to_quat<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>(
    m: S::Vector<8>,
) -> S::Vector<4> {
    S::Vector::<4>::from_array([m.elem(0), m.elem(1), m.elem(2), m.elem(3)])
}

#[allow(dead_code)]
fn sl2c_from_rot_boost<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>(
    q: S::Vector<4>,
    beta: S::Vector<3>,
) -> S::Vector<8> {
    let rot = quat_to_sl2c::<S, BATCH, DM, DN>(q);

    let beta_sq = beta.squared_norm();
    let beta_norm = beta_sq.sqrt();

    let near_zero = beta_sq.less_equal(&S::from_f64(EPS_F64));
    let inv_b = S::zeros().select(&near_zero, S::from_f64(1.0) / beta_norm);
    let n = beta.scaled(inv_b);

    let phi = ((S::from_f64(1.0) + beta_norm)
        / (S::from_f64(1.0) - beta_norm))
        .ln()
        * S::from_f64(0.5);
    let half_phi = phi * S::from_f64(0.5);
    let exp_pos = half_phi.exp();
    let exp_neg = (-half_phi).exp();
    let cosh = (exp_pos + exp_neg) * S::from_f64(0.5);
    let sinh = (exp_pos - exp_neg) * S::from_f64(0.5);

    let sigma_x = S::Vector::<8>::from_f64_array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    let sigma_y = S::Vector::<8>::from_f64_array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0]);
    let sigma_z = S::Vector::<8>::from_f64_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]);

    let n_sigma = sigma_x.scaled(n.elem(0))
        + sigma_y.scaled(n.elem(1))
        + sigma_z.scaled(n.elem(2));

    let boost = Sl2cImpl::<S, BATCH, DM, DN>::one().scaled(cosh) + n_sigma.scaled(sinh);
    Sl2cImpl::<S, BATCH, DM, DN>::mult(&boost, rot)
}

#[allow(dead_code)]
fn sl2c_to_rot_boost<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>(
    sl: S::Vector<8>,
) -> (S::Vector<4>, S::Vector<3>) {
    let sl_dag = Sl2cImpl::<S, BATCH, DM, DN>::conjugate_transpose(&sl);
    let m = Sl2cImpl::<S, BATCH, DM, DN>::mult(&sl, sl_dag);

    let t = (m.elem(0) + m.elem(6)) * S::from_f64(0.5);
    let inv_t = S::from_f64(1.0) / t;
    let x = m.elem(2) * inv_t;
    let y = -m.elem(3) * inv_t;
    let z = (m.elem(0) - m.elem(6)) * S::from_f64(0.5) * inv_t;
    let beta = S::Vector::<3>::from_array([x, y, z]);

    let beta_sq = beta.squared_norm();
    let beta_norm = beta_sq.sqrt();

    let near_zero = beta_sq.less_equal(&S::from_f64(EPS_F64));
    let inv_b = S::zeros().select(&near_zero, S::from_f64(1.0) / beta_norm);
    let n = beta.scaled(inv_b);

    let phi = ((S::from_f64(1.0) + beta_norm)
        / (S::from_f64(1.0) - beta_norm))
        .ln()
        * S::from_f64(0.5);
    let half_phi = phi * S::from_f64(0.5);
    let exp_pos = half_phi.exp();
    let exp_neg = (-half_phi).exp();
    let cosh = (exp_pos + exp_neg) * S::from_f64(0.5);
    let sinh = (exp_pos - exp_neg) * S::from_f64(0.5);

    let sigma_x = S::Vector::<8>::from_f64_array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    let sigma_y = S::Vector::<8>::from_f64_array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0]);
    let sigma_z = S::Vector::<8>::from_f64_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]);

    let n_sigma = sigma_x.scaled(n.elem(0))
        + sigma_y.scaled(n.elem(1))
        + sigma_z.scaled(n.elem(2));

    let boost_inv = Sl2cImpl::<S, BATCH, DM, DN>::one().scaled(cosh) - n_sigma.scaled(sinh);
    let rot = Sl2cImpl::<S, BATCH, DM, DN>::mult(&boost_inv, sl);
    let q = sl2c_to_quat::<S, BATCH, DM, DN>(rot);

    (q, beta)
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    HasDisambiguate<S, 7, BATCH, DM, DN> for Lorenz3Impl<S, BATCH, DM, DN>
{
    fn disambiguate(params: S::Vector<7>) -> S::Vector<7> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::disambiguate(params)
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsParamsImpl<S, 7, BATCH, DM, DN> for Lorenz3Impl<S, BATCH, DM, DN>
{
    fn params_examples() -> alloc::vec::Vec<S::Vector<7>> {
        let id = Self::identity_params();
        let rx = Rotation3::<S, BATCH, DM, DN>::rot_x(S::from_f64(0.10));
        let ry = Rotation3::<S, BATCH, DM, DN>::rot_y(S::from_f64(-0.12));
        let rz = Rotation3::<S, BATCH, DM, DN>::rot_z(S::from_f64(0.08));
        alloc::vec![
            S::Vector::<7>::block_vec2(*(rx * ry).params(), S::Vector::<3>::from_f64_array([0.02, 0.01, -0.01])),
            id,
            S::Vector::<7>::block_vec2(*ry.params(), S::Vector::<3>::zeros()),
            S::Vector::<7>::block_vec2(*rz.params(), S::Vector::<3>::zeros()),
            {
                let beta = S::Vector::<3>::from_f64_array([0.05, 0.00, 0.00]);
                S::Vector::<7>::block_vec2(*Rotation3::<S, BATCH, DM, DN>::identity().params(), beta)
            },
            S::Vector::<7>::block_vec2(*ry.params(), S::Vector::<3>::from_f64_array([0.00, 0.03, 0.01])),
            S::Vector::<7>::block_vec2(*rz.params(), S::Vector::<3>::from_f64_array([0.01, -0.04, 0.00])),
            {
                let q = Rotation3::<S, BATCH, DM, DN>::exp(S::Vector::<3>::from_f64_array([0.25, -0.18, 0.12]));
                S::Vector::<7>::block_vec2(*q.params(), S::Vector::<3>::from_f64_array([0.06, 0.02, -0.05]))
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
    IsTangent<S, 6, BATCH, DM, DN> for Lorenz3Impl<S, BATCH, DM, DN>
{
    fn tangent_examples() -> alloc::vec::Vec<S::Vector<6>> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::tangent_examples()
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsLieGroupImpl<S, 6, 7, 4, 4, BATCH, DM, DN> for Lorenz3Impl<S, BATCH, DM, DN>
{
    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = false;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<7> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::identity_params()
    }

    fn adj(params: &S::Vector<7>) -> S::Matrix<6, 6> {
        let mat_g = Self::matrix(params);
        let inv_params = Self::inverse(params);
        let mat_inv = Self::matrix(&inv_params);

        let mut ad = S::Matrix::<6, 6>::zeros();
        for i in 0..6 {
            let mut basis = S::Vector::<6>::zeros();
            *basis.elem_mut(i) = S::ones();
            let col = Self::vee(mat_g.mat_mul(Self::hat(basis)).mat_mul(mat_inv));
            ad.set_col_vec(i, col);
        }
        ad
    }

    fn ad(tangent: S::Vector<6>) -> S::Matrix<6, 6> {
        let hat_xi = Self::hat(tangent);
        let mut ad = S::Matrix::<6, 6>::zeros();
        for i in 0..6 {
            let mut basis = S::Vector::<6>::zeros();
            *basis.elem_mut(i) = S::ones();
            let col = Self::vee(hat_xi.mat_mul(Self::hat(basis)) - Self::hat(basis).mat_mul(hat_xi));
            ad.set_col_vec(i, col);
        }
        ad
    }

    fn exp(xi: S::Vector<6>) -> S::Vector<7> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::exp(xi)
    }

    fn log(params: &S::Vector<7>) -> S::Vector<6> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::log(params)
    }

    fn hat(xi: S::Vector<6>) -> S::Matrix<4, 4> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::hat(xi)
    }

    fn vee(h: S::Matrix<4, 4>) -> S::Vector<6> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::vee(h)
    }

    fn group_mul(lhs: &S::Vector<7>, rhs: S::Vector<7>) -> S::Vector<7> {
        let q1 = lhs.get_fixed_subvec::<4>(0);
        let q2 = rhs.get_fixed_subvec::<4>(0);
        let sl1 = quat_to_sl2c::<S, BATCH, DM, DN>(q1);
        let sl2 = quat_to_sl2c::<S, BATCH, DM, DN>(q2);
        let sl = Sl2cImpl::<S, BATCH, DM, DN>::mult(&sl1, sl2);
        let q = sl2c_to_quat::<S, BATCH, DM, DN>(sl);

        let beta1 = lhs.get_fixed_subvec::<3>(4);
        let beta2 = rhs.get_fixed_subvec::<3>(4);
        let r1 = Rotation3Impl::<S, BATCH, DM, DN>::matrix(&q1);
        let beta = beta1 + r1 * beta2;
        S::Vector::block_vec2(q, beta)
    }

    fn inverse(params: &S::Vector<7>) -> S::Vector<7> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::inverse(params)
    }

    fn transform(params: &S::Vector<7>, point: S::Vector<4>) -> S::Vector<4> {
        Self::matrix(params) * point
    }

    fn to_ambient(point: S::Vector<4>) -> S::Vector<4> {
        point
    }

    fn compact(params: &S::Vector<7>) -> S::Matrix<4, 4> {
        Self::matrix(params)
    }

    fn matrix(params: &S::Vector<7>) -> S::Matrix<4, 4> {
        let q = params.get_fixed_subvec::<4>(0);
        let beta = params.get_fixed_subvec::<3>(4);

        let mat_r = Rotation3Impl::<S, BATCH, DM, DN>::matrix(&q);

        let beta_sq = beta.squared_norm();
        let beta_norm = beta_sq.sqrt();

        let exp_pos = beta_norm.exp();
        let exp_neg = (-beta_norm).exp();
        let cosh = (exp_pos + exp_neg) * S::from_f64(0.5);
        let sinh = (exp_pos - exp_neg) * S::from_f64(0.5);

        let near_zero = beta_sq.less_equal(&S::from_f64(EPS_F64));

        let sinh_by_norm = S::from_f64(1.0).select(&near_zero, sinh / beta_norm);
        let cosh_minus_one_by_norm_sq =
            S::from_f64(0.5).select(&near_zero, (cosh - S::from_f64(1.0)) / beta_sq);

        let outer = beta.outer(&beta);
        let boost_br = S::Matrix::<3, 3>::identity() + outer.scaled(cosh_minus_one_by_norm_sq);
        let boost_tr = beta.to_mat().transposed().scaled(sinh_by_norm);
        let boost_bl = beta.to_mat().scaled(sinh_by_norm);

        let mat_tr = boost_tr.mat_mul(mat_r);
        let mat_br = boost_br.mat_mul(mat_r);

        S::Matrix::block_mat2x2::<1, 3, 1, 3>(
            (S::Matrix::from_array2([[cosh]]), mat_tr),
            (boost_bl, mat_br),
        )
    }

    type GenG<S2: IsScalar<BATCH, DM, DN>> = Lorenz3Impl<S2, BATCH, DM, DN>;
    type RealG = Lorenz3Impl<S::RealScalar, BATCH, 0, 0>;
    type DualG<const M: usize, const N: usize> = Lorenz3Impl<S::DualScalar<M, N>, BATCH, M, N>;
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsLieFactorGroupImpl<S, 6, 7, 4, BATCH, DM, DN> for Lorenz3Impl<S, BATCH, DM, DN>
{
    type GenFactorG<S2: IsScalar<BATCH, M, N>, const M: usize, const N: usize> =
        Lorenz3Impl<S2, BATCH, M, N>;
    type RealFactorG = Lorenz3Impl<S::RealScalar, BATCH, 0, 0>;
    type DualFactorG<const M: usize, const N: usize> =
        Lorenz3Impl<S::DualScalar<M, N>, BATCH, M, N>;

    fn mat_v(tan: S::Vector<6>) -> S::Matrix<4, 4> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::mat_v(tan)
    }

    fn mat_v_inverse(tan: S::Vector<6>) -> S::Matrix<4, 4> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::mat_v_inverse(tan)
    }

    fn ad_of_translation(point: S::Vector<4>) -> S::Matrix<4, 6> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::ad_of_translation(point)
    }

    fn adj_of_translation(params: &S::Vector<7>, p: S::Vector<4>) -> S::Matrix<4, 6> {
        RotationBoost3Impl::<S, BATCH, DM, DN>::adj_of_translation(params, p)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use sophus_autodiff::linalg::VecF64;

    use super::*;

    #[test]
    fn compose_id() {
        let id = Lorenz3F64::identity();
        let p = Lorenz3F64::exp(VecF64::from_array([0.1, 0.2, -0.1, 0.05, 0.0, 0.0]));
        let q = id * p;
        assert_relative_eq!(p.params(), q.params(), epsilon = 1e-12);
    }

    #[test]
    fn lorenz3_prop_tests() {
        #[cfg(feature = "simd")]
        use sophus_autodiff::dual::DualBatchScalar;
        use sophus_autodiff::dual::DualScalar;
        #[cfg(feature = "simd")]
        use sophus_autodiff::linalg::BatchScalarF64;

        Lorenz3F64::test_suite();
        #[cfg(feature = "simd")]
        Lorenz3::<BatchScalarF64<8>, 8, 0, 0>::test_suite();
        Lorenz3::<DualScalar<3, 1>, 1, 3, 1>::test_suite();
        #[cfg(feature = "simd")]
        Lorenz3::<DualBatchScalar<8, 1, 1>, 8, 1, 1>::test_suite();
    }
}

