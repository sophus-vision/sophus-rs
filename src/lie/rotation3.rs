use dfdx::prelude::*;

use crate::calculus::batch_types::*;
use crate::calculus::make::*;
use crate::calculus::params::*;
use crate::calculus::points::*;
use crate::lie::group::*;
use crate::lie::traits::*;
use crate::manifold::traits::*;

#[derive(Debug, Clone)]
pub struct Rotation3Impl<const BATCH: usize> {
    _phantom: std::marker::PhantomData<()>,
}

impl<const BATCH: usize> Rotation3Impl<BATCH> {}



impl<const BATCH: usize> ParamsTestUtils<BATCH, 4> for Rotation3Impl<BATCH> {
    fn tutil_params_examples() -> Vec<V<BATCH, 4>> {
        let mut params = vec![];

        let dev = dfdx::tensor::Cpu::default();

        let o: V<BATCH, 1> = dev.ones();
        let l: V<BATCH, 1> = dev.ones();
        let o_01: V<BATCH, 1> = l.clone() * 0.01;
        let o_2: V<BATCH, 1> = l.clone() * 0.2;
        let o_99: V<BATCH, 1> = l.clone() * 0.99;

        params.push(Rotation3Impl::exp(make_blockvec3(
            o.clone(),
            o.clone(),
            o.clone(),
        )));
        params.push(Rotation3Impl::exp(make_blockvec3(
            o.clone(),
            l.clone(),
            o.clone(),
        )));
        params.push(Rotation3Impl::exp(make_blockvec3(
            o.clone(),
            o.clone(),
            l.clone(),
        )));
        params.push(Rotation3Impl::exp(make_blockvec3(
            o.clone(),
            o_2.clone(),
            o_2.clone(),
        )));

        params.push(normalized(make_blockvec4(
            o_99.clone(),
            o_01.clone(),
            o_01.clone(),
            o_01.clone(),
        )));
        params.push(normalized(make_blockvec4(
            o_99.clone(),
            o_01.clone(),
            o_01.clone(),
            o_01.clone(),
        )));
        params.push(normalized(make_blockvec4(
            o_99.clone(),
            o_01.clone(),
            o.clone(),
            o.clone(),
        )));
        params.push(normalized(make_blockvec4(
            o_99.clone(),
            o_01.clone().negate(),
            o.clone(),
            o.clone(),
        )));
        params.push(normalized(make_blockvec4(
            o_99.clone(),
            o_01.clone(),
            o.clone(),
            o.clone(),
        )));
        params.push(normalized(make_blockvec4(
            o_99.clone().negate(),
            o_01.clone(),
            o.clone(),
            o_01.clone(),
        )));
        params.push(normalized(make_blockvec4(
            o_99.clone().negate(),
            o_01.clone().negate(),
            o.clone(),
            o_01.clone(),
        )));

        params
    }

    fn tutil_invalid_params_examples() -> Vec<V<BATCH, 4>> {
        let dev = dfdx::tensor::Cpu::default();
        vec![dev.zeros(), dev.ones() * 0.5, dev.ones().negate() * 0.5]
    }
}

impl<const BATCH: usize> ParamsImpl<BATCH, 4> for Rotation3Impl<BATCH> {
    fn are_params_valid(params: &V<BATCH, 4>) -> bool {
        let params = params.clone();
        let norm: Tensor<Rank1<BATCH>, _, _, _> = params.square().sum::<_, Axis<1>>();
        let one = 1.0;
        let eps = 1e-6;
        (norm - one).abs().array().iter().all(|x| x <= &eps)
    }
}

impl<const BATCH: usize> TangentTestUtil<BATCH, 3> for Rotation3Impl<BATCH> {
    fn tutil_tangent_examples() -> Vec<V<BATCH, 3>> {
        let dev = dfdx::tensor::Cpu::default();
        vec![
            dev.zeros(),
            dev.ones(),
            dev.ones().negate(),
            dev.ones() * 0.5,
            dev.ones().negate() * 0.5,
        ]
    }
}

impl<const BATCH: usize> LieGroupImplTrait<BATCH, 3, 4, 3, 3> for Rotation3Impl<BATCH> {
    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = true;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> V<BATCH, 4> {
        let zero = 0.0;
        let one = 1.0;
        let dev = dfdx::tensor::Cpu::default();
        dev.tensor([zero, zero, zero, one]).broadcast()
    }

    fn into_group_adjoint<Tape: SophusTape>(
        params: GenV<BATCH, 4, Tape>,
    ) -> GenM<BATCH, 3, 3, Tape> {
        Self::into_matrix(params)
    }

    fn exp<Tape: SophusTape>(omega: GenV<BATCH, 3, Tape>) -> GenV<BATCH, 4, Tape> {
        let theta_sq: GenS<BATCH, Tape> = omega.clone().square().sum();
        let theta_po4 = theta_sq.clone().square();
        let theta = theta_sq.clone().sqrt();
        let half_theta = theta.clone() * 0.5;

        let imag_factor: GenS<BATCH, Tape> = half_theta.clone().sin() / theta;
        let real_factor: GenS<BATCH, Tape> = half_theta.clone().cos();

        let one: GenS<BATCH, Tape> = half_theta.device().ones().retaped();

        let small_imag_factor = one.clone() * 0.5 - theta_sq.clone() * (1.0 / 48.0)
            + theta_po4.clone() * (1.0 / 3840.0);
        let small_real_factor = one - theta_sq.clone() * (1.0 / 8.0) + theta_po4 * (1.0 / 384.0);

        let theta_is_small = theta_sq.le(1e-6);
        let imag_factor = theta_is_small
            .clone()
            .choose(small_imag_factor, imag_factor);
        let real_factor: GenV<BATCH, 1, Tape> = theta_is_small
            .choose(small_real_factor, real_factor)
            .reshape();

        let im = imag_factor.broadcast() * omega;

        make_blockvec2(im, real_factor)
    }

    fn into_log<Tape: SophusTape>(params: GenV<BATCH, 4, Tape>) -> GenV<BATCH, 3, Tape> {
        let ivec: GenV<BATCH, 3, Tape> = params.clone().slice((.., ..3)).realize();
        let squared_n: GenS<BATCH, Tape> = ivec.clone().square().sum();
        let w: GenV<BATCH, 1, Tape> = params.slice((.., 3..4)).realize();
        let w: GenS<BATCH, Tape> = w.reshape();
        let w_sq = w.clone().square();

        let n = squared_n.clone().sqrt();
        let w_is_positive = w.le(1e-6);
        let mod_n = w_is_positive.clone().choose(n.clone(), n.clone().negate());
        let mod_w = w_is_positive.choose(w.clone(), w.clone().negate());
        let atan_nbyw = mod_n.atan2(mod_w);

        let n_is_small = squared_n.le(1e-6);

        let two_atan_nbyd_by_n: GenS<BATCH, Tape> = (atan_nbyw / n * 2.0).reshape();

        let one: GenS<BATCH, Tape> = two_atan_nbyd_by_n.device().ones().retaped();
        let small_two_atan_nbyd_by_n: GenS<BATCH, Tape> =
            one * 2.0 / w.clone() - squared_n / (w_sq * w) * 2.0 / 3.0;

        let two_atan_nbyd_by_n = n_is_small.choose(small_two_atan_nbyd_by_n, two_atan_nbyd_by_n);

        two_atan_nbyd_by_n.broadcast() * ivec
    }

    fn hat<Tape: SophusTape>(omega: GenV<BATCH, 3, Tape>) -> GenM<BATCH, 3, 3, Tape> {
        let zero: GenV<BATCH, 1, Tape> = omega.device().zeros().retaped();
        let omega_x = omega.clone().slice((.., 0..1)).realize();
        let omega_y = omega.clone().slice((.., 1..2)).realize();
        let omega_z = omega.clone().slice((.., 2..3)).realize();
        let neg_omega_x = omega_x.clone().negate();
        let neg_omega_y = omega_y.clone().negate();
        let neg_omega_z = omega_z.clone().negate();

        make_3rowblock_mat(
            make_3colvec_mat(zero.clone(), neg_omega_z, omega_y),
            make_3colvec_mat(omega_z, zero.clone(), neg_omega_x),
            make_3colvec_mat(neg_omega_y, omega_x, zero),
        )
    }

    fn vee<Tape: SophusTape>(omega_hat: GenM<BATCH, 3, 3, Tape>) -> GenV<BATCH, 3, Tape> {
        let omega_x: GenM<BATCH, 1, 1, Tape> = omega_hat.clone().slice((.., 2..3, 1..2)).realize();
        let omega_y: GenM<BATCH, 1, 1, Tape> = omega_hat.clone().slice((.., 0..1, 2..)).realize();
        let omega_z: GenM<BATCH, 1, 1, Tape> = omega_hat.slice((.., 1..2, 0..1)).realize();
        make_vec3(omega_x.reshape(), omega_y.reshape(), omega_z.reshape())
    }

    fn mul_assign<LeftTape: SophusTape + Merge<RightTape>, RightTape: SophusTape>(
        lhs: GenV<BATCH, 4, LeftTape>,
        rhs: GenV<BATCH, 4, RightTape>,
    ) -> GenV<BATCH, 4, LeftTape> {
        let lhs_ivec: GenV<BATCH, 3, LeftTape> = lhs.clone().slice((.., 0..3)).realize();
        let rhs_ivec: GenV<BATCH, 3, RightTape> = rhs.clone().slice((.., 0..3)).realize();

        let lhs_re: GenV<BATCH, 1, LeftTape> = lhs.clone().slice((.., 3..4)).realize();
        let lhs_re: GenS<BATCH, LeftTape> = lhs_re.reshape();
        let rhs_re: GenV<BATCH, 1, RightTape> = rhs.clone().slice((.., 3..4)).realize();
        let rhs_re: GenS<BATCH, RightTape> = rhs_re.reshape();

        let ivec: GenV<BATCH, 3, LeftTape> = lhs_re.clone().broadcast() * rhs_ivec.clone()
            +   lhs_ivec.clone()*rhs_re.clone().broadcast()
            + cross(lhs_ivec.clone(), rhs_ivec.clone());

        let re: GenS<BATCH, LeftTape> = lhs_re * rhs_re - (lhs_ivec * rhs_ivec).sum();
        let re: GenV<BATCH, 1, LeftTape> = re.reshape();

        make_blockvec2(ivec, re)
    }

    fn into_inverse<Tape: SophusTape>(params: GenV<BATCH, 4, Tape>) -> GenV<BATCH, 4, Tape> {
        let neg_im: GenV<BATCH, 3, Tape> = params.clone().slice((.., 0..3)).negate().realize();
        let re: GenV<BATCH, 1, Tape> = params.slice((.., 3..4)).realize();
        make_blockvec2(neg_im, re)
    }

    fn point_action<
        const NUM_POINTS: usize,
        Tape: SophusTape + Merge<PointTape>,
        PointTape: SophusTape,
    >(
        params: GenV<BATCH, 4, Tape>,
        points: GenM<BATCH, 3, NUM_POINTS, PointTape>,
    ) -> GenM<BATCH, 3, NUM_POINTS, Tape> {
        Self::into_matrix(params).matmul(points)
    }

    fn into_ambient<const NUM_POINTS: usize, Tape: SophusTape>(
        params: GenM<BATCH, 3, NUM_POINTS, Tape>,
    ) -> GenM<BATCH, 3, NUM_POINTS, Tape> {
        params
    }

    fn into_compact<Tape: SophusTape>(params: GenV<BATCH, 4, Tape>) -> GenM<BATCH, 3, 3, Tape> {
        Self::into_matrix(params)
    }

    fn into_matrix<Tape: SophusTape>(params: GenV<BATCH, 4, Tape>) -> GenM<BATCH, 3, 3, Tape> {
        let ivec: GenV<BATCH, 3, Tape> = params.clone().slice((.., 0..3)).realize();
        let re: GenV<BATCH, 1, Tape> = params.slice((.., 3..4)).realize();
        let re: GenS<BATCH, Tape> = re.reshape();

        let uv_x = cross(ivec.clone(), unit_x()) * 2.0;
        let uv_y = cross(ivec.clone(), unit_y()) * 2.0;
        let uv_z = cross(ivec.clone(), unit_z()) * 2.0;

        let col_x: GenV<BATCH, 3, Tape> =
            unit_x().retaped() + cross(ivec.clone(), uv_x.clone()) + uv_x * re.clone().broadcast();
        let col_y: GenV<BATCH, 3, Tape> =
            unit_y().retaped() + cross(ivec.clone(), uv_y.clone()) + uv_y * re.clone().broadcast();
        let col_z: GenV<BATCH, 3, Tape> =
            unit_z().retaped() + cross(ivec, uv_z.clone()) + uv_z * re.broadcast();

        make_3colvec_mat(col_x, col_y, col_z)
    }

    fn algebra_adjoint(tangent: V<BATCH, 3>) -> M<BATCH, 3, 3> {
        Self::hat(tangent)
    }

    fn dx_exp_x_at_0() -> M<BATCH, 4, 3> {
        let o: M<BATCH, 1, 1> = Cpu::default().zeros();
        let o_1x3: M<BATCH, 1, 3> = Cpu::default().zeros();
        let o_5: M<BATCH, 1, 1> = Cpu::default().ones() * 0.5;

        make_4rowblock_mat(
            make_3blockcol_mat(o_5.clone(), o.clone(), o.clone()),
            make_3blockcol_mat(o.clone(), o_5.clone(), o.clone()),
            make_3blockcol_mat(o.clone(), o.clone(), o_5.clone()),
            o_1x3,
        )
    }

    fn dx_exp_x_times_point_at_0(point: V<BATCH, 3>) -> M<BATCH, 3, 3> {
        Self::hat(point.negate())
    }

    fn dx_self_times_exp_x_at_0(params: &V<BATCH, 4>) -> M<BATCH, 4, 3> {
        let half_params = params.clone() * 0.5;
        let half_re: V<BATCH, 1> = half_params.clone().slice((.., 3..4)).realize();

        let half_im_x = half_params.clone().slice((.., 0..1)).realize();
        let half_im_y = half_params.clone().slice((.., 1..2)).realize();
        let half_im_z = half_params.clone().slice((.., 2..3)).realize();
        let half_neg_im_x = half_im_x.clone().negate();
        let half_neg_im_y = half_im_y.clone().negate();
        let half_neg_im_z = half_im_z.clone().negate();

        make_4rowblock_mat(
            make_3colvec_mat(half_re.clone(), half_neg_im_z, half_im_y.clone()),
            make_3colvec_mat(half_im_z.clone(), half_re.clone(), half_neg_im_x),
            make_3colvec_mat(half_neg_im_y, half_im_x.clone(), half_re),
            make_3colvec_mat(half_im_x, half_im_y, half_im_z),
        )
    }

    fn dx_log_exp_x_times_self_at_0(_params: &V<BATCH, 4>) -> M<BATCH, 3, 3> {
        todo!()

        // fn dx_log_exp_x_times_self_at_0(q: &V<T, 4>) -> M<T, 3, 3> {
        //     let l = T::one();
        //     let two = T::one() + T::one();

        //     let r = q[3];
        //     let ivec = V::<T, 3>::new(q[0], q[1], q[2]);

        //     let c0 = hyperdual::Float::powi(r, 4);
        //     let c1 = hyperdual::Float::powi(ivec[1], 4);
        //     let c2 = hyperdual::Float::powi(ivec[2], 4);
        //     let c3 = hyperdual::Float::powi(ivec[0], 2);
        //     let c4 = hyperdual::Float::powi(ivec[1], 2);
        //     let c5 = c3 * c4;
        //     let c6 = hyperdual::Float::powi(ivec[2], 2);
        //     let c7 = c3 * c6;
        //     let c8 = c4 * c6;
        //     let c9 = l / (c0 + c1 + c2 + two * c5 + two * c7 + two * c8);
        //     let c10 = hyperdual::Float::sqrt(c3 + c4 + c6);
        //     let c11 = c10 * hyperdual::Float::atan(c10);
        //     let c12 = c11 * r;
        //     let c13 = c12 * c6 + c5;
        //     let c14 = c12 * c4 + c7;
        //     let c15 = hyperdual::Float::powi(ivec[1], 3);
        //     let c16 = hyperdual::Float::powi(ivec[0], 3);
        //     let c17 = c6 * ivec[1];
        //     let c18 = c12 * ivec[0];
        //     let c19 = c18 * ivec[1];
        //     let c20 = hyperdual::Float::powi(ivec[2], 3);
        //     let c21 = c3 * ivec[2];
        //     let c22 = c4 * ivec[2];
        //     let c23 = c11 * c20 + c11 * c21 + c11 * c22;
        //     let c24 = c18 * ivec[2];
        //     let c25 = c11 * c15 + c11 * c17 + c11 * c3 * ivec[1];
        //     let c26 = c12 * c3 + c8;
        //     let c27 = c12 * ivec[1] * ivec[2];
        //     let c28 = c11 * ivec[0];
        //     let c29 = c11 * c16 + c28 * c4 + c28 * c6;

        //     let mut result = M::zeros();
        //     result[(0, 0)] = c9 * (c0 + c13 + c14);
        //     result[(1, 0)] = c9 * (c15 * ivec[0] + c16 * ivec[1] + c17 * ivec[0] - c19 + c23);
        //     result[(2, 0)] = c9 * (c16 * ivec[2] + c20 * ivec[0] - c24 - c25 + c4 * ivec[0] * ivec[2]);
        //     result[(0, 1)] = c9 * (c15 * ivec[0] + c16 * ivec[1] - c19 - c23 + c6 * ivec[0] * ivec[1]);
        //     result[(1, 1)] = c9 * (c1 + c13 + c26);
        //     result[(2, 1)] = c9 * (c15 * ivec[2] + c20 * ivec[1] + c21 * ivec[1] - c27 + c29);
        //     result[(0, 2)] = c9 * (c16 * ivec[2] + c20 * ivec[0] + c22 * ivec[0] - c24 + c25);
        //     result[(1, 2)] = c9 * (c15 * ivec[2] + c20 * ivec[1] - c27 - c29 + c3 * ivec[1] * ivec[2]);
        //     result[(2, 2)] = c9 * (c14 + c2 + c26);
        //     result
        // }
    }
}

pub type Rotation3<const BATCH: usize> = LieGroup<BATCH, 3, 4, 3, 3, Rotation3Impl<BATCH>>;

mod tests {

    #[test]
    fn rotation3_batch_tests() {
        use crate::lie::rotation3::*;

        Rotation3::<1>::test_suite();
        Rotation3::<2>::test_suite();
        Rotation3::<4>::test_suite();
    }
}
