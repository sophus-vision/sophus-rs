use dfdx_core::prelude::*;

use crate::calculus::batch_types::*;
use crate::calculus::make::*;
use crate::calculus::params::*;
use crate::lie::group::*;
use crate::lie::traits::*;
use crate::manifold::traits::*;

#[derive(Debug, Clone)]
pub struct Rotation2Impl<const BATCH: usize> {
    _phantom: std::marker::PhantomData<()>,
}

impl<const BATCH: usize> Rotation2Impl<BATCH> {}

impl<const BATCH: usize> ParamsTestUtils<BATCH, 2> for Rotation2Impl<BATCH> {
    fn tutil_params_examples() -> Vec<V<BATCH, 2>> {
        let mut params = vec![];
        for i in 2..10 {
            let angle = Into::into(i as f64 * std::f64::consts::PI / 5.0);
            let dev = dfdx_core::tensor::Cpu::default();
            let mut batch = dev.zeros();
            for i in 0..BATCH {
                batch[[i, 0]] = angle;
            }
            let foo = Rotation2::<BATCH>::exp(batch);
            params.push(foo.params().clone());
        }
        params
    }

    fn tutil_invalid_params_examples() -> Vec<V<BATCH, 2>> {
        let dev = dfdx_core::tensor::Cpu::default();
        vec![dev.zeros(), dev.ones() * 0.5, dev.ones().negate() * 0.5]
    }
}

impl<const BATCH: usize> ParamsImpl<BATCH, 2> for Rotation2Impl<BATCH> {
    fn are_params_valid(params: &V<BATCH, 2>) -> bool {
        let params = params.clone();
        let norm: Tensor<Rank1<BATCH>, _, _, _> = params.square().sum::<_, Axis<1>>();
        let one = 1.0;
        let eps = 1e-6;
        (norm - one).abs().array().iter().all(|x| x <= &eps)
    }
}

impl<const BATCH: usize> TangentTestUtil<BATCH, 1> for Rotation2Impl<BATCH> {
    fn tutil_tangent_examples() -> Vec<V<BATCH, 1>> {
        let dev = dfdx_core::tensor::Cpu::default();
        vec![
            dev.zeros(),
            dev.ones(),
            dev.ones().negate(),
            dev.ones() * 0.5,
            dev.ones().negate() * 0.5,
        ]
    }
}

impl<const BATCH: usize> LieGroupImplTrait<BATCH, 1, 2, 2, 2> for Rotation2Impl<BATCH> {
    const IS_ORIGIN_PRESERVING: bool = true;
    const IS_AXIS_DIRECTION_PRESERVING: bool = false;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = false;
    const IS_SHAPE_PRESERVING: bool = true;
    const IS_DISTANCE_PRESERVING: bool = true;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> V<BATCH, 2> {
        let zero = 0.0;
        let one = 1.0;
        let dev = dfdx_core::tensor::Cpu::default();
        dev.tensor([one, zero]).broadcast()
    }

    fn into_group_adjoint<Tape: SophusTape>(
        params: GenV<BATCH, 2, Tape>,
    ) -> GenM<BATCH, 1, 1, Tape> {
        params.device().ones().retaped()
    }

    fn exp<Tape: SophusTape>(omega: GenV<BATCH, 1, Tape>) -> GenV<BATCH, 2, Tape> {
        (omega.clone().cos(), omega.clone().sin()).concat_along(Axis::<1>)
    }

    fn into_log<Tape: SophusTape>(params: GenV<BATCH, 2, Tape>) -> GenV<BATCH, 1, Tape> {
        let x = params.clone().slice((.., 0..1));
        let y = params.slice((.., 1..2));
        atan2(y,x).realize()
    }

    fn hat<Tape: SophusTape>(omega: GenV<BATCH, 1, Tape>) -> GenM<BATCH, 2, 2, Tape> {
        let zero: V<BATCH, 1> = omega.device().zeros();
        // let omega: GenS<BATCH, Tape> = omega.reshape();

        make_2rowblock_mat(
            make_2colvec_mat(zero.retaped(), omega.clone().negate()),
            make_2colvec_mat(omega, zero),
        )
    }

    fn vee<Tape: SophusTape>(hat: GenM<BATCH, 2, 2, Tape>) -> GenV<BATCH, 1, Tape> {
        let vee_mat: GenM<BATCH, 1, 1, Tape> = hat.slice((.., 1..2, 0..1)).realize();
        vee_mat.reshape()
    }

    fn mul_assign<
        LeftTape:SophusTape + Merge<RightTape>,
        RightTape:SophusTape,
    >(
        lhs: GenV<BATCH, 2, LeftTape>,
        rhs: GenV<BATCH, 2, RightTape>,
    ) -> GenV<BATCH, 2, LeftTape> {
        let re_lhs: GenV<BATCH, 1, _> = lhs.clone().slice((.., 0..1)).realize();
        let im_lhs: GenV<BATCH, 1, _> = lhs.clone().slice((.., 1..2)).realize();
        let re_rhs: GenV<BATCH, 1, _> = rhs.clone().slice((.., 0..1)).realize();
        let im_rhs: GenV<BATCH, 1, _> = rhs.clone().slice((.., 1..2)).realize();

        let re: GenV<BATCH, 1, LeftTape> =
            re_lhs.clone() * re_rhs.clone() - im_lhs.clone() * im_rhs.clone();
        let im: GenV<BATCH, 1, LeftTape> = re_lhs * im_rhs + im_lhs * re_rhs;

        (re, im).concat_along(Axis::<1>)
    }

    fn into_inverse<Tape: SophusTape>(params: GenV<BATCH, 2, Tape>) -> GenV<BATCH, 2, Tape> {
        let im: GenV<BATCH, 1, _> = params.clone().slice((.., 1..2)).realize();
        let re: GenV<BATCH, 1, _> = params.slice((.., 0..1)).realize();

        (re, im.negate()).concat_along(Axis::<1>)
    }

    fn point_action<
        const NUM_POINTS: usize,
        Tape: SophusTape + Merge<PointTape>,
        PointTape: SophusTape,
    >(
        params: GenV<BATCH, 2, Tape>,
        points: GenM<BATCH, 2, NUM_POINTS, PointTape>,
    ) -> GenM<BATCH, 2, NUM_POINTS, Tape> {
        Self::into_matrix(params).matmul(points)
    }

    fn into_ambient<const NUM_POINTS: usize, Tape: SophusTape>(
        params: GenM<BATCH, 2, NUM_POINTS, Tape>,
    ) -> GenM<BATCH, 2, NUM_POINTS, Tape> {
        params
    }

    fn into_compact<Tape: SophusTape>(params: GenV<BATCH, 2, Tape>) -> GenM<BATCH, 2, 2, Tape> {
        Self::into_matrix(params)
    }

    fn into_matrix<Tape: SophusTape>(params: GenV<BATCH, 2, Tape>) -> GenM<BATCH, 2, 2, Tape> {
        let cos: Tensor<(Const<BATCH>, Const<1>), f64, _, Tape> =
            params.clone().slice((.., 0..1)).realize();

        let sin: Tensor<(Const<BATCH>, Const<1>), f64, _, Tape> =
            params.clone().slice((.., 1..2)).realize();

        make_2rowblock_mat(
            make_2colvec_mat(cos.clone(), sin.clone().negate()),
            make_2colvec_mat(sin, cos),
        )
    }

    fn algebra_adjoint(_tangent: V<BATCH, 1>) -> M<BATCH, 1, 1> {
        dfdx_core::tensor::Cpu::default().zeros()
    }

    fn dx_exp_x_at_0() -> M<BATCH, 2, 1> {
        let dev = dfdx_core::tensor::Cpu::default();
        let zero: V<BATCH, 1> = dev.zeros().retaped();
        let one: V<BATCH, 1> = dev.ones().retaped();
        [zero, one].stack().permute::<_, Axes3<1, 0, 2>>()
    }

    fn dx_exp_x_times_point_at_0(point: V<BATCH, 2>) -> M<BATCH, 2, 1> {
        let py: Tensor<(Const<BATCH>, Const<1>), f64, _, _> =
            point.clone().slice((.., 1..2)).realize();
        let px: Tensor<(Const<BATCH>, Const<1>), f64, _, _> = point.slice((.., 0..1)).realize();
        [py.negate(), px].stack().permute::<_, Axes3<1, 0, 2>>()
    }

    fn dx_self_times_exp_x_at_0(params: &V<BATCH, 2>) -> M<BATCH, 2, 1> {
        let py: Tensor<(Const<BATCH>, Const<1>), f64, _, _> =
            params.clone().slice((.., 1..2)).realize();
        let px: Tensor<(Const<BATCH>, Const<1>), f64, _, _> =
            params.clone().slice((.., 0..1)).realize();
        [py.negate(), px].stack().permute::<_, Axes3<1, 0, 2>>()
    }

    fn dx_log_exp_x_times_self_at_0(_params: &V<BATCH, 2>) -> M<BATCH, 1, 1> {
        dfdx_core::tensor::Cpu::default().ones()
    }
}

pub type Rotation2<const BATCH: usize> = LieGroup<BATCH, 1, 2, 2, 2, Rotation2Impl<BATCH>>;

mod tests {

    #[test]
    fn rotation2_batch_tests() {
        use crate::lie::rotation2::*;

        Rotation2::<1>::test_suite();
        Rotation2::<2>::test_suite();
        Rotation2::<4>::test_suite();
    }
}
