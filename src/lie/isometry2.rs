use dfdx::prelude::*;

use crate::calculus::batch_types::*;
use crate::calculus::make::*;
use crate::lie::group::*;
use crate::lie::rotation2::*;
use crate::lie::traits::*;
use crate::lie::translation_group_product::*;
use crate::manifold::traits::Manifold;

impl<const BATCH: usize> FactorGroupImplTrait<BATCH, 1, 2, 2, 2> for Rotation2Impl<BATCH> {
    fn mat_v<
        ParamsTape:SophusTape + std::fmt::Debug + Merge<Tape>,
        Tape:SophusTape,
    >(
        params: GenV<BATCH, 2, ParamsTape>,
        tangent: GenV<BATCH, 1, Tape>,
    ) -> GenM<BATCH, 2, 2, ParamsTape>
    where
        Tape:SophusTape + Merge<ParamsTape>,
    {
        let p0: GenV<BATCH, 1, _> = params.clone().slice((.., 0..1)).realize();
        let p1: GenV<BATCH, 1, _> = params.clone().slice((.., 1..2)).realize();
        let theta: GenS<BATCH, _> = tangent.reshape();
        let p0: GenS<BATCH, _> = p0.reshape();
        let p1: GenS<BATCH, _> = p1.reshape();

        let one: GenS<BATCH, ParamsTape> = theta.device().ones().retaped();
        let theta_sq = theta.clone().square();
        let theta_p3 = theta.clone() * theta.clone().square();

        let small_sin_theta_by_theta = one.clone() - theta_sq.clone() * (1.0f64 / 6.0f64);
        let small_one_minus_cos_theta_by_theta = theta.clone() * 0.5 - theta_p3 * (1.0 / 24.0);

        let sin_theta_by_theta = p1 / theta.clone();
        let one_minus_cos_theta_by_theta = (one - p0) / theta.clone();

        let theta_not_small = theta.clone().abs().ge(1e-6);

        // Scalar sin_theta_by_theta;
        // Scalar one_minus_cos_theta_by_theta;
        // using std::abs;

        // if (abs(theta[0]) < kEpsilon<Scalar>) {
        //   Scalar theta_sq = theta[0] * theta[0];
        //   sin_theta_by_theta = Scalar(1.) - Scalar(1. / 6.) * theta_sq;
        //   one_minus_cos_theta_by_theta =
        //       Scalar(0.5) * theta[0] - Scalar(1. / 24.) * theta[0] * theta_sq;
        // } else {
        //   sin_theta_by_theta = params.y() / theta[0];
        //   one_minus_cos_theta_by_theta = (Scalar(1.) - params.x()) / theta[0];
        // }
        // return Eigen::Matrix<Scalar, 2, 2>(
        //     {{sin_theta_by_theta, -one_minus_cos_theta_by_theta},
        //      {one_minus_cos_theta_by_theta, sin_theta_by_theta}});

        let sin_theta_by_theta = theta_not_small
            .clone()
            .choose(sin_theta_by_theta, small_sin_theta_by_theta);
        let one_minus_cos_theta_by_theta = theta_not_small.choose(
            one_minus_cos_theta_by_theta,
            small_one_minus_cos_theta_by_theta,
        );

        make_2rowblock_mat(
            make_2col_mat(
                sin_theta_by_theta.clone(),
                one_minus_cos_theta_by_theta.clone().negate(),
            ),
            make_2col_mat(one_minus_cos_theta_by_theta, sin_theta_by_theta),
        )
    }

    fn mat_v_inverse<
        ParamsTape:SophusTape + std::fmt::Debug + Merge<Tape>,
        Tape,
    >(
        params: GenV<BATCH, 2, ParamsTape>,
        tangent: GenV<BATCH, 1, Tape>,
    ) -> GenM<BATCH, 2, 2, ParamsTape>
    where
        Tape:SophusTape + Merge<ParamsTape>,
    {
        let halftheta: GenS<BATCH, _> = (tangent.clone() * 0.5).reshape();

        let real_minus_one: GenV<BATCH, 1, _> = (params.clone().slice((.., 0..1)) - 1.0).realize();
        let real_minus_one: GenS<BATCH, _> = real_minus_one.reshape();
        let im: GenV<BATCH, 1, _> = params.slice((.., 1..2)).realize();
        let im: GenS<BATCH, _> = im.reshape();

        let one: GenS<BATCH, ParamsTape> = tangent.device().ones().retaped();

        let real_not_small = real_minus_one.clone().abs().ge(1e-6);

        let a: GenS<BATCH, _> = one - (tangent.square() * (1.0 / 12.0)).reshape();
        let b = -(im * halftheta.clone()) / real_minus_one;
        let halftheta_by_tan_of_halftheta = real_not_small.choose(b, a);

        let row0 = make_2col_mat(halftheta_by_tan_of_halftheta.clone(), halftheta.clone());
        let row1 = make_2col_mat(halftheta.negate(), halftheta_by_tan_of_halftheta);

        make_2rowblock_mat(row0, row1)
    }

    fn group_adjoint_of_translation<
        ParamsTape:SophusTape + std::fmt::Debug + Merge<Tape>,
        Tape:SophusTape,
    >(
        params: GenV<BATCH, 2, ParamsTape>,
        point: GenV<BATCH, 2, Tape>,
    ) -> GenM<BATCH, 2, 1, ParamsTape> {
        let params_tape = params.split_tape().1;
        let (point, tape) = point.split_tape();
        let tape = params_tape.merge(tape);

        let (neg_px, tape) = point
            .clone()
            .put_tape(tape)
            .negate()
            .slice((.., 0..1))
            .realize()
            .split_tape();
        let py: GenV<BATCH, 1, ParamsTape> = point.put_tape(tape).slice((.., 1..2)).realize();
        [py, neg_px.retaped()]
            .stack()
            .permute::<_, Axes3<1, 0, 2>>()
    }

    fn algebra_adjoint_of_translation(point: V<BATCH, 2>) -> M<BATCH, 2, 1> {
        let px: V<BATCH, 1> = point.clone().slice((.., 0..1)).realize();
        let py: V<BATCH, 1> = point.slice((.., 1..2)).realize();

        make_vec2::<BATCH, NoneTape>(py.reshape(), px.negate().reshape()).reshape()
    }
}

pub type GenTapedIsometry2<const BATCH: usize, GenTape> = GenTapedLieGroup<
    BATCH,
    3,
    4,
    2,
    3,
    GenTape,
    TranslationGroupProductImpl<BATCH, 3, 4, 2, 3, 1, 2, Rotation2Impl<BATCH>>,
>;

pub type Isometry2<const BATCH: usize> = GenTapedIsometry2<BATCH, NoneTape>;

mod tests {

    #[test]
    fn isometry2_batch_tests() {
        use crate::lie::isometry2::*;

        Isometry2::<1>::test_suite();
        Isometry2::<2>::test_suite();
        Isometry2::<4>::test_suite();
    }
}

impl Manifold<3> for Isometry2<1> {
    fn into_oplus(self, tangent: V<1, 3>) -> Self {
        Isometry2::<1>::mul(Isometry2::exp(tangent), self)
    }
}
