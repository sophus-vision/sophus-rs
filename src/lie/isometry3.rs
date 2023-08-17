use dfdx::prelude::*;

use crate::calculus::batch_types::*;
use crate::calculus::points::*;
use crate::lie::group::*;
use crate::lie::rotation3::*;
use crate::lie::traits::*;
use crate::lie::translation_group_product::*;
use crate::manifold::traits::Manifold;

impl<const BATCH: usize> FactorGroupImplTrait<BATCH, 3, 4, 3, 3> for Rotation3Impl<BATCH> {
    fn mat_v<ParamsTape: SophusTape + std::fmt::Debug + Merge<Tape>, Tape: SophusTape>(
        _params: GenV<BATCH, 4, ParamsTape>,
        omega: GenV<BATCH, 3, Tape>,
    ) -> GenM<BATCH, 3, 3, ParamsTape>
    where
        Tape: SophusTape + Merge<ParamsTape>,
    {
        let theta_sq: GenS<BATCH, Tape> = omega.clone().square().sum::<_, Axis<1>>();
        let theta = theta_sq.clone().sqrt();
        let one: GenS<BATCH, Tape> = theta.device().ones().retaped();
        let zero: GenS<BATCH, Tape> = theta.device().zeros().retaped();

        let mat_omega = Self::hat(omega);
        let mat_omega_sq = mat_omega.clone().matmul(mat_omega.clone());

        let is_theta_sq_small = theta_sq.clone().le(1e-6);

        let factor1: GenS<BATCH, Tape> = (one.clone() - theta.clone().cos()) / theta_sq.clone();
        let factor2: GenS<BATCH, Tape> = (theta.clone() - theta.clone().sin()) / (theta_sq * theta);
        let small_factor1: GenS<BATCH, Tape> = one * 0.5;
        let small_factor2: GenS<BATCH, Tape> = zero;

        let factor1 = is_theta_sq_small.clone().choose(small_factor1, factor1);
        let factor2 = is_theta_sq_small.choose(small_factor2, factor2);

        let mat_v: GenM<BATCH, 3, 3, ParamsTape> =
            Identity3().retaped() + factor1.broadcast() * mat_omega + factor2.broadcast() * mat_omega_sq;

        mat_v
    }

    fn mat_v_inverse<ParamsTape: SophusTape + std::fmt::Debug + Merge<Tape>, Tape>(
        _params: GenV<BATCH, 4, ParamsTape>,
        omega: GenV<BATCH, 3, Tape>,
    ) -> GenM<BATCH, 3, 3, ParamsTape>
    where
        Tape: SophusTape + Merge<ParamsTape>,
    {
        let theta_sq: GenS<BATCH, Tape> = omega.clone().square().sum::<_, Axis<1>>();
        let theta = theta_sq.clone().sqrt();
        let half_theta = theta.clone() * 0.5;
        let one: GenS<BATCH, Tape> = theta.device().ones().retaped();
        let _zero: GenS<BATCH, Tape> = theta.device().zeros().retaped();

        let mat_omega = Self::hat(omega);
        let mat_omega_sq = mat_omega.clone().matmul(mat_omega.clone());

        let is_theta_sq_small = theta_sq.clone().le(1e-6);

        let factor1: GenS<BATCH, Tape> = one.clone() * 0.5;
        let factor2: GenS<BATCH, Tape> =
            (one.clone() - theta * 0.5 * half_theta.clone().cos() / half_theta.sin()) / (theta_sq);
        let small_factor1: GenS<BATCH, Tape> = one.clone() * 0.5;
        let small_factor2: GenS<BATCH, Tape> = one * 1. / 12.;

        let factor1 = is_theta_sq_small.clone().choose(small_factor1, factor1);
        let factor2 = is_theta_sq_small.choose(small_factor2, factor2);

        let inv_mat_v: GenM<BATCH, 3, 3, ParamsTape> =
            Identity3().retaped() - factor1.broadcast() * mat_omega + factor2.broadcast() * mat_omega_sq;

        inv_mat_v
    }

    fn group_adjoint_of_translation<
        ParamsTape: SophusTape + std::fmt::Debug + Merge<Tape>,
        Tape: SophusTape,
    >(
        _params: GenV<BATCH, 4, ParamsTape>,
        _point: GenV<BATCH, 3, Tape>,
    ) -> GenM<BATCH, 3, 3, ParamsTape> {
        // let params_tape = params.split_tape().1;
        // let (point, tape) = point.split_tape();
        // let tape = params_tape.merge(tape);

        // let (neg_px, tape) = point
        //     .clone()
        //     .put_tape(tape)
        //     .negate()
        //     .slice((.., 0..1))
        //     .realize()
        //     .split_tape();
        // let py: GenV<BATCH, 1, ParamsTape> = point.put_tape(tape).slice((.., 1..2)).realize();
        // [py, neg_px.retaped()]
        //     .stack()
        //     .permute::<_, Axes3<1, 0, 2>>()

        todo!()
    }

    fn algebra_adjoint_of_translation(_point: V<BATCH, 3>) -> M<BATCH, 3, 3> {
        // let px: V<BATCH, 1> = point.clone().slice((.., 0..1)).realize();
        // let py: V<BATCH, 1> = point.slice((.., 1..2)).realize();

        // make_vec2::<BATCH, NoneTape>(py.reshape(), px.negate().reshape()).reshape()

        todo!()
    }
}

pub type GenTapedIsometry3<const BATCH: usize, GenTape> = GenTapedLieGroup<
    BATCH,
    6,
    7,
    3,
    4,
    GenTape,
    TranslationGroupProductImpl<BATCH, 6, 7, 3, 4, 3, 4, Rotation3Impl<BATCH>>,
>;

pub type Isometry3<const BATCH: usize> = GenTapedIsometry3<BATCH, NoneTape>;

mod tests {
    use super::Isometry3;

    

    #[test]
    fn isometry3_batch_tests() {
        Isometry3::<1>::test_suite();
        Isometry3::<2>::test_suite();
        Isometry3::<4>::test_suite();
    }
}

impl Manifold<6> for Isometry3<1> {
    fn into_oplus(self, tangent: V<1, 6>) -> Self {
        Isometry3::<1>::mul(Isometry3::exp(tangent), self)
    }
}
