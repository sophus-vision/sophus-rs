use dfdx::prelude::*;

use crate::calculus::batch_types::*;
use crate::calculus::params::*;
use crate::manifold::traits::TangentTestUtil;

pub trait LieGroupImplTrait<
    const B: usize,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
>: ParamsImpl<B, PARAMS> + TangentTestUtil<B, DOF>+Clone
{
    const IS_ORIGIN_PRESERVING: bool;
    const IS_AXIS_DIRECTION_PRESERVING: bool;
    const IS_DIRECTION_VECTOR_PRESERVING: bool;
    const IS_SHAPE_PRESERVING: bool;
    const IS_DISTANCE_PRESERVING: bool;
    const IS_PARALLEL_LINE_PRESERVING: bool;

    fn identity_params() -> V<B, PARAMS>;

    // Manifold / Lie Group concepts

    fn into_group_adjoint<Tape: SophusTape>(
        params: GenV<B, PARAMS, Tape>,
    ) -> GenM<B, DOF, DOF, Tape>;

    fn exp<Tape: SophusTape>(omega: GenV<B, DOF, Tape>) -> GenV<B, PARAMS, Tape>;

    fn into_log<Tape: SophusTape>(params: GenV<B, PARAMS, Tape>) -> GenV<B, DOF, Tape>;

    fn hat<Tape: SophusTape>(omega: GenV<B, DOF, Tape>) -> GenM<B, AMBIENT, AMBIENT, Tape>;

    fn vee<Tape: SophusTape>(hat: GenM<B, AMBIENT, AMBIENT, Tape>) -> GenV<B, DOF, Tape>;

    // group operations
    fn mul_assign<LeftTape: SophusTape + Merge<RightTape>, RightTape: SophusTape>(
        params1: GenV<B, PARAMS, LeftTape>,
        params2: GenV<B, PARAMS, RightTape>,
    ) -> GenV<B, PARAMS, LeftTape>;

    fn into_inverse<Tape: SophusTape>(params: GenV<B, PARAMS, Tape>) -> GenV<B, PARAMS, Tape>;

    // Group actions

    fn point_action<
        const NUM_POINTS: usize,
        Tape: SophusTape + Merge<PointTape>,
        PointTape: SophusTape,
    >(
        params: GenV<B, PARAMS, Tape>,
        point: GenM<B, POINT, NUM_POINTS, PointTape>,
    ) -> GenM<B, POINT, NUM_POINTS, Tape>;

    fn into_ambient<const NUM_POINTS: usize, Tape: SophusTape>(
        params: GenM<B, POINT, NUM_POINTS, Tape>,
    ) -> GenM<B, AMBIENT, NUM_POINTS, Tape>;

    // Matrices
    fn into_compact<Tape: SophusTape>(
        params: GenV<B, PARAMS, Tape>,
    ) -> GenM<B, POINT, AMBIENT, Tape>;

    fn into_matrix<Tape: SophusTape>(
        params: GenV<B, PARAMS, Tape>,
    ) -> GenM<B, AMBIENT, AMBIENT, Tape>;

    // Manifold / Lie Group concepts

    //  fn group_adjoint(params: &V< B, PARAMS>) -> M< B, DOF, DOF>;

    // fn log(params: &V< B, PARAMS>) -> V< B, DOF>;

    // group operations

    // fn inverse(params: &V< B, PARAMS>) -> V< B, PARAMS>{
    //     let result = params.clone();
    //     Self::into_inverse(result)
    // }

    // Group actions

    // fn mul
    // (params1: &V< B, PARAMS>, params2: V< B, PARAMS>) -> V< B, PARAMS>{
    //     Self::mul_assign(params1.clone(), params2)
    // }

    // // Matrices
    // fn compact(params: &V< B, PARAMS>) -> M< B, POINT, AMBIENT>;

    // fn matrix(params: &V< B, PARAMS>) -> M< B, AMBIENT, AMBIENT>;

    // Derivatives

    fn algebra_adjoint(tangent: V<B, DOF>) -> M<B, DOF, DOF>;

    fn dx_exp_x_at_0() -> M<B, PARAMS, DOF>;

    fn dx_exp_x_times_point_at_0(point: V<B, POINT>) -> M<B, POINT, DOF>;

    fn dx_self_times_exp_x_at_0(params: &V<B, PARAMS>) -> M<B, PARAMS, DOF>;

    fn dx_log_exp_x_times_self_at_0(params: &V<B, PARAMS>) -> M<B, DOF, DOF>;
}

pub trait FactorGroupImplTrait<
    const B: usize,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT_DIM: usize,
>: LieGroupImplTrait<B, DOF, PARAMS, POINT, AMBIENT_DIM>
{
    fn mat_v<
        ParamsTape:SophusTape + Merge<Tape>,
        Tape:SophusTape,
    >(
        params: GenV<B, PARAMS, ParamsTape>,
        tangent: GenV<B, DOF, Tape>,
    ) -> GenM<B, POINT, POINT, ParamsTape>
    where
        Tape:SophusTape + Merge<ParamsTape>;

    fn mat_v_inverse<
        ParamsTape:SophusTape + Merge<Tape>,
        Tape,
    >(
        params: GenV<B, PARAMS, ParamsTape>,
        tangent: GenV<B, DOF, Tape>,
    ) -> GenM<B, POINT, POINT, ParamsTape>
    where
        Tape:SophusTape + Merge<ParamsTape>;

    fn group_adjoint_of_translation<
        ParamsTape:SophusTape + Merge<Tape>,
        Tape:SophusTape,
    >(
        params: GenV<B, PARAMS, ParamsTape>,
        point: GenV<B, POINT, Tape>,
    ) -> GenM<B, POINT, DOF, ParamsTape>;

    fn algebra_adjoint_of_translation(point: V<B, POINT>) -> M<B, POINT, DOF>;
}
