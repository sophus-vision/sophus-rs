use dfdx::prelude::*;

use crate::calculus::make::*;
use crate::calculus::params::*;
use crate::calculus::points::*;
use crate::calculus::batch_types::*;
use crate::lie::traits::*;
use crate::manifold::traits::*;

#[derive(Debug, Copy, Clone)]
pub struct TranslationGroupProductImpl<
    const B: usize,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    Factor: FactorGroupImplTrait<B, SDOF, SPARAMS, POINT, POINT>,
> {
    phantom: std::marker::PhantomData<(Factor)>,
}

impl<
        const B: usize,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        Factor: FactorGroupImplTrait<B, SDOF, SPARAMS, POINT, POINT>,
    > ParamsImpl<B, PARAMS>
    for TranslationGroupProductImpl<
        B,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        Factor,
    >
{
    fn are_params_valid(params: &V<B, PARAMS>) -> bool {
        let params = params.clone();
        Factor::are_params_valid(&Self::factor_group_params(params))
    }
}

impl<
        const B: usize,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        Factor: FactorGroupImplTrait<B, SDOF, SPARAMS, POINT, POINT>,
    > ParamsTestUtils<B, PARAMS>
    for TranslationGroupProductImpl<
        B,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        Factor,
    >
{
    fn tutil_params_examples() -> Vec<V<B, PARAMS>> {
        let mut examples = vec![];
        for factor_group_params in Factor::tutil_params_examples() {
            let factor_group_params = factor_group_params.clone();
            for translation in Self::tutil_translation_examples() {
                examples.push(Self::params_from(translation, factor_group_params.clone()));
            }
        }
        examples
    }

    fn tutil_invalid_params_examples() -> Vec<V<B, PARAMS>> {
        let dev = dfdx::tensor::Cpu::default();
        let mut examples = vec![];

        for factor_group_params in Factor::tutil_invalid_params_examples() {
            let factor_group_params = factor_group_params.clone();
            examples.push(Self::params_from(dev.zeros(), factor_group_params));
        }
        examples
    }

}



impl<
        const B: usize,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        Factor: FactorGroupImplTrait<B, SDOF, SPARAMS, POINT, POINT>,
    > TangentTestUtil<B, DOF>
    for TranslationGroupProductImpl<
        B,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        Factor,
    >
{
    fn tutil_tangent_examples() -> Vec<V<B, DOF>> {
        let mut examples = vec![];
        for group_tangent in Factor::tutil_tangent_examples() {
            let group_tangent = group_tangent.clone();
            for translation_tangent in Self::tutil_translation_examples() {
                examples.push(Self::tangent_from(
                    translation_tangent,
                    group_tangent.clone(),
                ));
            }
        }
        examples
    }
}

impl<
        const B: usize,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        Factor: FactorGroupImplTrait<B, SDOF, SPARAMS, POINT, POINT>,
    >
   TranslationGroupProductImpl<B, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, Factor>
{
    fn translation<Tape:SophusTape>(
        params: GenV<B, PARAMS, Tape>,
    ) -> GenV<B, POINT, Tape> {
        params.slice((.., 0..POINT)).realize()
    }

    fn factor_group_params<Tape:SophusTape>(
        params: GenV<B, PARAMS, Tape>,
    ) -> GenV<B, SPARAMS, Tape> {
        params.slice((.., POINT..)).realize()
    }

    pub fn params_from<Tape:SophusTape>(
        translation: GenV<B, POINT, Tape>,
        factor_group_params: GenV<B, SPARAMS, Tape>,
    ) -> GenV<B, PARAMS, Tape> {
        let factor_group_params: Tensor<(Const<B>, usize), f64, Cpu, _> =
            factor_group_params.realize();
        (translation, factor_group_params)
            .concat_along(Axis::<1>)
            .realize()
    }

    fn translation_tangent<Tape:SophusTape>(
        tangent: GenV<B, DOF, Tape>,
    ) -> GenV<B, POINT, Tape> {
        tangent.slice((.., 0..POINT)).realize()
    }

    fn factor_group_tangent<Tape:SophusTape>(
        tangent: GenV<B, DOF, Tape>,
    ) -> GenV<B, SDOF, Tape> {
        tangent.slice((.., POINT..)).realize()
    }

    pub fn tangent_from<Tape:SophusTape>(
        translation_tangent: GenV<B, POINT, Tape>,
        factor_group_tangent: GenV<B, SDOF, Tape>,
    ) -> GenV<B, DOF, Tape> {
        let factor_group_params: Tensor<(Const<B>, usize), f64, Cpu, _> =
            factor_group_tangent.realize();
        (translation_tangent, factor_group_params)
            .concat_along(Axis::<1>)
            .realize()
    }

    fn tutil_translation_examples() -> Vec<V<B, POINT>> {
        tutil_point_examples::<B, POINT>()
    }



 }


impl<
        const B: usize,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        Factor: FactorGroupImplTrait<B, SDOF, SPARAMS, POINT, POINT>,
    > LieGroupImplTrait<B, DOF, PARAMS, POINT, AMBIENT>
    for TranslationGroupProductImpl<B, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, Factor>
{

    const IS_AXIS_DIRECTION_PRESERVING: bool = Factor::IS_AXIS_DIRECTION_PRESERVING;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = Factor::IS_DIRECTION_VECTOR_PRESERVING;
    const IS_SHAPE_PRESERVING: bool = Factor::IS_SHAPE_PRESERVING;
    const IS_DISTANCE_PRESERVING: bool = Factor::IS_DISTANCE_PRESERVING;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    const IS_ORIGIN_PRESERVING: bool = false;

    fn identity_params() -> V<B, PARAMS> {
        let dev = dfdx::tensor::Cpu::default();
        Self::params_from(dev.zeros(), Factor::identity_params())
    }

    fn into_group_adjoint<Tape:SophusTape>(
        params: GenV<B, PARAMS, Tape>,
    ) -> GenM<B, DOF, DOF, Tape> {
        let factor_group_params = Self::factor_group_params(params.clone());
        let translation = Self::translation(params.clone());
        let group_adjoint = Factor::into_group_adjoint(factor_group_params.clone());

        let dev = group_adjoint.device();
        let zero: GenM<B, SDOF, POINT, Tape> = dev.zeros().retaped();
        let adj_t =
            Factor::group_adjoint_of_translation(factor_group_params.clone(), translation.clone());
        let mat = Factor::into_matrix(factor_group_params.clone());

        let top_row: GenM<B, POINT, DOF, Tape> = make_2blockcol_mat(mat, adj_t);
        let bottom_row: GenM<B, SDOF, DOF, Tape> = make_2blockcol_mat(zero, group_adjoint);

        make_2rowblock_mat(top_row, bottom_row)
    }

    fn exp<Tape:SophusTape>(
        omega: GenV<B, DOF, Tape>,
    ) -> GenV<B, PARAMS, Tape> {
        let translation: GenM<B, POINT, 1, Tape> =
            Self::translation_tangent(omega.clone()).reshape();
        let factor_group_params = Factor::exp(Self::factor_group_tangent(omega.clone()));

        let mat_v = Factor::mat_v(
            factor_group_params.clone(),
            Self::factor_group_tangent(omega),
        );

        println!("mat_v: {:?}", mat_v.array());

        let vv: Tensor<(Const<B>, Const<POINT>), f64, Cpu, Tape> =
            mat_v.matmul(translation).reshape();

        Self::params_from(vv, factor_group_params)
    }

    fn into_log<Tape:SophusTape>(
        params: GenV<B, PARAMS, Tape>,
    ) -> GenV<B, DOF, Tape> {
        let translation: GenM<B, POINT, 1, Tape> = Self::translation(params.clone()).reshape();
        let factor_group_params = Self::factor_group_params(params);

        let factor_group_tangent = Factor::into_log(factor_group_params.clone());
        let mat_v_inv = Factor::mat_v_inverse(factor_group_params, factor_group_tangent.clone());
        println!("mat_v_inv: {:?}", mat_v_inv.array());
        println!("translation: {:?}", translation.array());
        let translation_tangent: GenV<B, POINT, _> = mat_v_inv.matmul(translation).reshape();

        println!("translation_tangent: {:?}", translation_tangent.array());

        Self::tangent_from(translation_tangent, factor_group_tangent)
    }

    fn hat<Tape:SophusTape>(
        omega: GenV<B, DOF, Tape>,
    ) -> GenM<B, AMBIENT, AMBIENT, Tape> {
        let zero: M<B, 1, AMBIENT> = omega.device().zeros();

        let f_tangent: GenM<B, POINT, POINT, _> = Factor::hat(Self::factor_group_tangent(omega.clone()));
        let p_tangent: GenM<B, POINT, 1, _> = Self::translation_tangent(omega).reshape();

        make_2rowblock_mat(make_2blockcol_mat(f_tangent, p_tangent), zero)
    }

    fn vee<Tape:SophusTape>(
        hat: GenM<B, AMBIENT, AMBIENT, Tape>,
    ) -> GenV<B, DOF, Tape> {
        let point_hat_mat: GenM<B, POINT, POINT, Tape> =
            hat.clone().slice((.., 0..POINT, 0..POINT)).realize();

        let factor_group_tangent = Factor::vee(point_hat_mat);
        let translation_tangent: GenM<B, POINT, 1, Tape> =
            hat.slice((.., 0..POINT, POINT..POINT + 1)).realize();

        Self::tangent_from(translation_tangent.reshape(), factor_group_tangent)
    }

    fn mul_assign<
        LeftTape:SophusTape + Merge<RightTape>,
        RightTape:SophusTape,
    >(
        params1: GenV<B, PARAMS, LeftTape>,
        params2: GenV<B, PARAMS, RightTape>,
    ) -> GenV<B, PARAMS, LeftTape> {
        let factor_group_params1 = Self::factor_group_params(params1.clone());
        let factor_group_params2 = Self::factor_group_params(params2.clone());
        let translation1 = Self::translation(params1);
        let translation2 = Self::translation(params2);
        let factor_group_params = Factor::mul_assign(factor_group_params1.clone(), factor_group_params2);
        let translation: GenV<B, POINT, _> =
            Factor::point_action::<1, _, _>(factor_group_params1, translation2.reshape()).reshape();

        Self::params_from(translation + translation1, factor_group_params)
    }

    fn into_inverse<Tape:SophusTape>(
        params: GenV<B, PARAMS, Tape>,
    ) -> GenV<B, PARAMS, Tape> {
        let factor_group_params = Self::factor_group_params(params.clone());
        let translation = Self::translation(params);
        let factor_group_params = Factor::into_inverse(factor_group_params);

        let neg_t =
            Factor::point_action::<1, Tape, Tape>(factor_group_params.clone(), translation.reshape())
                .negate();

        Self::params_from(neg_t.reshape(), factor_group_params)
    }

    fn point_action<
        const NUM_POINTS: usize,
        Tape:SophusTape + Merge<PointTape>,
        PointTape:SophusTape,
    >(
        params: GenV<B, PARAMS, Tape>,
        point: GenM<B, POINT, NUM_POINTS, PointTape>,
    ) -> GenM<B, POINT, NUM_POINTS, Tape> {
        let factor_group_params = Self::factor_group_params(params.clone());
        let translation = Self::translation(params);
        Factor::point_action(factor_group_params, point) + translation.broadcast()
    }

    fn into_ambient<
        const NUM_POINTS: usize,
        Tape:SophusTape,
    >(
        params: GenM<B, POINT, NUM_POINTS, Tape>,
    ) -> GenM<B, AMBIENT, NUM_POINTS, Tape> {
        let one: M<B, 1, NUM_POINTS> = params.device().ones();
        make_2rowblock_mat(params, one)
    }

    fn into_compact<Tape:SophusTape>(
        params: GenV<B, PARAMS, Tape>,
    ) -> GenM<B, POINT, AMBIENT, Tape> {
        let zero: M<B, SPARAMS, AMBIENT> = params.device().zeros();

        let factor_mat: GenM<B, POINT, POINT, _> =
            Factor::into_matrix(Self::factor_group_params(params.clone()));
        let t: GenM<B, POINT, 1, _> = Self::translation(params).reshape();

        make_2blockcol_mat(factor_mat, t)
    }

    fn into_matrix<Tape:SophusTape>(
        params: GenV<B, PARAMS, Tape>,
    ) -> GenM<B, AMBIENT, AMBIENT, Tape> {
        let zero: M<B, 1, POINT> = params.device().zeros();
        let one: M<B, 1, 1> = params.device().ones();

        make_2rowblock_mat(Self::into_compact(params), make_2blockcol_mat(zero, one))
    }


    fn algebra_adjoint(tangent: V<B, DOF>) -> M<B, DOF, DOF> {
        let zero: M<B, SDOF, POINT> = tangent.device().zeros();
        let f_tangent = Self::factor_group_tangent(tangent.clone());

        make_2rowblock_mat(
            make_2blockcol_mat(
                Factor::hat(f_tangent.clone()),
                Factor::algebra_adjoint_of_translation(Self::translation_tangent(tangent)),
            ),
            make_2blockcol_mat(zero.retaped(), Factor::algebra_adjoint(f_tangent)),
        )


        // static auto ad(Tangent const& tangent) -> Eigen::Matrix<Scalar, kDof, kDof> {
        //     Eigen::Matrix<Scalar, kDof, kDof> ad;
        //     ad.template topLeftCorner<kPointDim, kPointDim>() =
        //         FactorGroup::hat(factorTangent(tangent));
        //     ad.template topRightCorner<kPointDim, FactorGroup::kDof>() =
        //         FactorGroup::adOfTranslation(translationTangent(tangent));
        
        //     ad.template bottomLeftCorner<FactorGroup::kDof, kPointDim>().setZero();
        
        //     ad.template bottomRightCorner<FactorGroup::kDof, FactorGroup::kDof>() =
        //         FactorGroup::ad(factorTangent(tangent));
        
        //     return ad;
        //   }
        
    }

    fn dx_exp_x_at_0() -> M<B, PARAMS, DOF> {
        todo!()
        //     fn dx_exp_x_at_0() -> M<T, PARAMS, DOF> {
        //         let mut j = M::<T, PARAMS, DOF>::zeros();
        //         j.view_mut((0, 0), (POINT, POINT)).fill_with_identity();
        //         j.view_mut((POINT, POINT), (SPARAMS, SDOF))
        //             .copy_from(&Factor::dx_exp_x_at_0());
        //         j
        //     }
    }

    fn dx_exp_x_times_point_at_0(point: V<B, POINT>) -> M<B, POINT, DOF> {
        todo!()
        //     fn dx_exp_x_times_point_at_0(point: &V<T, POINT>) -> M<T, POINT, DOF> {
        //         let mut j = M::<T, POINT, DOF>::zeros();
        //         j.view_mut((0, 0), (POINT, POINT)).fill_with_identity();
        //         j.view_mut((0, POINT), (POINT, SDOF))
        //             .copy_from(&Factor::dx_exp_x_times_point_at_0(point));
        //         j
        //     }
    }

    fn dx_self_times_exp_x_at_0(params: &V<B, PARAMS>) -> M<B, PARAMS, DOF> {
        todo!()
        //     fn dx_self_times_exp_x_at_0(params: &V<T, PARAMS>) -> M<T, PARAMS, DOF> {
        //         let mut j = M::<T, PARAMS, DOF>::zeros();
        //         j.view_mut((0, 0), (POINT, POINT))
        //             .copy_from(&Factor::matrix(&Self::factor_group_params(params)));
        //         j.view_mut((POINT, POINT), (SPARAMS, SDOF))
        //             .copy_from(&Factor::dx_self_times_exp_x_at_0(&Self::factor_group_params(
        //                 params,
        //             )));
        //         j
        //     }
    }

    fn dx_log_exp_x_times_self_at_0(params: &V<B, PARAMS>) -> M<B, DOF, DOF> {
        todo!()

        //     fn dx_log_exp_x_times_self_at_0(params: &V<T, PARAMS>) -> M<T, DOF, DOF> {
        //         let mut j = M::<T, DOF, DOF>::zeros();
        //         j.view_mut((0, 0), (POINT, POINT))
        //             .copy_from(&Factor::matrix(&Factor::inverse(&Self::factor_group_params(params))));

        //         println!("params: {}", params);

        //         println!("matrix: {}", Factor::matrix(&Self::factor_group_params(params)));
        //         println!(
        //             "dx_self_times_exp_x_at_0: {:?}",
        //             Factor::dx_self_times_exp_x_at_0(&Self::factor_group_params(params))
        //         );

        //         let v = Factor::matrix(&(Factor::inverse(&(Self::factor_group_params(params)))))
        //             * &Self::translation(params);
        //         j.view_mut((0, POINT), (POINT, SDOF))
        //             .copy_from(&Factor::dx_exp_x_times_point_at_0(&v));

        //         j.view_mut((POINT, POINT), (SDOF, SDOF))
        //             .copy_from(&Factor::dx_log_exp_x_times_self_at_0(
        //                 &Self::factor_group_params(params),
        //             ));
        //         j
        //     }
        // }
    }
}
