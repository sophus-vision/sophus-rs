use core::borrow::Borrow;

use sophus_autodiff::{
    manifold::IsTangent,
    params::IsParamsImpl,
    points::example_points,
};

use crate::{
    HasDisambiguate,
    IsLieFactorGroupImpl,
    IsLieGroupImpl,
    IsRealLieFactorGroupImpl,
    IsRealLieGroupImpl,
    lie_group::LieGroup,
    prelude::*,
};

extern crate alloc;

/// Template of an affine group.
///
/// It is a semi-direct product  `G ⋊ ℝᴺ` of a group `G ⊂ GL(M)` and a commutative translation group
/// (Euclidean vector space).
///
/// ## Overview
/// * **Tangent space:** K+N DoF – **[ α , ν ]**, with `α` being a tangent vector in `𝖌` and `ν` the
///   N-d **linear** rate.
/// * **Internal parameters:** M + N – **[ a , p ]**, with `a` being the `M`-dimensional parameter
///   vector of ``A`` and translation `p ∈ ℝ³`.
/// * **Action space:** N (G ⋊ ℝᴺ acts on 3-d points)
/// * **Matrix size:** N+1 (represented as 4 × 4 matrices)
///
/// ### Group structure
///
/// `G ⋊ ℝᴺ` has the following *matrix representation*
///
/// ```ascii
/// ---------
/// | A | p |
/// ---------
/// | 0 | 1 |
/// ---------
/// ```
///
/// We call ``A ∈ G`` an element of the group ``G`` and ``p ∈ ℝᴺ`` the translation vector.
/// Let `𝖌` be the `K`-dimensional Lie algebra of the group `G`. To emphasis that `G` is part of
/// the semi-direct product `G ⋊ ℝᴺ`, we call it the *factor group*. In order to use a Lie group
/// as a factor group, it must implement the trait [IsLieFactorGroupImpl].
///
/// It acts on points in `ℝᴺ` by the following affine transformation:
///
/// ```ascii
/// (A, p) ⊗ x = A·x + p
/// ```
///
/// *Group operation*
/// ```ascii
/// (Aₗ, pₗ) ⊗ (Aᵣ, pᵣ) = ( Aₗ·Aᵣ,  Aₗ·pᵣ + pₗ )
/// ```
/// *Inverse*
/// ```ascii
/// (A, p)⁻¹ = ( R⁻¹,  -A⁻¹·p )
/// ```
/// ### Lie-group properties
///
/// **Hat operator**
/// ```ascii
///           ----------
///  /α\^     | α^ | ν |
///  ---   =  ----------
///  \ν/      | O  | 0 |
///           ----------
/// ```
/// where `α^` is the hat operator of the factor group `G`.
///
/// **Exponential map** `exp : ℝᴷ⁺ᴺ → G ⋊ ℝ`
/// ```ascii
/// exp(α,ν) = ( exp_𝖌(α),  V(α) · ν )
/// ```
/// where  `exp_𝖌` is the exponential map of `G` and `V(α)` is [IsLieFactorGroupImpl::mat_v].
///
/// **Group adjoint** `Adj : G ⋊ ℝ → ℝ⁽ᴷ⁺ᴺ⁾ˣ⁽ᴷ⁺ᴺ⁾`
/// ```ascii
///              -----------------
///              | Adj_𝖌(A)  | O |
/// Adj(A,p)  =  |---------------|
///              | Adjt(p)·A | A |
///              -----------------
/// ```
/// where `Adj_𝖌(A)` is the adjoint representation of the factor group `G` and `Adjt(p)` is
/// [IsLieFactorGroupImpl::adj_of_translation]. `Adj(A,p)` acts on `(α; ν)`
///
/// ```ascii
/// Adj(A,p) · (α; ν)  =  ( Adj_𝖌(A)·α ;  Adjt(p)·A ν + A ν )
/// ```
///
///
/// **Lie-algebra adjoint** `ad : 𝖌 → ℝ⁽ᴷ⁺ᴺ⁾ˣ⁽ᴷ⁺ᴺ⁾`
/// ```ascii
///              |-------------|
///              | ad(α)  | O  |
/// ad(α; ν)  =  |-------------|
///              | adt(ν) | α^ |
///              |-------------|
/// ```
///
/// `ad(α)` is the adjoint representation of the factor group `G` and `adt(ν)` is
/// [IsLieFactorGroupImpl::ad_of_translation]. `ad(α; ν)` acts on `(φ; τ)`
///
/// ```ascii
/// ad(α; ν) · (φ; τ)  =  ( ad(α) φ ;  adt(ν) φ + α × τ )
/// ```
#[derive(Debug, Copy, Clone, Default)]
pub struct AffineGroupTemplateImpl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
> {
    phantom: core::marker::PhantomData<(S, F)>,
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
> AffineGroupTemplateImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, BATCH, DM, DN, F>
{
    /// factor part of the group parameters
    pub fn factor_params(params: &S::Vector<PARAMS>) -> S::Vector<SPARAMS> {
        params.get_fixed_subvec::<SPARAMS>(0)
    }

    /// translation part of the group parameters
    pub fn translation(params: &S::Vector<PARAMS>) -> S::Vector<POINT> {
        params.get_fixed_subvec::<POINT>(SPARAMS)
    }

    /// create group parameters from factor and translation parameters
    pub fn params_from(
        factor_params: &S::Vector<SPARAMS>,
        translation: &S::Vector<POINT>,
    ) -> S::Vector<PARAMS> {
        S::Vector::block_vec2(*factor_params, *translation)
    }

    /// factor part of the tangent vector
    fn factor_tangent(tangent: &S::Vector<DOF>) -> S::Vector<SDOF> {
        tangent.get_fixed_subvec::<SDOF>(0)
    }

    /// translation part of the tangent vector
    fn translation_tangent(tangent: &S::Vector<DOF>) -> S::Vector<POINT> {
        tangent.get_fixed_subvec::<POINT>(SDOF)
    }

    /// create tangent vector from factor and translation tangent
    fn tangent_from(
        factor_tangent: &S::Vector<SDOF>,
        translation: &S::Vector<POINT>,
    ) -> S::Vector<DOF> {
        S::Vector::block_vec2(*factor_tangent, *translation)
    }

    fn translation_examples() -> alloc::vec::Vec<S::Vector<POINT>> {
        example_points::<S, POINT, BATCH, DM, DN>()
    }

    /// create group parameters from factor parameters and translation
    pub fn params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>> {
        let mut examples = alloc::vec![];

        let factor_examples = F::params_examples();
        let translation_examples = Self::translation_examples();

        let max_len = core::cmp::max(factor_examples.len(), translation_examples.len());

        for i in 0..max_len {
            let factor_params = &factor_examples[i % factor_examples.len()];
            let translation = &translation_examples[i % translation_examples.len()];
            examples.push(Self::params_from(factor_params, translation));
        }
        examples
    }

    /// create tangent vector from factor tangent and translation tangent
    pub fn tangent_examples() -> alloc::vec::Vec<S::Vector<DOF>> {
        let mut examples = alloc::vec![];

        let factor_examples = F::tangent_examples();
        let translation_examples = Self::translation_examples();

        let max_len = core::cmp::max(factor_examples.len(), translation_examples.len());

        for i in 0..max_len {
            let factor_tangent = &factor_examples[i % factor_examples.len()];
            let translation = &translation_examples[i % translation_examples.len()];
            examples.push(Self::tangent_from(factor_tangent, translation));
        }
        examples
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
> HasDisambiguate<S, PARAMS, BATCH, DM, DN>
    for AffineGroupTemplateImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, BATCH, DM, DN, F>
{
    fn disambiguate(params: S::Vector<PARAMS>) -> S::Vector<PARAMS> {
        Self::params_from(
            &F::disambiguate(Self::factor_params(&params)),
            &Self::translation(&params),
        )
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
> IsParamsImpl<S, PARAMS, BATCH, DM, DN>
    for AffineGroupTemplateImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, BATCH, DM, DN, F>
{
    fn are_params_valid(params: S::Vector<PARAMS>) -> S::Mask {
        F::are_params_valid(Self::factor_params(params.borrow()))
    }

    fn params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>> {
        Self::params_examples()
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>> {
        alloc::vec![Self::params_from(
            &F::invalid_params_examples()[0],
            &S::Vector::zeros(),
        )]
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
> IsTangent<S, DOF, BATCH, DM, DN>
    for AffineGroupTemplateImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, BATCH, DM, DN, F>
{
    fn tangent_examples() -> alloc::vec::Vec<S::Vector<DOF>> {
        Self::tangent_examples()
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    Factor: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
> IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN>
    for AffineGroupTemplateImpl<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        DM,
        DN,
        Factor,
    >
{
    const IS_ORIGIN_PRESERVING: bool = false;
    const IS_AXIS_DIRECTION_PRESERVING: bool = Factor::IS_AXIS_DIRECTION_PRESERVING;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = Factor::IS_DIRECTION_VECTOR_PRESERVING;
    const IS_SHAPE_PRESERVING: bool = Factor::IS_SHAPE_PRESERVING;
    const IS_DISTANCE_PRESERVING: bool = Factor::IS_DISTANCE_PRESERVING;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<PARAMS> {
        Self::params_from(&Factor::identity_params(), &S::Vector::zeros())
    }

    fn adj(params: &S::Vector<PARAMS>) -> S::Matrix<DOF, DOF> {
        let factor_params = Self::factor_params(params);
        let translation = Self::translation(params);

        S::Matrix::block_mat2x2::<SDOF, POINT, SDOF, POINT>(
            (Factor::adj(&factor_params), S::Matrix::zeros()),
            (
                Factor::adj_of_translation(&factor_params, &translation),
                Factor::matrix(&factor_params),
            ),
        )
    }

    fn exp(omega: &S::Vector<DOF>) -> S::Vector<PARAMS> {
        let factor_tangent = Self::factor_tangent(omega);
        let translation = Self::translation_tangent(omega);
        let factor_params = Factor::exp(&factor_tangent);
        let mat_v = Factor::mat_v(&factor_tangent);
        Self::params_from(&factor_params, &(mat_v * translation))
    }

    fn log(params: &S::Vector<PARAMS>) -> S::Vector<DOF> {
        let factor_params = Self::factor_params(params);
        let translation = Self::translation(params);
        let factor_tangent = Factor::log(&factor_params);
        let mat_v_inv = Factor::mat_v_inverse(&factor_tangent);
        let translation_tangent = mat_v_inv * translation;
        Self::tangent_from(&factor_tangent, &translation_tangent)
    }

    fn hat(omega: &S::Vector<DOF>) -> S::Matrix<AMBIENT, AMBIENT> {
        S::Matrix::block_mat2x2::<POINT, 1, POINT, 1>(
            (
                Factor::hat(&Self::factor_tangent(omega)),
                Self::translation_tangent(omega).to_mat(),
            ),
            (S::Matrix::zeros(), S::Matrix::zeros()),
        )
    }

    fn vee(hat: &S::Matrix<AMBIENT, AMBIENT>) -> S::Vector<DOF> {
        let factor_tangent = Factor::vee(&hat.get_fixed_submat::<POINT, POINT>(0, 0));
        let translation_tangent = hat.get_fixed_submat::<POINT, 1>(0, POINT);
        Self::tangent_from(&factor_tangent, &translation_tangent.get_col_vec(0))
    }

    fn group_mul(params1: &S::Vector<PARAMS>, params2: &S::Vector<PARAMS>) -> S::Vector<PARAMS> {
        let factor_params1 = Self::factor_params(params1);
        let factor_params2 = Self::factor_params(params2);
        let translation1 = Self::translation(params1);
        let translation2 = Self::translation(params2);
        let factor_params = Factor::group_mul(&factor_params1, &factor_params2);
        let f = Factor::transform(&factor_params1, &translation2);
        let translation = f + translation1;
        Self::params_from(&factor_params, &translation)
    }

    fn inverse(params: &S::Vector<PARAMS>) -> S::Vector<PARAMS> {
        let factor_params = Self::factor_params(params);
        let translation = Self::translation(params);
        let factor_params = Factor::inverse(&factor_params);
        let translation = -Factor::transform(&factor_params, &translation);
        Self::params_from(&factor_params, &translation)
    }

    fn transform(params: &S::Vector<PARAMS>, point: &S::Vector<POINT>) -> S::Vector<POINT> {
        let factor_params = Self::factor_params(params);
        let translation = Self::translation(params);
        Factor::transform(&factor_params, point) + translation
    }

    fn to_ambient(params: &S::Vector<POINT>) -> S::Vector<AMBIENT> {
        S::Vector::block_vec2(*params, S::Vector::<1>::zeros())
    }

    fn compact(params: &S::Vector<PARAMS>) -> S::Matrix<POINT, AMBIENT> {
        S::Matrix::block_mat1x2::<POINT, 1>(
            Factor::matrix(&Self::factor_params(params)),
            Self::translation(params).to_mat(),
        )
    }

    fn matrix(params: &S::Vector<PARAMS>) -> S::Matrix<AMBIENT, AMBIENT> {
        S::Matrix::block_mat2x2::<POINT, 1, POINT, 1>(
            (
                Factor::matrix(&Self::factor_params(params)),
                Self::translation(params).to_mat(),
            ),
            (
                S::Matrix::<1, POINT>::zeros(),
                S::Matrix::<1, 1>::identity(),
            ),
        )
    }

    fn ad(tangent: &S::Vector<DOF>) -> S::Matrix<DOF, DOF> {
        let o = S::Matrix::<SDOF, POINT>::zeros();
        S::Matrix::block_mat2x2::<SDOF, POINT, SDOF, POINT>(
            (Factor::ad(&Self::factor_tangent(tangent)), o),
            (
                Factor::ad_of_translation(&Self::translation_tangent(tangent)),
                Factor::hat(&Self::factor_tangent(tangent)),
            ),
        )
    }

    type GenG<S2: IsScalar<BATCH, DM, DN>> = AffineGroupTemplateImpl<
        S2,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        DM,
        DN,
        Factor::GenFactorG<S2, DM, DN>,
    >;

    type RealG = AffineGroupTemplateImpl<
        S::RealScalar,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        0,
        0,
        Factor::GenFactorG<S::RealScalar, 0, 0>,
    >;

    type DualG<const M: usize, const N: usize> = AffineGroupTemplateImpl<
        S::DualScalar<M, N>,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        M,
        N,
        Factor::GenFactorG<S::DualScalar<M, N>, M, N>,
    >;
}

impl<
    S: IsRealScalar<BATCH>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    const BATCH: usize,
    Factor: IsRealLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH>,
> IsRealLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH>
    for AffineGroupTemplateImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, BATCH, 0, 0, Factor>
{
    fn dx_exp_x_at_0() -> S::Matrix<PARAMS, DOF> {
        S::Matrix::block_mat2x2::<SPARAMS, POINT, SDOF, POINT>(
            (
                Factor::dx_exp_x_at_0(),
                S::Matrix::<SPARAMS, POINT>::zeros(),
            ),
            (
                S::Matrix::<POINT, SDOF>::zeros(),
                S::Matrix::<POINT, POINT>::identity(),
            ),
        )
    }

    fn dx_exp_x_times_point_at_0(point: &S::Vector<POINT>) -> S::Matrix<POINT, DOF> {
        S::Matrix::block_mat1x2::<SDOF, POINT>(
            Factor::dx_exp_x_times_point_at_0(point),
            S::Matrix::<POINT, POINT>::identity(),
        )
    }

    fn dx_exp(tangent: &S::Vector<DOF>) -> S::Matrix<PARAMS, DOF> {
        let factor_tangent = &Self::factor_tangent(tangent);
        let trans_tangent = &Self::translation_tangent(tangent);

        let dx_mat_v = Factor::dx_mat_v(factor_tangent);
        let mut dx_mat_v_tangent = S::Matrix::<POINT, SDOF>::zeros();

        for i in 0..SDOF {
            dx_mat_v_tangent.set_col_vec(i, dx_mat_v[i] * *trans_tangent);
        }

        S::Matrix::block_mat2x2::<SPARAMS, POINT, SDOF, POINT>(
            (
                Factor::dx_exp(factor_tangent),
                S::Matrix::<SPARAMS, POINT>::zeros(),
            ),
            (dx_mat_v_tangent, Factor::mat_v(factor_tangent)),
        )
    }

    fn da_a_mul_b(a: &S::Vector<PARAMS>, b: &S::Vector<PARAMS>) -> S::Matrix<PARAMS, PARAMS> {
        let a_factor_params = &Self::factor_params(a);
        let b_factor_params = &Self::factor_params(b);

        let b_trans = &Self::translation(b);

        S::Matrix::block_mat2x2::<SPARAMS, POINT, SPARAMS, POINT>(
            (
                Factor::da_a_mul_b(a_factor_params, b_factor_params),
                S::Matrix::<SPARAMS, POINT>::zeros(),
            ),
            (
                Factor::dparams_matrix_times_point(a_factor_params, b_trans),
                S::Matrix::<POINT, POINT>::identity(),
            ),
        )
    }

    fn db_a_mul_b(a: &S::Vector<PARAMS>, b: &S::Vector<PARAMS>) -> S::Matrix<PARAMS, PARAMS> {
        let a_factor_params = &Self::factor_params(a);
        let b_factor_params = &Self::factor_params(b);

        S::Matrix::block_mat2x2::<SPARAMS, POINT, SPARAMS, POINT>(
            (
                Factor::db_a_mul_b(a_factor_params, b_factor_params),
                S::Matrix::<SPARAMS, POINT>::zeros(),
            ),
            (
                S::Matrix::<POINT, SPARAMS>::zeros(),
                Factor::matrix(a_factor_params),
            ),
        )
    }

    fn has_shortest_path_ambiguity(params: &<S>::Vector<PARAMS>) -> <S>::Mask {
        Factor::has_shortest_path_ambiguity(&Self::factor_params(params))
    }

    fn dparams_matrix(params: &<S>::Vector<PARAMS>, col_idx: usize) -> <S>::Matrix<POINT, PARAMS> {
        let factor_params = &Self::factor_params(params);

        if col_idx < POINT {
            S::Matrix::block_mat1x2::<SPARAMS, POINT>(
                Factor::dparams_matrix(factor_params, col_idx),
                S::Matrix::zeros(),
            )
        } else {
            S::Matrix::block_mat1x2::<SPARAMS, POINT>(S::Matrix::zeros(), S::Matrix::identity())
        }
    }

    fn left_jacobian(tangent: <S>::Vector<DOF>) -> <S>::Matrix<DOF, DOF> {
        // split ξ into factor and translation parts
        let phi = Self::factor_tangent(&tangent); // SDOF
        let rho = Self::translation_tangent(&tangent); // POINT

        // factor-group blocks
        let jl_phi = Factor::left_jacobian(phi); // SDOF × SDOF
        // translation block (V-matrix)
        let v_phi = Factor::mat_v(&phi); // POINT × POINT
        // coupling block
        let q_l = Factor::q_left_block(rho, phi); // POINT × SDOF

        // zero block (SDOF × POINT)
        let z_top = S::Matrix::<SDOF, POINT>::zeros();


        // assemble 2×2 block matrix
        S::Matrix::block_mat2x2::<SDOF, POINT, SDOF, POINT>((jl_phi.clone(), z_top), (q_l, v_phi))
    }

    fn inv_left_jacobian(tangent: <S>::Vector<DOF>) -> <S>::Matrix<DOF, DOF> {
        let phi = Self::factor_tangent(&tangent);
        let rho = Self::translation_tangent(&tangent);

        // factor-group inverse Jacobian
        let jl_inv = Factor::inv_left_jacobian(phi); // SDOF × SDOF
        // inverse of V-matrix
        let v_inv = Factor::mat_v_inverse(&phi); // POINT × POINT
        // coupling
        let q_l = Factor::q_left_block(rho, phi); // POINT × SDOF

        // zero block (SDOF × POINT)
        let z_top = S::Matrix::<SDOF, POINT>::zeros();

        // bottom-left = − V⁻¹ · Q_L · J_l⁻¹
        let bl = -(v_inv.clone().mat_mul(q_l).mat_mul(jl_inv.clone()));
        S::Matrix::block_mat2x2::<SDOF, POINT, SDOF, POINT>((jl_inv.clone(), z_top), (bl, v_inv))
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    FactorImpl: crate::IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
>
    IsTranslationProductGroup<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        DM,
        DN,
        LieGroup<S, SDOF, SPARAMS, POINT, POINT, BATCH, DM, DN, FactorImpl>,
    >
    for crate::lie_group::LieGroup<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        BATCH,
        DM,
        DN,
        AffineGroupTemplateImpl<
            S,
            DOF,
            PARAMS,
            POINT,
            AMBIENT,
            SDOF,
            SPARAMS,
            BATCH,
            DM,
            DN,
            FactorImpl,
        >,
    >
{
    type Impl = AffineGroupTemplateImpl<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        DM,
        DN,
        FactorImpl,
    >;

    fn from_factor_and_translation(
        factor: LieGroup<S, SDOF, SPARAMS, POINT, POINT, BATCH, DM, DN, FactorImpl>,
        translation: S::Vector<POINT>,
    ) -> Self {
        let params = Self::Impl::params_from(factor.borrow().params(), &translation);
        Self::from_params(params)
    }

    fn set_translation(&mut self, translation: S::Vector<POINT>) {
        self.set_params(Self::G::params_from(self.factor().params(), &translation))
    }

    fn translation(&self) -> <S as IsScalar<BATCH, DM, DN>>::Vector<POINT> {
        Self::Impl::translation(self.params())
    }

    fn set_factor(
        &mut self,
        factor: LieGroup<S, SDOF, SPARAMS, POINT, POINT, BATCH, DM, DN, FactorImpl>,
    ) {
        self.set_params(Self::G::params_from(
            factor.borrow().params(),
            &self.translation(),
        ))
    }

    fn factor(&self) -> LieGroup<S, SDOF, SPARAMS, POINT, POINT, BATCH, DM, DN, FactorImpl> {
        LieGroup::from_params(Self::Impl::factor_params(self.params()))
    }
}
