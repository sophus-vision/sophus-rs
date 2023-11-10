use std::collections::HashMap;
use std::time::Instant;

use dfdx_core::prelude::*;

use crate::calculus::batch_types::*;
use crate::calculus::matrix_valued_maps::*;
use crate::calculus::params::*;
use crate::calculus::points::*;
use crate::calculus::vector_valued_maps::*;
use crate::lie::traits::*;
use crate::*;

// A Lie group is a differentiable manifold with a group structure.
//
// The group might be "taped" or "untaped", hence the generic Tape is either OwnedTape or NoneTape
// respectively. A taped group is a group where the operations are stored in a tape in order to
// compute the derivatives of the operations.
//
#[derive(Debug, Clone)]
pub struct GenTapedLieGroup<
    const B: usize,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    Tape: SophusTape,
    G: LieGroupImplTrait<B, DOF, PARAMS, POINT, AMBIENT>,
> {
    pub(crate) params: GenV<B, PARAMS, Tape>,
    phantom: std::marker::PhantomData<G>,
}

// A taped Lie group with Tape = OwnedTape.
pub type TapedLieGroup<
    const B: usize,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    G,
> = GenTapedLieGroup<B, DOF, PARAMS, POINT, AMBIENT, OwnedTape<f64, Cpu>, G>;

impl<
        const B: usize,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: LieGroupImplTrait<B, DOF, PARAMS, POINT, AMBIENT>,
    > std::fmt::Display for LieGroup<B, DOF, PARAMS, POINT, AMBIENT, G>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.clone().into_compact().array())
    }
}

// An untaped Lie group with Tape = NoneTape.
pub type LieGroup<
    const B: usize,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    G,
> = GenTapedLieGroup<B, DOF, PARAMS, POINT, AMBIENT, NoneTape, G>;

impl<
        const B: usize,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        MaybeTape: SophusTape,
        G: LieGroupImplTrait<B, DOF, PARAMS, POINT, AMBIENT>,
    > GenTapedLieGroup<B, DOF, PARAMS, POINT, AMBIENT, MaybeTape, G>
{
    pub fn from_params<T: SophusTape>(
        params: GenV<B, PARAMS, T>,
    ) -> GenTapedLieGroup<B, DOF, PARAMS, POINT, AMBIENT, T, G> {
        let (params, tape) = params.split_tape();
        assert!(
            G::are_params_valid(&params.clone()),
            "Invalid parameters for:\n{:?}",
            params.array(),
        );
        GenTapedLieGroup::<B, DOF, PARAMS, POINT, AMBIENT, T, G> {
            params: params.put_tape(tape),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn into_params(self) -> GenV<B, PARAMS, MaybeTape> {
        self.params
    }

    pub fn into_group_adjoint(self) -> GenM<B, DOF, DOF, MaybeTape> {
        G::into_group_adjoint(self.params)
    }

    pub fn exp<TangentTape: SophusTape>(
        omega: GenV<B, DOF, TangentTape>,
    ) -> GenTapedLieGroup<B, DOF, PARAMS, POINT, AMBIENT, TangentTape, G> {
        let p = G::exp(omega);
        Self::from_params(p)
    }

    pub fn into_log(self) -> GenV<B, DOF, MaybeTape> {
        G::into_log(self.params)
    }

    pub fn hat<TangentTape: SophusTape>(
        omega: GenV<B, DOF, TangentTape>,
    ) -> GenM<B, AMBIENT, AMBIENT, TangentTape> {
        G::hat(omega)
    }

    pub fn vee<MatTape: SophusTape>(
        xi: GenM<B, AMBIENT, AMBIENT, MatTape>,
    ) -> GenV<B, DOF, MatTape> {
        G::vee(xi)
    }

    pub fn identity() -> GenTapedLieGroup<B, DOF, PARAMS, POINT, AMBIENT, NoneTape, G> {
        Self::from_params(G::identity_params())
    }

    pub fn mul<LeftTape: SophusTape + Merge<RightTape>, RightTape: SophusTape>(
        g1: GenTapedLieGroup<B, DOF, PARAMS, POINT, AMBIENT, LeftTape, G>,
        g2: GenTapedLieGroup<B, DOF, PARAMS, POINT, AMBIENT, RightTape, G>,
    ) -> GenTapedLieGroup<B, DOF, PARAMS, POINT, AMBIENT, LeftTape, G> {
        Self::from_params(G::mul_assign(g1.params, g2.into_params()))
    }

    pub fn into_inverse(self) -> Self {
        Self::from_params(G::into_inverse(self.params))
    }

    pub fn into_point_action<
        const NUM_POINTS: usize,
        Tape: SophusTape + Merge<PointTape>,
        PointTape: SophusTape,
    >(
        g: GenTapedLieGroup<B, DOF, PARAMS, POINT, AMBIENT, Tape, G>,
        point: GenM<B, POINT, NUM_POINTS, PointTape>,
    ) -> GenM<B, POINT, NUM_POINTS, Tape> {
        G::point_action(g.params, point)
    }

    pub fn into_compact(self) -> GenM<B, POINT, AMBIENT, MaybeTape> {
        G::into_compact(self.params)
    }

    pub fn into_matrix(self) -> GenM<B, AMBIENT, AMBIENT, MaybeTape> {
        let m = G::into_matrix(self.params);
        m
    }

    pub fn into_ambient<const NUM_POINTS: usize, PointTape: SophusTape>(
        point: GenM<B, POINT, NUM_POINTS, PointTape>,
    ) -> GenM<B, AMBIENT, NUM_POINTS, PointTape> {
        G::into_ambient(point)
    }
}

impl<
        const B: usize,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: LieGroupImplTrait<B, DOF, PARAMS, POINT, AMBIENT>,
    > ParamsTestUtils<B, PARAMS> for LieGroup<B, DOF, PARAMS, POINT, AMBIENT, G>
{
    fn tutil_params_examples() -> Vec<V<B, PARAMS>> {
        G::tutil_params_examples()
    }

    fn tutil_invalid_params_examples() -> Vec<V<B, PARAMS>> {
        G::tutil_invalid_params_examples()
    }
}

impl<
        const B: usize,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        G: LieGroupImplTrait<B, DOF, PARAMS, POINT, AMBIENT>,
    > LieGroup<B, DOF, PARAMS, POINT, AMBIENT, G>
{
    pub fn tutil_element_examples() -> Vec<LieGroup<B, DOF, PARAMS, POINT, AMBIENT, G>> {
        let mut elements = vec![];
        for params in Self::tutil_params_examples() {
            let p: Tensor<(Const<B>, Const<PARAMS>), f64, Cpu> = params;
            elements.push(Self::from_params(p));
        }
        elements
    }

    fn test_preservability() {
        let dev = dfdx_core::tensor::Cpu::default();
        if G::IS_ORIGIN_PRESERVING {
            for g in Self::tutil_element_examples() {
                let o: M<B, POINT, 4> = dev.zeros();
                let o2: M<B, POINT, 4> = dev.zeros();

                let r = g.point_action(o);
                assert_tensors_relative_eq_rank3!(r, o2, 0.0001);
            }
        } else {
            let mut num_preserves = 0;
            let mut num = 0;
            for g in &Self::tutil_element_examples() {
                let o: M<B, POINT, 4> = dev.zeros();
                let r = g.point_action(o);
                for b in 0..B {
                    for point in 0..POINT {
                        if Into::<f64>::into(r[[b, point, 0]]) < 0.0001 {
                            num_preserves += 1;
                        }
                        num += 1;
                    }
                }
            }
            let percentage = num_preserves as f64 / num as f64;
            assert!(percentage < 0.75);
        }
    }

    pub fn retaped(self: &Self) -> GenTapedLieGroup<B, DOF, PARAMS, POINT, AMBIENT, SharedTape, G> {
        let params = self.params.clone().retaped();
        GenTapedLieGroup::<B, DOF, PARAMS, POINT, AMBIENT, SharedTape, G>::from_params(params)
    }

    pub fn set_params(&mut self, params: &V<B, PARAMS>) {
        self.params = params.clone();
    }

    pub fn params(&self) -> &V<B, PARAMS> {
        &self.params
    }

    fn test_params() {
        for g in Self::tutil_element_examples() {
            let params = g.params();
            let params2 = g.params.clone();
            assert_tensors_relative_eq_rank2!(params, params2, 0.0001);
        }
    }

    fn test_params_taped() {
        for g in Self::tutil_element_examples() {
            let taped_g = g.retaped();
            let params = taped_g.into_params();
            let params2 = g.params.clone();
            assert_tensors_relative_eq_rank2!(params, params2, 0.0001);
        }
    }

    pub fn group_adjoint(&self) -> M<B, DOF, DOF> {
        self.clone().into_group_adjoint()
    }

    fn algebra_adjoint(tangent: V<B, DOF>) -> M<B, DOF, DOF> {
        G::algebra_adjoint(tangent)
    }

    fn test_adjoint() {
        // Lie Group adjoint tests
        //
        // Adj{g} * x ==  vee[ Adj{ g * hat(x) * inv(g) } ]
        for g in Self::tutil_element_examples() {
            for x in G::tutil_tangent_examples() {
                let mat = g.matrix();
                let mat_adj: M<B, DOF, DOF> = g.group_adjoint();
                let xx: M<B, DOF, 1> = x.clone().reshape();
                let mat_adj_x: V<B, DOF> = mat_adj.matmul(xx).reshape();
                let mat_adj_x2 =
                    Self::vee(mat.matmul(Self::hat(x).matmul(g.to_inverse().matrix())));
                assert_tensors_relative_eq_rank2!(mat_adj_x, mat_adj_x2, 0.0001);
            }
        }

        // // Lie Algebra adjoint tests
        // //
        // // ad{a} * b ==  vee[ hat(a) * hat(b) - hat(b) * hat(a) ]
        // for a in G::tutil_tangent_examples() {
        //     for b in G::tutil_tangent_examples() {
        //         let bb: M<B, DOF, 1> = b.clone().reshape();
        //         let ad_a: M<B, DOF, DOF> = Self::algebra_adjoint(a.clone());
        //         let ad_a_b: V<B, DOF> = ad_a.matmul(bb).reshape();
        //         let lie_bracket_a_b = Self::vee(
        //             Self::hat(a.clone()) * Self::hat(b.clone())
        //                 - Self::hat(b) * Self::hat(a.clone()),
        //         );
        //         assert_tensors_relative_eq_rank2!(ad_a_b, lie_bracket_a_b, 0.0001);
        //     }
        // }
    }

    fn test_adjoint_taped() {
        todo!()
    }

    pub fn log(self) -> V<B, DOF> {
        self.clone().into_log()
    }

    fn test_exp_log() {
        // exp(log(g)) == g
        for g in Self::tutil_element_examples() {
            let matrix_before = g.compact();
            let matrix_after = Self::exp(g.log()).compact();
            assert_tensors_relative_eq_rank3!(matrix_before, matrix_after, 0.0001);
        }

        for omega in G::tutil_tangent_examples() {
            let exp_inverse = Self::exp(omega.clone()).to_inverse();
            let exp_neg_omega = Self::exp(omega.clone().negate());

            let exp_inverse = exp_inverse.compact();
            let exp_neg_omega = exp_neg_omega.compact();
            // println!("exp_inverse: {:?}", exp_inverse.array());
            // println!("exp_neg_omega: {:?}", exp_neg_omega.array());
            assert_tensors_relative_eq_rank3!(exp_inverse, exp_neg_omega, 0.0001);
        }
    }

    fn test_exp_log_taped() {
        todo!()
    }

    fn test_hat_vee_taped() {
        for x in G::tutil_tangent_examples() {
            // x == vee(hat(x))
            let taped_x: TapedV<B, DOF> = x.retaped();
            let batch = Self::vee(Self::hat(taped_x));
            assert_tensors_relative_eq_rank2!(batch, x, 0.0001);

            // dx hat(x)
            {
                let dx_hatx = |v: TapedV<B, DOF>| -> TapedM<B, AMBIENT, AMBIENT> { Self::hat(v) };
                let df = MatrixValuedMapFromVector::sym_diff_quotient_from_taped(
                    dx_hatx,
                    x.clone(),
                    1e-6,
                );
                let auto_grad = MatrixValuedMapFromVector::auto_grad(dx_hatx, x.clone());
                assert_tensors_relative_eq_rank4!(df, auto_grad, 1e-3);
            }

            // dx vee(x)
            {
                let m = Self::hat(x);

                let dm_veem = |m: TapedM<B, AMBIENT, AMBIENT>| -> TapedV<B, DOF> { Self::vee(m) };
                let df = VectorValuedMapFromMatrix::sym_diff_quotient_from_taped(
                    dm_veem,
                    m.clone(),
                    1e-6,
                );
                let auto_grad = VectorValuedMapFromMatrix::auto_grad(dm_veem, m.clone());
                assert_tensors_relative_eq_rank4!(df, auto_grad, 1e-3);
            }
        }
    }

    // pub fn mul(&self, g2: Self) -> Self {
    //     Self::mul_assign(self.clone(), g2)
    // }

    pub fn to_inverse(&self) -> Self {
        let result = self.clone();
        Self::into_inverse(result)
    }

    pub fn test_mul_inverse() {
        let mut g_vec = Self::tutil_element_examples();
        g_vec.truncate(10);
        let mut i = 0;

        let mut map = HashMap::new();

        for g1 in g_vec.clone() {
            println!("g1 = {:?}", i);
            i += 1;

            for g2 in g_vec.clone() {
                let before = Instant::now();

                let g1_times_g2 = Self::mul(g1.clone(), g2.clone());
                let dt = before.elapsed().as_secs_f64();
                map.entry("g1_times_g2").or_insert(vec![dt]).push(dt);

                for g3 in g_vec.clone() {
                    let b0 = Instant::now();
                    let g2_times_g3 = Self::mul(g2.clone(), g3.clone());
                    let dt = b0.elapsed().as_secs_f64();
                    map.entry("g2_times_g3").or_insert(vec![dt]).push(dt);

                    let b1 = Instant::now();
                    let left_hugging = Self::mul(g1_times_g2.clone(), g3.clone()).compact();
                    let dt = b1.elapsed().as_secs_f64();
                    map.entry("left_hugging").or_insert(vec![dt]).push(dt);

                    let b2 = Instant::now();
                    let right_hugging = Self::mul(g1.clone(), g2_times_g3).compact();
                    let dt = b2.elapsed().as_secs_f64();
                    map.entry("right_hugging").or_insert(vec![dt]).push(dt);


                    let b3 = Instant::now();
                    assert_tensors_relative_eq_rank3!(left_hugging, right_hugging, 1e-3);
                    let dt = b3.elapsed().as_secs_f64();
                    map.entry("assert_tensors_relative_eq_rank3").or_insert(vec![dt]).push(dt);
                }
            }
        }
        for pair in map {
            let sum = pair.1.iter().sum::<f64>();
            println!("Elapsed time: {} {}", pair.0, sum / pair.1.len() as f64);
        }
        let mut i = 0;
        for g1 in g_vec.clone() {
            i += 1;
            for g2 in g_vec.clone() {
                let daz_from_foo_transform_1 =
                    Self::mul(g2.clone().into_inverse(), g1.clone().into_inverse()).compact();
                let daz_from_foo_transform_2 = Self::mul(g1.clone(), g2).into_inverse().compact();
                assert_tensors_relative_eq_rank3!(
                    daz_from_foo_transform_1,
                    daz_from_foo_transform_2,
                    1e-3
                );
            }
        }
    }

    pub fn test_mul_inverse_taped() {
        todo!()
    }

    pub fn point_action<const NUM_POINTS: usize>(
        self: &Self,
        point: M<B, POINT, NUM_POINTS>,
    ) -> M<B, POINT, NUM_POINTS> {
        Self::into_point_action(self.clone(), point)
    }

    fn test_point_action() {
        for g in Self::tutil_element_examples() {
            for p in tutil_points_examples::<B, POINT, 5>() {
                let p_prime = g.point_action(p.clone());
                let hom_p = Self::into_ambient(p.clone());
                let p_prime2: M<B, POINT, 5> = g.compact().matmul(hom_p);

                assert_tensors_relative_eq_rank3!(p_prime, p_prime2, 0.0001);
            }
        }
    }

    fn test_point_action_taped() {
        todo!()
    }

    pub fn compact(&self) -> M<B, POINT, AMBIENT> {
        self.clone().into_compact()
    }

    pub fn matrix(&self) -> M<B, AMBIENT, AMBIENT> {
        self.clone().into_matrix()
    }

    fn test_compact_matrix() {
        for g in Self::tutil_element_examples() {
            let compact = g.compact();
            let compact2: M<B, POINT, AMBIENT> = g.matrix().slice((.., 0..POINT, ..)).realize();
            assert_tensors_relative_eq_rank3!(compact, compact2, 0.0001);
        }
    }

    fn test_compact_matrix_taped() {
        // for a in G::tutil_tangent_examples() {
        //     let eps = 1e-6;
        //     {
        //         let f = |x: TapedV<B, DOF>| -> TapedM<B, AMBIENT, AMBIENT> {
        //             GenTapedLieGroup::<B, DOF, PARAMS, POINT, AMBIENT, SharedTape, G>::exp(x)
        //                 .into_matrix()
        //         };
        //         let df = MatrixValuedMapFromVector::sym_diff_quotient_from_taped(f, a.clone(), eps);
        //         let auto_grad = MatrixValuedMapFromVector::auto_grad(f, a.clone());
        //         assert_tensors_relative_eq_rank4!(df, auto_grad, 1e-6);
        //     }
        //     {
        //         let g = |x: TapedV<B, DOF>| -> TapedM<B, POINT, AMBIENT> {
        //             GenTapedLieGroup::<B, DOF, PARAMS, POINT, AMBIENT, SharedTape, G>::exp(x)
        //                 .into_compact()
        //         };
        //         let df = MatrixValuedMapFromVector::sym_diff_quotient_from_taped(g, a.clone(), eps);
        //         let auto_grad = MatrixValuedMapFromVector::auto_grad(g, a.clone());
        //         assert_tensors_relative_eq_rank4!(df, auto_grad, 1e-6);
        //     }
        // }
    }

    fn test_into_ambient_taped() {
        for a in tutil_points_examples::<B, POINT, 4>() {
            let eps = 1e-6;
            {
                let f = |x: TapedM<B, POINT, 4>| -> TapedM<B, AMBIENT, 4> {
                    GenTapedLieGroup::<B, DOF, PARAMS, POINT, AMBIENT, SharedTape, G>::into_ambient(
                        x,
                    )
                };
                // let df = MatrixValuedMapFromMatrix::sym_diff_quotient_from_taped(f, a.clone(), eps);
                // let auto_grad = MatrixValuedMapFromMatrix::auto_grad(f, a.clone());
                //assert_tensors_relative_eq_rank5!(df, auto_grad, 1e-6);
            }
        }
    }

    // Derivative of the exponential map at the identity:
    //
    //   Dx [ exp(x) ] x=0
    //
    // Note that exp(.) is a map from the tangent space V<DOF> to the Lie group
    // parameters space V<PARAMS>, hence the Jacobian is a matrix of size
    // PARAMS x DOF.
    pub fn dx_exp_x_at_0() -> M<B, PARAMS, DOF> {
        G::dx_exp_x_at_0()
    }

    fn test_dx_exp_x_at_0() {
        todo!()
    }

    // Derivative of the product of the exponential map and a certain point from the
    // tangent space, evaluated at the identity of the Lie group:
    //
    //   Dx [ exp(x) * point ] x=0
    //
    // The [exp(.) * point] goes from the tangent space V<DOF> to the vector space
    // V<POINT>. The resulting Jacobian matrix is of size
    // POINT x DOF.
    pub fn dx_exp_x_times_point_at_0(point: V<B, POINT>) -> M<B, POINT, DOF> {
        G::dx_exp_x_times_point_at_0(point)
    }

    fn test_dx_exp_x_times_point_at_0() {
        todo!()
    }

    // Derivative of the product of the current instance and the exponential map,
    // evaluated at the identity of the Lie group:
    //
    //   Dx [ g * exp(x) ] x=0
    //
    // The map [g * exp(.)] goes from the tangent space V<DOF> to the Lie group
    // parameters space V<PARAMS>. The resulting Jacobian matrix is of size
    // PARAMS x DOF.
    pub fn dx_self_times_exp_x_at_0(&self) -> M<B, PARAMS, DOF> {
        G::dx_self_times_exp_x_at_0(&self.params)
    }

    pub fn test_dx_self_times_exp_x_at_0() {
        todo!()
    }

    pub fn test_all_taped() {
        Self::test_params_taped();
        Self::test_compact_matrix_taped();
        Self::test_into_ambient_taped();
        Self::test_hat_vee_taped();

        // Self::test_adjoint_taped();
        // Self::test_exp_log_taped();
        // Self::test_mul_inverse_taped();
        // Self::test_point_action_taped();
    }

    pub fn test_suite() {
        Self::test_params();
        Self::test_mul_inverse();
        Self::test_point_action();
        Self::test_compact_matrix();
        Self::test_exp_log();
        Self::test_preservability();
        Self::test_adjoint();
        // Self::test_dx_exp_x_at_0();
        // Self::test_dx_exp_x_times_point_at_0();
        // Self::test_dx_self_times_exp_x_at_0();

        Self::test_all_taped();
    }
}

//     // TODO: reactivate this

//     pub fn exp_jacobian_tests() {
//         let device = dfdx_core::tensor::Cpu::default();

//         let dxfx = Self::dx_exp_x_at_0();
//         for b in 0..B {
//             for p in 0..PARAMS {
//                 let zero: V<B, DOF, _> = device.zeros();

//                 let fx = Self::exp(zero.leaky_trace())
//                     .into_params()
//                     .select(device.tensor(b))
//                     .select(device.tensor(p));
//                 let g = fx.backward();
//                 let g_at_x = g.get(&zero).select(device.tensor(b));

//                 for dof in 0..DOF {
//                     assert_relative_eq!(dxfx[[b, p, dof]], g_at_x[[dof]], epsilon = 0.0001);
//                 }
//             }
//         }

//         // for point in example_points() {
//         //     // let j = Self::dx_exp_x_times_point_at_0(&point);

//         //     // let j_num = VectorField::sym_diff_quotient(
//         //     //     |x| Self::exp(x).transform(&point),
//         //     //     V::<DOF>::zeros(),
//         //     //     1e-6,
//         //     // );
//         //     // assert_relative_eq!(j, j_num, epsilon = 0.0001);
//         // }

//         for g in &Self::element_examples() {
//             let dxfx = g.dx_self_times_exp_x_at_0();
//             for b in 0..B {
//                 for p in 0..PARAMS {
//                     let zero: V<B, DOF, _> = device.zeros();

//                     // let fx = g
//                     //     .multiply(&Self::exp(zero.leaky_trace()))
//                     //     .into_params()
//                     //     .select(device.tensor(b))
//                     //     .select(device.tensor(p));
//                     //   let g = fx.backward();
//                     // println!("g: {:?}", g);
//                     // let g_at_x = g.get(&zero).select(device.tensor(b));

//                     // println!("g_at_x: {:?}", g_at_x);

//                     // for dof in 0..DOF {
//                     //     assert_relative_eq!(dxfx[[b, p, dof]], g_at_x[[dof]], epsilon = 0.0001);
//                     // }
//                 }
//             }
//             println!("dxfx: {:?}", dxfx.array());
//             panic!("foo");
//         }

//         // for g in &Self::element_examples() {
//         //     let j = g.dx_exp_x_times_self_at_0();
//         //     let j_num = VectorField::sym_diff_quotient(
//         //         |x| *(Self::exp(x).multiply(&g)).params(),
//         //         V::<DOF>::zeros(),
//         //         1e-6,
//         //     );
//         //     assert_relative_eq!(j, j_num, epsilon = 0.0001);
//         // }

//         // for g in &Self::element_examples() {
//         //     for h in &Self::element_examples() {
//         //         let j = g.dx_self_times_exp_x_times_other_at_0(h);
//         //         let j_num = VectorField::sym_diff_quotient(
//         //             |x| *(g.multiply(&Self::exp(x).multiply(&h))).params(),
//         //             V::<DOF>::zeros(),
//         //             1e-6,
//         //         );
//         //         assert_relative_eq!(j, j_num, epsilon = 0.0001);
//         //     }
//         // }

//         // for g in &Self::element_examples() {
//         //     //for a in &G::tangent_examples() {
//         //     println!("g: {}", g);
//         //     //println!("a: {}", a);
//         //     let j = g.dx_log_exp_x_times_self_at_0();
//         //     let j_num = Self::sym_diff_quotient(
//         //         |x| (Self::exp(x).multiply(g)).log(),
//         //         V::<DOF>::zeros(),
//         //         1e-6,
//         //     );
//         //     assert_relative_eq!(j, j_num, epsilon = 0.0001);
//         //     // }
//         // }
//     }

//     pub fn test_suite() {
//         Self::presentability_tests();
//         Self::group_operation_tests();
//         Self::hat_tests();
//         Self::exp_tests();
//         Self::adjoint_tests();
//         Self::exp_jacobian_tests();
//     }

// }
