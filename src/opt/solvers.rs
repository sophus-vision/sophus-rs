use crate::opt::variables::VarKind;

use super::cost::IsCost;
use super::ldlt::SparseLdlt;
use super::ldlt::SymmetricTripletMatrix;
use super::variables::VarPool;

/// Normal equation
pub struct SparseNormalEquation {
    sparse_hessian: SymmetricTripletMatrix,
    neg_gradient: nalgebra::DVector<f64>,
}

impl SparseNormalEquation {
    fn from_families_and_cost(
        variables: &VarPool,
        costs: Vec<Box<dyn IsCost>>,
        nu: f64,
    ) -> SparseNormalEquation {
        assert!(variables.num_of_kind(VarKind::Marginalized) == 0);
        assert!(variables.num_of_kind(VarKind::Free) >= 1);

        // Note let's first focus on these special cases, before attempting a
        // general version covering all cases holistically. Also, it might not be trivial
        // to implement VarKind::Marginalized > 1.
        //  -  Example, the the arrow-head sparsity uses a recursive application of the Schur-Complement.
        let num_var_params = variables.num_free_params();
        let mut upper_hessian_triplet = Vec::new();
        let mut neg_grad = nalgebra::DVector::<f64>::zeros(num_var_params);

        for evaluated_cost in costs.iter() {
            evaluated_cost.populate_normal_equation(
                variables,
                nu,
                &mut upper_hessian_triplet,
                &mut neg_grad,
            );
        }

        Self {
            sparse_hessian: SymmetricTripletMatrix {
                upper_triplets: upper_hessian_triplet,
                size: num_var_params,
            },

            neg_gradient: neg_grad,
        }
    }

    fn solve(&mut self) -> nalgebra::DVector<f64> {
        // TODO: perform permutation to reduce fill-in and symbolic factorization only once
        SparseLdlt::from_triplets(&self.sparse_hessian).solve(&self.neg_gradient)
    }
}

/// Solve the normal equation
pub fn solve(variables: &VarPool, costs: Vec<Box<dyn IsCost>>, nu: f64) -> VarPool {
    assert!(variables.num_of_kind(VarKind::Marginalized) <= 1);
    assert!(variables.num_of_kind(VarKind::Free) >= 1);

    if variables.num_of_kind(VarKind::Marginalized) == 0 {
        let mut sne = SparseNormalEquation::from_families_and_cost(variables, costs, nu);
        sne.solve();
        let delta = sne.solve();
        variables.update(delta)
    } else {
        todo!("Schur complement")
    }
}

// fn solveWithSchur<
//     const NUM: usize,
//     const NUM_ARGS: usize,
//     VarTuple: IsVarTuple<NUM_ARGS> + 'static,
// >(
//     variables_in: &VarPool,
//     evaluated_costs: Vec<EvaluatedCost<NUM, NUM_ARGS>>,
//     nu: f64,
// ) -> VarPool {
//     let mut variables = variables_in.clone();
//     assert!(variables.num_of_kind(VarKind::Marginalized) == 1);
//     assert!(variables.num_of_kind(VarKind::Free) == 1);
//     assert!(evaluated_costs.len() == 1);
//     let evaluated_cost = &evaluated_costs[0];
//     assert!(evaluated_cost.num_of_kind(VarKind::Marginalized, &variables) == 1);
//     assert!(evaluated_cost.num_of_kind(VarKind::Free, &variables) == 1);

//     let num_args = evaluated_cost.family_names.len();

//     let mut term_idx = 0;

//     let mut var_kind_array =
//         VarTuple::var_kind_array(&variables, evaluated_cost.family_names.clone());
//     let c_array = c_from_var_kind(&var_kind_array);
//     let less = CompareIdx { c: c_array };

//     let dof_tuple = evaluated_cost.dof_tuple;

//     let mut marg_arg_id = usize::MAX;
//     let mut free_arg_id = usize::MAX;

//     for i in 0..var_kind_array.len() {
//         let kind = var_kind_array[i];
//         if kind == VarKind::Marginalized {
//             marg_arg_id = i
//         }
//         if kind == VarKind::Free {
//             free_arg_id = i
//         }
//     }
//     assert_eq!(marg_arg_id, usize::MAX);
//     assert_eq!(free_arg_id, usize::MAX);

//     let marg_dof = dof_tuple[marg_arg_id] as usize;
//     let free_dof = dof_tuple[free_arg_id] as usize;

//     let marg_family = variables
//         .families
//         .get(&evaluated_cost.family_names[marg_dof])
//         .unwrap()
//         .clone();

//     let free_family = variables
//         .families
//         .get(&evaluated_cost.family_names[free_dof])
//         .unwrap()
//         .clone();
//     let free_start_idx_vec = free_family.get_start_indices();

//     let mut gradient_free = nalgebra::DVector::<f64>::zeros(free_family.num_free_scalars());
//     let mut hessian_free = nalgebra::DMatrix::<f64>::zeros(
//         free_family.num_free_scalars(),
//         free_family.num_free_scalars(),
//     );

//     #[derive(Debug)]
//     struct Track {
//         start_idx: usize,
//         len: usize,
//         gradient_free: nalgebra::DVector<f64>,
//         delta: nalgebra::DVector<f64>,
//         hessian_marg_free_vec: Vec<nalgebra::DMatrix<f64>>,
//         schur_block_marg_free_vec: Vec<nalgebra::DMatrix<f64>>,
//     }

//     let mut tracks = vec![];
//     tracks.reserve(marg_family.len());
//     while term_idx < evaluated_cost.terms.len() {
//         let outer_term = &evaluated_cost.terms[term_idx];
//         assert_eq!(outer_term.idx.len(), 1);
//         let outer_idx = &outer_term.idx[0];

//         term_idx += 1;

//         let start_idx = term_idx;

//         while term_idx < evaluated_cost.terms.len() {
//             let inner_term = &evaluated_cost.terms[term_idx];
//             assert_eq!(inner_term.idx.len(), 1);
//             let inner_idx = &inner_term.idx[0];

//             if !less.are_all_marg_vars_equal(outer_idx, inner_idx) {
//                 break;
//             }
//             term_idx += 1;
//         }

//         let mut track = Track {
//             start_idx,
//             len: term_idx - start_idx,
//             gradient_free: nalgebra::DVector::zeros(marg_dof),
//             delta: nalgebra::DVector::zeros(marg_dof),
//             hessian_marg_free_vec: vec![],
//             schur_block_marg_free_vec: vec![],
//         };

//         track.len = term_idx - track.start_idx;
//         track.hessian_marg_free_vec.reserve(track.len);
//         track.schur_block_marg_free_vec.reserve(track.len);

//         let mut orig_marg_hessian = nalgebra::DMatrix::<f64>::zeros(marg_dof, marg_dof);

//         for j in 0..track.len {
//             let term = &evaluated_cost.terms[track.start_idx + j];
//             gradient_free -= term.gradient.block(marg_arg_id);
//             orig_marg_hessian += term.hessian.block(marg_arg_id, marg_arg_id);

//             let free_idx = term.idx[0][free_arg_id];

//             let free_start_idx = free_start_idx_vec[free_idx];
//             if free_start_idx != -1 {
//                 let free_start_idx = free_start_idx as usize;
//                 // if it is not marked const

//                 // setting diagonal of h1 mat
//                 hessian_free
//                     .view_mut((free_start_idx, free_start_idx), (free_dof, free_dof))
//                     .copy_from(&term.hessian.block(free_arg_id, free_arg_id));
//                 track
//                     .gradient_free
//                     .rows_mut(free_start_idx, free_dof)
//                     .add_assign(term.gradient.block(free_arg_id));
//                 track.hessian_marg_free_vec[j] += term.hessian.block(marg_arg_id, free_arg_id);
//             }
//         }

//         let inv_orig_marg_hessian = orig_marg_hessian.try_inverse().unwrap();
//         track.delta = inv_orig_marg_hessian.clone() * track.gradient_free.clone();

//         // Eigen::Matrix<double, kD2, kD2> inv_orig_marg_hessian = orig_marg_hessian.inverse();
//         //     track.delta =   * track.gradient_free;

//         // outer prod
//         for j in 0..track.len {
//             let j_term = &evaluated_cost.terms[track.start_idx + j];
//             let j_free_idx = j_term.idx[0][free_arg_id];
//             let j_free_start_idx = free_start_idx_vec[j_free_idx];
//             if j_free_start_idx != -1 {
//                 let j_free_start_idx = j_free_start_idx as usize;

//                 let val = track.hessian_marg_free_vec[j].transpose() * &track.delta;
//                 gradient_free
//                     .rows_mut(j_free_start_idx, free_dof)
//                     .add_assign(-val);

//                 let schur_block_free_marg =
//                     track.schur_block_marg_free_vec[j].transpose() * inv_orig_marg_hessian.clone();
//                 track.schur_block_marg_free_vec[j] = schur_block_free_marg.clone();

//                 hessian_free
//                     .view_mut((j_free_start_idx, j_free_start_idx), (free_dof, free_dof))
//                     .add_assign(
//                         schur_block_free_marg.clone() * track.hessian_marg_free_vec[j].clone(),
//                     );

//                 for k in 0..track.len {
//                     let k_term = &evaluated_cost.terms[track.start_idx + k];
//                     let k_free_idx = k_term.idx[0][free_arg_id];
//                     let k_free_start_idx = free_start_idx_vec[k_free_idx];
//                     if k_free_start_idx != -1 {
//                         let k_free_start_idx = k_free_start_idx as usize;

//                         let mat_w23 =
//                             schur_block_free_marg.clone() * track.hessian_marg_free_vec[k].clone();

//                         hessian_free
//                             .view_mut((j_free_start_idx, k_free_start_idx), (free_dof, free_dof))
//                             .add_assign(-mat_w23.clone());

//                         hessian_free
//                             .view_mut((k_free_start_idx, j_free_start_idx), (free_dof, free_dof))
//                             .add_assign(-mat_w23.transpose());
//                     }
//                 }
//             }
//         } // outer prod

//         term_idx += track.len;

//         tracks.push(track);
//     }

//     // use sparse solver here
//     let delta = hessian_free.try_inverse().unwrap() * gradient_free;

//     // TODO
//     //  free_family.update(delta);

//     for j in 0..tracks.len() {
//         let marg_family = variables
//             .families
//             .get_mut(&evaluated_cost.family_names[marg_dof])
//             .unwrap();

//         let track = &mut tracks[j];
//         let j_term = &evaluated_cost.terms[track.start_idx + j];

//         let j_free_idx = j_term.idx[0][free_arg_id];
//         let j_free_start_idx = free_start_idx_vec[j_free_idx];
//         if j_free_start_idx != -1 {
//             let j_free_start_idx = j_free_start_idx as usize;
//             track.delta.add_assign(
//                 track.schur_block_marg_free_vec[j].clone()
//                     * delta.clone().rows(j_free_start_idx, free_dof),
//             );
//         }

//         marg_family.update_i(j, track.delta.clone());
//     }

//     variables
// }
