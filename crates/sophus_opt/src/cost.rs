use crate::term::Term;
use crate::variables::VarKind;
use crate::variables::VarPool;
use dyn_clone::DynClone;
use std::fmt::Debug;
use std::ops::AddAssign;

/// Evaluated cost
pub trait IsCost: Debug + DynClone {
    /// squared error
    fn calc_square_error(&self) -> f64;

    /// populate the normal equation
    fn populate_normal_equation(
        &self,
        variables: &VarPool,
        nu: f64,
        upper_hessian_triplet: &mut Vec<(usize, usize, f64)>,
        neg_grad: &mut nalgebra::DVector<f64>,
    );

    /// solve the normal equation with Schur complement
    fn solve_with_schur(&mut self, variables_in: &VarPool, nu: f64) -> VarPool;
}

/// Generic evaluated cost
#[derive(Debug, Clone)]
pub struct Cost<const NUM: usize, const NUM_ARGS: usize> {
    /// one name (of the corresponding variable family) for each argument (of the cost function
    pub family_names: [String; NUM_ARGS],
    /// evaluated terms of the cost function
    pub terms: Vec<Term<NUM, NUM_ARGS>>,
    /// degrees of freedom for each argument
    pub dof_tuple: [i64; NUM_ARGS],
}

impl<const NUM: usize, const NUM_ARGS: usize> Cost<NUM, NUM_ARGS> {
    /// create a new evaluated cost
    pub fn new(family_names: [String; NUM_ARGS], dof_tuple: [i64; NUM_ARGS]) -> Self {
        Cost {
            family_names,
            terms: Vec::new(),
            dof_tuple,
        }
    }

    /// add a term to the evaluated cost
    pub fn num_of_kind(&self, kind: VarKind, pool: &VarPool) -> usize {
        let mut c = 0;

        for name in self.family_names.iter() {
            c += if pool.families.get(name).unwrap().get_var_kind() == kind {
                1
            } else {
                0
            };
        }
        c
    }
}

impl<const NUM: usize, const NUM_ARGS: usize> IsCost for Cost<NUM, NUM_ARGS> {
    fn calc_square_error(&self) -> f64 {
        let mut error = 0.0;
        for term in self.terms.iter() {
            error += term.cost;
        }
        error
    }

    fn populate_normal_equation(
        &self,
        variables: &VarPool,
        nu: f64,
        upper_hessian_triplet: &mut Vec<(usize, usize, f64)>,
        neg_grad: &mut nalgebra::DVector<f64>,
    ) {
        let num_args = self.family_names.len();

        let mut start_indices_per_arg = Vec::new();

        let mut dof_per_arg = Vec::new();
        for name in self.family_names.iter() {
            let family = variables.families.get(name).unwrap();
            start_indices_per_arg.push(family.get_start_indices().clone());
            dof_per_arg.push(family.free_or_marg_dof());
        }

        for evaluated_term in self.terms.iter() {
            let idx = evaluated_term.idx;
            assert_eq!(idx.len(), num_args);

            for arg_id_alpha in 0..num_args {
                let dof_alpha = dof_per_arg[arg_id_alpha];
                if dof_alpha == 0 {
                    continue;
                }

                let var_idx_alpha = idx[arg_id_alpha];
                let start_idx_alpha = start_indices_per_arg[arg_id_alpha][var_idx_alpha];

                if start_idx_alpha == -1 {
                    continue;
                }

                let grad_block = evaluated_term.gradient.block(arg_id_alpha);
                let start_idx_alpha = start_idx_alpha as usize;
                assert_eq!(dof_alpha, grad_block.nrows());

                neg_grad
                    .rows_mut(start_idx_alpha, dof_alpha)
                    .add_assign(-grad_block);

                let hessian_block = evaluated_term.hessian.block(arg_id_alpha, arg_id_alpha);
                assert_eq!(dof_alpha, hessian_block.nrows());
                assert_eq!(dof_alpha, hessian_block.ncols());

                // block diagonal
                for r in 0..dof_alpha {
                    for c in 0..dof_alpha {
                        let mut d = 0.0;
                        if r == c {
                            d = nu;
                        }

                        if r <= c {
                            // upper triangular
                            upper_hessian_triplet.push((
                                start_idx_alpha + r,
                                start_idx_alpha + c,
                                hessian_block[(r, c)] + d,
                            ));
                        }
                    }
                }

                // off diagonal hessian
                for arg_id_beta in 0..num_args {
                    // skip diagonal blocks
                    if arg_id_alpha == arg_id_beta {
                        continue;
                    }
                    let dof_beta = dof_per_arg[arg_id_beta];
                    if dof_beta == 0 {
                        continue;
                    }

                    let var_idx_beta = idx[arg_id_beta];
                    let start_idx_beta = start_indices_per_arg[arg_id_beta][var_idx_beta];
                    if start_idx_beta == -1 {
                        continue;
                    }
                    let start_idx_beta = start_idx_beta as usize;
                    if start_idx_beta < start_idx_alpha {
                        // upper triangular, skip lower triangular
                        continue;
                    }

                    let hessian_block_alpha_beta =
                        evaluated_term.hessian.block(arg_id_alpha, arg_id_beta);
                    let hessian_block_beta_alpha =
                        evaluated_term.hessian.block(arg_id_beta, arg_id_alpha);

                    assert_eq!(dof_alpha, hessian_block_alpha_beta.nrows());
                    assert_eq!(dof_beta, hessian_block_alpha_beta.ncols());
                    assert_eq!(dof_beta, hessian_block_beta_alpha.nrows());
                    assert_eq!(dof_alpha, hessian_block_beta_alpha.ncols());

                    // alpha-beta off-diagonal
                    for r in 0..dof_alpha {
                        for c in 0..dof_beta {
                            // upper triangular
                            upper_hessian_triplet.push((
                                start_idx_alpha + r,
                                start_idx_beta + c,
                                hessian_block_alpha_beta[(r, c)],
                            ));
                        }
                    }
                }
            }
        }
    }

    fn solve_with_schur(&mut self, variables_in: &VarPool, nu: f64) -> VarPool {
        let mut variables = variables_in.clone();
        assert!(variables.num_of_kind(VarKind::Marginalized) == 1);
        assert!(variables.num_of_kind(VarKind::Free) == 1);
        let cost = self;
        assert!(variables_in.num_of_kind(VarKind::Marginalized,) == 1);
        assert!(variables_in.num_of_kind(VarKind::Free) == 1);

        //let num_args = cost.family_names.len();

        let mut term_idx = 0;

        // let mut var_kind_array = VarTuple::var_kind_array(&variables, cost.family_names.clone());
        // // let c_array = c_from_var_kind(&var_kind_array);
        // // let less = CompareIdx::new(&c_array);

        // let dof_tuple = cost.dof_tuple;

        // let mut marg_arg_id = usize::MAX;
        // let mut free_arg_id = usize::MAX;

        // for i in 0..var_kind_array.len() {
        //     let kind = var_kind_array[i];
        //     if kind == VarKind::Marginalized {
        //         marg_arg_id = i
        //     }
        //     if kind == VarKind::Free {
        //         free_arg_id = i
        //     }
        // }
        // assert_eq!(marg_arg_id, usize::MAX);
        // assert_eq!(free_arg_id, usize::MAX);

        // let marg_dof = dof_tuple[marg_arg_id] as usize;
        // let free_dof = dof_tuple[free_arg_id] as usize;

        let mut marg_family = None;

        // = variables
        //     .families
        //     .get(&cost.family_names[marg_dof])
        //     .unwrap()
        //     .clone();

        let mut free_family = None;

        let mut marg_arg_id = usize::MAX;
        let mut free_arg_id = usize::MAX;

        let mut free_family_name = String::new();
        let mut marg_family_name = String::new();

        for i in 0..cost.family_names.len() {
            let family = variables.families.get(&cost.family_names[i]).unwrap();

            if family.get_var_kind() == VarKind::Marginalized {
                marg_family = Some(family.clone());
                marg_arg_id = i;
                marg_family_name = cost.family_names[i].clone();
            } else if family.get_var_kind() == VarKind::Free {
                free_family = Some(family.clone());
                free_arg_id = i;
                free_family_name = cost.family_names[i].clone();
            }
        }

        let free_family = free_family.unwrap();
        let marg_family = marg_family.unwrap();

        let marg_dof = marg_family.free_or_marg_dof();
        let free_dof = free_family.free_or_marg_dof();

        println!("marg_dof: {:?}", marg_dof);
        println!("free_dof: {:?}", free_dof);

        // variables
        //     .families
        //     .get(&cost.family_names[free_dof])
        //     .unwrap()
        //     .clone();
        let free_start_idx_vec = free_family.get_start_indices().clone();

        let mut gradient_free = nalgebra::DVector::<f64>::zeros(free_family.num_free_scalars());
        let mut hessian_free = nalgebra::DMatrix::<f64>::zeros(
            free_family.num_free_scalars(),
            free_family.num_free_scalars(),
        );

        println!(
            "hessian_free: {:?} {:?}",
            hessian_free.nrows(),
            hessian_free.ncols()
        );

        #[derive(Debug)]
        struct Track {
            start_idx: usize,
            len: usize,
            gradient_free: nalgebra::DVector<f64>,
            delta: nalgebra::DVector<f64>,
            hessian_marg_free_vec: Vec<nalgebra::DMatrix<f64>>,
            schur_block_marg_free_vec: Vec<nalgebra::DMatrix<f64>>,
        }

        let mut tracks = Vec::with_capacity(marg_family.len());
        while term_idx < cost.terms.len() {
            let outer_term = &cost.terms[term_idx];
            let outer_idx = &outer_term.idx;

            term_idx += 1;

            let start_idx = term_idx;

            while term_idx < cost.terms.len()
                && outer_idx[free_arg_id] == cost.terms[term_idx].idx[free_arg_id]
            {
                // let inner_term = &cost.terms[term_idx];
                // let inner_idx = &inner_term.idx;

                // if !less.are_all_marg_vars_equal(outer_idx, inner_idx) {
                //     break;
                // }
                term_idx += 1;
            }

            let mut track = Track {
                start_idx,
                len: term_idx - start_idx,
                gradient_free: nalgebra::DVector::zeros(free_dof),
                delta: nalgebra::DVector::zeros(marg_dof),
                hessian_marg_free_vec: vec![],
                schur_block_marg_free_vec: vec![],
            };

            track.len = term_idx - track.start_idx;
            track.hessian_marg_free_vec.reserve(track.len);
            track.schur_block_marg_free_vec.reserve(track.len);

            let mut orig_marg_hessian = nalgebra::DMatrix::<f64>::zeros(marg_dof, marg_dof);
            let mut gradient_marg = nalgebra::DVector::<f64>::zeros(marg_dof);

            for i in 0..orig_marg_hessian.nrows() {
                orig_marg_hessian[(i, i)] = nu;
            }

            println!("!!!!!!track len: {:?}", track.len);

            for j in 0..track.len {
                let term = &cost.terms[track.start_idx + j];
                gradient_marg -= term.gradient.block(marg_arg_id);
                orig_marg_hessian += term.hessian.block(marg_arg_id, marg_arg_id);

                let free_idx = term.idx[free_arg_id];

                let free_start_idx = free_start_idx_vec[free_idx];
                if free_start_idx != -1 {
                    let free_start_idx = free_start_idx as usize;
                    // if it is not marked const

                    // println!("free_start_idx: {:?}", free_start_idx);
                    // println!("free_dof: {:?}", free_dof);

                    // println!("free_arg_id: {:?}", free_arg_id);

                    // let free_block =term.hessian.block(free_arg_id, free_arg_id);

                    // println!("marg_free_block: {:?} {:?}", free_block.nrows(), free_block.ncols());

                    // setting diagonal of h1 mat
                    hessian_free
                        .view_mut((free_start_idx, free_start_idx), (free_dof, free_dof))
                        .copy_from(&term.hessian.block(free_arg_id, free_arg_id));

                    println!(
                        "grad free; {:?} {:?}",
                        track.gradient_free.nrows(),
                        track.gradient_free.ncols()
                    );
                    println!("free start idx: {:?}", free_start_idx);
                    println!("free dof: {:?}", free_dof);
                    track
                        .gradient_free
                        .rows_mut(free_start_idx, free_dof)
                        .add_assign(term.gradient.block(free_arg_id));
                    println!("j: {:?}", j);
                    track.hessian_marg_free_vec[j] += term.hessian.block(marg_arg_id, free_arg_id);
                }
            }

            println!("orig_marg_hessian: {}", orig_marg_hessian);
            println!("orig_marg_hessian: {}", orig_marg_hessian.determinant());

            let inv_orig_marg_hessian = orig_marg_hessian.try_inverse().unwrap();
            println!("inv_orig_marg_hessian: {}", inv_orig_marg_hessian);
            track.delta = inv_orig_marg_hessian.clone() * gradient_marg;

            println!("track len: {:?}", track.len);
            // outer prod
            for j in 0..track.len {
                let j_term = &cost.terms[track.start_idx + j];
                let j_free_idx = j_term.idx[free_arg_id];
                let j_free_start_idx = free_start_idx_vec[j_free_idx];
                if j_free_start_idx != -1 {
                    let j_free_start_idx = j_free_start_idx as usize;

                    println!(
                        "hessian_marg_free_vec: {:?}",
                        track.hessian_marg_free_vec[j]
                    );
                    println!("delta: {:?}", track.delta);

                    let val = track.hessian_marg_free_vec[j].transpose() * &track.delta;
                    gradient_free
                        .rows_mut(j_free_start_idx, free_dof)
                        .add_assign(-val);

                    let schur_block_free_marg = track.schur_block_marg_free_vec[j].transpose()
                        * inv_orig_marg_hessian.clone();
                    track.schur_block_marg_free_vec[j] = schur_block_free_marg.clone();

                    hessian_free
                        .view_mut((j_free_start_idx, j_free_start_idx), (free_dof, free_dof))
                        .add_assign(
                            schur_block_free_marg.clone() * track.hessian_marg_free_vec[j].clone(),
                        );

                    for k in 0..track.len {
                        let k_term = &cost.terms[track.start_idx + k];
                        let k_free_idx = k_term.idx[free_arg_id];
                        let k_free_start_idx = free_start_idx_vec[k_free_idx];
                        if k_free_start_idx != -1 {
                            let k_free_start_idx = k_free_start_idx as usize;

                            let mat_w23 = schur_block_free_marg.clone()
                                * track.hessian_marg_free_vec[k].clone();

                            hessian_free
                                .view_mut(
                                    (j_free_start_idx, k_free_start_idx),
                                    (free_dof, free_dof),
                                )
                                .add_assign(-mat_w23.clone());

                            hessian_free
                                .view_mut(
                                    (k_free_start_idx, j_free_start_idx),
                                    (free_dof, free_dof),
                                )
                                .add_assign(-mat_w23.transpose());
                        }
                    }
                }
            } // outer prod

            term_idx += track.len;

            tracks.push(track);
        }

        // use sparse solver here
        println!("hessian_free: {:?}", hessian_free);
        let delta = hessian_free.try_inverse().unwrap() * gradient_free;

        // TODO
        let free_family = variables.families.get_mut(&free_family_name).unwrap();
        free_family.update(delta.as_view());

        for j in 0..tracks.len() {
            let marg_family = variables.families.get_mut(&marg_family_name).unwrap();

            let track = &mut tracks[j];
            let j_term = &cost.terms[track.start_idx + j];

            let j_free_idx = j_term.idx[free_arg_id];
            let j_free_start_idx = free_start_idx_vec[j_free_idx];
            if j_free_start_idx != -1 {
                let j_free_start_idx = j_free_start_idx as usize;
                track.delta.add_assign(
                    track.schur_block_marg_free_vec[j].clone()
                        * delta.clone().rows(j_free_start_idx, free_dof),
                );
            }

            marg_family.update_i(j, track.delta.clone());
        }

        variables
    }
}
