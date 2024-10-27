use crate::term::Term;
use crate::variables::VarKind;
use crate::variables::VarPool;
use core::fmt::Debug;
use core::ops::AddAssign;
use dyn_clone::DynClone;

extern crate alloc;

/// Evaluated cost
pub trait IsCost: Debug + DynClone {
    /// squared error
    fn calc_square_error(&self) -> f64;

    /// populate the normal equation
    fn populate_normal_equation(
        &self,
        variables: &VarPool,
        nu: f64,
        upper_hessian_triplet: &mut alloc::vec::Vec<(usize, usize, f64)>,
        neg_grad: &mut nalgebra::DVector<f64>,
    );
}

/// Generic evaluated cost
#[derive(Debug, Clone)]
pub struct Cost<const NUM: usize, const NUM_ARGS: usize> {
    /// one name (of the corresponding variable family) for each argument (of the cost function
    pub family_names: [alloc::string::String; NUM_ARGS],
    /// evaluated terms of the cost function
    pub terms: alloc::vec::Vec<Term<NUM, NUM_ARGS>>,
    /// degrees of freedom for each argument
    pub dof_tuple: [i64; NUM_ARGS],
}

impl<const NUM: usize, const NUM_ARGS: usize> Cost<NUM, NUM_ARGS> {
    /// create a new evaluated cost
    pub fn new(
        family_names: [alloc::string::String; NUM_ARGS],
        dof_tuple: [i64; NUM_ARGS],
    ) -> Self {
        Cost {
            family_names,
            terms: alloc::vec::Vec::new(),
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
        upper_hessian_triplet: &mut alloc::vec::Vec<(usize, usize, f64)>,
        neg_grad: &mut nalgebra::DVector<f64>,
    ) {
        let num_args = self.family_names.len();

        let mut start_indices_per_arg = alloc::vec::Vec::new();

        let mut dof_per_arg = alloc::vec::Vec::new();
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
}
