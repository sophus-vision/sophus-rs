use dyn_clone::DynClone;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::calculus::types::M;
use crate::calculus::types::V;
use crate::lie::rotation2::Isometry2;
use crate::opt::block::SophusAddAssign;

use super::block::NewBlockMatrix;
use super::block::NewBlockVector;

pub trait NewCostTermSignature<const N: usize> {
    type Constants;

    const DOF_TUPLE: [i64; N];

    fn c_ref(&self) -> &Self::Constants;

    fn idx_vec(&self) -> Vec<usize>;

    fn idx_ref(&self) -> &[usize; N];
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum NewVariableKind {
    Free,
    Conditioned,
    Marginalized,
}

pub trait NewVariableTrait: Clone + Debug {
    const DOF: usize;
    type Arg;

    fn update(&mut self, delta: &[f64]);
}

pub trait NewVTuple<const NUM_ARGS: usize> {
    const DOF_T: [usize; NUM_ARGS];

    fn from(
        families: &NewVariableFamilies,
        names: [String; NUM_ARGS],
        ids: [usize; NUM_ARGS],
    ) -> Self;

    fn var_kind_array(
        families: &NewVariableFamilies,
        names: [String; NUM_ARGS],
    ) -> [NewVariableKind; NUM_ARGS];
}

impl<M0: NewVariableTrait + 'static> NewVTuple<1> for M0 {
    const DOF_T: [usize; 1] = [M0::DOF];

    fn from(families: &NewVariableFamilies, names: [String; 1], ids: [usize; 1]) -> Self {
        families
            .get::<NewGenVariableFamily<M0>>(names[0].clone())
            .members[ids[0]]
            .clone()
    }

    fn var_kind_array(families: &NewVariableFamilies, names: [String; 1]) -> [NewVariableKind; 1] {
        [families.families.get(&names[0]).unwrap().get_var_kind()]
    }
}

impl<M0: NewVariableTrait + 'static, M1: NewVariableTrait + 'static> NewVTuple<2> for (M0, M1) {
    const DOF_T: [usize; 2] = [M0::DOF, M1::DOF];

    fn from(families: &NewVariableFamilies, names: [String; 2], ids: [usize; 2]) -> Self {
        (
            families
                .get::<NewGenVariableFamily<M0>>(names[0].clone())
                .members[ids[0]]
                .clone(),
            families
                .get::<NewGenVariableFamily<M1>>(names[1].clone())
                .members[ids[1]]
                .clone(),
        )
    }

    fn var_kind_array(families: &NewVariableFamilies, names: [String; 2]) -> [NewVariableKind; 2] {
        [
            families.families.get(&names[0]).unwrap().get_var_kind(),
            families.families.get(&names[1]).unwrap().get_var_kind(),
        ]
    }
}

impl NewVariableTrait for Isometry2<f64> {
    const DOF: usize = 3;

    fn update(&mut self, delta: &[f64]) {
        let mut delta_vec = V::<3>::zeros();
        for d in 0..Self::DOF {
            delta_vec[d] = delta[d];
        }
        self.set_params(
            (Isometry2::<f64>::group_mul(&Isometry2::<f64>::exp(&delta_vec), &self.clone()))
                .params(),
        );
    }

    type Arg = Isometry2<f64>;
}

#[derive(Debug, Clone)]
pub struct NewGenVariableFamily<NV: NewVariableTrait> {
    kind: NewVariableKind,
    pub members: Vec<NV>,
    constant_members: HashMap<usize, ()>,
}

impl<NV: NewVariableTrait> NewGenVariableFamily<NV> {
    pub fn new(
        kind: NewVariableKind,
        members: Vec<NV>,
        constant_members: HashMap<usize, ()>,
    ) -> Self {
        NewGenVariableFamily {
            kind,
            members,
            constant_members,
        }
    }

    fn c(&self) -> NewVariableKind {
        self.kind
    }
}

pub trait NewVariableFamilyTrait: as_any::AsAny + Debug + DynClone {
    fn update(&mut self, delta: &[f64]);
    fn num_scalars(&self) -> usize;
    fn calc_start_indices(&self) -> Vec<i64>;
    fn dof(&self) -> usize;
    fn get_var_kind(&self) -> NewVariableKind;
}

impl<NV: NewVariableTrait + 'static> NewVariableFamilyTrait for NewGenVariableFamily<NV> {
    fn update(&mut self, delta: &[f64]) {
        let start_indices = self.calc_start_indices();
        let dof = self.dof();

        assert_eq!(start_indices.len(), self.members.len());

        for i in 0..start_indices.len() {
            let start_idx = start_indices[i];
            if start_idx == -1 {
                continue;
            }
            let start_idx = start_idx as usize;
            self.members[i].update(&delta[start_idx..start_idx + dof]);
        }
    }

    fn num_scalars(&self) -> usize {
        (self.members.len() - self.constant_members.len()) * NV::DOF
    }

    // returns -1 if variable is not free
    fn calc_start_indices(&self) -> Vec<i64> {
        let mut indices = vec![];
        let mut idx: usize = 0;
        for i in 0..self.members.len() {
            if self.constant_members.contains_key(&i) {
                indices.push(-1);
            } else {
                indices.push(idx as i64);
                idx += NV::DOF;
            }
        }

        assert_eq!(indices.len(), self.members.len());
        indices
    }

    fn dof(&self) -> usize {
        NV::DOF
    }

    fn get_var_kind(&self) -> NewVariableKind {
        self.c()
    }
}

dyn_clone::clone_trait_object!(NewVariableFamilyTrait);

#[derive(Debug, Clone)]
pub struct NewVariableFamilies {
    pub families: std::collections::HashMap<String, Box<dyn NewVariableFamilyTrait>>,
}

#[derive(Debug, Clone)]
pub struct GenEvalCost {
    pub family_names: Vec<String>,
    pub terms: Vec<NewCostTerm>,
}

impl GenEvalCost {
    fn new(family_names: Vec<String>) -> Self {
        GenEvalCost {
            family_names,
            terms: Vec::new(),
        }
    }
}

fn c_from_var_kind<const N: usize>(var_kind_array: &[NewVariableKind; N]) -> [char; N] {
    let mut c_array: [char; N] = ['0'; N];

    for i in 0..N {
        c_array[i] = match var_kind_array[i] {
            NewVariableKind::Free => 'f',
            NewVariableKind::Conditioned => 'c',
            NewVariableKind::Marginalized => 'm',
        };
    }

    c_array
}

impl NewVariableFamilies {
    pub fn num_free_params(&self) -> usize {
        let mut num = 0;

        for f in self.families.iter() {
            num += f.1.num_scalars();
        }

        num
    }

    fn update(&self, delta: Vec<f64>) -> NewVariableFamilies {
        let mut updated = self.clone();
        for family in updated.families.iter_mut() {
            family.1.update(&delta[..]);
        }
        updated
    }

    pub fn get<T: NewVariableFamilyTrait>(&self, name: String) -> &T {
        as_any::Downcast::downcast_ref::<T>(self.families.get(&name).unwrap().as_ref()).unwrap()
    }

    pub fn get_members<T: NewVariableTrait + 'static>(&self, name: String) -> Vec<T> {
        as_any::Downcast::downcast_ref::<NewGenVariableFamily<T>>(
            self.families.get(&name).unwrap().as_ref(),
        )
        .unwrap()
        .members
        .clone()
    }

    fn apply<
        const NUM_ARGS: usize,
        R,
        CCC,
        CTS: NewCostTermSignature<NUM_ARGS, Constants = CCC>,
        NVT: NewVTuple<NUM_ARGS> + 'static,
    >(
        &self,
        cost: &GenCostSignature<NUM_ARGS, CCC, CTS>,
        res_fn: R,
    ) -> GenEvalCost
    where
        R: NewResidualFn<NUM_ARGS, NVT, CCC>,
    {
        use crate::opt::cost_args::CompareIdx;
        let var_kind_array = NVT::var_kind_array(self, cost.family_names.clone());
        let c_array = c_from_var_kind(&var_kind_array);
        let less = CompareIdx { c: c_array };

        let mut evaluated_terms = GenEvalCost::new(cost.family_names.clone().into());

        let mut i = 0;

        while i < cost.terms.len() {
            let t = &cost.terms[i];

            let outer_idx = t.idx_ref();

            let mut term = res_fn.cost(
                NVT::from(self, cost.family_names.clone(), *t.idx_ref()),
                var_kind_array,
                t.c_ref(),
            );

            term.idx.push(t.idx_vec());
            i += 1;

            // perform reduction over conditioned variables
            while i < cost.terms.len() {
                let t = &cost.terms[i];

                if !less.all_var_eq(outer_idx, t.idx_ref()) {
                    break;
                }

                i += 1;

                let term2 = res_fn.cost(
                    NVT::from(self, cost.family_names.clone(), *t.idx_ref()),
                    var_kind_array,
                    t.c_ref(),
                );

                term.hessian.mat.add_assign(term2.hessian.mat);
                term.gradient.vec.add_assign(term2.gradient.vec);
                term.cost += term2.cost;
            }

            evaluated_terms.terms.push(term);
        }

        evaluated_terms
    }
}

#[derive(Debug, Clone)]
pub struct NewCostTerm {
    pub hessian: NewBlockMatrix,
    pub gradient: NewBlockVector,
    pub cost: f64,
    pub idx: Vec<Vec<usize>>,
}

impl NewCostTerm {
    pub fn new1<const D0: usize, const R: usize>(
        maybe_dx0: Option<M<R, D0>>,
        residual: V<R>,
    ) -> Self {
        let dims = vec![D0];
        let mut hessian = NewBlockMatrix::new(&dims);
        let mut gradient = NewBlockVector::new(&dims);

        if maybe_dx0.is_some() {
            let dx0: M<R, D0> = maybe_dx0.unwrap();
            let dx0_t: M<D0, R> = dx0.transpose();

            let grad0 = dx0_t * residual;
            gradient.set_block(0, grad0);

            let h00 = dx0_t * dx0;
            hessian.set_block(0, 0, h00);
        }

        Self {
            hessian,
            gradient,
            cost: residual.norm(),
            idx: Vec::new(),
        }
    }
    pub fn new2<const D0: usize, const D1: usize, const R: usize>(
        maybe_dx0: Option<M<R, D0>>,
        maybe_dx1: Option<M<R, D1>>,
        residual: V<R>,
    ) -> Self {
        let dims = vec![D0, D1];
        let mut hessian = NewBlockMatrix::new(&dims);
        let mut gradient = NewBlockVector::new(&dims);

        if maybe_dx0.is_some() {
            let dx0: M<R, D0> = maybe_dx0.unwrap();
            let dx0_t: M<D0, R> = dx0.transpose();

            let grad0 = dx0_t * residual;
            gradient.set_block(0, grad0);

            let h00 = dx0_t * dx0;
            hessian.set_block(0, 0, h00);
        }

        if maybe_dx1.is_some() {
            let dx1: M<R, D1> = maybe_dx1.unwrap();
            let dx1_t: M<D1, R> = dx1.transpose();

            let grad1 = dx1_t * residual;
            gradient.set_block(1, grad1);

            let h11 = dx1_t * dx1;
            hessian.set_block(1, 1, h11);

            // off-diagonal
            if maybe_dx0.is_some() {
                let dx0: M<R, D0> = maybe_dx0.unwrap();
                let dx0_t: M<D0, R> = dx0.transpose();

                let h01 = dx0_t * dx1;
                let h10 = dx1_t * dx0;
                hessian.set_block(0, 1, h01);
                hessian.set_block(1, 0, h10);
            }
        }

        Self {
            hessian,
            gradient,
            cost: residual.norm(),
            idx: Vec::new(),
        }
    }

    pub fn new3<const D0: usize, const D1: usize, const D2: usize, const R: usize>(
        _maybe_dx0: Option<M<R, D0>>,
        _maybe_dx1: Option<M<R, D1>>,
        _maybe_dx2: Option<M<R, D2>>,
        _residual: V<R>,
    ) -> Self {
        todo!()
    }
}

pub trait NewResidualFn<const NUM_ARGS: usize, Args: NewVTuple<NUM_ARGS>, Constants>: Copy {
    fn cost(
        &self,
        args: Args,
        derivatives: [NewVariableKind; NUM_ARGS],
        constants: &Constants,
    ) -> NewCostTerm;
}

#[derive(Debug, Clone)]
pub struct GenCostSignature<
    const NUM_ARGS: usize,
    CCC,
    CTS: NewCostTermSignature<NUM_ARGS, Constants = CCC>,
> {
    pub family_names: [String; NUM_ARGS],
    pub terms: Vec<CTS>,
}

impl<const NUM_ARGS: usize, CCC, CTS: NewCostTermSignature<NUM_ARGS, Constants = CCC>>
    GenCostSignature<NUM_ARGS, CCC, CTS>
{
    fn sort(&mut self, var_kind_array: &[NewVariableKind; NUM_ARGS]) {
        use crate::opt::cost_args::CompareIdx;

        let c_array = c_from_var_kind(var_kind_array);

        let less = CompareIdx { c: c_array };

        assert!(!self.terms.is_empty());

        self.terms
            .sort_by(|a, b| less.less_than(*a.idx_ref(), *b.idx_ref()));

        for t in 0..self.terms.len() - 1 {
            assert!(
                less.less_than(*self.terms[t].idx_ref(), *self.terms[t + 1].idx_ref())
                    == std::cmp::Ordering::Less
            );
        }
    }
}
pub struct NewSparseNormalEquation {
    sparse_hessian: sprs::CsMat<f64>,
    neg_gradient: Vec<f64>,
}

impl NewSparseNormalEquation {
    fn from_families_and_cost(
        variable_families: &NewVariableFamilies,
        costs: Vec<GenEvalCost>,
        nu: f64,
    ) -> NewSparseNormalEquation {
        let num_var_params = variable_families.num_free_params();
        let mut hessian_triplet = sprs::TriMat::new((num_var_params, num_var_params));
        let mut neg_grad = Vec::with_capacity(num_var_params);

        for _ in 0..num_var_params {
            neg_grad.push(0.0);
        }

        for evaluated_cost in costs.iter() {
            let num_args = evaluated_cost.family_names.len();

            let mut start_indices_per_arg = Vec::new();
            let mut dof_per_arg = Vec::new();

            for name in evaluated_cost.family_names.iter() {
                let family = variable_families.families.get(name).unwrap();
                start_indices_per_arg.push(family.calc_start_indices());
                dof_per_arg.push(family.dof());
            }

            for term in evaluated_cost.terms.iter() {
                assert_eq!(term.idx.len(), 1);
                let idx = term.idx[0].clone();
                assert_eq!(idx.len(), num_args);

                for arg_id_alpha in 0..num_args {
                    let dof_alpha = dof_per_arg[arg_id_alpha];

                    let var_idx_alpha = idx[arg_id_alpha];
                    let start_idx_alpha = start_indices_per_arg[arg_id_alpha][var_idx_alpha];

                    if start_idx_alpha == -1 {
                        continue;
                    }

                    let grad_block = term.gradient.block(arg_id_alpha);
                    let start_idx_alpha = start_idx_alpha as usize;
                    assert_eq!(dof_alpha, grad_block.nrows());

                    for r in 0..dof_alpha {
                        neg_grad[start_idx_alpha + r] -= grad_block.read(r, 0);
                    }

                    let hessian_block = term.hessian.block(arg_id_alpha, arg_id_alpha);
                    assert_eq!(dof_alpha, hessian_block.nrows());
                    assert_eq!(dof_alpha, hessian_block.ncols());

                    // block diagonal
                    for r in 0..dof_alpha {
                        for c in 0..dof_alpha {
                            let mut d = 0.0;
                            if r == c {
                                d = nu;
                            }
                            hessian_triplet.add_triplet(
                                start_idx_alpha + r,
                                start_idx_alpha + c,
                                hessian_block.read(r, c) + d,
                            );
                        }
                    }

                    // off diagonal hessian
                    for arg_id_beta in 0..num_args {
                        // skip diagonal blocks
                        if arg_id_alpha == arg_id_beta {
                            continue;
                        }
                        let dof_beta = dof_per_arg[arg_id_beta];

                        let var_idx_beta = idx[arg_id_beta];
                        let start_idx_beta = start_indices_per_arg[arg_id_beta][var_idx_beta];
                        if start_idx_beta == -1 {
                            continue;
                        }
                        let start_idx_beta = start_idx_beta as usize;

                        let hessian_block_alpha_beta =
                            term.hessian.block(arg_id_alpha, arg_id_beta);
                        let hessian_block_beta_alpha =
                            term.hessian.block(arg_id_beta, arg_id_alpha);

                        assert_eq!(dof_alpha, hessian_block_alpha_beta.nrows());
                        assert_eq!(dof_beta, hessian_block_alpha_beta.ncols());
                        assert_eq!(dof_beta, hessian_block_beta_alpha.nrows());
                        assert_eq!(dof_alpha, hessian_block_beta_alpha.ncols());

                        // alpha-beta off-diagonal
                        for r in 0..dof_alpha {
                            for c in 0..dof_beta {
                                hessian_triplet.add_triplet(
                                    start_idx_alpha + r,
                                    start_idx_beta + c,
                                    hessian_block_alpha_beta.read(r, c),
                                );
                            }
                        }
                    }
                }
            }
        }

        Self {
            sparse_hessian: hessian_triplet.to_csr(),
            neg_gradient: neg_grad,
        }
    }

    fn solve(&mut self) -> Vec<f64> {
        let ldl = sprs_ldl::LdlNumeric::new(self.sparse_hessian.view()).unwrap();

        ldl.solve(self.neg_gradient.clone())
    }
}

fn calc_mse(cost: Vec<GenEvalCost>) -> f64 {
    let mut c = 0.0;
    for g in cost {
        for t in g.terms {
            c += t.cost;
        }
    }
    c
}

#[derive(Copy, Clone, Debug)]
pub struct OptParams {
    pub num_iter: usize,
    pub initial_lm_nu: f64,
}

impl Default for OptParams {
    fn default() -> Self {
        Self {
            num_iter: 20,
            initial_lm_nu: 10.0,
        }
    }
}

// Takes a single n-nary cost function
//
// TODO: This needs to be generalized to take N m-ary cost functions
pub fn new_optimize_c<
    const NUM_ARGS: usize,
    NVT: NewVTuple<NUM_ARGS> + 'static,
    CCC,
    CTS: NewCostTermSignature<NUM_ARGS, Constants = CCC>,
    R,
>(
    mut variable_families: NewVariableFamilies,
    mut cost: (GenCostSignature<NUM_ARGS, CCC, CTS>, R),
    params: OptParams,
) -> NewVariableFamilies
where
    R: NewResidualFn<NUM_ARGS, NVT, CCC>,
{
    cost.0.sort(&NVT::var_kind_array(
        &variable_families,
        cost.0.family_names.clone(),
    ));
    let mut init_costs: Vec<GenEvalCost> = Vec::new();
    init_costs.push(variable_families.apply(&cost.0, cost.1));
    let mut nu = params.initial_lm_nu;

    let mut mse = calc_mse(init_costs);
    println!("mse: {:?}", mse);

    for _i in 0..params.num_iter {
        println!("nu: {:?}", nu);

        let mut evaluated_costs: Vec<GenEvalCost> = Vec::new();
        evaluated_costs.push(variable_families.apply(&cost.0, cost.1));
        let mut normal_eq = NewSparseNormalEquation::from_families_and_cost(
            &variable_families,
            evaluated_costs,
            nu,
        );
        let delta = normal_eq.solve();
        let updated_families = variable_families.update(delta);

        let mut new_costs: Vec<GenEvalCost> = Vec::new();
        new_costs.push(updated_families.apply(&cost.0, cost.1));
        let new_mse = calc_mse(new_costs);

        println!("new_mse: {:?}", new_mse);

        if new_mse < mse {
            nu *= 0.0333;
            variable_families = updated_families;
            mse = new_mse;
        } else {
            nu *= 2.0;
        }
    }

    variable_families
}
