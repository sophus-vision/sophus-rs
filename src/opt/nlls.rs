use dyn_clone::DynClone;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::AddAssign;
use std::ops::Deref;

use crate::calculus::types::M;
use crate::calculus::types::V;
use crate::calculus::types::params::HasParams;
use crate::lie::rotation2::Isometry2;
use crate::lie::rotation3::Isometry3;

use super::block::BlockVector;
use super::block::NewBlockMatrix;

pub trait IsTermSignature<const N: usize> {
    type Constants;

    const DOF_TUPLE: [i64; N];

    fn c_ref(&self) -> &Self::Constants;

    fn idx_ref(&self) -> &[usize; N];
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum VarKind {
    Free,
    Conditioned,
    Marginalized,
}

pub trait IsVariable: Clone + Debug {
    const DOF: usize;
    type Arg;

    fn update(&mut self, delta: &[f64]);
}

pub trait IsVarTuple<const NUM_ARGS: usize> {
    const DOF_T: [usize; NUM_ARGS];
    type VarFamilyTupleRef<'a>;

    fn ref_var_family_tuple(
        families: &VarPool,
        names: [String; NUM_ARGS],
    ) -> Self::VarFamilyTupleRef<'_>;

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; NUM_ARGS]) -> Self;

    fn var_kind_array(families: &VarPool, names: [String; NUM_ARGS]) -> [VarKind; NUM_ARGS];
}

impl<M0: IsVariable + 'static> IsVarTuple<1> for M0 {
    const DOF_T: [usize; 1] = [M0::DOF];
    type VarFamilyTupleRef<'a> = &'a VarFamily<M0>;

    fn var_kind_array(families: &VarPool, names: [String; 1]) -> [VarKind; 1] {
        [families.families.get(&names[0]).unwrap().get_var_kind()]
    }

    fn ref_var_family_tuple(families: &VarPool, names: [String; 1]) -> Self::VarFamilyTupleRef<'_> {
        families.get::<VarFamily<M0>>(names[0].clone())
    }

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; 1]) -> Self {
        family_tuple.members[ids[0]].clone()
    }
}

impl<M0: IsVariable + 'static, M1: IsVariable + 'static> IsVarTuple<2> for (M0, M1) {
    const DOF_T: [usize; 2] = [M0::DOF, M1::DOF];
    type VarFamilyTupleRef<'a> = (&'a VarFamily<M0>, &'a VarFamily<M1>);

    fn var_kind_array(families: &VarPool, names: [String; 2]) -> [VarKind; 2] {
        [
            families.families.get(&names[0]).unwrap().get_var_kind(),
            families.families.get(&names[1]).unwrap().get_var_kind(),
        ]
    }

    fn ref_var_family_tuple(families: &VarPool, names: [String; 2]) -> Self::VarFamilyTupleRef<'_> {
        (
            families.get::<VarFamily<M0>>(names[0].clone()),
            families.get::<VarFamily<M1>>(names[1].clone()),
        )
    }

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; 2]) -> Self {
        (
            family_tuple.0.members[ids[0]].clone(),
            family_tuple.1.members[ids[1]].clone(),
        )
    }
}

impl<M0: IsVariable + 'static, M1: IsVariable + 'static, M2: IsVariable + 'static> IsVarTuple<3>
    for (M0, M1, M2)
{
    const DOF_T: [usize; 3] = [M0::DOF, M1::DOF, M2::DOF];
    type VarFamilyTupleRef<'a> = (&'a VarFamily<M0>, &'a VarFamily<M1>, &'a VarFamily<M2>);

    fn var_kind_array(families: &VarPool, names: [String; 3]) -> [VarKind; 3] {
        [
            families.families.get(&names[0]).unwrap().get_var_kind(),
            families.families.get(&names[1]).unwrap().get_var_kind(),
            families.families.get(&names[2]).unwrap().get_var_kind(),
        ]
    }

    fn ref_var_family_tuple(families: &VarPool, names: [String; 3]) -> Self::VarFamilyTupleRef<'_> {
        (
            families.get::<VarFamily<M0>>(names[0].clone()),
            families.get::<VarFamily<M1>>(names[1].clone()),
            families.get::<VarFamily<M2>>(names[2].clone()),
        )
    }

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; 3]) -> Self {
        (
            family_tuple.0.members[ids[0]].clone(),
            family_tuple.1.members[ids[1]].clone(),
            family_tuple.2.members[ids[2]].clone(),
        )
    }
}

impl<const N: usize> IsVariable for V<N> {
    const DOF: usize = N;

    fn update(&mut self, delta: &[f64]) {
        for d in 0..Self::DOF {
            self[d] += delta[d];
        }
    }

    type Arg = V<N>;
}

impl IsVariable for Isometry2<f64> {
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

impl IsVariable for Isometry3<f64> {
    const DOF: usize = 6;

    fn update(&mut self, delta: &[f64]) {
        let mut delta_vec = V::<6>::zeros();
        for d in 0..Self::DOF {
            delta_vec[d] = delta[d];
        }
        self.set_params(
            (Isometry3::<f64>::group_mul(&Isometry3::<f64>::exp(&delta_vec), &self.clone()))
                .params(),
        );
    }

    type Arg = Isometry3<f64>;
}

#[derive(Debug, Clone)]
pub struct VarFamily<Var: IsVariable> {
    kind: VarKind,
    pub members: Vec<Var>,
    constant_members: HashMap<usize, ()>,
    start_indices: Vec<i64>,
}

impl<Var: IsVariable> VarFamily<Var> {
    pub fn new(kind: VarKind, members: Vec<Var>, constant_members: HashMap<usize, ()>) -> Self {
        VarFamily {
            kind,
            members,
            constant_members,
            start_indices: vec![],
        }
    }

    fn c(&self) -> VarKind {
        self.kind
    }
}

pub trait IsVarFamily: as_any::AsAny + Debug + DynClone {
    fn update(&mut self, delta: &[f64]);
    fn num_scalars(&self) -> usize;
    fn calc_start_indices(&mut self, offset: &mut usize);
    fn get_start_indices(&self) -> &Vec<i64>;

    fn dof(&self) -> usize;
    fn get_var_kind(&self) -> VarKind;
}

impl<Var: IsVariable + 'static> IsVarFamily for VarFamily<Var> {
    fn update(&mut self, delta: &[f64]) {
        let dof = self.dof();

        assert_eq!(self.start_indices.len(), self.members.len());

        for i in 0..self.start_indices.len() {
            let start_idx = self.start_indices[i];
            if start_idx == -1 {
                continue;
            }
            let start_idx = start_idx as usize;
            self.members[i].update(&delta[start_idx..start_idx + dof]);
        }
    }

    fn num_scalars(&self) -> usize {
        (self.members.len() - self.constant_members.len()) * Var::DOF
    }

    // returns -1 if variable is not free
    fn calc_start_indices(&mut self, inout_offset: &mut usize) {
        assert_eq!(
            self.start_indices.len(),
            0,
            "Ths function must ony called once"
        );

        let mut indices = vec![];
        let mut idx: usize = *inout_offset;
        for i in 0..self.members.len() {
            if self.constant_members.contains_key(&i) {
                indices.push(-1);
            } else {
                indices.push(idx as i64);
                idx += Var::DOF;
            }
        }
        *inout_offset = idx;

        assert_eq!(indices.len(), self.members.len());
        self.start_indices = indices;
    }

    fn dof(&self) -> usize {
        Var::DOF
    }

    fn get_var_kind(&self) -> VarKind {
        self.c()
    }

    fn get_start_indices(&self) -> &Vec<i64> {
        &self.start_indices
    }
}

dyn_clone::clone_trait_object!(IsVarFamily);

#[derive(Debug, Clone)]
pub struct VarPoolBuilder {
    families: std::collections::BTreeMap<String, Box<dyn IsVarFamily>>,
}

impl VarPoolBuilder {
    pub fn new() -> Self {
        Self {
            families: BTreeMap::new(),
        }
    }

    pub fn add_family<S: Into<String>, Var: IsVariable + 'static>(
        mut self,
        name: S,
        family: VarFamily<Var>,
    ) -> Self {
        self.families.insert(name.into(), Box::new(family));
        self
    }

    pub fn build(self) -> VarPool {
        VarPool::new(self)
    }
}

#[derive(Debug, Clone)]
pub struct VarPool {
    families: std::collections::BTreeMap<String, Box<dyn IsVarFamily>>,
}

impl VarPool {
    fn new(mut builder: VarPoolBuilder) -> Self {
        let mut offset = 0;
        for (_name, family) in builder.families.iter_mut() {
            family.calc_start_indices(&mut offset);
        }

        Self {
            families: builder.families,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvaluatedCost<const NUM: usize, const NUM_ARGS: usize> {
    pub family_names: Vec<String>,
    pub terms: Vec<EvaluatedTerm<NUM, NUM_ARGS>>,
}

impl<const NUM: usize, const NUM_ARGS: usize> EvaluatedCost<NUM, NUM_ARGS> {
    fn new(family_names: Vec<String>) -> Self {
        EvaluatedCost {
            family_names,
            terms: Vec::new(),
        }
    }
}

fn c_from_var_kind<const N: usize>(var_kind_array: &[VarKind; N]) -> [char; N] {
    let mut c_array: [char; N] = ['0'; N];

    for i in 0..N {
        c_array[i] = match var_kind_array[i] {
            VarKind::Free => 'f',
            VarKind::Conditioned => 'c',
            VarKind::Marginalized => 'm',
        };
    }

    c_array
}

impl VarPool {
    pub fn num_free_params(&self) -> usize {
        let mut num = 0;

        for f in self.families.iter() {
            num += f.1.num_scalars();
        }

        num
    }

    fn update(&self, delta: Vec<f64>) -> VarPool {
        let mut updated = self.clone();
        for family in updated.families.iter_mut() {
            family.1.update(&delta[..]);
        }
        updated
    }

    pub fn get<T: IsVarFamily>(&self, name: String) -> &T {
        as_any::Downcast::downcast_ref::<T>(self.families.get(&name).unwrap().as_ref()).unwrap()
    }

    pub fn get_members<T: IsVariable + 'static>(&self, name: String) -> Vec<T> {
        as_any::Downcast::downcast_ref::<VarFamily<T>>(self.families.get(&name).unwrap().as_ref())
            .unwrap()
            .members
            .clone()
    }

    fn apply<
        const NUM: usize,
        const NUM_ARGS: usize,
        ResidualFn,
        Constants,
        TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants>,
        VarTuple: IsVarTuple<NUM_ARGS> + 'static,
    >(
        &self,
        cost: &Cost<NUM, NUM_ARGS, Constants, TermSignature, ResidualFn, VarTuple>,
        calc_derivatives: bool,
    ) -> EvaluatedCost<NUM, NUM_ARGS>
    where
        ResidualFn: IsResidualFn<NUM, NUM_ARGS, VarTuple, Constants>,
    {
        use crate::opt::cost_args::CompareIdx;
        let mut var_kind_array = VarTuple::var_kind_array(self, cost.signature.family_names.clone());
        let c_array = c_from_var_kind(&var_kind_array);

        if !calc_derivatives {
            var_kind_array = var_kind_array.map(|_x| VarKind::Conditioned)
        }
        let less = CompareIdx { c: c_array };

        let mut evaluated_terms = EvaluatedCost::new(cost.signature.family_names.clone().into());

        let mut i = 0;

        let var_family_tuple =
            VarTuple::ref_var_family_tuple(self, cost.signature.family_names.clone());

        let eval_res = |term_signature: &TermSignature| {
            cost.residual_fn.eval(
                VarTuple::extract(&var_family_tuple, *term_signature.idx_ref()),
                var_kind_array,
                term_signature.c_ref(),
            )
        };

        evaluated_terms.terms.reserve(cost.signature.terms.len());

        while i < cost.signature.terms.len() {
            let term_signature = &cost.signature.terms[i];

            let outer_idx = term_signature.idx_ref();

            let mut evaluated_term = eval_res(term_signature);
            evaluated_term.idx.push(*term_signature.idx_ref());

            i += 1;

            // perform reduction over conditioned variables
            while i < cost.signature.terms.len() {
                let inner_term_signature = &cost.signature.terms[i];

                if !less.are_all_free_cars_equal(outer_idx, inner_term_signature.idx_ref()) {
                    // end condition for reduction over conditioned variables
                    break;
                }

                i += 1;

                let inner_evaluated_term = eval_res(inner_term_signature);

                evaluated_term.hessian.mat += inner_evaluated_term.hessian.mat;
                evaluated_term.gradient.vec += inner_evaluated_term.gradient.vec;
                evaluated_term.cost += inner_evaluated_term.cost;
            }

            evaluated_terms.terms.push(evaluated_term);
        }

        evaluated_terms
    }
}

#[derive(Debug, Clone)]
pub struct EvaluatedTerm<const NUM: usize, const NUM_ARGS: usize> {
    pub hessian: NewBlockMatrix<NUM>,
    pub gradient: BlockVector<NUM>,
    pub cost: f64,
    pub idx: Vec<[usize; NUM_ARGS]>,
}

impl<const NUM: usize, const NUM_ARGS: usize> EvaluatedTerm<NUM, NUM_ARGS> {
    pub fn new1<const D0: usize, const R: usize>(
        maybe_dx0: Option<M<R, D0>>,
        residual: V<R>,
    ) -> Self {
        let dims = vec![D0];
        let mut hessian = NewBlockMatrix::new(&dims);
        let mut gradient = BlockVector::new(&dims);

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
        let mut gradient = BlockVector::new(&dims);

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
                hessian.set_block(0, 1, h01);
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
        maybe_dx0: Option<M<R, D0>>,
        maybe_dx1: Option<M<R, D1>>,
        maybe_dx2: Option<M<R, D2>>,
        residual: V<R>,
    ) -> Self {
        let dims = vec![D0, D1, D2];
        let mut hessian = NewBlockMatrix::new(&dims);
        let mut gradient = BlockVector::new(&dims);

        if maybe_dx0.is_some() {
            let dx0: M<R, D0> = maybe_dx0.unwrap();
            let dx0_t: M<D0, R> = dx0.transpose();

            let grad0 = dx0_t * residual;
            gradient.set_block(0, grad0);
            hessian.set_block(0, 0, dx0_t * dx0);
        }

        if maybe_dx1.is_some() {
            let dx1: M<R, D1> = maybe_dx1.unwrap();
            let dx1_t: M<D1, R> = dx1.transpose();

            let grad1 = dx1_t * residual;
            gradient.set_block(1, grad1);

            let h11 = dx1_t * dx1;
            hessian.set_block(1, 1, h11);

            // off-diagonal 01
            if maybe_dx0.is_some() {
                let dx0: M<R, D0> = maybe_dx0.unwrap();
                let dx0_t: M<D0, R> = dx0.transpose();

                hessian.set_block(0, 1, dx0_t * dx1);
            }
        }

        if maybe_dx2.is_some() {
            let dx2: M<R, D2> = maybe_dx2.unwrap();
            let dx2_t: M<D2, R> = dx2.transpose();

            let grad2 = dx2_t * residual;
            gradient.set_block(2, grad2);

            hessian.set_block(2, 2, dx2_t * dx2);

            // off-diagonal 02
            if maybe_dx0.is_some() {
                let dx0: M<R, D0> = maybe_dx0.unwrap();
                let dx0_t: M<D0, R> = dx0.transpose();

                hessian.set_block(0, 2, dx0_t * dx2);
            }

            // off-diagonal 12
            if maybe_dx1.is_some() {
                let dx1: M<R, D1> = maybe_dx1.unwrap();
                let dx1_t: M<D1, R> = dx1.transpose();

                hessian.set_block(1, 2, dx1_t * dx2);
            }
        }

        Self {
            hessian,
            gradient,
            cost: residual.norm(),
            idx: Vec::new(),
        }
    }
}

pub trait IsResidualFn<
    const NUM: usize,
    const NUM_ARGS: usize,
    Args: IsVarTuple<NUM_ARGS>,
    Constants,
>: Copy
{
    fn eval(
        &self,
        args: Args,
        derivatives: [VarKind; NUM_ARGS],
        constants: &Constants,
    ) -> EvaluatedTerm<NUM, NUM_ARGS>;
}

#[derive(Debug, Clone)]
pub struct CostSignature<
    const NUM_ARGS: usize,
    Constants,
    TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants>,
> {
    pub family_names: [String; NUM_ARGS],
    pub terms: Vec<TermSignature>,
}

#[derive(Debug, Clone)]
pub struct Cost<
    const NUM: usize,
    const NUM_ARGS: usize,
    Constants,
    TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants>,
    ResidualFn,
    VarTuple: IsVarTuple<NUM_ARGS> + 'static,
> where
    ResidualFn: IsResidualFn<NUM, NUM_ARGS, VarTuple, Constants>,
{
    signature: CostSignature<NUM_ARGS, Constants, TermSignature>,
    residual_fn: ResidualFn,
    phantom: PhantomData<VarTuple>,
}

impl<
        const NUM: usize,
        const NUM_ARGS: usize,
        Constants,
        TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants>,
        ResidualFn,
        VarTuple: IsVarTuple<NUM_ARGS> + 'static,
    > Cost<NUM, NUM_ARGS, Constants, TermSignature, ResidualFn, VarTuple>
where
    ResidualFn: IsResidualFn<NUM, NUM_ARGS, VarTuple, Constants>,
{
    pub fn new(
        signature: CostSignature<NUM_ARGS, Constants, TermSignature>,
        residual_fn: ResidualFn,
    ) -> Self {
        Self {
            signature,
            residual_fn,
            phantom: PhantomData::default(),
        }
    }

    fn sort(&mut self, variables: &VarPool) {
        let var_kind_array =
            &VarTuple::var_kind_array(&variables, self.signature.family_names.clone());
        use crate::opt::cost_args::CompareIdx;

        let c_array = c_from_var_kind(var_kind_array);

        let less = CompareIdx { c: c_array };

        assert!(!self.signature.terms.is_empty());

        self.signature
            .terms
            .sort_by(|a, b| less.le_than(*a.idx_ref(), *b.idx_ref()));

        for t in 0..self.signature.terms.len() - 1 {
            assert!(
                less.le_than(
                    *self.signature.terms[t].idx_ref(),
                    *self.signature.terms[t + 1].idx_ref()
                ) == std::cmp::Ordering::Less
            );
        }
    }
}
pub struct SparseNormalEquation {
    sparse_hessian: sprs::CsMat<f64>,
    neg_gradient: nalgebra::DVector<f64>,
}

impl SparseNormalEquation {
    fn from_families_and_cost<const NUM: usize, const NUM_ARGS: usize>(
        variables: &VarPool,
        costs: Vec<EvaluatedCost<NUM, NUM_ARGS>>,
        nu: f64,
    ) -> SparseNormalEquation {
        let num_var_params = variables.num_free_params();
        let mut hessian_triplet = sprs::TriMat::new((num_var_params, num_var_params));
        let mut neg_grad = nalgebra::DVector::<f64>::zeros(num_var_params);

        let mut start_indices_per_arg = Vec::new();

        assert_eq!(costs.len(), 1);
        for evaluated_cost in costs.iter() {
            let num_args = evaluated_cost.family_names.len();

            let mut dof_per_arg = Vec::new();
            for name in evaluated_cost.family_names.iter() {
                let family = variables.families.get(name).unwrap();
                start_indices_per_arg.push(family.get_start_indices());
                dof_per_arg.push(family.dof());
            }

            for evaluated_term in evaluated_cost.terms.iter() {
                assert_eq!(evaluated_term.idx.len(), 1);
                let idx = evaluated_term.idx[0].clone();
                assert_eq!(idx.len(), num_args);

                for arg_id_alpha in 0..num_args {
                    let dof_alpha = dof_per_arg[arg_id_alpha];

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
                            hessian_triplet.add_triplet(
                                start_idx_alpha + r,
                                start_idx_alpha + c,
                                hessian_block[(r, c)] + d,
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
                                hessian_triplet.add_triplet(
                                    start_idx_alpha + r,
                                    start_idx_beta + c,
                                    hessian_block_alpha_beta[(r, c)],
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

    pub fn is_symmetric<N, I, Iptr, IpStorage, IStorage, DStorage>(
        mat: &sprs::CsMatBase<N, I, IpStorage, IStorage, DStorage, Iptr>,
    ) -> bool
    where
        N: PartialEq + std::fmt::Display,
        I: sprs::SpIndex,
        Iptr: sprs::SpIndex,
        IpStorage: Deref<Target = [Iptr]>,
        IStorage: Deref<Target = [I]>,
        DStorage: Deref<Target = [N]>,
    {
        if mat.rows() != mat.cols() {
            return false;
        }
        for (outer_ind, vec) in mat.outer_iterator().enumerate() {
            for (inner_ind, value) in vec.iter() {
                match mat.get_outer_inner(inner_ind, outer_ind) {
                    None => return false,
                    Some(transposed_val) => {
                        if transposed_val != value {
                            println!("{} != {}", transposed_val, value);
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    fn solve(&mut self) -> Vec<f64> {
        Self::is_symmetric(&self.sparse_hessian.view());

        let ldl = sprs_ldl::Ldl::new().check_symmetry(sprs::SymmetryCheck::DontCheckSymmetry);
        let ldl_num = ldl.numeric(self.sparse_hessian.view()).unwrap();

        ldl_num.solve(
            self.neg_gradient
                .iter()
                .map(|x: &f64| *x)
                .collect::<Vec<f64>>(),
        )
    }
}

fn calc_square_error<const NUM: usize, const NUM_ARGS: usize>(
    cost: Vec<EvaluatedCost<NUM, NUM_ARGS>>,
) -> f64 {
    let mut c = 0.0;
    for g in cost {
        for eval_term in g.terms {
            c += eval_term.cost;
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
pub fn optimize_one_cost<
    const NUM: usize,
    const NUM_ARGS: usize,
    VarTuple: IsVarTuple<NUM_ARGS> + 'static,
    Constants,
    TermSignature: IsTermSignature<NUM_ARGS, Constants = Constants>,
    ResidualFn,
>(
    mut variables: VarPool,
    mut cost: Cost<NUM, NUM_ARGS, Constants, TermSignature, ResidualFn, VarTuple>,
    params: OptParams,
) -> VarPool
where
    ResidualFn: IsResidualFn<NUM, NUM_ARGS, VarTuple, Constants>,
{
    cost.sort(&variables);
    let mut init_costs: Vec<EvaluatedCost<NUM, NUM_ARGS>> = Vec::new();
    
    init_costs.push(variables.apply(&cost, false));
    let mut nu = params.initial_lm_nu;

    let mut mse = calc_square_error(init_costs);
    println!("e^2: {:?}", mse);

    for _i in 0..params.num_iter {
        use std::time::Instant;
        let now = Instant::now();
        println!("nu: {:?}", nu);

        let mut evaluated_costs: Vec<EvaluatedCost<NUM, NUM_ARGS>> = Vec::new();
        evaluated_costs.push(variables.apply(&cost, true));
        println!("evaluate costs: {:.2?}", now.elapsed());
        let now = Instant::now();

        let mut normal_eq =
            SparseNormalEquation::from_families_and_cost(&variables, evaluated_costs, nu);
        println!("build normal eq {:.2?}", now.elapsed());
        let now = Instant::now();

        let delta = normal_eq.solve();
        println!("solve {:.2?}", now.elapsed());
        let now = Instant::now();
        let updated_families = variables.update(delta);

        let mut new_costs: Vec<EvaluatedCost<NUM, NUM_ARGS>> = Vec::new();
        new_costs.push(updated_families.apply(&cost, false));
        let new_mse = calc_square_error(new_costs);
        println!("update and new cost {:.2?}", now.elapsed());

        println!("new e^2: {:?}", new_mse);

        if new_mse < mse {
            nu *= 0.0333;
            variables = updated_families;
            mse = new_mse;
        } else {
            nu *= 2.0;
        }
    }

    variables
}
