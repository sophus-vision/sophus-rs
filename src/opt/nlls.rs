use std::fmt::Debug;

use crate::opt::block::SophusAddAssign;

use super::{
    block::{BlockMatrix, BlockVector},
    cost_args::{CostArgTuple, ManifoldVTuple},
};

#[derive(Debug)]
pub struct CostTerm<const MAX_DIM: usize, const NUM_BLOCKS: usize> {
    pub hessian: BlockMatrix<NUM_BLOCKS>,
    pub gradient: BlockVector<NUM_BLOCKS>,
    pub cost: f64,
    pub idx: Vec<[usize; NUM_BLOCKS]>,
}

impl<const MAX_DIM: usize, const NUM_BLOCKS: usize> CostTerm<MAX_DIM, NUM_BLOCKS> {
    pub fn new(dims: [usize; NUM_BLOCKS]) -> Self {
        Self {
            hessian: BlockMatrix::new(&dims),
            gradient: BlockVector::new(&dims),
            cost: 0.0,
            idx: Vec::new(),
        }
    }
}

pub trait ResidualFn<const MAX_DIM: usize, const NUM_BLOCKS: usize>: Copy {
    type Args: CostArgTuple;
    type Constants;

    fn cost(&self, args: &Self::Args, constants: &Self::Constants)
        -> CostTerm<MAX_DIM, NUM_BLOCKS>;
}

pub trait CostTermSignature<const N: usize> {
    type Constants;
    //type EntityIndexTuple: SimpleTuple<usize>;

    const DOF_TUPLE: [i64; N];

    fn c_ref(&self) -> &Self::Constants;

    fn idx_ref(&self) -> &[usize; N];
}

pub fn sort_and_apply<R, M, C, CC, AA: Debug, const NNN: usize, const NN: usize>(
    res_fn: R,
    cost_terms: &mut Vec<C>,
    families: &M,
) -> Vec<CostTerm<NNN, NN>>
where
    R: ResidualFn<NNN, NN, Constants = CC, Args = AA>,
    M: ManifoldVTuple<Idx = [usize; NN], GetElemTReturn = AA>,
    C: CostTermSignature<NN, Constants = CC>,
    <M as ManifoldVTuple>::CatArray: AsRef<[char]>,
{
    use crate::opt::cost_args::CompareIdx;
    let less = CompareIdx { c: M::CAT };

    assert!(cost_terms.len() > 0);

    cost_terms.sort_by(|a, b| less.less_than(*a.idx_ref(), *b.idx_ref()));

    apply(res_fn, cost_terms, families)
}

pub fn apply<R, M, C, CC, AA: Debug, const NNN: usize, const NN: usize>(
    res_fn: R,
    cost_terms: &Vec<C>,
    families: &M,
) -> Vec<CostTerm<NNN, NN>>
where
    R: ResidualFn<NNN, NN, Constants = CC, Args = AA>,
    M: ManifoldVTuple<Idx = [usize; NN], GetElemTReturn = AA>,
    C: CostTermSignature<NN, Constants = CC>,
    <M as ManifoldVTuple>::CatArray: AsRef<[char]>,
{
    println!("{:?}", M::CAT);
    use crate::opt::cost_args::CompareIdx;
    let less = CompareIdx { c: M::CAT };

    assert!(cost_terms.len() > 0);

    for t in 0..cost_terms.len() - 1 {
        assert!(
            less.less_than(*cost_terms[t].idx_ref(), *cost_terms[t + 1].idx_ref())
                == std::cmp::Ordering::Less
        );
    }

    let mut terms = Vec::new();

    let mut i = 0;

    while i < cost_terms.len() {
        let t = &cost_terms[i];

        let outer_idx = t.idx_ref();

        let elem = families.get_elem_t(t.idx_ref());
        let mut term = res_fn.cost(&elem, t.c_ref());
        term.idx.push(*t.idx_ref());
        i += 1; // skip first, is that right?

        // perform reduction over conditioned variables
        while i < cost_terms.len() {
            let t = &cost_terms[i];

            if !less.all_var_eq(outer_idx, t.idx_ref()) {
                break;
            }

            i += 1;

            let elem = families.get_elem_t(t.idx_ref());
            let term2 = res_fn.cost(&elem, t.c_ref());

            term.hessian.mat.add_assign(term2.hessian.mat);
            term.gradient.vec.add_assign(term2.gradient.vec);
            term.cost += term2.cost;
        }

        terms.push(term);
    }

    terms
}

pub struct SparseNormalEquation {
    sparse_hessian: sprs::CsMat<f64>,
    neg_gradient: Vec<f64>,
}

impl SparseNormalEquation {
    pub fn from_one_family<
        'a,
        M: ManifoldVTuple<
            Idx = [usize; 1],
            DofArray = [usize; 1],
            CatArray = [char; 1],
            GetElemTReturn = AA,
        >,
        const NNN: usize,
        AA: Debug,
    >(
        cost_terms: &Vec<CostTerm<NNN, 1>>,
        family: &M,
    ) -> Self {
        assert!(cost_terms.len() == 1);

        let num_var_params = family.len_t()[0] * M::DOF_T[0];

        // let cost = cost_terms[0].clone();

        // let mut total_rows = 0;
        // let mut total_cols = 0;
        // for t in cost_terms.iter() {
        //     let shape = t.hessian.mat.shape();
        //     assert_eq!(t.gradient.vec.shape().0, shape.0);

        //     total_rows += shape.0;
        //     total_cols += shape.1;
        // }

        let mut hessian_triplet = sprs::TriMat::new((num_var_params, num_var_params));
        let mut grad = Vec::with_capacity(num_var_params);

        for term in cost_terms.iter() {
            assert_eq!(term.idx.len(), 1);
            let idx = term.idx[0];
            let id = idx[0];

            let b = term.hessian.block(id, id);

            let gb = term.gradient.block(id);

            for i in 0..M::DOF_T[0] {
                println!("g: {}", gb.read(i, 0));
                grad.push(-gb.read(i, 0));
                for j in 0..M::DOF_T[0] {
                    println!("i: {}, j: {}", i, j);
                    println!("b: {}", b.read(i, j));
                    hessian_triplet.add_triplet(i, j, b.read(i, j));
                }
            }
        }

        Self {
            sparse_hessian: hessian_triplet.to_csr(),
            neg_gradient: grad,
        }
    }

    pub fn solve(&self) -> Vec<f64> {
        let ldl = sprs_ldl::LdlNumeric::new(self.sparse_hessian.view()).unwrap();
        let result = ldl.solve(self.neg_gradient.clone());

        result
    }
}

pub struct OneFamilyProblem {
    //family: Vec<M>,
}

impl OneFamilyProblem {
    pub fn optimize<R, M, C, CC, AA: Debug, const NNN: usize>(
        res_fn: R,
        mut cost_signature: Vec<C>,
        mut families: M,
    ) -> M
    where
        R: ResidualFn<NNN, 1, Constants = CC, Args = AA>,
        M: ManifoldVTuple<
            Idx = [usize; 1],
            DofArray = [usize; 1],
            CatArray = [char; 1],
            GetElemTReturn = AA,
        >,
        C: CostTermSignature<1, Constants = CC>,
        <M as ManifoldVTuple>::CatArray: AsRef<[char]>,
    {
        use crate::opt::cost_args::CompareIdx;
        let less = CompareIdx { c: M::CAT };
        assert!(cost_signature.len() > 0);
        cost_signature.sort_by(|a, b| less.less_than(*a.idx_ref(), *b.idx_ref()));

        for _i in 0..10 {
            let cost_terms = apply(res_fn, &cost_signature, &families);
            let normal_eq = SparseNormalEquation::from_one_family(&cost_terms, &families);
            let update_vec = normal_eq.solve();

            families.update_t(&update_vec);

            println!("update: {:?}", update_vec);
        }

        families
    }
}
