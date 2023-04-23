use std::fmt::Debug;

trait SimpleElem: Debug {}

impl SimpleElem for usize {}

impl SimpleElem for char {}

pub trait SimpleTuple<T>: Debug {
    const N: usize;
}

impl<T: SimpleElem> SimpleTuple<T> for () {
    const N: usize = 0;
}

impl<T: SimpleElem> SimpleTuple<T> for T {
    const N: usize = 1;
}

impl<T: SimpleElem, Tail> SimpleTuple<T> for (T, Tail)
where
    Tail: SimpleTuple<T>,
{
    const N: usize = 1 + Tail::N;
}

pub trait Manifold: std::fmt::Debug {
    const DOF: usize;
}

impl Manifold for f64 {
    const DOF: usize = 1;
}
impl Manifold for nalgebra::Vector1<f64> {
    const DOF: usize = 1;
}
impl Manifold for nalgebra::Vector2<f64> {
    const DOF: usize = 2;
}

impl Manifold for nalgebra::Vector3<f64> {
    const DOF: usize = 3;
}

impl Manifold for nalgebra::Vector6<f64> {
    const DOF: usize = 6;
}

#[derive(Copy, Clone, Debug)]
pub struct CostTermArg<T, const ET: char> {
    pub arg: T,
}

pub trait CostArg: std::fmt::Debug {
    const ENTITY_TYPE: char;
    const DOF: usize;
}

impl<const ET: char> CostArg for CostTermArg<f64, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 1;
}

impl<const ET: char> CostArg for CostTermArg<nalgebra::Vector1<f64>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 1;
}

impl<const ET: char> CostArg for CostTermArg<nalgebra::Vector2<f64>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 2;
}

impl<const ET: char> CostArg for CostTermArg<nalgebra::Vector3<f64>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 3;
}

impl<const ET: char> CostArg for CostTermArg<nalgebra::Vector6<f64>, ET> {
    const ENTITY_TYPE: char = ET;
    const DOF: usize = 6;
}

pub trait CostArgTuple: std::fmt::Debug {
    const N: usize;

    fn populate_dims(arr: &mut [usize], i: usize);

    fn get_dims<const M: usize>() -> [usize; M];
}

// Now we have to implement trait for an empty tuple,
// thus defining initial condition.
impl CostArgTuple for () {
    const N: usize = 0;

    fn populate_dims(arr: &mut [usize], i: usize) {
        assert!(arr.len() == i);
    }

    fn get_dims<const M: usize>() -> [usize; M] {
        [0; M]
    }
}

// Now we can implement trait for a non-empty tuple list,
// thus defining recursion and supporting tuple lists of arbitrary length.
impl<Head, Tail> CostArgTuple for (Head, Tail)
where
    Head: CostArg,
    Tail: CostArgTuple,
{
    const N: usize = 1 + Tail::N;

    fn populate_dims(arr: &mut [usize], i: usize) {
        arr[i] = if Head::ENTITY_TYPE == 'v' {
            Head::DOF
        } else {
            0
        };
        Tail::populate_dims(arr, i + 1);
    }

    fn get_dims<const M: usize>() -> [usize; M] {
        let mut arr = [0; M];
        Self::populate_dims(&mut arr, 0);
        arr
    }
}

pub struct CostFnArg<'a, T, const AT: char> {
    v: &'a Vec<T>,
}

impl<'a, T: Manifold, const AT: char> CostFnArg<'a, T, AT> {}

impl<'a, T: Manifold> CostFnArg<'a, T, 'v'> {
    pub fn var(v: &'a Vec<T>) -> CostFnArg<'a, T, 'v'> {
        CostFnArg::<'a, T, 'v'> { v }
    }
}

impl<'a, T: Manifold> CostFnArg<'a, T, 'c'> {
    pub fn cond(v: &'a Vec<T>) -> CostFnArg<'a, T, 'c'> {
        CostFnArg::<'a, T, 'c'> { v }
    }
}

pub trait ManifoldV {
    type Arg: CostArg;
    const CAT: char;

    fn get_elem(&self, idx: usize) -> Self::Arg;
}

impl<'a, const AT: char> ManifoldV for CostFnArg<'a, nalgebra::Vector1<f64>, AT> {
    type Arg = CostTermArg<nalgebra::Vector1<f64>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg { arg: self.v[idx] }
    }

    const CAT: char = AT;
}
impl<'a, const AT: char> ManifoldV for CostFnArg<'a, nalgebra::Vector2<f64>, AT> {
    type Arg = CostTermArg<nalgebra::Vector2<f64>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg { arg: self.v[idx] }
    }
    const CAT: char = AT;
}

impl<'a, const AT: char> ManifoldV for CostFnArg<'a, nalgebra::Vector3<f64>, AT> {
    type Arg = CostTermArg<nalgebra::Vector3<f64>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg { arg: self.v[idx] }
    }
    const CAT: char = AT;
}

impl<'a, const AT: char> ManifoldV for CostFnArg<'a, nalgebra::Vector6<f64>, AT> {
    type Arg = CostTermArg<nalgebra::Vector6<f64>, AT>;

    fn get_elem(&self, idx: usize) -> Self::Arg {
        CostTermArg { arg: self.v[idx] }
    }
    const CAT: char = AT;
}

pub trait ManifoldVTuple {
    const N: usize;
    type ArgTuple: CostArgTuple;
    type EntityIndexTuple: SimpleTuple<usize>;

    type C: SimpleTuple<char>;

    //  fn get_dofs() -> Self::EntityIndexTuple;

    fn get_elem(&self, idx: &Self::EntityIndexTuple) -> Self::ArgTuple;

    fn get_var_at() -> Self::C;
}

// Now we have to implement trait for an empty tuple,
// thus defining initial condition.
impl ManifoldVTuple for () {
    //fn plus_one(&mut self) {}
    const N: usize = 0;

    type ArgTuple = ();
    type EntityIndexTuple = ();
    type C = ();

    //fn get_dofs() -> () {}

    fn get_elem(&self, _idx: &()) {}

    fn get_var_at() {}
}

// Now we can implement trait for a non-empty tuple list,
// thus defining recursion and supporting tuple lists of arbitrary length.
impl<Head, Tail> ManifoldVTuple for (Head, Tail)
where
    Head: ManifoldV,
    Tail: ManifoldVTuple,
{
    const N: usize = 1 + Tail::N;
    type ArgTuple = (Head::Arg, Tail::ArgTuple);
    type EntityIndexTuple = (usize, Tail::EntityIndexTuple);
    type C = (char, Tail::C);

    // fn get_dofs() -> Self::EntityIndexTuple {
    //     (Head::Arg::DOF, Tail::get_dofs())
    // }

    fn get_elem(&self, idx: &(usize, Tail::EntityIndexTuple)) -> Self::ArgTuple {
        (self.0.get_elem(idx.0), self.1.get_elem(&idx.1))
    }

    fn get_var_at() -> Self::C {
        (Head::CAT, Tail::get_var_at())
    }
}

// struct TermSig<const L: usize> {
//     idx_tuple: [i64; L],
// }

#[derive(Copy, Clone, Debug)]
pub struct BlockVector<const MAX_DIM: usize, const NUM_BLOCKS: usize> {
    pub vec: nalgebra::SVector<f64, MAX_DIM>,
    pub indices: [i64; NUM_BLOCKS],
    pub dims: [usize; NUM_BLOCKS],
}

impl<const MAX_DIM: usize, const NUM_BLOCKS: usize> BlockVector<MAX_DIM, NUM_BLOCKS> {
    pub fn block(
        &self,
        i: usize,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Dyn,
        nalgebra::Const<1>,
        nalgebra::ViewStorage<
            '_,
            f64,
            nalgebra::Dyn,
            nalgebra::Const<1>,
            nalgebra::Const<1>,
            nalgebra::Const<MAX_DIM>,
        >,
    > {
        let idx = self.indices[i] as usize;
        self.vec.index((idx..idx + self.dims[i], ..))
    }

    pub fn mut_block(
        &mut self,
        i: usize,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Dyn,
        nalgebra::Const<1>,
        nalgebra::ViewStorageMut<
            '_,
            f64,
            nalgebra::Dyn,
            nalgebra::Const<1>,
            nalgebra::Const<1>,
            nalgebra::Const<MAX_DIM>,
        >,
    > {
        let idx = self.indices[i] as usize;
        self.vec.index_mut((idx..idx + self.dims[i], ..))
    }
}

impl<const MAX_DIM: usize, const NUM_BLOCKS: usize> BlockVector<MAX_DIM, NUM_BLOCKS> {
    fn new(dims: &[usize; NUM_BLOCKS]) -> Self {
        let mut idx = [0; NUM_BLOCKS];
        for i in 1..NUM_BLOCKS {
            idx[i] = idx[i - 1] + dims[i - 1] as i64;
        }
        Self {
            vec: nalgebra::SMatrix::repeat(0.0),
            indices: idx,
            dims: *dims,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BlockMatrix<const MAX_DIM: usize, const NUM_BLOCKS: usize> {
    pub mat: nalgebra::SMatrix<f64, MAX_DIM, MAX_DIM>,
    pub indices: [i64; NUM_BLOCKS],
    pub dims: [usize; NUM_BLOCKS],
}

impl<const MAX_DIM: usize, const NUM_BLOCKS: usize> BlockMatrix<MAX_DIM, NUM_BLOCKS> {
    pub fn block(
        &self,
        i: usize,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Dyn,
        nalgebra::Dyn,
        nalgebra::ViewStorage<
            '_,
            f64,
            nalgebra::Dyn,
            nalgebra::Dyn,
            nalgebra::Const<1>,
            nalgebra::Const<MAX_DIM>,
        >,
    > {
        let idx = self.indices[i] as usize;
        self.mat
            .index((idx..idx + self.dims[i], idx..idx + self.dims[i]))
    }

    pub fn mut_block(
        &mut self,
        i: usize,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Dyn,
        nalgebra::Dyn,
        nalgebra::ViewStorageMut<
            '_,
            f64,
            nalgebra::Dyn,
            nalgebra::Dyn,
            nalgebra::Const<1>,
            nalgebra::Const<MAX_DIM>,
        >,
    > {
        let idx = self.indices[i] as usize;
        self.mat
            .index_mut((idx..idx + self.dims[i], idx..idx + self.dims[i]))
    }
}

impl<const MAX_DIM: usize, const NUM_BLOCKS: usize> BlockMatrix<MAX_DIM, NUM_BLOCKS> {
    fn new(dims: &[usize; NUM_BLOCKS]) -> Self {
        let mut idx = [0; NUM_BLOCKS];
        for i in 1..NUM_BLOCKS {
            idx[i] = idx[i - 1] + dims[i - 1] as i64;
        }
        Self {
            mat: nalgebra::SMatrix::repeat(0.0),
            indices: idx,
            dims: *dims,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CostTerm<const MAX_DIM: usize, const NUM_BLOCKS: usize> {
    pub hessian: BlockMatrix<MAX_DIM, NUM_BLOCKS>,
    pub gradient: BlockVector<MAX_DIM, NUM_BLOCKS>,
    pub cost: f64,
}

impl<const MAX_DIM: usize, const NUM_BLOCKS: usize> CostTerm<MAX_DIM, NUM_BLOCKS> {
    pub fn new(dims: [usize; NUM_BLOCKS]) -> Self {
        Self {
            hessian: BlockMatrix::new(&dims),
            gradient: BlockVector::new(&dims),
            cost: 0.0,
        }
    }
}

pub trait ResidualFn<const MAX_DIM: usize, const NUM_BLOCKS: usize>: Copy {
    type Args: CostArgTuple;
    type Constants;

    fn cost(args: &Self::Args, constants: &Self::Constants) -> CostTerm<MAX_DIM, NUM_BLOCKS>;
}

pub trait CostTermSignature<const N: usize> {
    type Constants;
    type EntityIndexTuple: SimpleTuple<usize>;

    const DOF_TUPLE: [i64; N];

    fn c_ref(&self) -> &Self::Constants;

    fn idx_ref(&self) -> &Self::EntityIndexTuple;
}

pub fn apply<R, I, M, C, CC, AA: Debug, const NNN: usize, const NN: usize, const N: usize>(
    //cost_terms: Vec<TermSig<L>>,
    _res_fn: R,
    cost_terms: &Vec<C>,
    families: &M,
) where
    R: ResidualFn<NNN, NN, Constants = CC, Args = AA>,
    I: SimpleTuple<usize>,
    M: ManifoldVTuple<EntityIndexTuple = I, ArgTuple = AA>,
    C: CostTermSignature<N, EntityIndexTuple = I, Constants = CC>,
{
    println!("{:?}", M::get_var_at());
    //println!("{:?}", M::get_dofs());

    for t in cost_terms {
        println!("{:?}", families.get_elem(t.idx_ref()));

        R::cost(&families.get_elem(t.idx_ref()), t.c_ref());
    }
}
