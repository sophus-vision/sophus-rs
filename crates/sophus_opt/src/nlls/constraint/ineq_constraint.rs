use core::ops::Range;
use std::fmt::Debug;

use sophus_autodiff::linalg::MatF64;

use crate::{
    block::BlockJacobian,
    variables::{
        IsVarTuple,
        VarKind,
    },
};

extern crate alloc;

/// Inequality constraint function.
///
/// This trait is implemented by the user to define the concrete inequality constraint.
///
/// The constraint is of the form `h(x) >= 0`, where `h` is a scalar-valued function.
///
/// ## Generic parameters
///
///  * `INPUT_DIM`
///    - Total input dimension of the constraint function `h`. It is the sum of argument dimensions:
///      |Vⁱ₀| + |Vⁱ₁| + ... + |Vⁱₙ₋₁|.
///  * `N`
///    - Number of arguments of the constraint function `h`.
///  * `GlobalConstants`
///    - Type of the global constants which are passed to the constraint function. If no global
///      constants are needed, use `()`.
///  * `Args`
///    - Tuple of input argument types: `(Vⁱ₀, Vⁱ₁, ..., Vⁱₙ₋₁)`.
pub trait HasIneqConstraintFn<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<N>,
>: Send + Sync + 'static + Debug
{
    /// Returns an array of variable indices for each argument in this constraint.
    ///
    /// For example, if `idx_ref() = [2, 7]` and `Args` is `(Foo, Bar)`, then:
    ///
    /// - Argument 0 is the 2nd variable of the `Foo` family.
    /// - Argument 1 is the 7th variable of the `Bar` family.
    fn idx_ref(&self) -> &[usize; N];

    /// Evaluate the inequality constraint, returning `h(x)` and its Jacobian.
    fn eval(
        &self,
        global_constants: &GlobalConstants,
        idx: [usize; N],
        args: Args,
        derivatives: [VarKind; N],
    ) -> EvaluatedIneqConstraint<INPUT_DIM, N>;
}

/// Evaluated inequality constraint.
///
/// Represents the scalar constraint `h(x) >= 0` together with its gradient.
///
/// ## Generic parameters
///
///  * `INPUT_DIM`
///    - Total input dimension of the constraint function `h`.
///  * `N`
///    - Number of arguments of the constraint function `h`.
#[derive(Debug, Clone)]
pub struct EvaluatedIneqConstraint<const INPUT_DIM: usize, const N: usize> {
    /// The scalar constraint value `h(x)`. Positive means feasible.
    pub h: f64,
    /// The Jacobian of `h` with respect to all variables (1-row Jacobian).
    pub jacobian: BlockJacobian<1, INPUT_DIM, N>,
    /// Array of variable indices for each argument.
    pub idx: [usize; N],
}

/// A collection of unevaluated inequality constraints sharing a common constraint function.
///
/// ## Generic parameters
///
///  * `INPUT_DIM`
///    - Total input dimension of the constraint function `h`.
///  * `N`
///    - Number of arguments.
///  * `GlobalConstants`
///    - Type of global constants passed to the constraint function.
///  * `Args`
///    - Tuple of input argument types.
///  * `Constraint`
///    - The concrete constraint function.
#[derive(Debug, Clone)]
pub struct IneqConstraints<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<N>,
    Constraint: HasIneqConstraintFn<INPUT_DIM, N, GlobalConstants, Args>,
> {
    /// Variable family name for each argument.
    pub family_names: [String; N],
    /// Collection of unevaluated constraints.
    pub collection: alloc::vec::Vec<Constraint>,
    pub(crate) reduction_ranges: Option<alloc::vec::Vec<Range<usize>>>,
    phantom: core::marker::PhantomData<(GlobalConstants, Args, Constraint)>,
}

impl<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<N>,
    Constraint: HasIneqConstraintFn<INPUT_DIM, N, GlobalConstants, Args>,
> IneqConstraints<INPUT_DIM, N, GlobalConstants, Args, Constraint>
{
    /// Create a new set of inequality constraints.
    pub fn new(family_names: [impl ToString; N], constraints: alloc::vec::Vec<Constraint>) -> Self {
        IneqConstraints {
            family_names: family_names.map(|name| name.to_string()),
            collection: constraints,
            reduction_ranges: None,
            phantom: core::marker::PhantomData,
        }
    }

    /// Sort constraints by variable indices and compute reduction ranges.
    ///
    /// Must be called before `eval`. Groups constraints that share the same
    /// free-variable index set into reduction ranges for efficient evaluation.
    pub(crate) fn sort_and_reduce(&mut self, variables: &crate::variables::VarFamilies) {
        use crate::nlls::cost::compare_idx::{
            CompareIdx,
            c_from_var_kind,
        };

        let var_kind_array = &<Args as crate::variables::IsVarTuple<N>>::var_kind_array(
            variables,
            self.family_names.clone(),
        );
        let c_array = c_from_var_kind(var_kind_array);
        let less = CompareIdx::new(&c_array);

        if self.collection.is_empty() {
            self.reduction_ranges = Some(alloc::vec![]);
            return;
        }

        self.collection
            .sort_by(|a, b| less.le_than(*a.idx_ref(), *b.idx_ref()));

        let mut reduction_ranges: alloc::vec::Vec<Range<usize>> = alloc::vec![];
        let mut i = 0;
        while i < self.collection.len() {
            let start = i;
            let anchor_idx = *self.collection[i].idx_ref();
            while i < self.collection.len()
                && less.free_vars_equal(&anchor_idx, self.collection[i].idx_ref())
            {
                i += 1;
            }
            reduction_ranges.push(start..i);
        }

        self.reduction_ranges = Some(reduction_ranges);
    }

    /// Record the block-sparsity pattern of this constraint set into a symbolic builder.
    ///
    /// Iterates over every constraint and marks which (row, col) block pairs
    /// will be written by the Hessian contribution. Called once before the first
    /// optimizer iteration to pre-build the sparse matrix pattern.
    pub(crate) fn populate_symbolic(
        &self,
        variables: &crate::variables::VarFamilies,
        sym_builder: &mut sophus_solver::matrix::block_sparse::BlockSparseSymmetricSymbolicBuilder,
    ) {
        let family_names = &self.family_names;
        let num_args = family_names.len();
        let mut scalar_start_indices_per_arg = alloc::vec::Vec::new();
        let mut block_start_indices_per_arg = alloc::vec::Vec::new();
        let mut dof_per_arg = alloc::vec::Vec::new();
        let mut arg_ids = alloc::vec::Vec::new();

        for name in family_names.iter() {
            let family = variables.collection.get(name).unwrap();
            scalar_start_indices_per_arg.push(family.get_scalar_start_indices().clone());
            block_start_indices_per_arg.push(family.get_block_start_indices().clone());
            dof_per_arg.push(family.free_or_marg_dof());
            arg_ids.push(variables.index(name).unwrap());
        }

        for constraint in self.collection.iter() {
            let idx = constraint.idx_ref();
            for arg_id_alpha in 0..num_args {
                let dof_alpha = dof_per_arg[arg_id_alpha];
                let family_alpha = arg_ids[arg_id_alpha];
                if dof_alpha == 0 {
                    continue;
                }
                let var_idx_alpha = idx[arg_id_alpha];
                let scalar_start_idx_alpha =
                    scalar_start_indices_per_arg[arg_id_alpha][var_idx_alpha];
                let block_start_idx_alpha =
                    block_start_indices_per_arg[arg_id_alpha][var_idx_alpha];
                if scalar_start_idx_alpha == -1 {
                    continue;
                }
                let block_start_idx_alpha = block_start_idx_alpha as usize;
                let idx_alpha = sophus_solver::matrix::PartitionBlockIndex {
                    partition: variables.partition_idx_by_family[family_alpha],
                    block: block_start_idx_alpha,
                };

                sym_builder.add_lower_block(idx_alpha, idx_alpha);

                for arg_id_beta in 0..num_args {
                    let family_beta = arg_ids[arg_id_beta];
                    if arg_id_alpha == arg_id_beta {
                        continue;
                    }
                    let dof_beta = dof_per_arg[arg_id_beta];
                    if dof_beta == 0 {
                        continue;
                    }
                    let var_idx_beta = idx[arg_id_beta];
                    let scalar_start_idx_beta =
                        scalar_start_indices_per_arg[arg_id_beta][var_idx_beta];
                    if scalar_start_idx_beta == -1 {
                        continue;
                    }
                    let scalar_start_idx_alpha_usize = scalar_start_idx_alpha as usize;
                    let scalar_start_idx_beta_usize = scalar_start_idx_beta as usize;
                    if scalar_start_idx_beta_usize > scalar_start_idx_alpha_usize {
                        continue;
                    }
                    let block_start_idx_beta =
                        block_start_indices_per_arg[arg_id_beta][var_idx_beta] as usize;
                    let idx_beta = sophus_solver::matrix::PartitionBlockIndex {
                        partition: variables.partition_idx_by_family[family_beta],
                        block: block_start_idx_beta,
                    };
                    sym_builder.add_lower_block(idx_alpha, idx_beta);
                }
            }
        }
    }
}

/// Helper trait for making an N-ary evaluated inequality constraint.
///
/// Analogous to [`crate::nlls::MakeEvaluatedEqConstraint`] but for inequality constraints.
pub trait MakeEvaluatedIneqConstraint<const INPUT_DIM: usize, const N: usize> {
    /// Make an inequality constraint from a scalar `h` value and its derivatives (= `self`).
    ///
    /// This function shall be called inside the
    /// [`HasIneqConstraintFn::eval`] implementation of a user-defined constraint.
    ///
    ///  * `self`
    ///    - A tuple of closures that compute the Jacobian of `h` w.r.t. each argument.
    ///  * `idx`
    ///    - Variable indices for each argument.
    ///  * `var_kinds`
    ///    - An array of `VarKind` for each argument. A Jacobian will be computed for each argument
    ///      that is not `Conditioned`.
    ///  * `h`
    ///    - The scalar constraint value `h(x)`.
    fn make_ineq(
        self,
        idx: [usize; N],
        var_kinds: [VarKind; N],
        h: f64,
    ) -> EvaluatedIneqConstraint<INPUT_DIM, N>;
}

impl<F0, const INPUT_DIM: usize> MakeEvaluatedIneqConstraint<INPUT_DIM, 1> for (F0,)
where
    F0: FnOnce() -> MatF64<1, INPUT_DIM>,
{
    fn make_ineq(
        self,
        idx: [usize; 1],
        var_kinds: [VarKind; 1],
        h: f64,
    ) -> EvaluatedIneqConstraint<INPUT_DIM, 1> {
        let mut jacobian = BlockJacobian::new(&[INPUT_DIM]);
        if var_kinds[0] != VarKind::Conditioned {
            jacobian.set_block::<INPUT_DIM>(0, self.0());
        }
        EvaluatedIneqConstraint { h, jacobian, idx }
    }
}

impl<F0, F1, const C0: usize, const C1: usize, const INPUT_DIM: usize>
    MakeEvaluatedIneqConstraint<INPUT_DIM, 2> for (F0, F1)
where
    F0: FnOnce() -> MatF64<1, C0>,
    F1: FnOnce() -> MatF64<1, C1>,
{
    fn make_ineq(
        self,
        idx: [usize; 2],
        var_kinds: [VarKind; 2],
        h: f64,
    ) -> EvaluatedIneqConstraint<INPUT_DIM, 2> {
        let mut jacobian = BlockJacobian::new(&[C0, C1]);
        if var_kinds[0] != VarKind::Conditioned {
            jacobian.set_block::<C0>(0, self.0());
        }
        if var_kinds[1] != VarKind::Conditioned {
            jacobian.set_block::<C1>(1, self.1());
        }
        EvaluatedIneqConstraint { h, jacobian, idx }
    }
}

impl<F0, F1, F2, const C0: usize, const C1: usize, const C2: usize, const INPUT_DIM: usize>
    MakeEvaluatedIneqConstraint<INPUT_DIM, 3> for (F0, F1, F2)
where
    F0: FnOnce() -> MatF64<1, C0>,
    F1: FnOnce() -> MatF64<1, C1>,
    F2: FnOnce() -> MatF64<1, C2>,
{
    fn make_ineq(
        self,
        idx: [usize; 3],
        var_kinds: [VarKind; 3],
        h: f64,
    ) -> EvaluatedIneqConstraint<INPUT_DIM, 3> {
        let mut jacobian = BlockJacobian::new(&[C0, C1, C2]);
        if var_kinds[0] != VarKind::Conditioned {
            jacobian.set_block::<C0>(0, self.0());
        }
        if var_kinds[1] != VarKind::Conditioned {
            jacobian.set_block::<C1>(1, self.1());
        }
        if var_kinds[2] != VarKind::Conditioned {
            jacobian.set_block::<C2>(2, self.2());
        }
        EvaluatedIneqConstraint { h, jacobian, idx }
    }
}

impl<
    F0,
    F1,
    F2,
    F3,
    const C0: usize,
    const C1: usize,
    const C2: usize,
    const C3: usize,
    const INPUT_DIM: usize,
> MakeEvaluatedIneqConstraint<INPUT_DIM, 4> for (F0, F1, F2, F3)
where
    F0: FnOnce() -> MatF64<1, C0>,
    F1: FnOnce() -> MatF64<1, C1>,
    F2: FnOnce() -> MatF64<1, C2>,
    F3: FnOnce() -> MatF64<1, C3>,
{
    fn make_ineq(
        self,
        idx: [usize; 4],
        var_kinds: [VarKind; 4],
        h: f64,
    ) -> EvaluatedIneqConstraint<INPUT_DIM, 4> {
        let mut jacobian = BlockJacobian::new(&[C0, C1, C2, C3]);
        if var_kinds[0] != VarKind::Conditioned {
            jacobian.set_block::<C0>(0, self.0());
        }
        if var_kinds[1] != VarKind::Conditioned {
            jacobian.set_block::<C1>(1, self.1());
        }
        if var_kinds[2] != VarKind::Conditioned {
            jacobian.set_block::<C2>(2, self.2());
        }
        if var_kinds[3] != VarKind::Conditioned {
            jacobian.set_block::<C3>(3, self.3());
        }
        EvaluatedIneqConstraint { h, jacobian, idx }
    }
}
