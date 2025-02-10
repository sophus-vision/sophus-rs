use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};
use sophus_geo::region::box_region::NonEmptyBoxRegion;

use crate::{
    block::block_jacobian::BlockJacobian,
    variables::VarKind,
};

/// Evaluated constraint
#[derive(Debug, Clone)]
pub struct EvaluatedIneqConstraint<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const NUM_ARGS: usize,
> {
    /// lower and upper bound
    pub bounds: NonEmptyBoxRegion<RESIDUAL_DIM>,
    /// The current constraint value g(x) of dimension DIM
    pub constraint_value: VecF64<RESIDUAL_DIM>,
    /// The jacobian of the inequality constraint at x.
    pub jacobian: BlockJacobian<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>,
    /// The indices of the variable families this constraint touches
    pub idx: [usize; NUM_ARGS],
}

impl<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const NUM_ARGS: usize>
    EvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>
{
    pub(crate) fn reduce(
        &mut self,
        other: EvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>,
    ) {
        self.jacobian.mat += other.jacobian.mat;
        self.constraint_value += other.constraint_value;
    }
}

/// Trait for making n-ary constraint.
pub trait MakeEvaluatedIneqConstraint<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const NUM_ARGS: usize,
>
{
    /// Make an equality from a constraint_value value, and derivatives (=self)
    ///
    /// In more detail, this function computes the Jacobian of the
    /// corresponding constraint given the following inputs:
    ///
    /// - `self`:          A tuple of functions that return the Jacobian of the constraint function
    ///   with respect to each argument.
    /// - `var_kinds`:     An array of `VarKind` for each argument of the cost function. A Jacobian
    ///   will be computed for each argument that is not `Conditioned`.
    /// - `constraint_value`:      The constraint_value of the corresponding eq constraint.
    fn make_ineq(
        self,
        idx: [usize; NUM_ARGS],
        var_kinds: [VarKind; NUM_ARGS],
        constraint_value: VecF64<RESIDUAL_DIM>,
        bounds: NonEmptyBoxRegion<RESIDUAL_DIM>,
    ) -> EvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>;
}

impl<F0, const RESIDUAL_DIM: usize, const INPUT_DIM: usize>
    MakeEvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, 1> for (F0,)
where
    F0: FnOnce() -> MatF64<RESIDUAL_DIM, INPUT_DIM>,
{
    fn make_ineq(
        self,
        idx: [usize; 1],
        var_kinds: [VarKind; 1],
        constraint_value: VecF64<RESIDUAL_DIM>,
        bounds: NonEmptyBoxRegion<RESIDUAL_DIM>,
    ) -> EvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, 1> {
        let mut jacobian = BlockJacobian::new(&[INPUT_DIM]);
        if var_kinds[0] != VarKind::Conditioned {
            jacobian.set_block::<INPUT_DIM>(0, self.0());
        }
        EvaluatedIneqConstraint {
            constraint_value,
            jacobian,
            idx,
            bounds,
        }
    }
}

impl<
        F0,
        F1,
        const RESIDUAL_DIM: usize,
        const C0: usize,
        const C1: usize,
        const INPUT_DIM: usize,
    > MakeEvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, 2> for (F0, F1)
where
    F0: FnOnce() -> MatF64<RESIDUAL_DIM, C0>,
    F1: FnOnce() -> MatF64<RESIDUAL_DIM, C1>,
{
    fn make_ineq(
        self,
        idx: [usize; 2],
        var_kinds: [VarKind; 2],
        constraint_value: VecF64<RESIDUAL_DIM>,
        bounds: NonEmptyBoxRegion<RESIDUAL_DIM>,

    ) -> EvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, 2> {
        let mut jacobian = BlockJacobian::new(&[C0, C1]);
        if var_kinds[0] != VarKind::Conditioned {
            jacobian.set_block::<C0>(0, self.0());
        }
        if var_kinds[1] != VarKind::Conditioned {
            jacobian.set_block::<C1>(1, self.1());
        }
        EvaluatedIneqConstraint {
            constraint_value,
            jacobian,
            idx,
            bounds
        }
    }
}

impl<
        F0,
        F1,
        F2,
        const RESIDUAL_DIM: usize,
        const C0: usize,
        const C1: usize,
        const C2: usize,
        const INPUT_DIM: usize,
    > MakeEvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, 3> for (F0, F1, F2)
where
    F0: FnOnce() -> MatF64<RESIDUAL_DIM, C0>,
    F1: FnOnce() -> MatF64<RESIDUAL_DIM, C1>,
    F2: FnOnce() -> MatF64<RESIDUAL_DIM, C2>,
{
    fn make_ineq(
        self,
        idx: [usize; 3],
        var_kinds: [VarKind; 3],
        constraint_value: VecF64<RESIDUAL_DIM>,
        bounds: NonEmptyBoxRegion<RESIDUAL_DIM>,
    ) -> EvaluatedIneqConstraint<RESIDUAL_DIM, INPUT_DIM, 3> {
        let mut jacobian = BlockJacobian::new(&[C0, C1, C2]);
        if var_kinds[0] != VarKind::Conditioned {
            jacobian.set_block::<C0>(0, self.0());
        }
        if var_kinds[1] != VarKind::Conditioned {
            jacobian.set_block::<C1>(1, self.1());
        }
        if var_kinds[2] != VarKind::Conditioned {
            jacobian.set_block::<C2>(1, self.2());
        }
        EvaluatedIneqConstraint {
            constraint_value,
            jacobian,
            idx,
            bounds
        }
    }
}
