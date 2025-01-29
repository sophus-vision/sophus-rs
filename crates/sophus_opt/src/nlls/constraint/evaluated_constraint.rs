use sophus_autodiff::linalg::MatF64;
use sophus_autodiff::linalg::VecF64;

use crate::block::block_jacobian::BlockJacobian;
use crate::variables::VarKind;

/// Evaluated constraint
#[derive(Debug, Clone)]
pub struct EvaluatedConstraint<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const NUM_ARGS: usize,
> {
    /// The current constraint residual vector g(x) of dimension DIM
    pub residual: VecF64<RESIDUAL_DIM>,
    /// The jacobian of the equality constraint.
    pub jacobian: BlockJacobian<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>,
    /// The indices of the variable families this constraint touches
    pub idx: [usize; NUM_ARGS],
}

impl<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const NUM_ARGS: usize>
    EvaluatedConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>
{
    pub(crate) fn reduce(&mut self, other: EvaluatedConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>) {
        self.jacobian.mat += other.jacobian.mat;
        self.residual += other.residual;
    }
}

/// Trait for making n-ary constraint.
pub trait MakeEvaluatedConstraint<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const NUM_ARGS: usize,
>
{
    /// Make an equality from a residual value, and derivatives (=self)
    ///
    /// In more detail, this function computes the Jacobian of the
    /// corresponding constraint given the following inputs:
    ///
    /// - `self`:          A tuple of functions that return the Jacobian of the constraint function
    ///                    with respect to each argument.
    /// - `var_kinds`:     An array of `VarKind` for each argument of the cost function. A Jacobian
    ///                    will be computed for each argument that is not `Conditioned`.
    /// - `residual`:      The residual of the corresponding eq constraint.
    fn make_eq(
        self,
        idx: [usize; NUM_ARGS],
        var_kinds: [VarKind; NUM_ARGS],
        residual: VecF64<RESIDUAL_DIM>,
    ) -> EvaluatedConstraint<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>;
}

impl<F0, const RESIDUAL_DIM: usize, const INPUT_DIM: usize>
    MakeEvaluatedConstraint<RESIDUAL_DIM, INPUT_DIM, 1> for (F0,)
where
    F0: FnOnce() -> MatF64<RESIDUAL_DIM, INPUT_DIM>,
{
    fn make_eq(
        self,
        idx: [usize; 1],
        var_kinds: [VarKind; 1],
        residual: VecF64<RESIDUAL_DIM>,
    ) -> EvaluatedConstraint<RESIDUAL_DIM, INPUT_DIM, 1> {
        let mut jacobian = BlockJacobian::new(&[INPUT_DIM]);
        if var_kinds[0] != VarKind::Conditioned {
            jacobian.set_block::<INPUT_DIM>(0, self.0());
        }
        EvaluatedConstraint {
            residual,
            jacobian,
            idx,
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
    > MakeEvaluatedConstraint<RESIDUAL_DIM, INPUT_DIM, 2> for (F0, F1)
where
    F0: FnOnce() -> MatF64<RESIDUAL_DIM, C0>,
    F1: FnOnce() -> MatF64<RESIDUAL_DIM, C1>,
{
    fn make_eq(
        self,
        idx: [usize; 2],
        var_kinds: [VarKind; 2],
        residual: VecF64<RESIDUAL_DIM>,
    ) -> EvaluatedConstraint<RESIDUAL_DIM, INPUT_DIM, 2> {
        let mut jacobian = BlockJacobian::new(&[C0, C1]);
        if var_kinds[0] != VarKind::Conditioned {
            jacobian.set_block::<C0>(0, self.0());
        }
        if var_kinds[1] != VarKind::Conditioned {
            jacobian.set_block::<C1>(1, self.1());
        }
        EvaluatedConstraint {
            residual,
            jacobian,
            idx,
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
    > MakeEvaluatedConstraint<RESIDUAL_DIM, INPUT_DIM, 3> for (F0, F1, F2)
where
    F0: FnOnce() -> MatF64<RESIDUAL_DIM, C0>,
    F1: FnOnce() -> MatF64<RESIDUAL_DIM, C1>,
    F2: FnOnce() -> MatF64<RESIDUAL_DIM, C2>,
{
    fn make_eq(
        self,
        idx: [usize; 3],
        var_kinds: [VarKind; 3],
        residual: VecF64<RESIDUAL_DIM>,
    ) -> EvaluatedConstraint<RESIDUAL_DIM, INPUT_DIM, 3> {
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
        EvaluatedConstraint {
            residual,
            jacobian,
            idx,
        }
    }
}
