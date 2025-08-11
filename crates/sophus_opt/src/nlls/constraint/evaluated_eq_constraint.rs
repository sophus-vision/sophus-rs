use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};
use sophus_block::BlockJacobian;

use crate::variables::VarKind;

/// Evaluated equality constraint.
///
/// ## Generic parameters
///
///  * `RESIDUAL_DIM`
///    - Dimension of the constraint residual vector `c`.
///  * `INPUT_DIM`
///    - Total input dimension of the constraint residual function `c`. It is the sum of argument
///      dimensions: |Vⁱ₀| + |Vⁱ₁| + ... + |Vⁱₙ₋₁|.
///  * `N`
///    - Number of arguments of the constraint residual function `c`.
#[derive(Debug, Clone)]
pub struct EvaluatedEqConstraint<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const N: usize>
{
    /// The constraint residual vector `c`.
    pub residual: VecF64<RESIDUAL_DIM>,
    /// The Jacobian of the equality constraint residual function `c`.
    pub jacobian: BlockJacobian<RESIDUAL_DIM, INPUT_DIM, N>,
    /// Array of variable indices for each argument of the constraint residual function.
    ///
    /// For example, if `idx = [2, 7, 3]` and `Args` is `(Foo, Bar, Bar)`, then:
    ///
    /// - Argument 0 is the 2nd variable of the `Foo` family.
    /// - Argument 1 is the 7th variable of the `Bar` family.
    /// - Argument 2 is the 3rd variable of the `Bar` family.
    pub idx: [usize; N],
}

impl<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const N: usize>
    EvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, N>
{
    pub(crate) fn reduce(&mut self, other: EvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, N>) {
        self.jacobian.mat += other.jacobian.mat;
        self.residual += other.residual;
    }
}

/// Trait for making an N-ary evaluated equality constraint.
pub trait MakeEvaluatedEqConstraint<
    const RESIDUAL_DIM: usize,
    const INPUT_DIM: usize,
    const N: usize,
>
{
    /// Make an equality from a residual value, and derivatives (=self). This function shall be
    /// called inside the [crate::nlls::IsEqConstraintsFn::eval] function of a user-defined
    /// constraint.
    ///
    /// This function computes the Jacobian of the constraint to produce the
    /// [EvaluatedEqConstraint] - given the following inputs:
    ///
    ///  * `self`
    ///     - A tuple of functions that return the Jacobian of the constraint function with respect
    ///       to each argument.
    ///  * `var_kinds`
    ///     - An array of `VarKind` for each argument of the constraint. A Jacobian will be computed
    ///       for each argument that is not `Conditioned`.
    ///  * `residual`
    ///     - The residual of the corresponding equality constraint.
    fn make_eq(
        self,
        idx: [usize; N],
        var_kinds: [VarKind; N],
        residual: VecF64<RESIDUAL_DIM>,
    ) -> EvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, N>;
}

impl<F0, const RESIDUAL_DIM: usize, const INPUT_DIM: usize>
    MakeEvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, 1> for (F0,)
where
    F0: FnOnce() -> MatF64<RESIDUAL_DIM, INPUT_DIM>,
{
    fn make_eq(
        self,
        idx: [usize; 1],
        var_kinds: [VarKind; 1],
        residual: VecF64<RESIDUAL_DIM>,
    ) -> EvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, 1> {
        let mut jacobian = BlockJacobian::new(&[INPUT_DIM]);
        if var_kinds[0] != VarKind::Conditioned {
            jacobian.set_block::<INPUT_DIM>(0, self.0());
        }
        EvaluatedEqConstraint {
            residual,
            jacobian,
            idx,
        }
    }
}

impl<F0, F1, const RESIDUAL_DIM: usize, const C0: usize, const C1: usize, const INPUT_DIM: usize>
    MakeEvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, 2> for (F0, F1)
where
    F0: FnOnce() -> MatF64<RESIDUAL_DIM, C0>,
    F1: FnOnce() -> MatF64<RESIDUAL_DIM, C1>,
{
    fn make_eq(
        self,
        idx: [usize; 2],
        var_kinds: [VarKind; 2],
        residual: VecF64<RESIDUAL_DIM>,
    ) -> EvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, 2> {
        let mut jacobian = BlockJacobian::new(&[C0, C1]);
        if var_kinds[0] != VarKind::Conditioned {
            jacobian.set_block::<C0>(0, self.0());
        }
        if var_kinds[1] != VarKind::Conditioned {
            jacobian.set_block::<C1>(1, self.1());
        }
        EvaluatedEqConstraint {
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
> MakeEvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, 3> for (F0, F1, F2)
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
    ) -> EvaluatedEqConstraint<RESIDUAL_DIM, INPUT_DIM, 3> {
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
        EvaluatedEqConstraint {
            residual,
            jacobian,
            idx,
        }
    }
}
