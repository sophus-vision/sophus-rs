use super::evaluated_term::EvaluatedCostTerm;
use crate::robust_kernel::RobustKernel;
use crate::variables::IsVarTuple;
use crate::variables::VarKind;

/// Residual function
///
/// This trait is the main customization point for the user.
pub trait IsResidualFn<
    const NUM: usize,
    const NUM_ARGS: usize,
    GlobalConstants: 'static + Send + Sync,
    Args: IsVarTuple<NUM_ARGS>,
    Constants,
>: Copy + Send + Sync + 'static
{
    /// evaluate the residual function which shall be defined by the user
    fn eval(
        &self,
        global_constants: &GlobalConstants,
        idx: [usize; NUM_ARGS],
        args: Args,
        derivatives: [VarKind; NUM_ARGS],
        robust_kernel: Option<RobustKernel>,
        constants: &Constants,
    ) -> EvaluatedCostTerm<NUM, NUM_ARGS>;
}
