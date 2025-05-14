use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};

use crate::{
    block::{
        block_gradient::BlockGradient,
        block_hessian::BlockHessian,
    },
    prelude::*,
    robust_kernel,
    variables::VarKind,
};

extern crate alloc;

/// Evaluated cost term.
///
/// ## Generic parameters
///
///  * `INPUT_DIM`
///    - Total input dimension of the residual function `g`. It is the sum of argument dimensions:
///      |Vⁱ₀| + |Vⁱ₁| + ... + |Vⁱₙ₋₁|.
///  * `N`
///    - Number of arguments of the residual function `g`.
#[derive(Debug, Clone)]
pub struct EvaluatedCostTerm<const INPUT_DIM: usize, const N: usize> {
    /// Hessian matrix: `J^T * W * J` where `J` is the Jacobian matrix of the residual function
    /// and `W` is the precision matrix.
    pub hessian: BlockHessian<INPUT_DIM, N>,
    /// Gradient vector: `J^T * W * g` where `J` is the Jacobian matrix of the residual
    /// function and `W` is the precision matrix, and `g` is the residual vector.
    pub gradient: BlockGradient<INPUT_DIM, N>,
    /// Least squares cost: `0.5 * g^T * W * g` where `g` is the residual vector and `W` is the
    /// precision matrix.
    pub cost: f64,
    /// Array of variable indices for each argument of the residual function.
    ///
    /// For example, if `idx = [2, 7, 3]` and `Args` is `(Foo, Bar, Bar)`, then:
    ///
    /// - Argument 0 is the 2nd variable of the `Foo` family.
    /// - Argument 1 is the 7th variable of the `Bar` family.
    /// - Argument 2 is the 3rd variable of the `Bar` family.
    pub idx: [usize; N],
    /// Number of sub-terms. There is more than one sub-term if this term was created by
    /// reducing multiple terms - which share the same set of free variables.
    pub num_sub_terms: usize,
}

impl<const INPUT_DIM: usize, const N: usize> EvaluatedCostTerm<INPUT_DIM, N> {
    pub(crate) fn reduce(&mut self, other: EvaluatedCostTerm<INPUT_DIM, N>) {
        self.hessian.mat += other.hessian.mat;
        self.gradient.vec += other.gradient.vec;
        self.cost += other.cost;
        self.num_sub_terms += other.num_sub_terms;
    }
}

trait RowLoop<const INPUT_DIM: usize, const N: usize, const R: usize> {
    fn set_off_diagonal(
        self,
        idx: usize,
        i: usize,
        j: usize,
        hessian: &mut BlockHessian<INPUT_DIM, N>,
        precision_mat: Option<MatF64<R, R>>,
    );

    fn set_diagonal(
        self,
        idx: usize,
        i: usize,
        lambda_res: &VecF64<R>,
        gradient: &mut BlockGradient<INPUT_DIM, N>,
        hessian: &mut BlockHessian<INPUT_DIM, N>,
        precision_mat: Option<MatF64<R, R>>,
    );
}

impl<const INPUT_DIM: usize, const N: usize, const R: usize> RowLoop<INPUT_DIM, N, R> for () {
    fn set_off_diagonal(
        self,
        _idx: usize,
        _i: usize,
        _j: usize,
        _hessian: &mut BlockHessian<INPUT_DIM, N>,
        _precision_mat: Option<MatF64<R, R>>,
    ) {
    }

    fn set_diagonal(
        self,
        _idx: usize,
        _i: usize,
        _lambda_res: &VecF64<R>,
        _gradient: &mut BlockGradient<INPUT_DIM, N>,
        _hessian: &mut BlockHessian<INPUT_DIM, N>,
        _precision_mat: Option<MatF64<R, R>>,
    ) {
    }
}

impl<
    const INPUT_DIM: usize,
    const N: usize,
    const R: usize,
    const DX: usize,
    Tail: RowLoop<INPUT_DIM, N, R> + ColLoop<INPUT_DIM, N, R, DX>,
> RowLoop<INPUT_DIM, N, R> for (Option<MatF64<R, DX>>, Tail)
{
    fn set_off_diagonal(
        self,
        idx: usize,
        i: usize,
        j: usize,
        hessian: &mut BlockHessian<INPUT_DIM, N>,
        precision_mat: Option<MatF64<R, R>>,
    ) {
        if idx == i {
            if let Some(self_0) = self.0 {
                self.set_off_diagonal_from_lhs(idx, i, j, hessian, self_0, precision_mat);
            }
        } else {
            self.1
                .set_off_diagonal(idx + 1, i, j, hessian, precision_mat);
        }
    }

    fn set_diagonal(
        self,
        idx: usize,
        i: usize,
        lambda_res: &VecF64<R>,
        gradient: &mut BlockGradient<INPUT_DIM, N>,
        hessian: &mut BlockHessian<INPUT_DIM, N>,
        precision_mat: Option<MatF64<R, R>>,
    ) {
        if idx == i {
            if self.0.is_some() {
                let dx: MatF64<R, DX> = self.0.unwrap();
                let dx_t: MatF64<DX, R> = dx.transpose();
                let lambda_dx = match precision_mat {
                    Some(precision_mat) => precision_mat * dx,
                    None => dx,
                };

                let grad0 = dx_t * lambda_res;
                gradient.set_block(i, grad0);
                hessian.set_block(i, i, dx_t * lambda_dx);
            }
        } else {
            self.1
                .set_diagonal(idx + 1, i, lambda_res, gradient, hessian, precision_mat);
        }
    }
}

trait ColLoop<const INPUT_DIM: usize, const N: usize, const R: usize, const DJ: usize> {
    fn set_off_diagonal_from_lhs(
        self,
        idx: usize,
        i: usize,
        j: usize,
        hessian: &mut BlockHessian<INPUT_DIM, N>,
        lhs: MatF64<R, DJ>,
        precision_mat: Option<MatF64<R, R>>,
    );
}

impl<const INPUT_DIM: usize, const N: usize, const R: usize, const DJ: usize>
    ColLoop<INPUT_DIM, N, R, DJ> for ()
{
    fn set_off_diagonal_from_lhs(
        self,
        _idx: usize,
        _i: usize,
        _j: usize,
        _hessian: &mut BlockHessian<INPUT_DIM, N>,
        _lhs: MatF64<R, DJ>,
        _precision_mat: Option<MatF64<R, R>>,
    ) {
    }
}

impl<
    const INPUT_DIM: usize,
    const N: usize,
    const R: usize,
    const DI: usize,
    const DJ: usize,
    Tail: ColLoop<INPUT_DIM, N, R, DJ>,
> ColLoop<INPUT_DIM, N, R, DJ> for (Option<MatF64<R, DI>>, Tail)
{
    fn set_off_diagonal_from_lhs(
        self,
        idx: usize,
        i: usize,
        j: usize,
        hessian: &mut BlockHessian<INPUT_DIM, N>,
        lhs: MatF64<R, DJ>,
        precision_mat: Option<MatF64<R, R>>,
    ) {
        if idx == j {
            if let Some(self_0) = self.0 {
                if let Some(precision) = precision_mat {
                    hessian.set_block(i, j, lhs.transpose() * precision * self_0);
                } else {
                    hessian.set_block(i, j, lhs.transpose() * self_0);
                }
            }
        } else {
            self.1
                .set_off_diagonal_from_lhs(idx + 1, i, j, hessian, lhs, precision_mat);
        }
    }
}

/// Trait for making an N-ary cost term.
pub trait MakeEvaluatedCostTerm<const R: usize, const N: usize> {
    /// Make a term from a residual value, and derivatives (=self). This function shall be called
    /// inside the [HasResidualFn::eval] function of a user-defined residual.
    ///
    /// This function computes the Hessian, gradient and least-squares cost to produce the
    /// [EvaluatedCostTerm] - given the following inputs:
    ///
    ///  * `self`
    ///    - A tuple of functions that return the Jacobian of the residual function with respect to
    ///      each argument.
    ///  * `var_kinds`
    ///    - An array of `VarKind` for each argument of the cost function. A gradient and Hessian
    ///      will be computed for each argument that is not `Conditioned`.
    ///  * `residual`
    ///    - The residual of the corresponding cost term.
    ///  * `robust_kernel`
    ///    - An optional robust kernel to apply to the residual.
    ///  * `precision_mat`
    ///    - Precision matrix `W` - i.e. inverse of the covariance matrix - to compute the
    ///      least-squares cost: `0.5 * g^T * precision_mat * g`. If `None`, the identity matrix is
    ///      used: `0.5 * g^T * g`.
    fn make<const INPUT_DIM: usize>(
        self,
        idx: [usize; N],
        var_kinds: [VarKind; N],
        residual: VecF64<R>,
        robust_kernel: Option<robust_kernel::RobustKernel>,
        precision_mat: Option<MatF64<R, R>>,
    ) -> EvaluatedCostTerm<INPUT_DIM, N>;
}

macro_rules! nested_option_tuple {
    // Base case
    ($var_kinds:ident; $F:ident, $D:ident, $idx:tt) => {
        (
            if $var_kinds[$idx] != VarKind::Conditioned {
                Some($F())
            } else {
                None
            },
            ()
        )
    };
    // Recursive case
    ($var_kinds:ident; $F:ident, $D:ident, $idx:tt, $($rest:tt)*) => {
        (
            if $var_kinds[$idx] != VarKind::Conditioned {
                Some($F())
            } else {
                None
            },
            nested_option_tuple!($var_kinds; $($rest)*)
        )
    };
}

macro_rules! impl_make_evaluated_cost_term_for_tuples {
    ($($N:literal: ($($F:ident, $D:ident, $idx:tt),+),)+) => {
        $(
            #[allow(non_snake_case)]
            impl<const R: usize, $($F,)+ $(const $D: usize,)+> MakeEvaluatedCostTerm<R, $N> for ($($F,)+)
            where
                $(
                    $F: FnOnce() -> MatF64<R, $D>,
                )+
            {
                fn make<const INPUT_DIM: usize>(
                    self,
                    idx: [usize; $N],
                    var_kinds: [VarKind; $N],
                    residual: VecF64<R>,
                    robust_kernel: Option<robust_kernel::RobustKernel>,
                    precision_mat: Option<MatF64<R, R>>,
                ) -> EvaluatedCostTerm<INPUT_DIM, $N> {
                    let residual = match robust_kernel {
                        Some(rk) => rk.weight(residual.norm()) * residual,
                        None => residual,
                    };

                    let ($($F,)+) = self;

                    let maybe_dx = nested_option_tuple!(var_kinds; $($F, $D, $idx),+);

                    let dims = [$($D,)+];
                    let mut hessian = BlockHessian::new(&dims);
                    let mut gradient = BlockGradient::new(&dims);

                    let lambda_res = match precision_mat {
                        Some(pm) => pm * residual,
                        None => residual,
                    };

                    for i in 0..$N {
                        maybe_dx.set_diagonal(
                            0, i, &lambda_res, &mut gradient, &mut hessian, precision_mat);
                        for j in (i+1)..$N {
                            maybe_dx.set_off_diagonal(0, i, j, &mut hessian, precision_mat);
                        }
                    }

                    EvaluatedCostTerm {
                        hessian,
                        gradient,
                        cost: (residual.transpose() * lambda_res)[0],
                        idx,
                        num_sub_terms: 1,
                    }
                }
            }
        )+
    }
}

// implement MakeEvaluatedCostTerm for up to 25 arguments.
impl_make_evaluated_cost_term_for_tuples! {
    1: ( F0, D0, 0),
    2: ( F0, D0, 0, F1 ,D1, 1),
    3: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2),
    4: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3),
    5: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4),
    6: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5),
    7: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6),
    8: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7),
    9: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8),
   10: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9),
   11: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10),
   12: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11),
   13: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12),
   14: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13),
   15: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13,F14,D14,14),
   16: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13,F14,D14,14,F15,D15,15),
   17: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13,F14,D14,14,F15,D15,15,
        F16,D16,16),
   18: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13,F14,D14,14,F15,D15,15,
        F16,D16,16,F17,D17,17),
   19: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13,F14,D14,14,F15,D15,15,
        F16,D16,16,F17,D17,17,F18,D18,18),
   20: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13,F14,D14,14,F15,D15,15,
        F16,D16,16,F17,D17,17,F18,D18,18,F19,D19,19),
   21: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13,F14,D14,14,F15,D15,15,
        F16,D16,16,F17,D17,17,F18,D18,18,F19,D19,19,F20,D20,20),
   22: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13,F14,D14,14,F15,D15,15,
        F16,D16,16,F17,D17,17,F18,D18,18,F19,D19,19,F20,D20,20,F21,D21,21),
   23: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13,F14,D14,14,F15,D15,15,
        F16,D16,16,F17,D17,17,F18,D18,18,F19,D19,19,F20,D20,20,F21,D21,21,F22,D22,22),
   24: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13,F14,D14,14,F15,D15,15,
        F16,D16,16,F17,D17,17,F18,D18,18,F19,D19,19,F20,D20,20,F21,D21,21,F22,D22,22,F23,D23,23),
   25: ( F0, D0, 0, F1 ,D1, 1, F2, D2, 2, F3, D3, 3, F4, D4, 4, F5, D5, 5, F6, D6, 6, F7, D7, 7,
         F8, D8, 8, F9, D9, 9,F10,D10,10,F11,D11,11,F12,D12,12,F13,D13,13,F14,D14,14,F15,D15,15,
        F16,D16,16,F17,D17,17,F18,D18,18,F19,D19,19,F20,D20,20,F21,D21,21,F22,D22,22,F23,D23,23,
        F24,D24,24),
}
