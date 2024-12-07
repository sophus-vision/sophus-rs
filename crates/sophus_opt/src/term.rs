use crate::block::BlockMatrix;
use crate::block::BlockVector;
use crate::prelude::*;
use crate::robust_kernel;
use crate::variables::VarKind;
use sophus_core::linalg::MatF64;
use sophus_core::linalg::VecF64;

extern crate alloc;

/// Evaluated cost term
#[derive(Debug, Clone)]
pub struct Term<const DIM: usize, const NUM_ARGS: usize> {
    /// Hessian
    pub hessian: BlockMatrix<DIM, NUM_ARGS>,
    /// Gradient
    pub gradient: BlockVector<DIM, NUM_ARGS>,
    /// cost: 0.5 * residual^T * precision_mat * residual
    pub cost: f64,
    /// indices of the variable families
    pub idx: [usize; NUM_ARGS],
    /// number of sub-terms
    pub num_sub_terms: usize,
}

impl<const DIM: usize, const NUM_ARGS: usize> Term<DIM, NUM_ARGS> {
    pub(crate) fn reduce(&mut self, other: Term<DIM, NUM_ARGS>) {
        self.hessian.mat += other.hessian.mat;
        self.gradient.vec += other.gradient.vec;
        self.cost += other.cost;
        self.num_sub_terms += other.num_sub_terms;
    }
}

trait RowLoop<const DIM: usize, const NUM_ARGS: usize, const R: usize> {
    fn set_off_diagonal(
        self,
        idx: usize,
        i: usize,
        j: usize,
        hessian: &mut BlockMatrix<DIM, NUM_ARGS>,
        precision_mat: Option<MatF64<R, R>>,
    );

    fn set_diagonal(
        self,
        idx: usize,
        i: usize,
        lambda_res: &VecF64<R>,
        gradient: &mut BlockVector<DIM, NUM_ARGS>,
        hessian: &mut BlockMatrix<DIM, NUM_ARGS>,
        precision_mat: Option<MatF64<R, R>>,
    );
}

impl<const DIM: usize, const NUM_ARGS: usize, const R: usize> RowLoop<DIM, NUM_ARGS, R> for () {
    fn set_off_diagonal(
        self,
        _idx: usize,
        _i: usize,
        _j: usize,
        _hessian: &mut BlockMatrix<DIM, NUM_ARGS>,
        _precision_mat: Option<MatF64<R, R>>,
    ) {
    }

    fn set_diagonal(
        self,
        _idx: usize,
        _i: usize,
        _lambda_res: &VecF64<R>,
        _gradient: &mut BlockVector<DIM, NUM_ARGS>,
        _hessian: &mut BlockMatrix<DIM, NUM_ARGS>,
        _precision_mat: Option<MatF64<R, R>>,
    ) {
    }
}

impl<
        const DIM: usize,
        const NUM_ARGS: usize,
        const R: usize,
        const DX: usize,
        Tail: RowLoop<DIM, NUM_ARGS, R> + ColLoop<DIM, NUM_ARGS, R, DX>,
    > RowLoop<DIM, NUM_ARGS, R> for (Option<MatF64<R, DX>>, Tail)
{
    fn set_off_diagonal(
        self,
        idx: usize,
        i: usize,
        j: usize,
        hessian: &mut BlockMatrix<DIM, NUM_ARGS>,
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
        gradient: &mut BlockVector<DIM, NUM_ARGS>,
        hessian: &mut BlockMatrix<DIM, NUM_ARGS>,
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

trait ColLoop<const DIM: usize, const NUM_ARGS: usize, const R: usize, const DJ: usize> {
    fn set_off_diagonal_from_lhs(
        self,
        idx: usize,
        i: usize,
        j: usize,
        hessian: &mut BlockMatrix<DIM, NUM_ARGS>,
        lhs: MatF64<R, DJ>,
        precision_mat: Option<MatF64<R, R>>,
    );
}

impl<const DIM: usize, const NUM_ARGS: usize, const R: usize, const DJ: usize>
    ColLoop<DIM, NUM_ARGS, R, DJ> for ()
{
    fn set_off_diagonal_from_lhs(
        self,
        _idx: usize,
        _i: usize,
        _j: usize,
        _hessian: &mut BlockMatrix<DIM, NUM_ARGS>,
        _lhs: MatF64<R, DJ>,
        _precision_mat: Option<MatF64<R, R>>,
    ) {
    }
}

impl<
        const DIM: usize,
        const NUM_ARGS: usize,
        const R: usize,
        const DI: usize,
        const DJ: usize,
        Tail: ColLoop<DIM, NUM_ARGS, R, DJ>,
    > ColLoop<DIM, NUM_ARGS, R, DJ> for (Option<MatF64<R, DI>>, Tail)
{
    fn set_off_diagonal_from_lhs(
        self,
        idx: usize,
        i: usize,
        j: usize,
        hessian: &mut BlockMatrix<DIM, NUM_ARGS>,
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

/// Trait for making n-ary terms
pub trait MakeTerm<const R: usize, const N: usize> {
    /// make a term from a residual value, and derivatives (=self)
    ///
    /// In more detail, this function computes the Hessian, gradient and least-squares cost of the
    /// corresponding term given the following inputs:
    ///
    /// - `self`:          A tuple of functions that return the Jacobian of the cost function with
    ///                    respect to each argument.
    /// - `var_kinds`:     An array of `VarKind` for each argument of the cost function. A gradient
    ///                    and Hessian will be computed for each argument that is not `Conditioned`.
    /// - `residual`:      The residual of the corresponding cost term.
    /// - `robust_kernel`: An optional robust kernel to apply to the residual.
    /// - `precision_mat`: Precision matrix - i.e. inverse of the covariance matrix - to compute the
    ///                    least-squares cost: `0.5 * residual^T * precision_mat * residual`.
    ///                    If `None`, the identity matrix is used: `0.5 * residual^T * residual`.
    fn make_term<const DIM: usize>(
        self,
        idx: [usize; N],
        var_kinds: [VarKind; N],
        residual: VecF64<R>,
        robust_kernel: Option<robust_kernel::RobustKernel>,
        precision_mat: Option<MatF64<R, R>>,
    ) -> Term<DIM, N>;
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

macro_rules! impl_make_term_for_tuples {
    ($($N:literal: ($($F:ident, $D:ident, $idx:tt),+),)+) => {
        $(
            #[allow(non_snake_case)]
            impl<const R: usize, $($F,)+ $(const $D: usize,)+> MakeTerm<R, $N> for ($($F,)+)
            where
                $(
                    $F: FnOnce() -> MatF64<R, $D>,
                )+
            {
                fn make_term<const DIM: usize>(
                    self,
                    idx: [usize; $N],
                    var_kinds: [VarKind; $N],
                    residual: VecF64<R>,
                    robust_kernel: Option<robust_kernel::RobustKernel>,
                    precision_mat: Option<MatF64<R, R>>,
                ) -> Term<DIM, $N> {
                    let residual = match robust_kernel {
                        Some(rk) => rk.weight(residual.norm()) * residual,
                        None => residual,
                    };

                    let ($($F,)+) = self;

                    let maybe_dx = nested_option_tuple!(var_kinds; $($F, $D, $idx),+);

                    let dims = [$($D,)+];
                    let mut hessian = BlockMatrix::new(&dims);
                    let mut gradient = BlockVector::new(&dims);

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

                    Term {
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

// implement MakeTerm for up to 25 arguments.
impl_make_term_for_tuples! {
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
