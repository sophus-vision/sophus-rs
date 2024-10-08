use crate::block::BlockMatrix;
use crate::block::BlockVector;
use crate::prelude::*;
use crate::robust_kernel;
use crate::variables::VarKind;
use sophus_core::linalg::MatF64;
use sophus_core::linalg::VecF64;

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

// TODO: Improve MakeTerm implementations:
//  - Figure out how to make this work for arbitrary length tuples without code duplication.
//  - Benchmark the performance the implementation against hand-written versions to rule out
// pessimization.

impl<F0, const D0: usize, const R: usize> MakeTerm<R, 1> for (F0,)
where
    F0: FnOnce() -> MatF64<R, D0>,
{
    fn make_term<const DIM: usize>(
        self,
        idx: [usize; 1],
        var_kinds: [VarKind; 1],
        residual: VecF64<R>,
        robust_kernel: Option<robust_kernel::RobustKernel>,
        precision_mat: Option<MatF64<R, R>>,
    ) -> Term<DIM, 1> {
        const SIZE: usize = 1;

        let residual = match robust_kernel {
            Some(robust_kernel) => robust_kernel.weight(residual.norm()) * residual,
            None => residual,
        };

        let maybe_dx = (
            if var_kinds[0] != VarKind::Conditioned {
                Some(self.0())
            } else {
                None
            },
            (),
        );

        let dims = vec![D0];
        let mut hessian = BlockMatrix::new(&dims);
        let mut gradient = BlockVector::new(&[D0]);

        let lambda_res = match precision_mat {
            Some(precision_mat) => precision_mat * residual,
            None => residual,
        };

        for i in 0..SIZE {
            maybe_dx.set_diagonal(
                0,
                i,
                &lambda_res,
                &mut gradient,
                &mut hessian,
                precision_mat,
            );
            for j in i + 1..SIZE {
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

impl<F0, F1, const D0: usize, const D1: usize, const R: usize> MakeTerm<R, 2> for (F0, F1)
where
    F0: FnOnce() -> MatF64<R, D0>,
    F1: FnOnce() -> MatF64<R, D1>,
{
    fn make_term<const DIM: usize>(
        self,
        idx: [usize; 2],
        var_kinds: [VarKind; 2],
        residual: VecF64<R>,
        robust_kernel: Option<robust_kernel::RobustKernel>,
        precision_mat: Option<MatF64<R, R>>,
    ) -> Term<DIM, 2> {
        const SIZE: usize = 2;

        let residual = match robust_kernel {
            Some(robust_kernel) => robust_kernel.weight(residual.norm()) * residual,
            None => residual,
        };

        let maybe_dx = (
            if var_kinds[0] != VarKind::Conditioned {
                Some(self.0())
            } else {
                None
            },
            (
                if var_kinds[1] != VarKind::Conditioned {
                    Some(self.1())
                } else {
                    None
                },
                (),
            ),
        );

        let dims = [D0, D1];
        let mut hessian = BlockMatrix::new(&dims);
        let mut gradient = BlockVector::new(&dims);

        let lambda_res = match precision_mat {
            Some(precision_mat) => precision_mat * residual,
            None => residual,
        };

        for i in 0..SIZE {
            maybe_dx.set_diagonal(
                0,
                i,
                &lambda_res,
                &mut gradient,
                &mut hessian,
                precision_mat,
            );
            for j in i + 1..SIZE {
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

impl<F0, F1, F2, const D0: usize, const D1: usize, const D2: usize, const R: usize> MakeTerm<R, 3>
    for (F0, F1, F2)
where
    F0: FnOnce() -> MatF64<R, D0>,
    F1: FnOnce() -> MatF64<R, D1>,
    F2: FnOnce() -> MatF64<R, D2>,
{
    fn make_term<const DIM: usize>(
        self,
        idx: [usize; 3],
        var_kinds: [VarKind; 3],
        residual: VecF64<R>,
        robust_kernel: Option<robust_kernel::RobustKernel>,
        precision_mat: Option<MatF64<R, R>>,
    ) -> Term<DIM, 3> {
        const SIZE: usize = 3;

        let residual = match robust_kernel {
            Some(robust_kernel) => robust_kernel.weight(residual.norm()) * residual,
            None => residual,
        };

        let maybe_dx = (
            if var_kinds[0] != VarKind::Conditioned {
                Some(self.0())
            } else {
                None
            },
            (
                if var_kinds[1] != VarKind::Conditioned {
                    Some(self.1())
                } else {
                    None
                },
                (
                    if var_kinds[2] != VarKind::Conditioned {
                        Some(self.2())
                    } else {
                        None
                    },
                    (),
                ),
            ),
        );

        let dims = [D0, D1, D2];
        let mut hessian = BlockMatrix::new(&dims);
        let mut gradient = BlockVector::new(&dims);

        let lambda_res = match precision_mat {
            Some(precision_mat) => precision_mat * residual,
            None => residual,
        };

        for i in 0..SIZE {
            maybe_dx.set_diagonal(
                0,
                i,
                &lambda_res,
                &mut gradient,
                &mut hessian,
                precision_mat,
            );
            for j in i + 1..SIZE {
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

impl<
        F0,
        F1,
        F2,
        F3,
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const R: usize,
    > MakeTerm<R, 4> for (F0, F1, F2, F3)
where
    F0: FnOnce() -> MatF64<R, D0>,
    F1: FnOnce() -> MatF64<R, D1>,
    F2: FnOnce() -> MatF64<R, D2>,
    F3: FnOnce() -> MatF64<R, D3>,
{
    fn make_term<const DIM: usize>(
        self,
        idx: [usize; 4],
        var_kinds: [VarKind; 4],
        residual: VecF64<R>,
        robust_kernel: Option<robust_kernel::RobustKernel>,
        precision_mat: Option<MatF64<R, R>>,
    ) -> Term<DIM, 4> {
        const SIZE: usize = 4;

        let residual = match robust_kernel {
            Some(robust_kernel) => robust_kernel.weight(residual.norm()) * residual,
            None => residual,
        };

        let maybe_dx = (
            if var_kinds[0] != VarKind::Conditioned {
                Some(self.0())
            } else {
                None
            },
            (
                if var_kinds[1] != VarKind::Conditioned {
                    Some(self.1())
                } else {
                    None
                },
                (
                    if var_kinds[2] != VarKind::Conditioned {
                        Some(self.2())
                    } else {
                        None
                    },
                    (
                        if var_kinds[3] != VarKind::Conditioned {
                            Some(self.3())
                        } else {
                            None
                        },
                        (),
                    ),
                ),
            ),
        );

        let dims = [D0, D1, D2, D3];
        let mut hessian = BlockMatrix::new(&dims);
        let mut gradient = BlockVector::new(&dims);

        let lambda_res = match precision_mat {
            Some(precision_mat) => precision_mat * residual,
            None => residual,
        };

        for i in 0..SIZE {
            maybe_dx.set_diagonal(
                0,
                i,
                &lambda_res,
                &mut gradient,
                &mut hessian,
                precision_mat,
            );
            for j in i + 1..SIZE {
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
