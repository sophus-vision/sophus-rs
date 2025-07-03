## Sparse non-linear least squares optimization

This crate provides a general sparse non-linear least squares optimization
library, similar to the C++ libraries [ceres-solver](http://ceres-solver.org/),
[g2o](https://github.com/RainerKuemmerle/g2o) and [gtsam](https://gtsam.org/).
It supports automatic differentiation through the
[sophus-autodiff](https://crates.io/crates/sophus-autodiff) crate, as well
optimization on manifolds  and Lie groups through the
[sophus_autodiff::manifold::IsVariable] trait and the
[sophus-lie](https://crates.io/crates/sophus-lie) crate.

## Example

```rust
use sophus_opt::prelude::*;
use sophus_autodiff::linalg::{MatF64, VecF64};
use sophus_autodiff::dual::DualVector;
use sophus_lie::{Isometry2, Isometry2F64, Rotation2F64};
use sophus_opt::nlls::{CostFn, CostTerms, EvaluatedCostTerm, optimize_nlls, OptParams};
use sophus_opt::robust_kernel;
use sophus_opt::variables::{VarBuilder, VarFamily, VarKind};

// We want to fit the isometry `T ∈ SE(2)` to a prior distribution
// `N(E(T), W⁻¹)`, where `E(T)` is the prior mean and `W⁻¹` is the prior
// covariance matrix.

// (1) First we define the residual cost term.
#[derive(Clone, Debug)]
pub struct Isometry2PriorCostTerm {
    // Prior mean, `E(T)` of type [Isometry2F64].
    pub isometry_prior_mean: Isometry2F64,
    // `W`, which is the inverse of the prior covariance matrix.
    pub isometry_prior_precision: MatF64<3, 3>,
    // We only have one variable, so this will be `[0]`.
    pub entity_indices: [usize; 1],
}

impl Isometry2PriorCostTerm {
    // (2) Then we define  residual function for the cost term:
    //
    // `g(T) = log[T * E(T)⁻¹]`
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        isometry: Isometry2<Scalar, 1, DM, DN>,
        isometry_prior_mean: Isometry2<Scalar, 1, DM, DN>,
    ) -> Scalar::Vector<3> {
        (isometry * isometry_prior_mean.inverse()).log()
    }
}

// (3) Implement the `HasResidualFn` trait for the cost term.
impl HasResidualFn<3, 1, (), Isometry2F64> for Isometry2PriorCostTerm {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: Isometry2F64,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<robust_kernel::RobustKernel>,
    ) -> EvaluatedCostTerm<3, 1> {
        let isometry: Isometry2F64 = args;

        let residual = Self::residual(isometry, self.isometry_prior_mean);
        let dx_res_fn = |x: DualVector<3, 3, 1>| -> DualVector<3, 3, 1> {
            Self::residual(
                Isometry2::exp(x) * isometry.to_dual_c(),
                self.isometry_prior_mean.to_dual_c(),
            )
        };

        (|| dx_res_fn(DualVector::var(VecF64::<3>::zeros())).jacobian(),).make(
            idx,
            var_kinds,
            residual,
            robust_kernel,
            Some(self.isometry_prior_precision),
        )
    }
}

let prior_world_from_robot = Isometry2F64::from_translation(
    VecF64::<2>::new(1.0, 2.0),
);

// (4) Define the cost terms, by specifying the residual function
// `g(T) = Isometry2PriorCostTerm` as well as providing the prior distribution.
const POSE: &str = "poses";
let obs_pose_a_from_pose_b_poses = CostTerms::new(
    [POSE],
    vec![Isometry2PriorCostTerm {
        isometry_prior_mean: prior_world_from_robot,
        isometry_prior_precision: MatF64::<3, 3>::identity(),
        entity_indices: [0],
    }],
);

// (5) Define the decision variables. In this case, we only have one variable,
// and we initialize it with the identity transformation.
let est_world_from_robot = Isometry2F64::identity();
let variables = VarBuilder::new()
    .add_family(
        POSE,
        VarFamily::new(VarKind::Free, vec![est_world_from_robot]),
    )
    .build();

// (6) Perform the non-linear least squares optimization.
let solution = optimize_nlls(
    variables,
    vec![CostFn::new_boxed((), obs_pose_a_from_pose_b_poses.clone(),)],
    OptParams::default(),
)
.unwrap();

// (7) Retrieve the refined transformation and compare it with the prior one.
let refined_world_from_robot
    = solution.variables.get_members::<Isometry2F64>(POSE)[0];
approx::assert_abs_diff_eq!(
    prior_world_from_robot.compact(),
    refined_world_from_robot.compact(),
    epsilon = 1e-6,
);
```

In the code above, we define an [crate::nlls::costs::Isometry2PriorCostTerm]
struct that imposes a Gaussian prior on an [sophus_lie::Isometry2F64]
pose, specifying a prior mean `E(T)` and a precision matrix `W`. The key
operation is the residual function `g(T) = log[T * E(T)⁻¹]`. This maps the
difference between the current estimate \(T\) and the prior mean \(E(T)\) into
the tangent space of the [Lie group](sophus_lie::LieGroup).

We then implement the [nlls::HasResidualFn] trait for our cost term
so that it can be evaluated inside the solver. In the `eval` method, automatic
differentiation with [sophus_autodiff::dual::DualVector] computes the Jacobian
of the residual. The returned [nlls::EvaluatedCostTerm] includes
both the residual vector and its associated covariance.

Next, we group our cost terms with [crate::nlls::CostTerms] and associate
them with one or more variable “families” via the [crate::variables::VarBuilder].
Here, we create a single family [crate::variables::VarFamily] named
`"poses"` for the free variable, initialized to the identity transform. Finally,
we call the sparse non-linear least squares solver [crate::nlls::optimize_nlls]
with these variables and the wrapped cost function [crate::nlls::CostFn],
passing in tuning parameters [crate::nlls::OptParams]; we are using the
default here. Upon convergence, the solver returns updated variables that we
retrieve from the solution using [crate::variables::VarFamilies::get_members].

## Integration with sophus-rs

This crate is part of the [sophus umbrella crate](https://crates.io/crates/sophus).
It re-exports the relevant prelude types under [prelude], so you can
seamlessly interoperate with the rest of the sophus-rs types.

Also check out alternative non-linear least squares crates:

* [fact-rs](https://crates.io/crates/factrs)
* [tiny-solver-rs](https://crates.io/crates/tiny-solver)
