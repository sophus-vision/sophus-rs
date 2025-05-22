use log::warn;

use crate::{
    AffineGroupTemplateImpl,
    EmptySliceError,
    HasAverage,
    Rotation2,
    Rotation2Impl,
    lie_group::{
        LieGroup,
        average::{
            IterativeAverageError,
            iterative_average,
        },
    },
    prelude::*,
};

/// 2-d **isometry** – element of the Special Euclidean group **SE(2)**
///
/// ## Generic parameters
/// * **BATCH** – batch dimension ( = 1 for plain `f64` or `DualScalar`).
/// * **DM, DN** – static Jacobian shape when `S = DualScalar<DM,DN>` (both 0 when `S = f64`).
///
/// ## Overview
///
/// * **Tangent space:** 3 DoF – **[ ϑ , ν ]**, with `ϑ` the **angular** rate and `ν` the 2-d
///   **linear** rate.
/// * **Internal parameters:** 4 – **[ z , t ]**, complex number `z` ( |q| = 1 ) and translation `t
///   ∈ ℝ²`.
/// * **Action space:** 2 (SE(2) acts on 2-d points)
/// * **Matrix size:** 2 (represented as 2 × 2 matrices)
///
/// ### Group structure
///
/// *Matrix representation*
/// ```ascii
/// ---------
/// | R | p |
/// ---------
/// | O | 1 |
/// ---------
/// ```
/// `R ∈ SO(2)`, `p ∈ ℝ²`.
///
/// *Group operation*
/// ```ascii
/// (Rₗ, pₗ) ⊗ (Rᵣ, pᵣ) = ( Rₗ·Rᵣ,  Rₗ·pᵣ + pₗ )
/// ```
/// *Inverse*
/// ```ascii
/// (R, p)⁻¹ = ( Rᵀ,  -Rᵀ·p )
/// ```
///
/// ### Lie-group properties
///
/// **Hat operator** `x = [ ϑ, ν₀, ν₁, ν₂ ]`
/// ```ascii
///            -------------------
///            |  0  | -ϑ  |  ν₀ |
///  /ϑ \^     -------------------
/// | ν₀ | =   |  ϑ  |  0  |  ν₁ |
///  \ν₁/      -------------------
///            |  0  |  0  |  0  |
///            -------------------
/// ```
///
/// **Exponential map** `exp : ℝ³ → SE(2)`
/// ```ascii
/// exp(ϑ,ν) = ( exp_so2(ϑ),  V(ϑ) · ν )
/// ```
/// where `V(ϑ)` is `Rotation2::mat_v`.
///
/// **Group adjoint** `Adj : SE(2) → ℝ³ˣ³` (acts on `[ ϑ ; ν ]`)
/// ```ascii
///               -----------------
///               |  1    |  O₁ₓ₂ |
///     /ϑ \      |---------------|
/// Adj| ν₀ |  =  |  p₁   |       |
///     \ν₁/      |       |   R   |
///               | -p₀   |       |
///               -----------------
/// ```
///
/// `Adj(A,p)` acts on `(ϑ; ν)`
/// ```ascii
/// Adj(R,p) · (ϑ; ν₀; ν₁)  =  ( ϑ , p₁ϑ - ν₁ϑ + Rν )
/// ```
///
/// **Lie-algebra adjoint** `ad : se(2) →  ℝ³ˣ³`
/// ```ascii
///              -------------------
///              |  0  |  0  |  0  |
///    /φ \      |-----------------|
/// ad| τ₀ |  =  |  ν₁ |  0  | -ϑ  |
///    \τ₁/      |-----------------|
///              | -ν₀ |  ϑ  |  0  |
///              -------------------
/// ```
///
/// ad(α; ν)` acts on `(φ; τ)`
/// ```ascii
/// ad(α; ν) · (φ; τ₀; τ₁)  =  ( ν₁φ - ν₀φ, -ϑτ₀, ϑτ₁ )
/// ```
pub type Isometry2<S, const BATCH: usize, const DM: usize, const DN: usize> =
    LieGroup<S, 3, 4, 2, 3, BATCH, DM, DN, Isometry2Impl<S, BATCH, DM, DN>>;

/// 2d isometry with f64 scalar type - element of the Special Euclidean group SE(2)
///
/// See [Isometry2] for details.
pub type Isometry2F64 = Isometry2<f64, 1, 0, 0>;

/// 2d isometry implementation.
///
/// See [Isometry2] for details.
pub type Isometry2Impl<S, const BATCH: usize, const DM: usize, const DN: usize> =
    AffineGroupTemplateImpl<S, 3, 4, 2, 3, 1, 2, BATCH, DM, DN, Rotation2Impl<S, BATCH, DM, DN>>;

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Isometry2<S, BATCH, DM, DN>
{
    /// create isometry from rotation and translation
    pub fn from_rotation_and_translation(
        rotation: Rotation2<S, BATCH, DM, DN>,
        translation: S::Vector<2>,
    ) -> Self {
        Self::from_factor_and_translation(rotation, translation)
    }

    /// create isometry from translation
    pub fn from_translation(translation: S::Vector<2>) -> Self {
        Self::from_factor_and_translation(Rotation2::identity(), translation)
    }

    /// create isometry from rotation
    pub fn from_rotation(rotation: Rotation2<S, BATCH, DM, DN>) -> Self {
        Self::from_factor_and_translation(rotation, S::Vector::<2>::zeros())
    }

    /// translate along x axis
    pub fn trans_x<U>(x: S) -> Self {
        Self::from_translation(S::Vector::from_array([x, S::zero()]))
    }

    /// translate along y axis
    pub fn trans_y<U>(y: S) -> Self {
        Self::from_translation(S::Vector::from_array([S::zero(), y]))
    }

    /// Rotate by angle
    pub fn rot(theta: S) -> Self {
        Self::from_rotation(Rotation2::rot(theta))
    }

    /// set rotation
    pub fn set_rotation(&mut self, rotation: Rotation2<S, BATCH, DM, DN>) {
        self.set_factor(rotation)
    }

    /// get rotation
    pub fn rotation(&self) -> Rotation2<S, BATCH, DM, DN> {
        self.factor()
    }
}

impl From<nalgebra::Isometry2<f64>> for Isometry2F64 {
    fn from(isometry: nalgebra::Isometry2<f64>) -> Self {
        Self::from_rotation_and_translation(isometry.rotation.into(), isometry.translation.vector)
    }
}

impl From<Isometry2F64> for nalgebra::Isometry2<f64> {
    fn from(val: Isometry2F64) -> Self {
        let translation = val.translation();
        let rotation = val.rotation();
        nalgebra::Isometry2::from_parts(translation.into(), rotation.into())
    }
}

impl<S: IsSingleScalar<DM, DN> + PartialOrd, const DM: usize, const DN: usize>
    HasAverage<S, 3, 4, 2, 3, DM, DN> for Isometry2<S, 1, DM, DN>
{
    /// Average Isometry2 poses [parent_from_body0, ..., ].
    ///
    /// Note: This function can be used when there is no well-defined body center, since
    ///       this average is right-hand invariance. It does nor depend on what frame on the
    ///       body is chosen.
    ///       If there is a well defined body center for the purpose of averaging, it is likely
    ///       better to average body center positions - using "1/n sum_i pos_i" - and rotations
    ///       independently.
    fn average(
        parent_from_body_transforms: &[Isometry2<S, 1, DM, DN>],
    ) -> Result<Self, EmptySliceError> {
        // todo: Implement close form solution from
        //       ftp://ftp-sop.inria.fr/epidaure/Publications/Arsigny/arsigny_rr_biinvariant_average.pdf.
        match iterative_average(parent_from_body_transforms, 50) {
            Ok(parent_from_body_average) => Ok(parent_from_body_average),
            Err(err) => match err {
                IterativeAverageError::EmptySlice => Err(EmptySliceError),
                IterativeAverageError::NotConverged {
                    max_iteration_count,
                    parent_from_body_estimate,
                } => {
                    warn!(
                        "iterative_average did not converge (iters={max_iteration_count}), returning best guess."
                    );
                    Ok(parent_from_body_estimate)
                }
            },
        }
    }
}

#[test]
fn isometry2_prop_tests() {
    #[cfg(feature = "simd")]
    use sophus_autodiff::dual::DualBatchScalar;
    use sophus_autodiff::dual::DualScalar;
    #[cfg(feature = "simd")]
    use sophus_autodiff::linalg::BatchScalarF64;

    use crate::lie_group::real_lie_group::RealLieGroupTest;

    Isometry2F64::test_suite();
    #[cfg(feature = "simd")]
    Isometry2::<BatchScalarF64<8>, 8, 0, 0>::test_suite();
    Isometry2::<DualScalar<1, 1>, 1, 1, 1>::test_suite();
    #[cfg(feature = "simd")]
    Isometry2::<DualBatchScalar<8, 1, 1>, 8, 1, 1>::test_suite();

    Isometry2F64::run_real_tests();
    #[cfg(feature = "simd")]
    Isometry2::<BatchScalarF64<8>, 8, 0, 0>::run_real_tests();
}

#[test]
fn test_nalgebra_interop() {
    use approx::assert_relative_eq;
    use sophus_autodiff::linalg::VecF64;

    use crate::Rotation2F64;

    let isometry = Isometry2F64::from_rotation_and_translation(
        Rotation2F64::exp(VecF64::<1>::new(0.5)),
        VecF64::from_array([1.0, 2.0]),
    );
    let na_isometry: nalgebra::Isometry2<f64> = isometry.into();
    assert_eq!(isometry.translation(), na_isometry.translation.vector);
    assert_relative_eq!(
        isometry.rotation().log()[0],
        na_isometry.rotation.angle(),
        epsilon = 1e-10
    );

    let roundtrip_isometry = Isometry2F64::from(na_isometry);
    assert_relative_eq!(isometry.translation(), roundtrip_isometry.translation());
    assert_relative_eq!(
        isometry.rotation().params(),
        roundtrip_isometry.rotation().params(),
        epsilon = 1e-10
    );
}
