use core::borrow::Borrow;

use log::warn;

use crate::{
    lie_group::{
        average::{
            iterative_average,
            IterativeAverageError,
        },
        LieGroup,
    },
    prelude::*,
    EmptySliceError,
    HasAverage,
    Rotation2,
    Rotation2Impl,
    TranslationProductGroupImpl,
};

/// 2d isometry - element of the Special Euclidean group SE(2)
///
///  * BATCH
///     - batch dimension. If S is f64 or [sophus_autodiff::dual::DualScalar] then BATCH=1.
///  * DM, DN
///     - DM x DN is the static shape of the Jacobian to be computed if S == DualScalar<DM,DN>. If S
///       == f64, then DM==0, DN==0.
pub type Isometry2<S, const BATCH: usize, const DM: usize, const DN: usize> =
    LieGroup<S, 3, 4, 2, 3, BATCH, DM, DN, Isometry2Impl<S, BATCH, DM, DN>>;

/// 2d isometry with f64 scalar type - element of the Special Euclidean group SE(2)
///
/// See [Isometry2] for details.
pub type Isometry2F64 = Isometry2<f64, 1, 0, 0>;

/// 2d isometry implementation details
pub type Isometry2Impl<S, const BATCH: usize, const DM: usize, const DN: usize> =
    TranslationProductGroupImpl<
        S,
        3,
        4,
        2,
        3,
        1,
        2,
        BATCH,
        DM,
        DN,
        Rotation2Impl<S, BATCH, DM, DN>,
    >;

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Isometry2<S, BATCH, DM, DN>
{
    /// create isometry from translation and rotation
    pub fn from_translation_and_rotation<F>(translation: S::Vector<2>, rotation: F) -> Self
    where
        F: Borrow<Rotation2<S, BATCH, DM, DN>>,
    {
        Self::from_translation_and_factor(translation, rotation)
    }

    /// create isometry from translation
    pub fn from_translation(translation: S::Vector<2>) -> Self {
        Self::from_translation_and_factor(translation, Rotation2::identity())
    }

    /// create isometry from rotation
    pub fn from_rotation<F>(rotation: F) -> Self
    where
        F: Borrow<Rotation2<S, BATCH, DM, DN>>,
    {
        Self::from_translation_and_factor(S::Vector::<2>::zeros(), rotation)
    }

    /// translate along x axis
    pub fn trans_x<U>(x: U) -> Self
    where
        U: Borrow<S>,
    {
        let x: &S = x.borrow();
        Self::from_translation(S::Vector::from_array([*x, S::zero()]))
    }

    /// translate along y axis
    pub fn trans_y<U>(y: U) -> Self
    where
        U: Borrow<S>,
    {
        let y: &S = y.borrow();
        Self::from_translation(S::Vector::from_array([S::zero(), *y]))
    }

    /// Rotate by angle
    pub fn rot<U>(theta: U) -> Self
    where
        U: Borrow<S>,
    {
        let theta: &S = theta.borrow();
        Self::from_rotation(Rotation2::rot(theta))
    }

    /// set rotation
    pub fn set_rotation<F>(&mut self, rotation: F)
    where
        F: Borrow<Rotation2<S, BATCH, DM, DN>>,
    {
        self.set_factor(rotation)
    }

    /// get rotation
    pub fn rotation(&self) -> Rotation2<S, BATCH, DM, DN> {
        self.factor()
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
                        "iterative_average did not converge (iters={}), returning best guess.",
                        max_iteration_count
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
