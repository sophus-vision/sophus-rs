use core::borrow::Borrow;

use crate::groups::rotation2::Rotation2Impl;
use crate::groups::translation_product_product::TranslationProductGroupImpl;
use crate::lie_group::average::iterative_average;
use crate::lie_group::average::IterativeAverageError;
use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::traits::EmptySliceError;
use crate::traits::HasAverage;
use crate::Rotation2;

use log::warn;

/// 2D isometry group implementation struct - SE(2)
pub type Isometry2Impl<S, const BATCH: usize> =
    TranslationProductGroupImpl<S, 3, 4, 2, 3, 1, 2, BATCH, Rotation2Impl<S, BATCH>>;

/// 2D isometry group - SE(2)
pub type Isometry2<S, const BATCH: usize> = LieGroup<S, 3, 4, 2, 3, BATCH, Isometry2Impl<S, BATCH>>;

/// 2D isometry group with f64 scalar type
pub type Isometry2F64 = Isometry2<f64, 1>;

impl<S: IsScalar<BATCH>, const BATCH: usize> Isometry2<S, BATCH> {
    /// create isometry from translation and rotation
    pub fn from_translation_and_rotation<P, F>(translation: P, rotation: F) -> Self
    where
        P: Borrow<S::Vector<2>>,
        F: Borrow<Rotation2<S, BATCH>>,
    {
        Self::from_translation_and_factor(translation, rotation)
    }

    /// create isometry from translation
    pub fn from_translation<P>(translation: P) -> Self
    where
        P: Borrow<S::Vector<2>>,
    {
        Self::from_translation_and_factor(translation, Rotation2::identity())
    }

    /// create isometry from rotation
    pub fn from_rotation<F>(rotation: F) -> Self
    where
        F: Borrow<Rotation2<S, BATCH>>,
    {
        Self::from_translation_and_factor(S::Vector::<2>::zeros(), rotation)
    }

    /// translate along x axis
    pub fn trans_x<U>(x: U) -> Self
    where
        U: Borrow<S>,
    {
        let x: &S = x.borrow();
        Self::from_translation(S::Vector::from_array([x.clone(), S::zero()]))
    }

    /// translate along y axis
    pub fn trans_y<U>(y: U) -> Self
    where
        U: Borrow<S>,
    {
        let y: &S = y.borrow();
        Self::from_translation(S::Vector::from_array([S::zero(), y.clone()]))
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
        F: Borrow<Rotation2<S, BATCH>>,
    {
        self.set_factor(rotation)
    }

    /// get rotation
    pub fn rotation(&self) -> Rotation2<S, BATCH> {
        self.factor()
    }
}

impl<S: IsSingleScalar + PartialOrd> HasAverage<S, 3, 4, 2, 3> for Isometry2<S, 1> {
    /// Average Isometry2 poses [parent_from_body0, ..., ].
    ///
    /// Note: This function can be used when there is no well-defined body center, since
    ///       this average is right-hand invariance. It does nor depend on what frame on the
    ///       body is chosen.
    ///       If there is a well defined body center for the purpose of averaging, it is likely
    ///       better to average body center positions - using "1/n sum_i pos_i" - and rotations
    ///       independently.
    fn average(parent_from_body_transforms: &[Isometry2<S, 1>]) -> Result<Self, EmptySliceError> {
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
    use crate::lie_group::real_lie_group::RealLieGroupTest;
    use sophus_core::calculus::dual::dual_scalar::DualScalar;

    #[cfg(feature = "simd")]
    use sophus_core::calculus::dual::dual_batch_scalar::DualBatchScalar;
    #[cfg(feature = "simd")]
    use sophus_core::linalg::BatchScalarF64;

    Isometry2::<f64, 1>::test_suite();
    #[cfg(feature = "simd")]
    Isometry2::<BatchScalarF64<8>, 8>::test_suite();
    Isometry2::<DualScalar, 1>::test_suite();
    #[cfg(feature = "simd")]
    Isometry2::<DualBatchScalar<8>, 8>::test_suite();

    Isometry2::<f64, 1>::run_real_tests();
    #[cfg(feature = "simd")]
    Isometry2::<BatchScalarF64<8>, 8>::run_real_tests();
}
