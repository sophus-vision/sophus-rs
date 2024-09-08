use log::warn;

use super::rotation3::Rotation3Impl;
use super::translation_product_product::TranslationProductGroupImpl;
use crate::average::iterative_average;
use crate::average::IterativeAverageError;
use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::traits::EmptySliceError;
use crate::traits::HasAverage;
use crate::Rotation3;

/// 3D isometry group implementation struct - SE(3)
pub type Isometry3Impl<S, const BATCH: usize> =
    TranslationProductGroupImpl<S, 6, 7, 3, 4, 3, 4, BATCH, Rotation3Impl<S, BATCH>>;
/// 3d isometry group - SE(3)
pub type Isometry3<S, const BATCH: usize> = LieGroup<S, 6, 7, 3, 4, BATCH, Isometry3Impl<S, BATCH>>;

/// 3D isometry group with f64 scalar type
pub type Isometry3F64 = Isometry3<f64, 1>;

impl<S: IsScalar<BATCH>, const BATCH: usize> Isometry3<S, BATCH> {
    /// create isometry from translation and rotation
    pub fn from_translation_and_rotation(
        translation: &S::Vector<3>,
        rotation: &Rotation3<S, BATCH>,
    ) -> Self {
        Self::from_translation_and_factor(translation, rotation)
    }

    /// rotate around x axis
    pub fn rot_x(theta: S) -> Self {
        Self::from_rotation(&Rotation3::rot_x(theta))
    }

    /// rotate around y axis
    pub fn rot_y(theta: S) -> Self {
        Self::from_rotation(&Rotation3::rot_y(theta))
    }

    /// rotate around z axis
    pub fn rot_z(theta: S) -> Self {
        Self::from_rotation(&Rotation3::rot_z(theta))
    }

    /// translate along x axis
    pub fn trans_x(x: S) -> Self {
        Self::from_translation(&S::Vector::from_array([x, S::zero(), S::zero()]))
    }

    /// translate along y axis
    pub fn trans_y(y: S) -> Self {
        Self::from_translation(&S::Vector::from_array([S::zero(), y, S::zero()]))
    }

    /// translate along z axis
    pub fn trans_z(z: S) -> Self {
        Self::from_translation(&S::Vector::from_array([S::zero(), S::zero(), z]))
    }

    /// create isometry from translation
    pub fn from_translation(translation: &S::Vector<3>) -> Self {
        Self::from_translation_and_factor(translation, &Rotation3::identity())
    }

    /// create isometry from rotation
    pub fn from_rotation(rotation: &Rotation3<S, BATCH>) -> Self {
        Self::from_translation_and_factor(&S::Vector::<3>::zeros(), rotation)
    }

    /// set rotation
    pub fn set_rotation(&mut self, rotation: &Rotation3<S, BATCH>) {
        self.set_factor(rotation)
    }

    /// get rotation
    pub fn rotation(&self) -> Rotation3<S, BATCH> {
        self.factor()
    }
}

impl<S: IsSingleScalar + PartialOrd> HasAverage<S, 6, 7, 3, 4> for Isometry3<S, 1> {
    /// Average Isometry3 poses [parent_from_body0, ..., ].
    ///
    /// Note: This function can be used when there is no well-defined body center, since
    ///       this average is right-hand invariance. It does nor depend on what frame on the
    ///       body is chosen.
    ///       If there is a well defined body center for the purpose of averaging, it is likely
    ///       better to average body center positions - using "1/n sum_i pos_i" - and rotations
    ///       independently.
    fn average(parent_from_body_transforms: &[Isometry3<S, 1>]) -> Result<Self, EmptySliceError> {
        match iterative_average(parent_from_body_transforms, 50) {
            Ok(parent_from_body_average) => Ok(parent_from_body_average),
            Err(err) => match err {
                IterativeAverageError::EmptySlice => Err(EmptySliceError),
                IterativeAverageError::NotConverged(not_conv) => {
                    warn!(
                        "iterative_average did not converge (iters={}), returning best guess.",
                        not_conv.max_iteration_count
                    );
                    Ok(not_conv.parent_from_body_estimate)
                }
            },
        }
    }
}

#[test]
fn isometry3_prop_tests() {
    use crate::real_lie_group::RealLieGroupTest;
    use sophus_core::calculus::dual::dual_scalar::DualScalar;

    #[cfg(feature = "simd")]
    use sophus_core::calculus::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
    use sophus_core::linalg::BatchScalarF64;

    Isometry3F64::test_suite();
    #[cfg(feature = "simd")]
    Isometry3::<BatchScalarF64<8>, 8>::test_suite();
    Isometry3::<DualScalar, 1>::test_suite();
    #[cfg(feature = "simd")]
    Isometry3::<DualBatchScalar<8>, 8>::test_suite();

    Isometry3F64::run_real_tests();
    #[cfg(feature = "simd")]
    Isometry3::<BatchScalarF64<8>, 8>::run_real_tests();
}
