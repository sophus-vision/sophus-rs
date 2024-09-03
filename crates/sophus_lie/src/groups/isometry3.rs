use super::rotation3::Rotation3Impl;
use super::translation_product_product::TranslationProductGroupImpl;
use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::Rotation3;

/// 3D isometry group implementation struct - SE(3)
pub type Isometry3Impl<S, const BATCH: usize> =
    TranslationProductGroupImpl<S, 6, 7, 3, 4, 3, 4, BATCH, Rotation3Impl<S, BATCH>>;
/// 3d isometry group - SE(3)
pub type Isometry3<S, const BATCH: usize> = LieGroup<S, 6, 7, 3, 4, BATCH, Isometry3Impl<S, BATCH>>;

/// 3D isometry group with f64 scalar type
pub type Isometry3F64 = Isometry3<f64, 1>;

impl Isometry3F64 {
    /// rotate around x axis
    pub fn rot_x(theta: f64) -> Isometry3F64 {
        Isometry3F64::from_rotation(&Rotation3::rot_x(theta))
    }

    /// rotate around y axis
    pub fn rot_y(theta: f64) -> Isometry3F64 {
        Isometry3F64::from_rotation(&Rotation3::rot_y(theta))
    }

    /// rotate around z axis
    pub fn rot_z(theta: f64) -> Isometry3F64 {
        Isometry3F64::from_rotation(&Rotation3::rot_z(theta))
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> Isometry3<S, BATCH> {
    /// create isometry from translation and rotation
    pub fn from_translation_and_rotation(
        translation: &S::Vector<3>,
        rotation: &Rotation3<S, BATCH>,
    ) -> Self {
        Self::from_translation_and_factor(translation, rotation)
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

#[test]
fn isometry3_prop_tests() {
    use crate::real_lie_group::RealLieGroupTest;
    use sophus_core::calculus::dual::dual_scalar::DualScalar;

    #[cfg(feature = "simd")]
    use sophus_core::calculus::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
    use sophus_core::linalg::BatchScalarF64;

    Isometry3::<f64, 1>::test_suite();
    #[cfg(feature = "simd")]
    Isometry3::<BatchScalarF64<8>, 8>::test_suite();
    Isometry3::<DualScalar, 1>::test_suite();
    #[cfg(feature = "simd")]
    Isometry3::<DualBatchScalar<8>, 8>::test_suite();

    Isometry3::<f64, 1>::run_real_tests();
    #[cfg(feature = "simd")]
    Isometry3::<BatchScalarF64<8>, 8>::run_real_tests();
}
