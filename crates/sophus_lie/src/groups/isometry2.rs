use crate::groups::rotation2::Rotation2Impl;
use crate::groups::translation_product_product::TranslationProductGroupImpl;
use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::Rotation2;

/// 2D isometry group implementation struct - SE(2)
pub type Isometry2Impl<S, const BATCH: usize> =
    TranslationProductGroupImpl<S, 3, 4, 2, 3, 1, 2, BATCH, Rotation2Impl<S, BATCH>>;

/// 2D isometry group - SE(2)
pub type Isometry2<S, const BATCH: usize> = LieGroup<S, 3, 4, 2, 3, BATCH, Isometry2Impl<S, BATCH>>;

impl<S: IsScalar<BATCH>, const BATCH: usize> Isometry2<S, BATCH> {
    /// create isometry from translation and rotation
    pub fn from_translation_and_rotation(
        translation: &S::Vector<2>,
        rotation: &Rotation2<S, BATCH>,
    ) -> Self {
        Self::from_translation_and_factor(translation, rotation)
    }

    /// set rotation
    pub fn set_rotation(&mut self, rotation: &Rotation2<S, BATCH>) {
        self.set_factor(rotation)
    }

    /// get rotation
    pub fn rotation(&self) -> Rotation2<S, BATCH> {
        self.factor()
    }
}

#[test]
fn isometry2_prop_tests() {
    use crate::real_lie_group::RealLieGroupTest;
    use sophus_core::calculus::dual::dual_scalar::DualBatchScalar;
    use sophus_core::calculus::dual::dual_scalar::DualScalar;
    use sophus_core::linalg::BatchScalarF64;

    Isometry2::<f64, 1>::test_suite();
    Isometry2::<BatchScalarF64<8>, 8>::test_suite();
    Isometry2::<DualScalar, 1>::test_suite();
    Isometry2::<DualBatchScalar<8>, 8>::test_suite();

    Isometry2::<f64, 1>::run_real_tests();
    Isometry2::<BatchScalarF64<8>, 8>::run_real_tests();
}
