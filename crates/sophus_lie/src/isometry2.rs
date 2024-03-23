use super::{
    lie_group::LieGroup, rotation2::Rotation2Impl,
    translation_product_product::TranslationProductGroupImpl,
};
use crate::rotation2::Rotation2;
use crate::traits::IsTranslationProductGroup;
use sophus_calculus::types::scalar::IsScalar;

/// 2D isometry group implementation struct - SE(2)
pub type Isometry2Impl<S> = TranslationProductGroupImpl<S, 3, 4, 2, 3, 1, 2, Rotation2Impl<S>>;

/// 2D isometry group - SE(2)
pub type Isometry2<S> = LieGroup<S, 3, 4, 2, 3, 1, Isometry2Impl<S>>;

impl<S: IsScalar<1>> Isometry2<S> {
    /// create isometry from translation and rotation
    pub fn from_translation_and_rotation(
        translation: &<S as IsScalar<1>>::Vector<2>,
        rotation: &Rotation2<S>,
    ) -> Self {
        Self::from_translation_and_factor(translation, rotation)
    }

    /// set rotation
    pub fn set_rotation(&mut self, rotation: &Rotation2<S>) {
        self.set_factor(rotation)
    }

    /// get rotation
    pub fn rotation(&self) -> Rotation2<S> {
        self.factor()
    }
}

mod tests {

    #[test]
    fn isometry2_prop_tests() {
        use super::Isometry2;
        use sophus_calculus::dual::dual_scalar::Dual;

        Isometry2::<f64>::test_suite();
        Isometry2::<Dual>::test_suite();
        Isometry2::<f64>::real_test_suite();
    }
}
