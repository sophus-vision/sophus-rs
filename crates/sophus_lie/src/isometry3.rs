use super::{
    lie_group::LieGroup,
    rotation3::{Isometry3Impl, Rotation3},
};
use crate::traits::IsTranslationProductGroup;
use sophus_calculus::types::scalar::IsScalar;

/// 3d isometry group - SE(3)
pub type Isometry3<S> = LieGroup<S, 6, 7, 3, 4, 1, Isometry3Impl<S>>;

impl<S: IsScalar<1>> Isometry3<S> {
    /// create isometry from translation and rotation
    pub fn from_translation_and_rotation(
        translation: &<S as IsScalar<1>>::Vector<3>,
        rotation: &Rotation3<S>,
    ) -> Self {
        Self::from_translation_and_factor(translation, rotation)
    }

    /// set rotation
    pub fn set_rotation(&mut self, rotation: &Rotation3<S>) {
        self.set_factor(rotation)
    }

    /// get translation
    pub fn rotation(&self) -> Rotation3<S> {
        self.factor()
    }
}

impl Default for Isometry3<f64> {
    fn default() -> Self {
        Self::identity()
    }
}

mod tests {

    #[test]
    fn isometry3_prop_tests() {
        use super::Isometry3;
        use crate::traits::IsTranslationProductGroup;
        use sophus_calculus::dual::dual_scalar::Dual;

        Isometry3::<f64>::test_suite();
        Isometry3::<Dual>::test_suite();
        Isometry3::<f64>::real_test_suite();

        for g in Isometry3::<f64>::element_examples() {
            let translation = g.translation();
            let rotation = g.rotation();

            let g2 = Isometry3::from_translation_and_rotation(&translation, &rotation);
            assert_eq!(g2.translation(), translation);
            assert_eq!(g2.rotation().matrix(), rotation.matrix());
        }
    }
}
