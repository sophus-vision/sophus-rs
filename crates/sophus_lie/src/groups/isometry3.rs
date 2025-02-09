use core::borrow::Borrow;

use log::warn;

use super::{
    rotation3::Rotation3Impl,
    translation_product_product::TranslationProductGroupImpl,
};
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
    Rotation3,
};

/// 3D isometry group implementation struct - SE(3)
pub type Isometry3Impl<S, const BATCH: usize, const DM: usize, const DN: usize> =
    TranslationProductGroupImpl<
        S,
        6,
        7,
        3,
        4,
        3,
        4,
        BATCH,
        DM,
        DN,
        Rotation3Impl<S, BATCH, DM, DN>,
    >;
/// 3d isometry group - SE(3)
pub type Isometry3<S, const BATCH: usize, const DM: usize, const DN: usize> =
    LieGroup<S, 6, 7, 3, 4, BATCH, DM, DN, Isometry3Impl<S, BATCH, DM, DN>>;

/// 3D isometry group with f64 scalar type
pub type Isometry3F64 = Isometry3<f64, 1, 0, 0>;

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Isometry3<S, BATCH, DM, DN>
{
    /// create isometry from translation and rotation
    pub fn from_translation_and_rotation<F>(translation: S::Vector<3>, rotation: F) -> Self
    where
        F: Borrow<Rotation3<S, BATCH, DM, DN>>,
    {
        Self::from_translation_and_factor(translation, rotation)
    }

    /// create isometry from translation
    pub fn from_translation(translation: S::Vector<3>) -> Self {
        Self::from_translation_and_factor(translation, Rotation3::identity())
    }

    /// create isometry from rotation
    pub fn from_rotation<F>(rotation: F) -> Self
    where
        F: Borrow<Rotation3<S, BATCH, DM, DN>>,
    {
        Self::from_translation_and_factor(S::Vector::<3>::zeros(), rotation)
    }

    /// translate along x axis
    pub fn trans_x<U>(x: U) -> Self
    where
        U: Borrow<S>,
    {
        let x: &S = x.borrow();
        Self::from_translation(S::Vector::from_array([*x, S::zero(), S::zero()]))
    }

    /// translate along y axis
    pub fn trans_y<U>(y: U) -> Self
    where
        U: Borrow<S>,
    {
        let y: &S = y.borrow();
        Self::from_translation(S::Vector::from_array([S::zero(), *y, S::zero()]))
    }

    /// translate along z axis
    pub fn trans_z<U>(z: U) -> Self
    where
        U: Borrow<S>,
    {
        let z: &S = z.borrow();
        Self::from_translation(S::Vector::from_array([S::zero(), S::zero(), *z]))
    }

    /// Rotate by angle
    pub fn rot_x<U>(theta: U) -> Self
    where
        U: Borrow<S>,
    {
        let theta: &S = theta.borrow();
        Self::from_rotation(Rotation3::rot_x(theta))
    }

    /// Rotate by angle
    pub fn rot_y<U>(theta: U) -> Self
    where
        U: Borrow<S>,
    {
        let theta: &S = theta.borrow();
        Self::from_rotation(Rotation3::rot_y(theta))
    }

    /// Rotate by angle
    pub fn rot_z<U>(theta: U) -> Self
    where
        U: Borrow<S>,
    {
        let theta: &S = theta.borrow();
        Self::from_rotation(Rotation3::rot_z(theta))
    }

    /// set rotation
    pub fn set_rotation<F>(&mut self, rotation: F)
    where
        F: Borrow<Rotation3<S, BATCH, DM, DN>>,
    {
        self.set_factor(rotation)
    }

    /// get rotation
    pub fn rotation(&self) -> Rotation3<S, BATCH, DM, DN> {
        self.factor()
    }
}

impl<S: IsSingleScalar<DM, DN> + PartialOrd, const DM: usize, const DN: usize>
    HasAverage<S, 6, 7, 3, 4, DM, DN> for Isometry3<S, 1, DM, DN>
{
    /// Average Isometry3 poses [parent_from_body0, ..., ].
    ///
    /// Note: This function can be used when there is no well-defined body center, since
    ///       this average is right-hand invariance. It does nor depend on what frame on the
    ///       body is chosen.
    ///       If there is a well defined body center for the purpose of averaging, it is likely
    ///       better to average body center positions - using "1/n sum_i pos_i" - and rotations
    ///       independently.
    fn average(
        parent_from_body_transforms: &[Isometry3<S, 1, DM, DN>],
    ) -> Result<Self, EmptySliceError> {
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
fn isometry3_prop_tests() {
    use sophus_autodiff::dual::dual_scalar::DualScalar;
    #[cfg(feature = "simd")]
    use sophus_autodiff::dual::DualBatchScalar;
    #[cfg(feature = "simd")]
    use sophus_autodiff::linalg::BatchScalarF64;

    use crate::lie_group::real_lie_group::RealLieGroupTest;

    Isometry3F64::test_suite();
    #[cfg(feature = "simd")]
    Isometry3::<BatchScalarF64<8>, 8, 0, 0>::test_suite();
    Isometry3::<DualScalar<1, 1>, 1, 1, 1>::test_suite();
    #[cfg(feature = "simd")]
    Isometry3::<DualBatchScalar<8, 1, 1>, 8, 1, 1>::test_suite();

    Isometry3F64::run_real_tests();
    #[cfg(feature = "simd")]
    Isometry3::<BatchScalarF64<8>, 8, 0, 0>::run_real_tests();
}
