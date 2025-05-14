use log::warn;

use super::{
    rotation3::Rotation3Impl,
    translation_product_product::TranslationProductGroupImpl,
};
use crate::{
    EmptySliceError,
    HasAverage,
    Rotation3,
    lie_group::{
        LieGroup,
        average::{
            IterativeAverageError,
            iterative_average,
        },
    },
    prelude::*,
};

/// 3d isometry - element of the Special Euclidean group SE(3)
///
///  * BATCH
///     - batch dimension. If S is f64 or [sophus_autodiff::dual::DualScalar] then BATCH=1.
///  * DM, DN
///     - DM x DN is the static shape of the Jacobian to be computed if S == DualScalar<DM,DN>. If S
///       == f64, then DM==0, DN==0.
pub type Isometry3<S, const BATCH: usize, const DM: usize, const DN: usize> =
    LieGroup<S, 6, 7, 3, 4, BATCH, DM, DN, Isometry3Impl<S, BATCH, DM, DN>>;

/// 3d isometry with f64 scalar type - element of the Special Euclidean group SE(3)
///
/// See [Isometry3] for details.
pub type Isometry3F64 = Isometry3<f64, 1, 0, 0>;

/// 3d isometry implementation details
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

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Isometry3<S, BATCH, DM, DN>
{
    /// create isometry from translation and rotation
    pub fn from_translation_and_rotation(
        translation: S::Vector<3>,
        rotation: Rotation3<S, BATCH, DM, DN>,
    ) -> Self {
        Self::from_translation_and_factor(translation, rotation)
    }

    /// create isometry from translation
    pub fn from_translation(translation: S::Vector<3>) -> Self {
        Self::from_translation_and_factor(translation, Rotation3::identity())
    }

    /// create isometry from rotation
    pub fn from_rotation(rotation: Rotation3<S, BATCH, DM, DN>) -> Self {
        Self::from_translation_and_factor(S::Vector::<3>::zeros(), rotation)
    }

    /// translate along x axis
    pub fn trans_x(x: S) -> Self {
        Self::from_translation(S::Vector::from_array([x, S::zero(), S::zero()]))
    }

    /// translate along y axis
    pub fn trans_y(y: S) -> Self {
        Self::from_translation(S::Vector::from_array([S::zero(), y, S::zero()]))
    }

    /// translate along z axis
    pub fn trans_z(z: S) -> Self {
        Self::from_translation(S::Vector::from_array([S::zero(), S::zero(), z]))
    }

    /// Rotate by angle
    pub fn rot_x(theta: S) -> Self {
        Self::from_rotation(Rotation3::rot_x(theta))
    }

    /// Rotate by angle
    pub fn rot_y(theta: S) -> Self {
        Self::from_rotation(Rotation3::rot_y(theta))
    }

    /// Rotate by angle
    pub fn rot_z(theta: S) -> Self {
        Self::from_rotation(Rotation3::rot_z(theta))
    }

    /// set rotation
    pub fn set_rotation(&mut self, rotation: Rotation3<S, BATCH, DM, DN>) {
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
                        "iterative_average did not converge (iters={max_iteration_count}), returning best guess."
                    );
                    Ok(parent_from_body_estimate)
                }
            },
        }
    }
}

impl From<nalgebra::Isometry3<f64>> for Isometry3F64 {
    fn from(isometry: nalgebra::Isometry3<f64>) -> Self {
        Self::from_translation_and_rotation(isometry.translation.vector, isometry.rotation.into())
    }
}

impl From<Isometry3F64> for nalgebra::Isometry3<f64> {
    fn from(val: Isometry3F64) -> Self {
        let translation = val.translation();
        let rotation = val.rotation();
        nalgebra::Isometry3::from_parts(translation.into(), rotation.into())
    }
}

#[test]
fn isometry3_prop_tests() {
    #[cfg(feature = "simd")]
    use sophus_autodiff::dual::DualBatchScalar;
    use sophus_autodiff::dual::DualScalar;
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

#[test]
fn test_nalgebra_interop() {
    use approx::assert_relative_eq;
    use sophus_autodiff::linalg::VecF64;

    let isometry = Isometry3F64::from_translation_and_rotation(
        VecF64::from_array([1.0, 2.0, 3.0]),
        Rotation3::rot_x(0.5),
    );
    let na_isometry: nalgebra::Isometry3<f64> = isometry.into();
    assert_eq!(isometry.translation(), na_isometry.translation.vector);
    assert_relative_eq!(
        isometry.rotation().params()[0],
        na_isometry.rotation.scalar(),
        epsilon = 1e-10
    );
    assert_relative_eq!(
        isometry.rotation().params()[1],
        na_isometry.rotation.vector().x,
        epsilon = 1e-10
    );
    assert_relative_eq!(
        isometry.rotation().params()[2],
        na_isometry.rotation.vector().y,
        epsilon = 1e-10
    );
    assert_relative_eq!(
        isometry.rotation().params()[3],
        na_isometry.rotation.vector().z,
        epsilon = 1e-10
    );

    let roundtrip_isometry = Isometry3F64::from(na_isometry);
    assert_relative_eq!(isometry.translation(), roundtrip_isometry.translation());
    assert_relative_eq!(
        isometry.rotation().params(),
        roundtrip_isometry.rotation().params(),
        epsilon = 1e-10
    );
}
