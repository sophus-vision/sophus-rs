use super::affine_group_template::AffineGroupTemplateImpl;
use crate::{
    groups::rotation_boost3::RotationBoost3Impl,
    lie_group::LieGroup,
};

/// 3d **Galilean transformations** – element of the Galilean group **Gal(3)**
pub type Galilean3<S, const BATCH: usize, const DM: usize, const DN: usize> =
    LieGroup<S, 10, 11, 4, 5, BATCH, DM, DN, Galilei3Impl<S, BATCH, DM, DN>>;

/// 3d Galilean transformations with f64 scalar type - element of the Galilean group **Gal(3)**
///
/// See [Galilean3] for details.
pub type Galilei3F64 = Galilean3<f64, 1, 0, 0>;

/// 3d Galilean transformations implementation details
///
/// See [Galilean3] for details.
pub type Galilei3Impl<S, const BATCH: usize, const DM: usize, const DN: usize> =
    AffineGroupTemplateImpl<
        S,
        10,
        11,
        4,
        5,
        6,
        7,
        BATCH,
        DM,
        DN,
        RotationBoost3Impl<S, BATCH, DM, DN>,
    >;

#[test]
fn galilei3_rop_tests() {
    #[cfg(feature = "simd")]
    use sophus_autodiff::dual::DualBatchScalar;
    use sophus_autodiff::dual::DualScalar;
    #[cfg(feature = "simd")]
    use sophus_autodiff::linalg::BatchScalarF64;

    Galilei3F64::test_suite();
    #[cfg(feature = "simd")]
    Galilean3::<BatchScalarF64<8>, 8, 0, 0>::test_suite();
    Galilean3::<DualScalar<1, 1>, 1, 1, 1>::test_suite();
    #[cfg(feature = "simd")]
    Galilean3::<DualBatchScalar<8, 1, 1>, 8, 1, 1>::test_suite();
}
