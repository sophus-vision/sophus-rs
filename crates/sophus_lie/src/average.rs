use crate::groups::isometry2::Isometry2F64;
use crate::groups::isometry3::Isometry3F64;
use crate::groups::rotation2::Rotation2F64;
use crate::groups::rotation3::Rotation3F64;
use crate::traits::IsLieGroupImpl;
use crate::LieGroup;
use snafu::prelude::*;
use sophus_core::linalg::EPS_F64;
use sophus_core::prelude::IsSingleScalar;
use sophus_core::prelude::IsVector;

/// error of iterative_average
#[derive(Snafu, Debug)]
pub enum IterativeAverageError<Group> {
    /// slice is empty
    #[snafu(display("empty slice"))]
    EmptySlice,
    /// not converged
    #[snafu(display("no convergence after {max_iteration_count:?} iterations"))]
    NotConverged {
        /// max iteration count
        max_iteration_count: u32,
        /// estimate after max number of iterations
        parent_from_body_estimate: Group,
    },
}

/// iterative Lie group average
pub fn iterative_average<
    S: IsSingleScalar + PartialOrd,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, 1>,
>(
    parent_from_body_transforms: &[LieGroup<S, DOF, PARAMS, POINT, AMBIENT, 1, G>],
    max_iteration_count: u32,
) -> Result<
    LieGroup<S, DOF, PARAMS, POINT, AMBIENT, 1, G>,
    IterativeAverageError<LieGroup<S, DOF, PARAMS, POINT, AMBIENT, 1, G>>,
> {
    if parent_from_body_transforms.is_empty() {
        return Err(IterativeAverageError::EmptySlice);
    }
    let mut parent_from_body_average = parent_from_body_transforms[0].clone();
    let w = S::from_f64(1.0 / parent_from_body_transforms.len() as f64);

    // This implements the algorithm in the beginning of Sec. 4.2 in
    // ftp://ftp-sop.inria.fr/epidaure/Publications/Arsigny/arsigny_rr_biinvariant_average.pdf.
    for _iter in 0..max_iteration_count {
        let mut average_tangent = S::Vector::<DOF>::zeros();

        for parent_from_body in parent_from_body_transforms {
            average_tangent = average_tangent
                + parent_from_body_average
                    .inverse()
                    .group_mul(parent_from_body)
                    .log()
                    .scaled(w.clone());
        }
        let refined_parent_from_body_average =
            parent_from_body_average.group_mul(&LieGroup::exp(&average_tangent));
        let step = refined_parent_from_body_average
            .inverse()
            .group_mul(&parent_from_body_average)
            .log();
        if step.squared_norm() < S::from_f64(EPS_F64) {
            return Ok(refined_parent_from_body_average);
        }

        parent_from_body_average = refined_parent_from_body_average;
    }

    Err(IterativeAverageError::NotConverged {
        max_iteration_count,
        parent_from_body_estimate: parent_from_body_average,
    })
}

/// Tests for Lie group average
pub trait LieGroupAverageTests {
    /// Run average tests.
    fn run_average_tests() {}
}

macro_rules! def_average_test_template {
    ( $group: ty
) => {
        impl LieGroupAverageTests for $group {
            fn run_average_tests() {
                use crate::traits::HasAverage;
                use approx::assert_relative_eq;
                use sophus_core::linalg::VecF64;

                // test: empty slice
                assert!(Self::average(&[]).is_err());

                for parent_from_a in Self::element_examples() {
                    // test: average of one elements is the element

                    let averaged_element = Self::average(&[parent_from_a]).unwrap();
                    assert_relative_eq!(
                        averaged_element.matrix(),
                        parent_from_a.matrix(),
                        epsilon = 0.001
                    );

                    for parent_from_b in Self::element_examples() {
                        // test: average of two elements is identical to interpolation

                        if parent_from_a
                            .inverse()
                            .group_mul(&parent_from_b)
                            .has_shortest_path_ambiguity()
                        {
                            continue;
                        }
                        let averaged_element =
                            Self::average(&[parent_from_a, parent_from_b]).unwrap();
                        let interp_element = parent_from_a.interpolate(&parent_from_b, 0.5);

                        assert_relative_eq!(
                            averaged_element.matrix(),
                            interp_element.matrix(),
                            epsilon = 0.001
                        );

                        for parent_from_c in Self::element_examples() {
                            // test: average of all tangents from each element to the average equal zero.

                            let list = [parent_from_a, parent_from_b, parent_from_c];
                            let parent_from_average = Self::average(&list).unwrap();
                            let mut average_tangent = VecF64::zeros();

                            let w = 1.0 / (list.len() as f64);
                            for parent_from_x in list {
                                average_tangent += parent_from_average
                                    .inverse()
                                    .group_mul(&parent_from_x)
                                    .log()
                                    .scaled(w);
                            }
                            assert_relative_eq!(average_tangent, VecF64::zeros(), epsilon = 0.001,);
                        }
                    }
                }
            }
        }
    };
}

def_average_test_template!(Rotation2F64);
def_average_test_template!(Rotation3F64);
def_average_test_template!(Isometry2F64);
def_average_test_template!(Isometry3F64);

#[test]
fn average_tests() {
    Rotation2F64::run_average_tests();
    Rotation3F64::run_average_tests();
    Isometry2F64::run_average_tests();
    Isometry3F64::run_average_tests();
}
