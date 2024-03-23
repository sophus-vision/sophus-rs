use crate::types::matrix::IsMatrix;
use crate::types::scalar::IsScalar;
use crate::types::vector::IsVector;
use crate::types::vector::IsVectorLike;

use std::marker::PhantomData;

/// cubic basis function
pub struct CubicBasisFunction<S> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<1>> CubicBasisFunction<S> {
    /// C matrix
    pub fn c() -> S::Matrix<3, 4> {
        S::Matrix::<3, 4>::from_c_array2([
            [5.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 3.0 / 6.0, 3.0 / 6.0, -2.0 / 6.0],
            [0.0, 0.0, 0.0, 1.0 / 6.0],
        ])
    }

    /// B(u) matrix
    pub fn b(u: S) -> S::Vector<3> {
        let u_sq = u.clone() * u.clone();
        Self::c() * S::Vector::<4>::from_array([1.0.into(), u.clone(), u_sq.clone(), u_sq * u])
    }

    /// derivative of B(u) matrix with respect to u
    pub fn du_b(u: S, delta_t: S) -> S::Vector<3> {
        let u_sq = u.clone() * u.clone();
        Self::c().scaled(S::c(1.0) / delta_t)
            * S::Vector::<4>::from_array([S::c(0.0), S::c(1.0), S::c(2.0) * u, S::c(3.0) * u_sq])
    }

    /// second derivative of B(u) matrix with respect to u
    pub fn du2_b(u: S, delta_t: S) -> S::Vector<3> {
        Self::c().scaled(S::c(1.0) / (delta_t.clone() * delta_t))
            * S::Vector::<4>::from_array([S::c(0.0), S::c(0.0), S::c(2.0), S::c(6.0) * u])
    }
}

/// cubic B-spline function
pub struct CubicBSplineFn<S: IsScalar<1>, const DIMS: usize> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<1>, const DIMS: usize> CubicBSplineFn<S, DIMS> {
    fn interpolate(
        control_point: S::Vector<DIMS>,
        control_points: [S::Vector<DIMS>; 3],
        u: S,
    ) -> S::Vector<DIMS> {
        let b = CubicBasisFunction::<S>::b(u);
        control_point
            + control_points[0].scaled(b.get(0))
            + control_points[1].scaled(b.get(1))
            + control_points[2].scaled(b.get(2))
    }

    fn dxi_interpolate(u: S, quadruple_idx: usize) -> S::Matrix<DIMS, DIMS> {
        let b = CubicBasisFunction::<S>::b(u.clone());
        if quadruple_idx == 0 {
            S::Matrix::<DIMS, DIMS>::identity()
        } else {
            S::Matrix::<DIMS, DIMS>::identity().scaled(b.get(quadruple_idx - 1))
        }
    }
}

/// Segment case
#[derive(Clone, Debug, Copy)]
pub enum SegmentCase {
    /// First segment
    First,
    /// segment in the middle
    Normal,
    /// Last segment
    Last,
}

/// Cubic B-spline segment
#[derive(Clone, Debug)]
pub struct CubicBSplineSegment<S: IsScalar<1>, const DIMS: usize> {
    pub(crate) case: SegmentCase,
    pub(crate) control_points: [S::Vector<DIMS>; 4],
}

impl<S: IsScalar<1>, const DIMS: usize> CubicBSplineSegment<S, DIMS> {
    /// Interpolate
    pub fn interpolate(&self, u: S) -> S::Vector<DIMS> {
        match self.case {
            SegmentCase::First => CubicBSplineFn::interpolate(
                self.control_points[1].clone(),
                [
                    S::Vector::<DIMS>::zero(),
                    self.control_points[2].clone() - self.control_points[1].clone(),
                    self.control_points[3].clone() - self.control_points[2].clone(),
                ],
                u,
            ),
            SegmentCase::Normal => CubicBSplineFn::interpolate(
                self.control_points[0].clone(),
                [
                    self.control_points[1].clone() - self.control_points[0].clone(),
                    self.control_points[2].clone() - self.control_points[1].clone(),
                    self.control_points[3].clone() - self.control_points[2].clone(),
                ],
                u,
            ),
            SegmentCase::Last => CubicBSplineFn::interpolate(
                self.control_points[0].clone(),
                [
                    self.control_points[1].clone() - self.control_points[0].clone(),
                    self.control_points[2].clone() - self.control_points[1].clone(),
                    S::Vector::<DIMS>::zero(),
                ],
                u,
            ),
        }
    }

    /// Derivative of the interpolation with respect to u
    pub fn dxi_interpolate(&self, u: S, quadruple_idx: usize) -> S::Matrix<DIMS, DIMS> {
        match self.case {
            SegmentCase::First => {
                if quadruple_idx == 0 {
                    S::Matrix::<DIMS, DIMS>::zero()
                } else if quadruple_idx == 1 {
                    CubicBSplineFn::dxi_interpolate(u.clone(), 0)
                        - CubicBSplineFn::dxi_interpolate(u.clone(), 2)
                } else if quadruple_idx == 2 {
                    CubicBSplineFn::dxi_interpolate(u.clone(), 2)
                        - CubicBSplineFn::dxi_interpolate(u.clone(), 3)
                } else {
                    CubicBSplineFn::dxi_interpolate(u.clone(), 3)
                }
            }
            SegmentCase::Normal => {
                if quadruple_idx == 0 {
                    CubicBSplineFn::dxi_interpolate(u.clone(), 0)
                        - CubicBSplineFn::dxi_interpolate(u.clone(), 1)
                } else if quadruple_idx == 1 {
                    CubicBSplineFn::dxi_interpolate(u.clone(), 1)
                        - CubicBSplineFn::dxi_interpolate(u.clone(), 2)
                } else if quadruple_idx == 2 {
                    CubicBSplineFn::dxi_interpolate(u.clone(), 2)
                        - CubicBSplineFn::dxi_interpolate(u.clone(), 3)
                } else {
                    CubicBSplineFn::dxi_interpolate(u.clone(), 3)
                }
            }
            SegmentCase::Last => {
                if quadruple_idx == 0 {
                    CubicBSplineFn::dxi_interpolate(u.clone(), 0)
                        - CubicBSplineFn::dxi_interpolate(u.clone(), 1)
                } else if quadruple_idx == 1 {
                    CubicBSplineFn::dxi_interpolate(u.clone(), 1)
                        - CubicBSplineFn::dxi_interpolate(u.clone(), 2)
                } else if quadruple_idx == 2 {
                    CubicBSplineFn::dxi_interpolate(u.clone(), 2)
                } else {
                    S::Matrix::<DIMS, DIMS>::zero()
                }
            }
        }
    }
}

mod test {

    #[test]
    fn test_spline_basis_fn() {
        use crate::dual::dual_scalar::Dual;
        use crate::dual::dual_vector::DualV;
        use crate::maps::vector_valued_maps::VectorValuedMapFromVector;
        use crate::points::example_points;
        use crate::spline::spline_segment::CubicBSplineFn;
        use crate::types::scalar::IsScalar;
        use crate::types::vector::IsVector;
        use crate::types::vector::IsVectorLike;
        use crate::types::VecF64;

        let points = &example_points::<f64, 3>();
        assert!(points.len() >= 8);

        let mut u = 0.0;
        loop {
            if u >= 1.0 {
                break;
            }

            for p_idx in 0..points.len() - 4 {
                let first_control_point = points[p_idx];
                let first_control_point_dual = DualV::c(points[p_idx]);
                let mut segment_control_points = [VecF64::<3>::zeros(); 3];
                let mut segment_control_points_dual =
                    [DualV::<3>::zero(), DualV::<3>::zero(), DualV::<3>::zero()];
                for i in 0..3 {
                    segment_control_points[i] = points[p_idx + 1];
                    segment_control_points_dual[i] = DualV::c(segment_control_points[i]);
                }

                let f0 = |x| -> DualV<3> {
                    CubicBSplineFn::<Dual, 3>::interpolate(
                        x,
                        segment_control_points_dual.clone(),
                        Dual::c(u),
                    )
                };
                let auto_dx0 =
                    VectorValuedMapFromVector::static_fw_autodiff(f0, first_control_point);
                let analytic_dx0 = CubicBSplineFn::<f64, 3>::dxi_interpolate(u, 0);
                approx::assert_abs_diff_eq!(auto_dx0, analytic_dx0, epsilon = 0.0001);

                for i in 0..3 {
                    let fi = |x| -> DualV<3> {
                        let mut seg = segment_control_points_dual.clone();
                        seg[i] = x;
                        CubicBSplineFn::<Dual, 3>::interpolate(
                            first_control_point_dual.clone(),
                            seg,
                            Dual::c(u),
                        )
                    };
                    let auto_dxi = VectorValuedMapFromVector::static_fw_autodiff(
                        fi,
                        segment_control_points[i],
                    );
                    let analytic_dxi = CubicBSplineFn::<f64, 3>::dxi_interpolate(u, i + 1);
                    approx::assert_abs_diff_eq!(auto_dxi, analytic_dxi, epsilon = 0.0001);
                }
            }
            u += 0.1;
        }
    }

    #[test]
    fn test_spline_segment() {
        use crate::dual::dual_scalar::Dual;
        use crate::dual::dual_vector::DualV;
        use crate::maps::vector_valued_maps::VectorValuedMapFromVector;
        use crate::points::example_points;
        use crate::spline::spline_segment::CubicBSplineSegment;
        use crate::spline::spline_segment::SegmentCase;
        use crate::types::scalar::IsScalar;
        use crate::types::vector::IsVector;
        use crate::types::vector::IsVectorLike;
        use crate::types::VecF64;

        let points = &example_points::<f64, 3>();
        assert!(points.len() >= 8);

        for p_idx in 0..points.len() - 4 {
            let mut segment_control_points = [VecF64::<3>::zeros(); 4];
            let mut segment_control_points_dual = [
                DualV::<3>::zero(),
                DualV::<3>::zero(),
                DualV::<3>::zero(),
                DualV::<3>::zero(),
            ];

            for i in 0..4 {
                segment_control_points[i] = points[p_idx];
                segment_control_points_dual[i] = DualV::c(segment_control_points[i]);
            }

            for case in [SegmentCase::First, SegmentCase::Normal, SegmentCase::Last] {
                let base = CubicBSplineSegment::<f64, 3> {
                    case,
                    control_points: segment_control_points,
                };

                let mut u = 0.0;
                loop {
                    if u >= 1.0 {
                        break;
                    }
                    for i in 0..4 {
                        let analytic_dx = base.dxi_interpolate(u, i);
                        let f = |v: VecF64<3>| {
                            let mut base_copy = base.clone();
                            base_copy.control_points[i] = v;
                            base_copy.interpolate(u)
                        };

                        let num_dx = VectorValuedMapFromVector::static_sym_diff_quotient(
                            f, points[0], 0.0001,
                        );

                        let f = |v: DualV<3>| {
                            let mut base_dual = CubicBSplineSegment::<Dual, 3> {
                                case,
                                control_points: segment_control_points_dual.clone(),
                            };

                            base_dual.control_points[i] = v;
                            base_dual.interpolate(Dual::c(u))
                        };

                        let auto_dx = VectorValuedMapFromVector::static_fw_autodiff(f, points[i]);

                        approx::assert_abs_diff_eq!(analytic_dx, num_dx, epsilon = 0.0001);
                        approx::assert_abs_diff_eq!(analytic_dx, auto_dx, epsilon = 0.0001);
                    }
                    u += 0.1;
                }
            }
        }
    }
}
