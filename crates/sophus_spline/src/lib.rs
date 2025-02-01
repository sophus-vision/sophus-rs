#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![no_std]
//! # Spline module

/// Cubic B-Spline details
pub mod spline_segment;

use log::debug;
use sophus_autodiff::prelude::*;

use crate::spline_segment::{
    CubicBSplineSegment,
    SegmentCase,
};

extern crate alloc;

/// Cubic B-Spline implementation
pub struct CubicBSplineImpl<
    S: IsSingleScalar<DM, DN>,
    const DIMS: usize,
    const DM: usize,
    const DN: usize,
> {
    /// Control points
    pub control_points: alloc::vec::Vec<S::SingleVector<DIMS>>,
    /// delta between control points
    pub delta_t: S,
}

impl<S: IsSingleScalar<DM, DN>, const DIMS: usize, const DM: usize, const DN: usize>
    CubicBSplineImpl<S, DIMS, DM, DN>
{
    /// indices involved
    pub fn idx_involved(&self, segment_idx: usize) -> alloc::vec::Vec<usize> {
        let num = self.num_segments();
        assert!(segment_idx < num);

        let idx_prev = if segment_idx == 0 { 0 } else { segment_idx - 1 };
        let idx_0 = segment_idx;
        let idx_1 = segment_idx + 1;
        let idx_2 = (segment_idx + 2).min(self.control_points.len() - 1);
        alloc::vec![idx_prev, idx_0, idx_1, idx_2]
    }

    /// Interpolate
    pub fn interpolate(&self, segment_idx: usize, u: S) -> S::SingleVector<DIMS> {
        let num = self.num_segments();
        assert!(segment_idx < num);

        let case = if segment_idx == 0 {
            SegmentCase::First
        } else if segment_idx == num - 1 {
            SegmentCase::Last
        } else {
            SegmentCase::Normal
        };

        let idx_prev = if segment_idx == 0 { 0 } else { segment_idx - 1 };
        let idx_0 = segment_idx;
        let idx_1 = segment_idx + 1;
        let idx_2 = (segment_idx + 2).min(self.control_points.len() - 1);
        CubicBSplineSegment::<S, DIMS, DM, DN> {
            case,
            control_points: [
                self.control_points[idx_prev],
                self.control_points[idx_0],
                self.control_points[idx_1],
                self.control_points[idx_2],
            ],
        }
        .interpolate(u)
    }

    fn num_segments(&self) -> usize {
        assert!(!self.control_points.is_empty());
        self.control_points.len() - 1
    }

    /// derivative of the interpolation
    pub fn dxi_interpolate(
        &self,
        segment_idx: usize,
        u: S,
        control_point_idx: usize,
    ) -> S::SingleMatrix<DIMS, DIMS> {
        let num = self.num_segments();
        assert!(segment_idx < num);

        let case = if segment_idx == 0 {
            SegmentCase::First
        } else if segment_idx == num - 1 {
            SegmentCase::Last
        } else {
            SegmentCase::Normal
        };

        let idx_prev = if segment_idx == 0 { 0 } else { segment_idx - 1 };
        let idx_0 = segment_idx;
        let idx_1 = segment_idx + 1;
        let idx_2 = (segment_idx + 2).min(self.control_points.len() - 1);
        let spline_segment = CubicBSplineSegment::<S, DIMS, DM, DN> {
            case,
            control_points: [
                self.control_points[idx_prev],
                self.control_points[idx_0],
                self.control_points[idx_1],
                self.control_points[idx_2],
            ],
        };

        let mut dxi: S::SingleMatrix<DIMS, DIMS> = S::SingleMatrix::<DIMS, DIMS>::zeros();
        if idx_prev == control_point_idx {
            dxi = dxi + spline_segment.dxi_interpolate(u, 0);
        }
        if idx_0 == control_point_idx {
            dxi = dxi + spline_segment.dxi_interpolate(u, 1);
        }
        if idx_1 == control_point_idx {
            dxi = dxi + spline_segment.dxi_interpolate(u, 2);
        }
        if idx_2 == control_point_idx {
            dxi = dxi + spline_segment.dxi_interpolate(u, 3);
        }
        dxi
    }
}

/// Index and u
#[derive(Clone, Debug, Copy)]
pub struct IndexAndU<S: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize> {
    /// segment index
    pub segment_idx: usize,
    /// u
    pub u: S,
}

/// Cubic B-Spline
pub struct CubicBSpline<
    S: IsSingleScalar<DM, DN>,
    const DIMS: usize,
    const DM: usize,
    const DN: usize,
> {
    /// Cubic B-Spline implementation
    pub spline_impl: CubicBSplineImpl<S, DIMS, DM, DN>,
    /// start time t0
    pub t0: S,
}

/// Cubic B-Spline parameters
#[derive(Clone, Debug, Copy)]
pub struct CubicBSplineParams<S: IsSingleScalar<DM, DN> + 'static, const DM: usize, const DN: usize>
{
    /// delta between control points
    pub delta_t: S,
    /// start time t0
    pub t0: S,
}

impl<S: IsSingleScalar<DM, DN> + 'static, const DIMS: usize, const DM: usize, const DN: usize>
    CubicBSpline<S, DIMS, DM, DN>
{
    /// create a new cubic B-Spline
    pub fn new(
        control_points: alloc::vec::Vec<S::SingleVector<DIMS>>,
        params: CubicBSplineParams<S, DM, DN>,
    ) -> Self {
        Self {
            spline_impl: CubicBSplineImpl {
                control_points,
                delta_t: params.delta_t,
            },
            t0: params.t0,
        }
    }

    /// interpolate
    pub fn interpolate(&self, t: S) -> S::SingleVector<DIMS> {
        let index_and_u = self.index_and_u(t);
        self.spline_impl
            .interpolate(index_and_u.segment_idx, index_and_u.u)
    }

    /// derivative of the interpolation
    pub fn dxi_interpolate(&self, t: S, control_point_idx: usize) -> S::SingleMatrix<DIMS, DIMS> {
        let index_and_u = self.index_and_u(t);
        self.spline_impl
            .dxi_interpolate(index_and_u.segment_idx, index_and_u.u, control_point_idx)
    }

    /// indices involved
    pub fn idx_involved(&self, t: S) -> alloc::vec::Vec<usize> {
        let index_and_u = self.index_and_u(t);
        self.spline_impl.idx_involved(index_and_u.segment_idx)
    }

    /// index and u
    pub fn index_and_u(&self, t: S) -> IndexAndU<S, DM, DN> {
        assert!(t.greater_equal(&self.t0).all());
        assert!(t.less_equal(&self.t_max()).all());

        let normalized_t: S = self.normalized_t(t);

        let mut idx_and_u = IndexAndU::<S, DM, DN> {
            segment_idx: normalized_t.i64_floor() as usize,
            u: normalized_t.clone().fract(),
        };
        debug!("{:?}", idx_and_u);

        let eps = 0.00001;

        if idx_and_u.u.single_real_scalar() > eps {
            debug!("case A");
            return idx_and_u;
        }

        // i < N/2
        if idx_and_u.segment_idx < self.num_segments() / 2 {
            debug!("case B");
            return idx_and_u;
        }

        debug!("case C");

        idx_and_u.segment_idx -= 1;
        idx_and_u.u += S::from_f64(1.0);

        idx_and_u
    }

    /// normalized between [0, N]
    pub fn normalized_t(&self, t: S) -> S {
        (t - self.t0) / self.spline_impl.delta_t
    }

    /// number of segments
    pub fn num_segments(&self) -> usize {
        self.spline_impl.num_segments()
    }

    /// t_max
    pub fn t_max(&self) -> S {
        self.t0 + S::from_f64(self.num_segments() as f64) * self.spline_impl.delta_t
    }
}

#[test]
fn test_pline() {
    use log::info;
    use sophus_autodiff::example_points;

    let points = example_points::<f64, 2, 1, 0, 0>();
    for (t0, delta_t) in [(0.0, 1.0)] {
        let params = CubicBSplineParams { delta_t, t0 };

        let mut points = points.clone();
        let spline = CubicBSpline::new(points.clone(), params);
        points.reverse();
        let rspline = CubicBSpline::new(points, params);

        info!("tmax: {}", spline.t_max());

        let mut t = t0;

        loop {
            if t >= spline.t_max() {
                break;
            }

            info!("t {}", t);
            info!("{:?}", spline.idx_involved(t));

            info!("i: {}", spline.interpolate(t));
            info!("i': {}", rspline.interpolate(spline.t_max() - t));

            for i in 0..spline.num_segments() {
                info!("dx: {} {}", i, spline.dxi_interpolate(t, i));
                info!(
                    "dx': {} {}",
                    i,
                    rspline.dxi_interpolate(spline.t_max() - t, spline.num_segments() - i)
                );
            }
            t += 0.1;
        }

        info!("{:?}", spline.idx_involved(1.01));
    }
}
