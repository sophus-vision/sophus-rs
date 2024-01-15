use crate::calculus::spline::spline_segment::CubicBSplineSegment;
use crate::calculus::spline::spline_segment::SegmentCase;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVectorLike;
use assertables::assert_ge;
use assertables::assert_ge_as_result;
use assertables::assert_le;
use assertables::assert_le_as_result;
use nalgebra::Scalar;

pub struct CubicBSplineImpl<S: IsScalar, const DIMS: usize> {
    pub control_points: Vec<S::Vector<DIMS>>,
    pub delta_t: S,
}

impl<S: IsScalar, const DIMS: usize> CubicBSplineImpl<S, DIMS> {
    pub fn idx_involved(&self, segment_idx: usize) -> Vec<usize> {
        let num = self.num_segments();
        assert!(segment_idx < num);

        let idx_prev = if segment_idx == 0 { 0 } else { segment_idx - 1 };
        let idx_0 = segment_idx;
        let idx_1 = segment_idx + 1;
        let idx_2 = (segment_idx + 2).min(self.control_points.len() - 1);
        vec![idx_prev, idx_0, idx_1, idx_2]
    }

    pub fn interpolate(&self, segment_idx: usize, u: S) -> S::Vector<DIMS> {
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
        CubicBSplineSegment::<S, DIMS> {
            case,
            control_points: [
                self.control_points[idx_prev].clone(),
                self.control_points[idx_0].clone(),
                self.control_points[idx_1].clone(),
                self.control_points[idx_2].clone(),
            ],
        }
        .interpolate(u)
    }

    fn num_segments(&self) -> usize {
        assert!(!self.control_points.is_empty());
        self.control_points.len() - 1
    }

    pub fn dxi_interpolate(
        &self,
        segment_idx: usize,
        u: S,
        control_point_idx: usize,
    ) -> S::Matrix<DIMS, DIMS> {
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
        let spline_segment = CubicBSplineSegment::<S, DIMS> {
            case,
            control_points: [
                self.control_points[idx_prev].clone(),
                self.control_points[idx_0].clone(),
                self.control_points[idx_1].clone(),
                self.control_points[idx_2].clone(),
            ],
        };

        let mut dxi = S::Matrix::<DIMS, DIMS>::zero();
        if idx_prev == control_point_idx {
            dxi = dxi + spline_segment.dxi_interpolate(u.clone(), 0);
        }
        if idx_0 == control_point_idx {
            dxi = dxi + spline_segment.dxi_interpolate(u.clone(), 1);
        }
        if idx_1 == control_point_idx {
            dxi = dxi + spline_segment.dxi_interpolate(u.clone(), 2);
        }
        if idx_2 == control_point_idx {
            dxi = dxi + spline_segment.dxi_interpolate(u, 3);
        }
        dxi
    }
}

#[derive(Clone, Debug, Copy)]
pub struct IndexAndU<S: Scalar> {
    pub segment_idx: usize,
    pub u: S,
}

pub struct CubicBSpline<S: IsScalar, const DIMS: usize> {
    pub spline_impl: CubicBSplineImpl<S, DIMS>,
    pub t0: S,
}

#[derive(Clone, Debug, Copy)]
pub struct CubicBSplineParams<S: IsScalar + 'static> {
    pub delta_t: S,
    pub t0: S,
}

impl<S: IsScalar + 'static, const DIMS: usize> CubicBSpline<S, DIMS> {
    pub fn new(control_points: Vec<S::Vector<DIMS>>, params: CubicBSplineParams<S>) -> Self {
        Self {
            spline_impl: CubicBSplineImpl {
                control_points,
                delta_t: params.delta_t,
            },
            t0: params.t0,
        }
    }
    pub fn interpolate(&self, t: S) -> S::Vector<DIMS> {
        let index_and_u = self.index_and_u(t);
        self.spline_impl
            .interpolate(index_and_u.segment_idx, index_and_u.u)
    }

    pub fn dxi_interpolate(&self, t: S, control_point_idx: usize) -> S::Matrix<DIMS, DIMS> {
        let index_and_u = self.index_and_u(t);
        self.spline_impl
            .dxi_interpolate(index_and_u.segment_idx, index_and_u.u, control_point_idx)
    }

    pub fn idx_involved(&self, t: S) -> Vec<usize> {
        let index_and_u = self.index_and_u(t);
        self.spline_impl.idx_involved(index_and_u.segment_idx)
    }

    pub fn index_and_u(&self, t: S) -> IndexAndU<S> {
        assert_ge!(t.real(), self.t0.real());
        assert_le!(t.real(), self.t_max().real());

        let normalized_t = self.normalized_t(t.clone());

        let mut idx_and_u = IndexAndU::<S> {
            segment_idx: normalized_t.floor() as usize,
            u: normalized_t.clone().fract(),
        };
        println!("{:?}", idx_and_u);

        let eps = 0.00001;

        if idx_and_u.u.real() > eps {
            println!("case A");
            return idx_and_u;
        }

        // i < N/2
        if idx_and_u.segment_idx < self.num_segments() / 2 {
            println!("case B");
            return idx_and_u;
        }

        println!("case C");

        idx_and_u.segment_idx -= 1;
        idx_and_u.u = idx_and_u.u + S::c(1.0);

        idx_and_u
    }

    // normalized between [0, N]
    pub fn normalized_t(&self, t: S) -> S {
        (t - self.t0.clone()) / self.spline_impl.delta_t.clone()
    }

    pub fn num_segments(&self) -> usize {
        self.spline_impl.num_segments()
    }

    pub fn t_max(&self) -> S {
        self.t0.clone() + S::c(self.num_segments() as f64) * self.spline_impl.delta_t.clone()
    }
}

mod test {

    #[test]
    fn test() {
        use crate::calculus::points::{example_points};
        use crate::calculus::spline::spline::{CubicBSpline, CubicBSplineParams};

        let points = example_points::<f64, 2>();
        for (t0, delta_t) in [(0.0, 1.0)] {
            let params = CubicBSplineParams { delta_t, t0 };

            let mut points = points.clone();
            let spline = CubicBSpline::new(points.clone(), params);
            points.reverse();
            let rspline = CubicBSpline::new(points, params);

            println!("tmax: {}", spline.t_max());

            let mut t = t0;

            loop {
                if t >= spline.t_max() {
                    break;
                }

                println!("t {}", t);
                println!("{:?}", spline.idx_involved(t));

                println!("i: {}", spline.interpolate(t));
                println!("i': {}", rspline.interpolate(spline.t_max() - t));

                for i in 0..spline.num_segments() {
                    println!("dx: {} {}", i, spline.dxi_interpolate(t, i));
                    println!(
                        "dx': {} {}",
                        i,
                        rspline.dxi_interpolate(spline.t_max() - t, spline.num_segments() - i)
                    );
                }
                t += 0.1;
            }

            println!("{:?}", spline.idx_involved(1.01));
        }
    }
}
