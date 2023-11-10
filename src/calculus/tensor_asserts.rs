use dfdx_core::shapes::{Rank0, Rank1, Rank2, Rank3, Rank4, Rank5};

pub trait ShapeRank<const RANK: usize> {
    fn rank(&self) -> usize;

    fn dims(&self) -> [usize; RANK];
}

impl ShapeRank<0> for &Rank0 {
    fn rank(&self) -> usize {
        0
    }

    fn dims(&self) -> [usize; 0] {
        []
    }
}

impl<const D: usize> ShapeRank<1> for &Rank1<D> {
    fn rank(&self) -> usize {
        1
    }

    fn dims(&self) -> [usize; 1] {
        [D; 1]
    }
}

impl<const M: usize, const N: usize> ShapeRank<2> for &Rank2<M, N> {
    fn rank(&self) -> usize {
        2
    }

    fn dims(&self) -> [usize; 2] {
        [M, N]
    }
}

impl<const M: usize, const N: usize, const O: usize> ShapeRank<3> for &Rank3<M, N, O> {
    fn rank(&self) -> usize {
        3
    }

    fn dims(&self) -> [usize; 3] {
        [M, N, O]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize> ShapeRank<4>
    for &Rank4<M, N, O, P>
{
    fn rank(&self) -> usize {
        4
    }

    fn dims(&self) -> [usize; 4] {
        [M, N, O, P]
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize> ShapeRank<5>
    for &Rank5<M, N, O, P, Q>
{
    fn rank(&self) -> usize {
        5
    }

    fn dims(&self) -> [usize; 5] {
        [M, N, O, P, Q]
    }
}

#[macro_export]
macro_rules! assert_tensors_relative_eq_rank0 {
    ($t1:expr, $t2:expr, $eps:expr) => {{
        use approx::relative_eq;
        if !relative_eq!($t1.array(), $t2.array(),epsilon = $eps) {
            panic!(
                "assert_tensors_relative_eq_rank0 FAILED\n{} {:?} != {:?}\n",
                $t1.array(),
                $t2.array(),
                $eps
            );
        }
    }};
}

#[macro_export]
macro_rules! assert_tensors_relative_eq_rank1 {
    ($t1:expr, $t2:expr, $eps:literal) => {{
        use crate::calculus::tensor_asserts::ShapeRank;
        use approx::relative_eq;
        use dfdx_core::shapes::HasShape;
        if !($t1.shape().dims() == $t2.shape().dims()) {
            panic!(
                "assert_tensors_relative_eq_rank1 FAILED: tensor shapes are not equal: {:?} != {:?}",
                $t1.shape().dims(),
                $t2.shape().dims()
            );
        }
        let mut details = String::new();
        for b in 0..$t1.shape().dims()[0] {
            if !relative_eq!($t1[[b]], $t2[[b]],epsilon = $eps) {
                details.push_str(&format!(
                    "{} != {} at [{}]\n",
                    $t1[[b]],
                    $t2[[b]],
                    b,
                ));
            }

        }
        if !details.is_empty() {
            panic!("assert_tensors_relative_eq_rank1 FAILED:\nlhs: {}\nrhs: {}\neps: {}; tensor shape: {:?}\ndetails:\n{}",
                stringify!($t1),
                stringify!($t2),
                $eps,
                $t1.shape().dims(),
                details
            );
        }
    }};
}

#[macro_export]
macro_rules! assert_tensors_relative_eq_rank2 {
    ($t1:expr, $t2:expr, $eps:literal) => {{
        use crate::calculus::tensor_asserts::ShapeRank;
        use approx::relative_eq;
        use dfdx_core::shapes::HasShape;
        if !($t1.shape().dims() == $t2.shape().dims()) {
            panic!(
                "assert_tensors_relative_eq_rank2 FAILED: tensor shapes are not equal: {:?} != {:?}",
                $t1.shape().dims(),
                $t2.shape().dims()
            );
        }
        let mut details = String::new();
        for b in 0..$t1.shape().dims()[0] {
            for r in 0..$t1.shape().dims()[1] {
                if !relative_eq!($t1[[b, r]], $t2[[b, r]], epsilon = $eps) {
                    details.push_str(&format!(
                        "{} != {} at [{}, {}]\n",
                        $t1[[b, r]],
                        $t2[[b, r]],
                        b,
                        r
                    ));
                }
            }
        }
        if !details.is_empty() {
            panic!("assert_tensors_relative_eq_rank2 FAILED:\nlhs: {}\n{:?}\nrhs: {}\n{:?}\neps: {}; tensor shape: {:?}\ndetails:\n{}",
            stringify!($t1),
            $t1.array(),
            stringify!($t2),
            $t2.array(),
            $eps,
            $t1.shape().dims(),
            details
        );
        }
    }};
}

#[macro_export]
macro_rules! assert_tensors_relative_eq_rank3 {
    ($t1:expr, $t2:expr, $eps:literal) => {{
        use crate::calculus::tensor_asserts::ShapeRank;
        use approx::relative_eq;
        use dfdx_core::shapes::HasShape;
        if !($t1.shape().dims() == $t2.shape().dims()) {
            panic!(
                "assert_tensors_relative_eq_rank3 FAILED: tensor shapes are not equal: {:?} != {:?}",
                $t1.shape().dims(),
                $t2.shape().dims()
            );
        }
        let mut details = String::new();
        for b in 0..$t1.shape().dims()[0] {
            for r in 0..$t1.shape().dims()[1] {
                for c in 0..$t1.shape().dims()[2] {
                    if !relative_eq!($t1[[b, r, c]], $t2[[b, r, c]], epsilon = $eps) {
                        details.push_str(&format!(
                            "{} != {} at [{}, {}, {}]\n",
                            $t1[[b, r, c]],
                            $t2[[b, r, c]],
                            b,
                            r,
                            c
                        ));
                    }
                }
            }
        }
        if !details.is_empty() {
            panic!("assert_tensors_relative_eq_rank3 FAILED:\nlhs: {}\n{:?}\nrhs: {}\n{:?}\neps: {}; tensor shape: {:?}\ndetails:\n{}",
                stringify!($t1),
                $t1.array(),
                stringify!($t2),
                $t2.array(),
                $eps,
                $t1.shape().dims(),
                details
            );
        }
    }};
}

#[macro_export]
macro_rules! assert_tensors_relative_eq_rank4 {
    ($t1:expr, $t2:expr, $eps:literal) => {{
        use crate::calculus::tensor_asserts::ShapeRank;
        use approx::relative_eq;
        use dfdx_core::shapes::HasShape;
        if !($t1.shape().dims() == $t2.shape().dims()) {
            panic!(
                "assert_tensors_relative_eq_rank4 FAILED: tensor shapes are not equal: {:?} != {:?}",
                $t1.shape().dims(),
                $t2.shape().dims()
            );
        }
        let mut details = String::new();
        for a in 0..$t1.shape().dims()[0] {
            for b in 0..$t1.shape().dims()[1] {
                for r in 0..$t1.shape().dims()[2] {
                    for c in 0..$t1.shape().dims()[3] {
                        if !relative_eq!($t1[[a, b, r, c]], $t2[[a, b, r, c]], epsilon = $eps) {
                            details.push_str(&format!(
                                "{} != {} at [{}, {}, {}, {}]\n",
                                $t1[[a, b, r, c]],
                                $t2[[a, b, r, c]],
                                a,
                                b,
                                r,
                                c
                            ));
                        }
                    }
                }
            }
        }
        if !details.is_empty() {
            panic!("assert_tensors_relative_eq_rank4 FAILED:\nlhs: {}\nrhs: {}\neps: {}; tensor shape: {:?}\ndetails:\n{}",
                stringify!($t1),
                stringify!($t2),
                $eps,
                $t1.shape().dims(),
                details
            );
        }
    }};
}

#[macro_export]
macro_rules! assert_tensors_relative_eq_rank5 {
    ($t1:expr, $t2:expr, $eps:literal) => {{
        use crate::calculus::tensor_asserts::ShapeRank;
        use approx::relative_eq;
        use dfdx_core::shapes::HasShape;
        if !($t1.shape().dims() == $t2.shape().dims()) {
            panic!(
                "assert_tensors_relative_eq_rank5 FAILED: tensor shapes are not equal: {:?} != {:?}",
                $t1.shape().dims(),
                $t2.shape().dims()
            );
        }
        let mut details = String::new();
        for i in 0..$t1.shape().dims()[0] {
            for j in 0..$t1.shape().dims()[1] {
                for k in 0..$t1.shape().dims()[2] {
                    for l in 0..$t1.shape().dims()[3] {
                        for m in 0..$t1.shape().dims()[4] {
                            if !relative_eq!($t1[[i, j, k, l, m]], $t2[[i, j, k, l, m]],max_relative=1.0, epsilon = $eps) {
                                details.push_str(&format!(
                                    "{} != {} at [{}, {}, {}, {}, {}]\n",
                                    $t1[[i, j, k, l, m]],
                                    $t2[[i, j, k, l, m]],
                                    i,
                                    j,
                                    k,
                                    l,
                                    m
                                ));
                            }
                        }
                    }
                }
            }
        }
        if !details.is_empty() {
            panic!("assert_tensors_relative_eq_rank5 FAILED:\nlhs: {}\nrhs: {}\neps: {}; tensor shape: {:?}\ndetails:\n{}",
                stringify!($t1),
                stringify!($t2),
                $eps,
                $t1.shape().dims(),
                details
            );
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensors_relative_eq_rank1;
    use dfdx_core::prelude::*;

    #[test]
    #[should_panic(expected = "assert_tensors_relative_eq_rank0 FAILED")]
    fn test_assert_tensors_relative_eq_rank0() {
        let dev = dfdx_core::tensor::Cpu::default();
        let t1 = dev.tensor(1.0);
        let t2 = dev.tensor(2.0);
        assert_tensors_relative_eq_rank0!(t1, t2, 0.00001);
    }

    #[test]
    fn test_assert_tensors_relative_eq_rank0_passes() {
        let dev = dfdx_core::tensor::Cpu::default();
        let t1 = dev.tensor(1.0);
        let t2 = dev.tensor(1.0);
        assert_tensors_relative_eq_rank0!(t1, t2, 0.00001);
    }

    #[test]
    #[should_panic(expected = "assert_tensors_relative_eq_rank1 FAILED")]
    fn test_assert_tensors_relative_eq_rank1() {
        let dev = dfdx_core::tensor::Cpu::default();
        let t1 = dev.tensor([1.0, 2.0, 3.0]);
        let t2 = dev.tensor([1.0, 2.0, 4.0]);
        assert_tensors_relative_eq_rank1!(t1, t2, 0.00001);
    }

    #[test]
    fn test_assert_tensors_relative_eq_rank1_passes() {
        let dev = dfdx_core::tensor::Cpu::default();
        let t1 = dev.tensor([1.0, 2.0, 3.0]);
        let t2 = dev.tensor([1.0, 2.0, 3.0]);
        assert_tensors_relative_eq_rank1!(t1, t2, 0.00001);
    }

    #[test]
    #[should_panic(expected = "assert_tensors_relative_eq_rank2 FAILED")]
    fn test_assert_tensors_relative_eq_rank2() {
        let dev = dfdx_core::tensor::Cpu::default();
        let t1: Tensor<Rank2<2, 3>, f64, Cpu> = dev.tensor([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]);
        let t2: Tensor<Rank2<2, 3>, f64, Cpu> = dev.tensor([[1.0, 2.0, 1.0], [4.0, 4.0, 1.0]]);
        assert_tensors_relative_eq_rank2!(t1, t2, 0.00001);
    }

    #[test]
    fn test_assert_tensors_relative_eq_rank2_passes() {
        let dev = dfdx_core::tensor::Cpu::default();
        let t1: Tensor<Rank2<2, 3>, f64, Cpu> = dev.tensor([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]);
        let t2: Tensor<Rank2<2, 3>, f64, Cpu> = dev.tensor([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]);
        assert_tensors_relative_eq_rank2!(t1, t2, 0.00001);
    }

    #[test]
    #[should_panic(expected = "assert_tensors_relative_eq_rank3 FAILED")]
    fn test_assert_tensors_relative_eq_rank3() {
        let dev = dfdx_core::tensor::Cpu::default();
        let t1: Tensor<Rank3<1, 2, 3>, f64, Cpu> = dev.tensor([[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]]);
        let t2: Tensor<Rank3<1, 2, 3>, f64, Cpu> = dev.tensor([[[1.0, 2.0, 1.0], [4.0, 4.0, 1.0]]]);
        assert_tensors_relative_eq_rank3!(t1, t2, 0.00001);
    }

    #[test]
    fn test_assert_tensors_relative_eq_rank3_passes() {
        let dev = dfdx_core::tensor::Cpu::default();
        let t1: Tensor<Rank3<1, 2, 3>, f64, Cpu> = dev.tensor([[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]]);
        let t2: Tensor<Rank3<1, 2, 3>, f64, Cpu> = dev.tensor([[[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]]);
        assert_tensors_relative_eq_rank3!(t1, t2, 0.00001);
    }

    #[test]
    #[should_panic(expected = "assert_tensors_relative_eq_rank4 FAILED")]
    fn test_assert_tensors_relative_eq_rank4() {
        let dev = dfdx_core::tensor::Cpu::default();
        let t1: Tensor<Rank4<1, 2, 2, 2>, f64, Cpu> =
            dev.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]);
        let t2: Tensor<Rank4<1, 2, 2, 2>, f64, Cpu> =
            dev.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [9.0, 8.0]]]]);
        assert_tensors_relative_eq_rank4!(t1, t2, 0.00001);
    }

    #[test]
    fn test_assert_tensors_relative_eq_rank4_passes() {
        let dev = dfdx_core::tensor::Cpu::default();
        let t1: Tensor<Rank4<1, 2, 2, 2>, f64, Cpu> =
            dev.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]);
        let t2: Tensor<Rank4<1, 2, 2, 2>, f64, Cpu> =
            dev.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]);
        assert_tensors_relative_eq_rank4!(t1, t2, 0.00001);
    }
}
