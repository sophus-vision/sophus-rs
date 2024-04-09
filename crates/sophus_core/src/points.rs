use crate::linalg::scalar::IsScalar;
use crate::linalg::vector::IsVector;

/// Example points
pub fn example_points<S: IsScalar<BATCH>, const POINT: usize, const BATCH: usize>(
) -> Vec<S::Vector<POINT>> {
    let points4 = vec![
        S::Vector::<4>::from_f64_array([0.1, 0.0, 0.0, 0.0]),
        S::Vector::<4>::from_f64_array([1.0, 4.0, 1.0, 0.5]),
        S::Vector::<4>::from_f64_array([0.7, 5.0, 1.1, (-5.0)]),
        S::Vector::<4>::from_f64_array([1.0, 3.0, 1.0, 0.5]),
        S::Vector::<4>::from_f64_array([0.7, 5.0, 0.8, (-5.0)]),
        S::Vector::<4>::from_f64_array([1.0, 3.0, 1.0, 0.5]),
        S::Vector::<4>::from_f64_array([-0.7, 5.0, 0.1, (-5.0)]),
        S::Vector::<4>::from_f64_array([2.0, (-3.0), 1.0, 0.5]),
    ];

    let mut out: Vec<S::Vector<POINT>> = vec![];
    for p4 in points4 {
        let mut v = S::Vector::<POINT>::zeros();
        for i in 0..POINT.min(4) {
            let val = p4.get_elem(i);
            v.set_elem(i, val);
        }
        out.push(v)
    }
    out
}
