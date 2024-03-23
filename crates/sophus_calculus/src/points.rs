use crate::types::scalar::IsScalar;
use crate::types::vector::IsVector;
use crate::types::vector::IsVectorLike;
use crate::types::VecF64;

/// Example points
pub fn example_points<S: IsScalar<1>, const POINT: usize>() -> Vec<S::Vector<POINT>> {
    let points4 = vec![
        VecF64::<4>::from_array([0.1, 0.0, 0.0, 0.0]),
        VecF64::<4>::from_array([1.0, 4.0, 1.0, 0.5]),
        VecF64::<4>::from_array([0.7, 5.0, 1.1, (-5.0)]),
        VecF64::<4>::from_array([1.0, 3.0, 1.0, 0.5]),
        VecF64::<4>::from_array([0.7, 5.0, 0.8, (-5.0)]),
        VecF64::<4>::from_array([1.0, 3.0, 1.0, 0.5]),
        VecF64::<4>::from_array([-0.7, 5.0, 0.1, (-5.0)]),
        VecF64::<4>::from_array([2.0, (-3.0), 1.0, 0.5]),
    ];

    let mut out: Vec<S::Vector<POINT>> = vec![];
    for p4 in points4 {
        let mut v = S::Vector::<POINT>::zero();
        for i in 0..POINT.min(4) {
            let val = p4[i];
            v.set_c(i, val);
        }
        out.push(v)
    }
    out
}
