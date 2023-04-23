use nalgebra::{SVector};
type V<const N: usize> = SVector<f64, N>;


pub fn example_points<const POINT: usize>() -> Vec<V<POINT>> {
    let points4 = vec![
        SVector::<f64, 4>::new(0.0, 0.0, 0.0, 0.0),
        SVector::<f64, 4>::new(1.0, 0.0, 1.0, 1.0),
        SVector::<f64, 4>::new(0.0, 5.0, 0.0, -5.0),
        SVector::<f64, 4>::new(2.0, -3.0, 1.0, 0.0),
    ];

    let mut out: Vec<V<POINT>> = vec![];
    for p4 in points4 {
        let p = SVector::<f64, POINT>::from_iterator(p4.iter().cloned());
        out.push(p);
    }
    out
}