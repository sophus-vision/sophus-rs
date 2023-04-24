use nalgebra::{SMatrix, SVector};
type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

pub struct Curve;

impl Curve {
    pub fn numeric_diff<TFn, const N: usize, const O: usize>(curve: TFn, t: f64, h: f64) -> M<N, O>
    where
        TFn: Fn(f64) -> M<N, O>,
    {
        (curve(t + h) - curve(t - h)) / (2.0 * h)
    }
}

pub struct VectorField;

impl VectorField {
    pub fn numeric_diff<TFn, const INPUT: usize, const OUTPUT: usize>(
        vector_field: TFn,
        a: V<INPUT>,
        eps: f64,
    ) -> M<OUTPUT, INPUT>
    where
        TFn: Fn(&V<INPUT>) -> V<OUTPUT>,
    {
        let mut result = M::<OUTPUT, INPUT>::zeros();
        for i in 0..INPUT {
            let mut a_plus = a.clone();
            a_plus[i] += eps;
            let mut a_minus = a.clone();
            a_minus[i] -= eps;
            let diff = vector_field(&a_plus) - vector_field(&a_minus);
            result.set_column(i, &(&diff / (2.0 * eps)));
        }
        result
    }
}
