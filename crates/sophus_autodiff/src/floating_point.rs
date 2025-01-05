/// floating point: f32 or f64
pub trait FloatingPointNumber:
    num_traits::Float + num_traits::FromPrimitive + num_traits::NumCast
{
}
impl FloatingPointNumber for f32 {}
impl FloatingPointNumber for f64 {}
