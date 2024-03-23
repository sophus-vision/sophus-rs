use crate::types::matrix::IsMatrix;
use crate::types::vector::IsVector;
use crate::types::MatF64;
use crate::types::VecF64;

use std::fmt::Debug;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

/// Scalar - either a real (f64) or a dual number
pub trait IsScalar<const BATCH_SIZE: usize>:
    PartialOrd
    + PartialEq
    + Debug
    + Clone
    + Add<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Sub<Output = Self>
    + Sized
    + Neg<Output = Self>
    + num_traits::One
    + num_traits::Zero
    + From<f64>
{
    /// Vector type
    type Vector<const ROWS: usize>: IsVector<Self, ROWS, BATCH_SIZE>;

    /// Matrix type
    type Matrix<const ROWS: usize, const COLS: usize>: IsMatrix<Self, ROWS, COLS, BATCH_SIZE>;

    /// create a constant scalar
    fn c(val: f64) -> Self;

    /// return the real part
    fn real(&self) -> f64;

    /// absolute value
    fn abs(self) -> Self;

    /// cosine
    fn cos(self) -> Self;

    /// sine
    fn sin(self) -> Self;

    /// tangent
    fn tan(self) -> Self;

    /// arccosine
    fn acos(self) -> Self;

    /// arcsine
    fn asin(self) -> Self;

    /// arctangent
    fn atan(self) -> Self;

    /// square root
    fn sqrt(self) -> Self;

    /// arctangent2
    fn atan2(self, x: Self) -> Self;

    /// value
    fn value(self) -> f64;

    /// return as a vector
    fn to_vec(self) -> Self::Vector<1>;

    /// fractional part
    fn fract(self) -> Self;

    /// floor
    fn floor(&self) -> i64;
}

impl IsScalar<1> for f64 {
    type Vector<const ROWS: usize> = VecF64<ROWS>;
    type Matrix<const ROWS: usize, const COLS: usize> = MatF64<ROWS, COLS>;

    fn abs(self) -> f64 {
        f64::abs(self)
    }

    fn cos(self) -> f64 {
        f64::cos(self)
    }

    fn sin(self) -> f64 {
        f64::sin(self)
    }

    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }

    fn c(val: f64) -> f64 {
        val
    }

    fn value(self) -> f64 {
        self
    }

    fn atan2(self, x: Self) -> Self {
        self.atan2(x)
    }

    fn real(&self) -> f64 {
        self.value()
    }

    fn to_vec(self) -> VecF64<1> {
        VecF64::<1>::new(self)
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn atan(self) -> Self {
        self.atan()
    }

    fn fract(self) -> Self {
        f64::fract(self)
    }

    fn floor(&self) -> i64 {
        f64::floor(*self) as i64
    }
}
