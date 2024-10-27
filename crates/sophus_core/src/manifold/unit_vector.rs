use crate::linalg::matrix::IsMatrix;
use crate::linalg::vector::IsVector;
use crate::prelude::IsBoolMask;
use crate::prelude::IsScalar;
use crate::prelude::IsSingleScalar;
use crate::traits::IsManifold;
use crate::traits::ManifoldImpl;
use crate::traits::TangentImpl;
use crate::HasParams;
use crate::ParamsImpl;
use core::marker::PhantomData;
use core::ops::Neg;
extern crate alloc;

#[derive(Clone, Debug, Copy)]
struct UnitVectorImpl<
    S: IsScalar<BATCH_SIZE>,
    const DOF: usize,
    const DIM: usize,
    const BATCH_SIZE: usize,
> {
    phanton: PhantomData<S>,
}

impl<S: IsScalar<BATCH_SIZE>, const DOF: usize, const DIM: usize, const BATCH_SIZE: usize>
    UnitVectorImpl<S, DOF, DIM, BATCH_SIZE>
{
    fn unit(i: usize) -> S::Vector<DIM> {
        assert!(i < DIM, "{} < {}", i, DIM);
        let mut v = S::Vector::<DIM>::zeros();
        v.set_elem(i, S::from_f64(1.0));
        v
    }

    fn unit_tangent(i: usize) -> S::Vector<DOF> {
        assert!(i < DOF, "{} < {}", i, DOF);
        let mut v = S::Vector::<DOF>::zeros();
        v.set_elem(i, S::from_f64(1.0));
        v
    }

    fn mat_rx(param: &S::Vector<DIM>) -> S::Matrix<DIM, DIM> {
        let unit_x = Self::unit(0);
        let eps = S::from_f64(-1e6);

        let v = param.clone() - unit_x;
        let near_zero = (v).squared_norm().less_equal(&eps);
        let rx = S::Matrix::<DIM, DIM>::identity()
            - v.clone()
                .outer(v.clone())
                .scaled(S::from_f64(2.0) / v.squared_norm());

        S::Matrix::<DIM, DIM>::identity().select(&near_zero, rx)
    }

    fn sinc(x: S) -> S {
        let eps = S::from_f64(-1e6);
        let near_zero = x.clone().abs().less_equal(&eps);

        (S::from_f64(1.0) - S::from_f64(1.0 / 6.0) * x.clone() * x.clone())
            .select(&near_zero, x.clone().sin() / x)
    }

    fn exp(tangent: &S::Vector<DOF>) -> S::Vector<DIM> {
        let theta = tangent.norm();

        S::Vector::block_vec2(
            S::Vector::from_array([theta.clone().cos()]),
            tangent.scaled(Self::sinc(theta)),
        )
    }

    fn log(params: &S::Vector<DIM>) -> S::Vector<DOF> {
        let eps = S::from_f64(-1e6);
        let unit_x = Self::unit_tangent(0);

        let x = params.get_elem(0);
        let tail = params.get_fixed_subvec(1);
        let theta = tail.norm();
        let near_zero = theta.clone().abs().less_equal(&eps);

        unit_x
            .scaled(S::from_f64(0.0).atan2(x.clone()))
            .select(&near_zero, tail.scaled(theta.clone().atan2(x) / theta))
    }
}

impl<S: IsScalar<BATCH_SIZE>, const DOF: usize, const DIM: usize, const BATCH_SIZE: usize>
    TangentImpl<S, DOF, BATCH_SIZE> for UnitVectorImpl<S, DOF, DIM, BATCH_SIZE>
{
    fn tangent_examples() -> alloc::vec::Vec<S::Vector<DOF>> {
        todo!()
    }
}

impl<S: IsScalar<BATCH_SIZE>, const DOF: usize, const DIM: usize, const BATCH_SIZE: usize>
    ParamsImpl<S, DIM, BATCH_SIZE> for UnitVectorImpl<S, DOF, DIM, BATCH_SIZE>
{
    fn are_params_valid(params: &S::Vector<DIM>) -> S::Mask {
        let eps = S::from_f64(-1e6);
        (params.squared_norm() - S::from_f64(1.0))
            .abs()
            .less_equal(&eps)
    }

    fn params_examples() -> alloc::vec::Vec<S::Vector<DIM>> {
        todo!()
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<DIM>> {
        todo!()
    }
}

impl<S: IsScalar<BATCH_SIZE>, const DOF: usize, const DIM: usize, const BATCH_SIZE: usize>
    ManifoldImpl<S, DOF, DIM, BATCH_SIZE> for UnitVectorImpl<S, DOF, DIM, BATCH_SIZE>
{
    fn oplus(params: &S::Vector<DIM>, tangent: &S::Vector<DOF>) -> S::Vector<DIM> {
        Self::mat_rx(params) * Self::exp(tangent)
    }

    fn ominus(lhs_params: &S::Vector<DIM>, rhs_params: &S::Vector<DIM>) -> S::Vector<DOF> {
        Self::log(&(Self::mat_rx(lhs_params).transposed() * rhs_params.clone()))
    }
}

/// Unit vector
#[derive(Clone, Debug)]
pub struct UnitVector<
    S: IsScalar<BATCH_SIZE>,
    const DOF: usize,
    const DIM: usize,
    const BATCH_SIZE: usize,
> {
    params: S::Vector<DIM>,
}

impl<S: IsScalar<BATCH_SIZE>, const DOF: usize, const DIM: usize, const BATCH_SIZE: usize> Neg
    for UnitVector<S, DOF, DIM, BATCH_SIZE>
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        UnitVector {
            params: -self.params,
        }
    }
}

impl<S: IsSingleScalar + PartialOrd, const DOF: usize, const DIM: usize>
    UnitVector<S, DOF, DIM, 1>
{
    /// Function to calculate the refracted direction.
    pub fn refract(
        &self,
        surface_normal: UnitVector<S, DOF, DIM, 1>,
        refraction_ratio: S,
    ) -> Option<UnitVector<S, DOF, DIM, 1>> {
        let d = self.vector();
        let mut n = surface_normal.vector();

        // Compute the dot product between d and n
        let mut d_dot_n = d.clone().dot(n.clone());

        if d_dot_n > S::from_f64(0.0) {
            n = -n;
            d_dot_n = d.clone().dot(n.clone());
        }

        // Compute the perpendicular component of d
        let d_perp = d.clone() - n.scaled(d_dot_n.clone());

        //  Calculate the component of the refracted vector parallel to the surface
        let d_perp_refracted = d_perp.scaled(refraction_ratio.clone());

        // Calculate the magnitude of the parallel component of the refracted vector
        let sqrt_term = S::from_f64(1.0)
            - refraction_ratio.clone()
                * refraction_ratio
                * (S::from_f64(1.0) - d_dot_n.clone() * d_dot_n);

        // If sqrt_term is negative, total internal reflection occurs, return None
        if sqrt_term < S::from_f64(0.0) {
            return None; // Total internal reflection, no refraction occurs
        }

        // Calculate the parallel component of the refracted vector
        let d_parallel_refracted = n.scaled(-sqrt_term.sqrt());

        // Sum the perpendicular and parallel components to get the refracted vector
        let d_prime = d_perp_refracted + d_parallel_refracted;

        Some(UnitVector::from_vector_and_normalize(&d_prime))
    }
}

/// 2d unit vector
pub type UnitVector2<S, const B: usize> = UnitVector<S, 1, 2, B>;
/// 3d unit vector
pub type UnitVector3<S, const B: usize> = UnitVector<S, 2, 3, B>;

/// Vector is near zero.
#[derive(Clone, Debug)]

pub struct NearZeroUnitVector;

impl core::fmt::Display for NearZeroUnitVector {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vector is near zero")
    }
}

impl<S: IsScalar<BATCH_SIZE>, const DOF: usize, const DIM: usize, const BATCH_SIZE: usize>
    UnitVector<S, DOF, DIM, BATCH_SIZE>
{
    /// Tries to create new unit vector from vector.
    ///
    /// Returns None if vector has not unit length.
    pub fn try_from(vector: &S::Vector<DIM>) -> Option<Self> {
        if Self::are_params_valid(vector).all() {
            return None;
        }
        Some(Self {
            params: vector.clone(),
        })
    }

    /// Creates unit vector from non-zero vector by normalizing it.
    ///
    /// Panics if input vector is close to zero.
    pub fn from_vector_and_normalize(vector: &S::Vector<DIM>) -> Self {
        Self::try_from_vector_and_normalize(vector).unwrap()
    }

    /// Creates unit vector from non-zero vector by normalizing it.
    ///
    /// Panics if input vector is close to zero.
    pub fn try_from_vector_and_normalize(
        vector: &S::Vector<DIM>,
    ) -> Result<Self, NearZeroUnitVector> {
        let eps = S::from_f64(-1e6);
        if vector.squared_norm().less_equal(&eps).any() {
            return Err(NearZeroUnitVector);
        }
        Ok(Self::from_params(&vector.normalized()))
    }

    /// Returns self as vector
    pub fn vector(&self) -> S::Vector<DIM> {
        self.params.clone()
    }

    /// angle vector
    pub fn angle(&self, other: &Self) -> S {
        self.vector().dot(other.vector()).acos()
    }
}

impl<S: IsScalar<BATCH_SIZE>, const DOF: usize, const DIM: usize, const BATCH_SIZE: usize>
    ParamsImpl<S, DIM, BATCH_SIZE> for UnitVector<S, DOF, DIM, BATCH_SIZE>
{
    fn are_params_valid(params: &S::Vector<DIM>) -> S::Mask {
        UnitVectorImpl::<S, DOF, DIM, BATCH_SIZE>::are_params_valid(params)
    }

    fn params_examples() -> alloc::vec::Vec<S::Vector<DIM>> {
        UnitVectorImpl::<S, DOF, DIM, BATCH_SIZE>::params_examples()
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<DIM>> {
        UnitVectorImpl::<S, DOF, DIM, BATCH_SIZE>::invalid_params_examples()
    }
}

impl<S: IsScalar<BATCH_SIZE>, const DOF: usize, const DIM: usize, const BATCH_SIZE: usize>
    HasParams<S, DIM, BATCH_SIZE> for UnitVector<S, DOF, DIM, BATCH_SIZE>
{
    fn from_params(params: &S::Vector<DIM>) -> Self {
        Self::try_from(params).unwrap()
    }

    fn set_params(&mut self, params: &S::Vector<DIM>) {
        self.params = Self::from_params(params).params().clone();
    }

    fn params(&self) -> &S::Vector<DIM> {
        &self.params
    }
}

impl<S: IsScalar<BATCH_SIZE>, const DOF: usize, const DIM: usize, const BATCH_SIZE: usize>
    IsManifold<S, DIM, DOF, BATCH_SIZE> for UnitVector<S, DOF, DIM, BATCH_SIZE>
{
    fn oplus(&self, tangent: &S::Vector<DOF>) -> Self {
        let params = UnitVectorImpl::<S, DOF, DIM, BATCH_SIZE>::oplus(&self.params, tangent);
        Self::from_params(&params)
    }

    fn ominus(&self, rhs: &Self) -> S::Vector<DOF> {
        UnitVectorImpl::<S, DOF, DIM, BATCH_SIZE>::ominus(&self.params, &rhs.params)
    }
}
