use core::{
    borrow::Borrow,
    marker::PhantomData,
    ops::Neg,
};

use sophus_autodiff::linalg::EPS_F64;
use sophus_lie::prelude::*;
extern crate alloc;

#[derive(Clone, Debug, Copy)]
struct UnitVectorImpl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phanton: PhantomData<S>,
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> UnitVectorImpl<S, DOF, DIM, BATCH, DM, DN>
{
    fn unit(i: usize) -> S::Vector<DIM> {
        assert!(i < DIM, "{i} < {DIM}");
        let mut v = S::Vector::<DIM>::zeros();
        *v.elem_mut(i) = S::from_f64(1.0);
        v
    }

    fn unit_tangent(i: usize) -> S::Vector<DOF> {
        assert!(i < DOF, "{i} < {DOF}");
        let mut v = S::Vector::<DOF>::zeros();
        *v.elem_mut(i) = S::from_f64(1.0);
        v
    }

    fn mat_rx(param: &S::Vector<DIM>) -> S::Matrix<DIM, DIM> {
        let unit_x = Self::unit(0);
        let eps = S::from_f64(EPS_F64);

        let v = *param - unit_x;
        let near_zero = (v).squared_norm().less_equal(&eps);
        let rx = S::Matrix::<DIM, DIM>::identity()
            - v.outer(v).scaled(S::from_f64(2.0) / v.squared_norm());

        S::Matrix::<DIM, DIM>::identity().select(&near_zero, rx)
    }

    fn sinc(x: S) -> S {
        let eps = S::from_f64(1e-6);
        let near_zero = x.abs().less_equal(&eps);

        (S::from_f64(1.0) - S::from_f64(1.0 / 6.0) * x * x).select(&near_zero, x.sin() / x)
    }

    fn exp(tangent: &S::Vector<DOF>) -> S::Vector<DIM> {
        let theta = tangent.norm();

        S::Vector::block_vec2(
            S::Vector::from_array([theta.cos()]),
            tangent.scaled(Self::sinc(theta)),
        )
    }

    fn log(params: &S::Vector<DIM>) -> S::Vector<DOF> {
        let eps = S::from_f64(EPS_F64);
        let unit_x = Self::unit_tangent(0);

        let x = params.elem(0);
        let tail = params.get_fixed_subvec(1);
        let theta = tail.norm();
        let near_zero = theta.abs().less_equal(&eps);

        unit_x
            .scaled(S::from_f64(0.0).atan2(x))
            .select(&near_zero, tail.scaled(theta.atan2(x) / theta))
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> IsTangent<S, DOF, BATCH, DM, DN> for UnitVectorImpl<S, DOF, DIM, BATCH, DM, DN>
{
    fn tangent_examples() -> alloc::vec::Vec<S::Vector<DOF>> {
        todo!()
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> IsParamsImpl<S, DIM, BATCH, DM, DN> for UnitVectorImpl<S, DOF, DIM, BATCH, DM, DN>
{
    fn are_params_valid(params: S::Vector<DIM>) -> S::Mask {
        let eps = S::from_f64(EPS_F64);
        (params.borrow().squared_norm() - S::from_f64(1.0))
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

/// A unit vector.
///
/// ## Generic parameters:
///
///  * S
///    - the underlying scalar such as [f64] or [sophus_autodiff::dual::DualScalar].
///  * DOF
///    - Degrees of freedom of the unit vector. A n-dimensional unit vector has n-1 degrees of
///      freedom.
///  * DIM
///    - Dimension of the unit vector. It holds that DIM == DOF + 1.
///  * BATCH
///     - Batch dimension. If S is [f64] or [sophus_autodiff::dual::DualScalar] then BATCH=1.
///  * DM, DN
///    - DM x DN is the static shape of the Jacobian to be computed if S == DualScalar<DM, DN>. If S
///      == f64, then DM==0, DN==0.
#[derive(Clone, Debug)]
pub struct UnitVector<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    params: S::Vector<DIM>,
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> Neg for UnitVector<S, DOF, DIM, BATCH, DM, DN>
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        UnitVector {
            params: -self.params,
        }
    }
}

impl<
    S: IsSingleScalar<DM, DN> + PartialOrd,
    const DOF: usize,
    const DIM: usize,
    const DM: usize,
    const DN: usize,
> UnitVector<S, DOF, DIM, 1, DM, DN>
{
    /// Function to calculate the refracted direction.
    ///
    /// Given the incident direction self, the surface_normal, and refraction_ratio, this
    /// function calculates the refracted direction.
    ///
    /// If the refracted direction is not possible (i.e. total internal reflection), then None is
    /// returned.
    pub fn refract(
        &self,
        surface_normal: UnitVector<S, DOF, DIM, 1, DM, DN>,
        refraction_ratio: S,
    ) -> Option<UnitVector<S, DOF, DIM, 1, DM, DN>> {
        let d = self.vector();
        let mut n = surface_normal.vector();

        // Compute the dot product between d and n
        let mut d_dot_n = d.dot(n);

        if d_dot_n > S::from_f64(0.0) {
            n = -n;
            d_dot_n = d.dot(n);
        }

        // Compute the perpendicular component of d
        let d_perp = d - n.scaled(d_dot_n);

        // Calculate the component of the refracted vector parallel to the surface
        let d_perp_refracted = d_perp.scaled(refraction_ratio);

        // Calculate the magnitude of the parallel component of the refracted vector
        let sqrt_term = S::from_f64(1.0)
            - refraction_ratio * refraction_ratio * (S::from_f64(1.0) - d_dot_n * d_dot_n);

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

/// 2d unit vector.
pub type UnitVector2<S, const B: usize, const DM: usize, const DN: usize> =
    UnitVector<S, 1, 2, B, DM, DN>;
/// 3d unit vector.
pub type UnitVector3<S, const B: usize, const DM: usize, const DN: usize> =
    UnitVector<S, 2, 3, B, DM, DN>;

/// Error type for near zero direction vector.
#[derive(Clone, Debug)]

pub struct NearZeroDirectionVectorError;

impl core::fmt::Display for NearZeroDirectionVectorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vector is near zero")
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> UnitVector<S, DOF, DIM, BATCH, DM, DN>
{
    /// Tries to create new unit vector from direction vector.
    ///
    /// Returns [None] if vector has not unit length.
    pub fn try_from(vector: S::Vector<DIM>) -> Option<Self> {
        if !Self::are_params_valid(vector).all() {
            return None;
        }
        Some(Self { params: vector })
    }

    /// Creates unit vector from non-zero direction vector by normalizing it.
    ///
    /// Panics if input vector is close to zero.
    pub fn from_vector_and_normalize(vector: &S::Vector<DIM>) -> Self {
        Self::try_from_vector_and_normalize(vector).unwrap()
    }

    /// Tries to create unit vector from non-zero direction vector by normalizing it.
    pub fn try_from_vector_and_normalize(
        vector: &S::Vector<DIM>,
    ) -> Result<Self, NearZeroDirectionVectorError> {
        let eps = S::from_f64(EPS_F64);
        if vector.squared_norm().less_equal(&eps).any() {
            return Err(NearZeroDirectionVectorError);
        }
        Ok(Self::from_params(vector.normalized()))
    }

    /// Returns self as vector.
    pub fn vector(&self) -> S::Vector<DIM> {
        self.params
    }

    /// Returns the angle between self and other unit vector.
    pub fn angle(&self, other: &Self) -> S {
        self.vector().dot(other.vector()).acos()
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> IsParamsImpl<S, DIM, BATCH, DM, DN> for UnitVector<S, DOF, DIM, BATCH, DM, DN>
{
    fn are_params_valid(params: S::Vector<DIM>) -> S::Mask {
        UnitVectorImpl::<S, DOF, DIM, BATCH, DM, DN>::are_params_valid(params)
    }

    fn params_examples() -> alloc::vec::Vec<S::Vector<DIM>> {
        UnitVectorImpl::<S, DOF, DIM, BATCH, DM, DN>::params_examples()
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<DIM>> {
        UnitVectorImpl::<S, DOF, DIM, BATCH, DM, DN>::invalid_params_examples()
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> HasParams<S, DIM, BATCH, DM, DN> for UnitVector<S, DOF, DIM, BATCH, DM, DN>
{
    fn from_params(params: S::Vector<DIM>) -> Self {
        Self::try_from(params).unwrap()
    }

    fn set_params(&mut self, params: S::Vector<DIM>) {
        self.params = *Self::from_params(params).params();
    }

    fn params(&self) -> &S::Vector<DIM> {
        &self.params
    }
}

impl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> IsManifold<S, DIM, DOF, BATCH, DM, DN> for UnitVector<S, DOF, DIM, BATCH, DM, DN>
{
    fn oplus(&self, tangent: &S::Vector<DOF>) -> Self {
        let params = UnitVectorImpl::<S, DOF, DIM, BATCH, DM, DN>::mat_rx(&self.params)
            * UnitVectorImpl::<S, DOF, DIM, BATCH, DM, DN>::exp(tangent);
        Self::from_params(params)
    }

    fn ominus(&self, rhs: &Self) -> S::Vector<DOF> {
        UnitVectorImpl::<S, DOF, DIM, BATCH, DM, DN>::log(
            &(UnitVectorImpl::<S, DOF, DIM, BATCH, DM, DN>::mat_rx(rhs.params()).transposed()
                * *rhs.params()),
        )
    }
}
