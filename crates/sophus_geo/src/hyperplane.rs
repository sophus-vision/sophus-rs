use std::borrow::Borrow;

use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};
use sophus_lie::{
    prelude::*,
    Isometry2,
    Isometry2F64,
    Isometry3,
    Isometry3F64,
};

use crate::unit_vector::UnitVector;

/// N-dimensional Hyperplane.
pub struct HyperPlane<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const DIM: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    /// Hyperplane origin, i.e. the origin of the (N-1)d subspace.
    pub origin: S::Vector<DIM>,
    /// Normal vector.
    pub normal: UnitVector<S, DOF, DIM, BATCH, DM, DN>,
}

impl<
        S: IsScalar<BATCH, DM, DN>,
        const DOF: usize,
        const DIM: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
    > HyperPlane<S, DOF, DIM, BATCH, DM, DN>
{
    /// Projects a point onto the hyperplane.
    ///
    /// Given a N-d point, this function is projecting the point onto the hyperplane (along the
    /// planar normal) and returning the result.
    pub fn proj_onto<B: Borrow<S::Vector<DIM>>>(&self, point: B) -> S::Vector<DIM> {
        let point = point.borrow();
        let diff = *point - self.origin;
        *point - self.normal.vector().scaled(diff.dot(self.normal.vector()))
    }

    /// Returns the Jacobian of `proj_onto(point)` w.r.t. `point` itself.
    pub fn dx_proj_x_onto(&self) -> S::Matrix<DIM, DIM> {
        // The normal is constant wrt. the input point.
        let n = self.normal.vector();
        // dx (x - n * dot(x-o, n))
        // = dx x - dx  n * dot(x, n)
        // = I    - n * dot(d x, n)  [since n is constant]
        // = I    - n * n^T
        S::Matrix::<DIM, DIM>::identity() - n.outer(n)
    }

    /// Distance of a point to the hyperplane.
    pub fn distance<B: Borrow<S::Vector<DIM>>>(&self, point: B) -> S {
        let point = point.borrow();
        (self.proj_onto(point) - *point).norm()
    }
}

/// A line in 2D - represented as a 2d hyperplane.
pub type Line<S, const B: usize, const DM: usize, const DN: usize> = HyperPlane<S, 1, 2, B, DM, DN>;
/// A line in 2D - for f64.
pub type LineF64 = Line<f64, 1, 0, 0>;
// A plane in 3D - represented as a 3d hyperplane.
pub type Plane<S, const B: usize, const DM: usize, const DN: usize> =
    HyperPlane<S, 2, 3, B, DM, DN>;
/// A plane in 3D - for f64.
pub type PlaneF64 = Plane<f64, 1, 0, 0>;

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Line<S, BATCH, DM, DN>
{
    /// Converting a 2d isometry to a line.
    ///
    /// Given an isometry "parent_from_line_origin", this function creates a line spanned by the
    /// X axis of the "line_origin" frame. The line is specified relative to the "parent"
    /// frame.
    pub fn from_isometry2<B: Borrow<Isometry2<S, BATCH, DM, DN>>>(
        parent_from_line_origin: B,
    ) -> Self {
        let parent_from_line_origin = parent_from_line_origin.borrow();
        Line {
            origin: parent_from_line_origin.translation(),
            normal: UnitVector::from_vector_and_normalize(
                &parent_from_line_origin.rotation().matrix().get_col_vec(1),
            ),
        }
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Plane<S, BATCH, DM, DN>
{
    /// Converting a 3d isometry to a plane.
    ///
    /// Given an isometry "parent_from_plane_origin", this function creates a plane spanned by the
    /// X-Y axis of the "plane_origin" frame. The plane is specified relative to the "parent"
    /// frame.
    pub fn from_isometry3<B: Borrow<Isometry3<S, BATCH, DM, DN>>>(
        parent_from_plane_origin: B,
    ) -> Self {
        let parent_from_plane_origin = parent_from_plane_origin.borrow();
        Plane {
            origin: parent_from_plane_origin.translation(),
            normal: UnitVector::from_vector_and_normalize(
                &parent_from_plane_origin.rotation().matrix().get_col_vec(2),
            ),
        }
    }
}

impl LineF64 {
    /// Returns the Jacobian of `line.proj_onto(point)` w.r.t. the "line pose" at the identity.
    ///
    /// In more details, this is the Jacobian w.r.t. x, with x=0, T' = exp(x) * T,
    /// plane normal n = T.col(1), plane origin o = T.col(2).
    pub fn dx_proj_onto_line_at_0(
        parent_from_plane_origin: &Isometry2F64,
        point: &VecF64<2>,
    ) -> MatF64<2, 3> {
        const NORMAL_IDX: usize = 1;
        const TRANSLATION_IDX: usize = 2;

        // dx (p - n * dot(p-o, n)) |x=0, T' = exp(x) * T, n = T.col(1), o = T.col(2)
        //  = dx [-n * dot(p-o, n)]

        let m = parent_from_plane_origin.compact();
        let o0 = m.get_col_vec(TRANSLATION_IDX);
        let n0 = m.get_col_vec(NORMAL_IDX);

        let p_minus_o0 = point - o0;
        let alpha = p_minus_o0.dot(&n0);

        let df_dplane_o = n0.outer(n0);
        let df_dplane_n = -(MatF64::<2, 2>::identity().scaled(alpha) + n0.outer(p_minus_o0));

        let do_dx: MatF64<2, 3> =
            parent_from_plane_origin.dx_exp_x_time_matrix_at_0(TRANSLATION_IDX);
        let dn_dx: MatF64<2, 3> = parent_from_plane_origin.dx_exp_x_time_matrix_at_0(NORMAL_IDX);

        df_dplane_o * do_dx + df_dplane_n * dn_dx
    }
}

impl PlaneF64 {
    /// Returns the Jacobian of `plane.proj_onto(point)` w.r.t. the "plane pose" at the identity.
    ///
    /// In more details, this is the Jacobian w.r.t. x, with x=0, T' = exp(x) * T,
    /// plane normal n = T.col(2), plane origin o = T.col(3).
    pub fn dx_proj_onto_plane_at_0(
        parent_from_plane_origin: &Isometry3F64,
        point: &VecF64<3>,
    ) -> MatF64<3, 6> {
        const NORMAL_IDX: usize = 2;
        const TRANSLATION_IDX: usize = 3;

        // dx (p - n * dot(p-o, n)) |x=0, T' = exp(x) * T, n = T.col(2), o = T.col(3)
        //  = dx [-n * dot(p-o, n)]

        let m = parent_from_plane_origin.compact();
        let o0 = m.get_col_vec(3);
        let n0 = m.get_col_vec(NORMAL_IDX);

        let p_minus_o0 = point - o0;
        let alpha = p_minus_o0.dot(&n0);

        let df_dplane_o = n0.outer(n0);
        let df_dplane_n = -(MatF64::<3, 3>::identity().scaled(alpha) + n0.outer(p_minus_o0));

        let do_dx: MatF64<3, 6> =
            parent_from_plane_origin.dx_exp_x_time_matrix_at_0(TRANSLATION_IDX);
        let dn_dx: MatF64<3, 6> = parent_from_plane_origin.dx_exp_x_time_matrix_at_0(NORMAL_IDX);

        df_dplane_o * do_dx + df_dplane_n * dn_dx
    }
}

#[test]
fn plane_test() {
    use sophus_autodiff::{
        dual::{
            DualScalar,
            DualVector,
        },
        linalg::{
            VecF64,
            EPS_F64,
        },
        maps::VectorValuedVectorMap,
    };
    {
        let plane = Plane::<f64, 1, 0, 0>::from_isometry3(Isometry3::rot_y(0.2));

        fn proj_x_onto<
            S: IsScalar<BATCH, DM, DN>,
            const BATCH: usize,
            const DM: usize,
            const DN: usize,
        >(
            v: S::Vector<3>,
        ) -> S::Vector<3> {
            let plane =
                Plane::<S, BATCH, DM, DN>::from_isometry3(Isometry3::rot_y(S::from_f64(0.2)));
            plane.proj_onto(v)
        }

        let a: sophus_autodiff::nalgebra::Matrix<
            f64,
            sophus_autodiff::nalgebra::Const<3>,
            sophus_autodiff::nalgebra::Const<1>,
            sophus_autodiff::nalgebra::ArrayStorage<f64, 3, 1>,
        > = VecF64::<3>::new(1.0, 2.0, 3.0);
        let finite_diff = VectorValuedVectorMap::<f64, 1>::sym_diff_quotient_jacobian(
            proj_x_onto::<f64, 1, 0, 0>,
            a,
            EPS_F64,
        );
        let auto_grad = proj_x_onto::<DualScalar<3, 1>, 1, 3, 1>(DualVector::var(a)).jacobian();

        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.00001);
        approx::assert_abs_diff_eq!(plane.dx_proj_x_onto(), auto_grad, epsilon = 0.00001);
        {
            fn proj_x_onto<
                S: IsScalar<BATCH, DM, DN>,
                const BATCH: usize,
                const DM: usize,
                const DN: usize,
            >(
                v: S::Vector<6>,
            ) -> S::Vector<3> {
                let plane = Plane::<S, BATCH, DM, DN>::from_isometry3(
                    Isometry3::exp(v) * Isometry3::rot_y(S::from_f64(0.2)),
                );
                let a = S::Vector::<3>::from_f64_array([1.0, 2.0, 3.0]);
                plane.proj_onto(a)
            }

            let auto_grad =
                proj_x_onto::<DualScalar<6, 1>, 1, 6, 1>(DualVector::var(VecF64::zeros()))
                    .jacobian();

            approx::assert_abs_diff_eq!(
                Plane::dx_proj_onto_plane_at_0(
                    &Isometry3::rot_y(0.2),
                    &VecF64::<3>::new(1.0, 2.0, 3.0)
                ),
                auto_grad,
                epsilon = 0.00001
            );
        }

        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.00001);
        approx::assert_abs_diff_eq!(plane.dx_proj_x_onto(), auto_grad, epsilon = 0.00001);
    }
    {
        fn proj_x_onto<
            S: IsScalar<BATCH, DM, DN>,
            const BATCH: usize,
            const DM: usize,
            const DN: usize,
        >(
            v: S::Vector<3>,
        ) -> S::Vector<2> {
            let line = Line::<S, BATCH, DM, DN>::from_isometry2(
                Isometry2::exp(v) * Isometry2::rot(S::from_f64(0.2)),
            );
            let a = S::Vector::<2>::from_f64_array([1.0, 2.0]);
            line.proj_onto(a)
        }

        let auto_grad =
            proj_x_onto::<DualScalar<3, 1>, 1, 3, 1>(DualVector::var(VecF64::zeros())).jacobian();

        approx::assert_abs_diff_eq!(
            Line::dx_proj_onto_line_at_0(&Isometry2::rot(0.2), &VecF64::<2>::new(1.0, 2.0)),
            auto_grad,
            epsilon = 0.00001
        );
    }
}
