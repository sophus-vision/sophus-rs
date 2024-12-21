use std::borrow::Borrow;

use sophus_core::unit_vector::UnitVector;
use sophus_lie::prelude::*;
use sophus_lie::Isometry2;
use sophus_lie::Isometry3;

/// N-dimensional Hyperplane.
pub struct HyperPlane<S: IsScalar<BATCH>, const DOF: usize, const DIM: usize, const BATCH: usize> {
    /// Hyperplane origin, i.e. the origin of the (N-1)d subspace.
    pub origin: S::Vector<DIM>,
    /// Normal vector.
    pub normal: UnitVector<S, DOF, DIM, BATCH>,
}

impl<S: IsScalar<BATCH>, const DOF: usize, const DIM: usize, const BATCH: usize>
    HyperPlane<S, DOF, DIM, BATCH>
{
    /// Projects a point onto the hyperplane.
    ///
    /// Given a N-d point, this function is projecting the point onto the hyperplane (along the planar
    /// normal) and returning the result.
    pub fn proj_onto<B: Borrow<S::Vector<DIM>>>(&self, point: B) -> S::Vector<DIM> {
        let point = point.borrow();
        let diff = point.clone() - self.origin.clone();
        point.clone() - self.normal.vector().scaled(diff.dot(self.normal.vector()))
    }

    /// Returns the Jacobian of `proj_onto(point)` w.r.t. `point` itself.
    pub fn dx_proj_x_onto(&self) -> S::Matrix<DIM, DIM> {
        // The normal is constant wrt. the input point.
        let n = self.normal.vector();
        // dx (x - n * dot(x-o, n))
        // = dx x - dx  n * dot(x, n)
        // = I    - n * dot(d x, n)  [since n is constant]
        // = I    - n * n^T
        S::Matrix::<DIM, DIM>::identity() - n.outer(&n)
    }

    /// Distance of a point to the hyperplane.
    pub fn distance<B: Borrow<S::Vector<DIM>>>(&self, point: B) -> S {
        let point = point.borrow();
        (self.proj_onto(point) - point.clone()).norm()
    }
}

/// A line in 2D - represented as a 2d hyperplane.
pub type Line<S, const B: usize> = HyperPlane<S, 1, 2, B>;
/// A line in 2D - for f64.
pub type LineF64 = Line<f64, 1>;
// A plane in 3D - represented as a 3d hyperplane.
pub type Plane<S, const B: usize> = HyperPlane<S, 2, 3, B>;
/// A plane in 3D - for f64.
pub type PlaneF64 = Plane<f64, 1>;

impl<S: IsScalar<BATCH>, const BATCH: usize> Line<S, BATCH> {
    /// Converting a 2d isometry to a line.
    ///
    /// Given an isometry "parent_from_line_origin", this function creates a line spanned by the
    /// X axis of the "line_origin" frame. The line is specified relative to the "parent"
    /// frame.
    pub fn from_isometry2<B: Borrow<Isometry2<S, BATCH>>>(parent_from_line_origin: B) -> Self {
        let parent_from_line_origin = parent_from_line_origin.borrow();
        Line {
            origin: parent_from_line_origin.translation(),
            normal: UnitVector::from_vector_and_normalize(
                &parent_from_line_origin.rotation().matrix().get_col_vec(2),
            ),
        }
    }
}

impl<S: IsScalar<BATCH>, const BATCH: usize> Plane<S, BATCH> {
    /// Converting a 3d isometry to a plane.
    ///
    /// Given an isometry "parent_from_plane_origin", this function creates a plane spanned by the
    /// X-Y axis of the "plane_origin" frame. The plane is specified relative to the "parent"
    /// frame.
    pub fn from_isometry3<B: Borrow<Isometry3<S, BATCH>>>(parent_from_plane_origin: B) -> Self {
        let parent_from_plane_origin = parent_from_plane_origin.borrow();
        Plane {
            origin: parent_from_plane_origin.translation(),
            normal: UnitVector::from_vector_and_normalize(
                &parent_from_plane_origin.rotation().matrix().get_col_vec(2),
            ),
        }
    }
}

#[test]
fn plane_test() {
    use sophus_core::calculus::dual::DualScalar;
    use sophus_core::calculus::maps::VectorValuedMapFromVector;
    use sophus_core::linalg::VecF64;
    use sophus_core::linalg::EPS_F64;

    let plane = Plane::<f64, 1>::from_isometry3(Isometry3::rot_y(0.2));

    fn proj_x_onto<S: IsScalar<BATCH>, const BATCH: usize>(v: S::Vector<3>) -> S::Vector<3> {
        let plane = Plane::<S, BATCH>::from_isometry3(Isometry3::rot_y(S::from_f64(0.2)));
        plane.proj_onto(v)
    }

    let a = VecF64::<3>::new(1.0, 2.0, 3.0);
    let finite_diff = VectorValuedMapFromVector::<f64, 1>::static_sym_diff_quotient(
        proj_x_onto::<f64, 1>,
        a,
        EPS_F64,
    );
    let auto_grad = VectorValuedMapFromVector::<DualScalar, 1>::static_fw_autodiff(
        proj_x_onto::<DualScalar, 1>,
        a,
    );

    approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.00001);
    approx::assert_abs_diff_eq!(plane.dx_proj_x_onto(), auto_grad, epsilon = 0.00001);
}
