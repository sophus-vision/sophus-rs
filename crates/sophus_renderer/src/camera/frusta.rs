//! Planar frusta covering a wide field of view.
//!
//! The scene is rasterized through an undistorted pinhole intermediate, which the distortion pass
//! then warps into the image. One plane covers a field of view of well under 180 degrees: a ray
//! at 90 degrees off axis is parallel to any plane in front of the camera, so the plane would have
//! to be infinitely wide to hold it.
//!
//! Several planes do cover it. These are the faces of a cube, of which the five with a forward
//! component are enough for a hemisphere - which is as wide as this is meant to go.

use sophus_autodiff::{
    linalg::{
        MatF64,
        VecF64,
    },
    prelude::IsMatrix,
};
use sophus_image::ImageSize;
use sophus_lie::{
    Isometry3F64,
    Rotation3F64,
};

use crate::{
    camera::RenderIntrinsics,
    prelude::*,
    types::TranslationAndScaling,
};

/// A face of the intermediate: a 90 degree frustum, identified by which axis it looks along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frustum {
    /// looking along +z, the optical axis
    Forward,
    /// looking along +x
    Right,
    /// looking along -x
    Left,
    /// looking along +y
    Down,
    /// looking along -y
    Up,
}

/// The faces which together cover the forward hemisphere.
pub const HEMISPHERE: [Frustum; 5] = [
    Frustum::Forward,
    Frustum::Right,
    Frustum::Left,
    Frustum::Down,
    Frustum::Up,
];

impl Frustum {
    /// Rotation taking a direction in the camera frame to the frame of this face, whose own
    /// optical axis is +z.
    pub fn face_from_camera(&self) -> Rotation3F64 {
        // rows of the rotation matrix: where each camera axis lands in the face frame
        let matrix = match self {
            Frustum::Forward => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            Frustum::Right => [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            Frustum::Left => [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            Frustum::Down => [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            Frustum::Up => [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        };
        Rotation3F64::try_from_mat(MatF64::<3, 3>::from_array2(matrix))
            .expect("the face rotations are rotations")
    }

    /// Pose to render this face from, given the pose of the camera.
    ///
    /// The scene renderer projects through `world_from_camera`, so handing it this puts the face's
    /// optical axis where the face looks.
    pub fn world_from_face(&self, world_from_camera: &Isometry3F64) -> Isometry3F64 {
        *world_from_camera * Isometry3F64::from_rotation(self.face_from_camera().inverse())
    }

    /// The face which observes `direction`: the one whose axis the direction is closest to, or
    /// [None] when that is the backward axis and no face of the forward hemisphere can see it.
    ///
    /// Ties on the boundary between two faces resolve to either of them; both hold the direction.
    ///
    /// Note that a direction is claimed by a side face for a little way *past* 90 degrees off the
    /// optical axis - which is where a 180 degree camera puts the corners of its image, give or
    /// take a rounding error. It lands on the very edge of that face, since being closest to the
    /// face's axis means being within 90 degrees of it, so there is no hole in the cover at the
    /// boundary - and the exact boundary is where such a camera is used.
    pub fn covering(direction: &VecF64<3>) -> Option<Frustum> {
        let (x, y, z) = (direction[0], direction[1], direction[2]);
        if -z > x.abs() && -z > y.abs() {
            return None;
        }
        // whichever axis the direction is closest to
        if z >= x.abs() && z >= y.abs() {
            Some(Frustum::Forward)
        } else if x.abs() >= y.abs() {
            Some(match x > 0.0 {
                true => Frustum::Right,
                false => Frustum::Left,
            })
        } else {
            Some(match y > 0.0 {
                true => Frustum::Down,
                false => Frustum::Up,
            })
        }
    }

    /// Index of this face among [HEMISPHERE].
    pub fn index(&self) -> usize {
        HEMISPHERE.iter().position(|f| f == self).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every direction in the forward hemisphere is held by the face which claims it - meaning it
    /// lands inside that face's 90 degree frustum.
    #[test]
    fn each_face_holds_the_directions_it_claims() {
        let steps = 40;
        let mut counts = [0usize; 5];
        for i in 0..=steps {
            for j in 0..=steps {
                // a direction sweeping the forward hemisphere
                let azimuth = core::f64::consts::TAU * i as f64 / steps as f64;
                let polar = core::f64::consts::FRAC_PI_2 * j as f64 / steps as f64;
                let direction = VecF64::<3>::new(
                    polar.sin() * azimuth.cos(),
                    polar.sin() * azimuth.sin(),
                    polar.cos(),
                );

                let face = Frustum::covering(&direction).expect("forward hemisphere");
                counts[face.index()] += 1;

                // in the face's own frame the direction must be in front, and within the 90
                // degree frustum - so |x| and |y| do not exceed z
                let in_face = face.face_from_camera().transform(direction);
                assert!(
                    in_face[2] > 0.0,
                    "{face:?} claims {direction:?} but it is behind the face"
                );
                let epsilon = 1e-9;
                assert!(
                    in_face[0].abs() <= in_face[2] + epsilon
                        && in_face[1].abs() <= in_face[2] + epsilon,
                    "{face:?} claims {direction:?}, which is outside its frustum: {in_face:?}"
                );
            }
        }
        // and the sweep actually exercised every face
        assert!(
            counts.iter().all(|c| *c > 0),
            "some face was never selected: {counts:?}"
        );
    }

    /// A direction is claimed exactly when some face can see it - which reaches a little past
    /// straight sideways, since each face spans a full 90 degrees around its own axis.
    #[test]
    fn a_direction_is_claimed_exactly_when_a_face_holds_it() {
        let held_by_some_face = |d: &VecF64<3>| {
            HEMISPHERE.iter().any(|face| {
                let in_face = face.face_from_camera().transform(*d);
                in_face[2] > 0.0
                    && in_face[0].abs() <= in_face[2] + 1e-12
                    && in_face[1].abs() <= in_face[2] + 1e-12
            })
        };

        for direction in [
            VecF64::<3>::new(0.0, 0.0, 1.0),
            // exactly sideways - the rim of a 180 degree field of view
            VecF64::<3>::new(1.0, 0.0, 0.0),
            VecF64::<3>::new(0.0, -1.0, 0.0),
            // past sideways, but still within 45 degrees of the +x axis
            VecF64::<3>::new(0.3, -0.2, -0.1),
            // straight back, and mostly back
            VecF64::<3>::new(0.0, 0.0, -1.0),
            VecF64::<3>::new(0.1, 0.1, -0.9),
        ] {
            assert_eq!(
                Frustum::covering(&direction).is_some(),
                held_by_some_face(&direction),
                "{direction:?}"
            );
        }
    }
}

/// What the scene is rasterized into, before the distortion pass warps it into the image.
#[derive(Clone, Debug)]
pub enum Intermediate {
    /// A single undistorted plane, with this zoom folded into its pinhole model.
    Plane(TranslationAndScaling),
    /// Several 90 degree frusta, sampled by ray direction - for a field of view no plane holds.
    Frusta(Vec<Frustum>),
}

/// Rays flatter than this have no usable place on any plane: it would have to be about `1 / cos`
/// wider to hold them, which is unbounded at 90 degrees.
const MIN_RAY_Z: f64 = 0.2;
/// Room for the quads of points and lines, whose extent is added around the projected position
/// and would otherwise be clipped at the border of the render target. Relative to the image,
/// since a few pixels of margin is most of a tiny one.
const MARGIN_FRACTION: f64 = 0.02;
/// Half a pixel of slack for the coverage test: the visible region is measured by unprojecting
/// and projecting again, so a view which exactly fills the plane - a pinhole camera, where that
/// round trip is the identity - lands on the boundary give or take floating point.
const COVERAGE_SLACK: f64 = 0.5;
/// Samples per axis over the visible region. Its undistorted image is curved, so the corners
/// alone neither bound it nor name every frustum it reaches.
const SAMPLES: usize = 32;

impl Intermediate {
    /// Decides what this view has to be rendered into, and with what.
    ///
    /// Both answers come out of one sweep of the visible region, so they cannot disagree about
    /// what is visible.
    pub fn choose(intrinsics: &RenderIntrinsics, zoom: TranslationAndScaling) -> Intermediate {
        let ImageSize { width, height } = intrinsics.image_size();
        let (width, height) = (width as f64, height as f64);
        let pinhole = intrinsics.pinhole_model();

        // the visible region, in unzoomed distorted image coordinates
        let min = zoom.apply_inverse(VecF64::<2>::new(0.0, 0.0));
        let max = zoom.apply_inverse(VecF64::<2>::new(width, height));

        let mut used = [false; HEMISPHERE.len()];
        let mut fits_on_a_plane = true;
        let (mut lo, mut hi) = (
            VecF64::<2>::new(f64::MAX, f64::MAX),
            VecF64::<2>::new(f64::MIN, f64::MIN),
        );

        for i in 0..=SAMPLES {
            for j in 0..=SAMPLES {
                let uv = VecF64::<2>::new(
                    min[0] + (max[0] - min[0]) * i as f64 / SAMPLES as f64,
                    min[1] + (max[1] - min[1]) * j as f64 / SAMPLES as f64,
                );
                // The question is asked of the *ray* the pixel observes: a ray at 90 degrees off
                // axis - which the enhanced unified model reaches - has no point on the z = 1
                // plane at all, and asking for one yields an ever larger number, not an answer.
                let ray = intrinsics.cam_unproj_to_unit_vector(&uv);

                if let Some(frustum) = Frustum::covering(&ray) {
                    used[frustum.index()] = true;
                }

                // where this ray lands on the undistorted plane, if it lands at all
                if ray[2] < MIN_RAY_Z {
                    fits_on_a_plane = false;
                    continue;
                }
                let undistorted = pinhole.cam_proj(ray);
                if !undistorted[0].is_finite() || !undistorted[1].is_finite() {
                    fits_on_a_plane = false;
                    continue;
                }
                lo = VecF64::<2>::new(lo[0].min(undistorted[0]), lo[1].min(undistorted[1]));
                hi = VecF64::<2>::new(hi[0].max(undistorted[0]), hi[1].max(undistorted[1]));
            }
        }

        let frusta = || {
            Intermediate::Frusta(
                HEMISPHERE
                    .iter()
                    .filter(|frustum| used[frustum.index()])
                    .copied()
                    .collect(),
            )
        };
        if !fits_on_a_plane {
            return frusta();
        }

        let (extent_x, extent_y) = (hi[0] - lo[0], hi[1] - lo[1]);
        if extent_x <= 0.0 || extent_y <= 0.0 {
            return frusta();
        }

        // Fit the undistorted region into the render target, isotropically so that points stay
        // round, and centered - but only when that *gains* resolution.
        //
        // For a distorted camera of moderate field of view the undistorted region is far
        // narrower than the image, since `pinhole_model` halves the focal length so that the
        // warp is covered at all; fitting then raises the resolution of the intermediate,
        // typically by about 2x. Without it the distortion pass magnifies the scene texture,
        // which turns its multisampled edges into visible stair steps.
        let margin = MARGIN_FRACTION * width.min(height);
        let scale = ((width - 2.0 * margin) / extent_x).min((height - 2.0 * margin) / extent_y);
        let plane = match scale.is_finite() && scale > zoom.scaling[0].max(zoom.scaling[1]) {
            true => TranslationAndScaling {
                translation: VecF64::<2>::new(
                    0.5 * width - scale * 0.5 * (lo[0] + hi[0]),
                    0.5 * height - scale * 0.5 * (lo[1] + hi[1]),
                ),
                scaling: VecF64::<2>::new(scale, scale),
            },
            false => zoom,
        };

        // Does that plane actually hold the whole view? Whatever lands outside the render target
        // is simply missing - which showed up as white corners on a camera wide enough that one
        // plane no longer reaches them. Asking this, rather than thresholding how far off axis a
        // ray may point, is what makes the switch happen exactly when it is needed.
        let corner_lo = plane.apply(lo);
        let corner_hi = plane.apply(hi);
        let covered = corner_lo[0] >= -COVERAGE_SLACK
            && corner_lo[1] >= -COVERAGE_SLACK
            && corner_hi[0] <= width + COVERAGE_SLACK
            && corner_hi[1] <= height + COVERAGE_SLACK;

        match covered {
            true => Intermediate::Plane(plane),
            false => frusta(),
        }
    }
}
