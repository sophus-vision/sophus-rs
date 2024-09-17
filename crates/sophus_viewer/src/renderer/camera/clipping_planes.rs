use num_traits::cast;
use sophus_core::floating_point::FloatingPointNumber;

/// Clipping planes for the Wgpu renderer
#[derive(Clone, Copy, Debug)]
pub struct ClippingPlanes<S: FloatingPointNumber> {
    /// Near clipping plane
    pub near: S,
    /// Far clipping plane
    pub far: S,
}

impl Default for ClippingPlanesF64 {
    fn default() -> Self {
        ClippingPlanes {
            near: ClippingPlanes::DEFAULT_NEAR,
            far: ClippingPlanes::DEFAULT_FAR,
        }
    }
}

impl<S: FloatingPointNumber> ClippingPlanes<S> {
    /// metric z from ndc z
    pub fn metric_z_from_ndc_z(&self, ndc_z: S) -> S {
        -(self.far * self.near) / (-self.far + ndc_z * self.far - ndc_z * self.near)
    }

    /// ndc z from metric z
    pub fn ndc_z_from_metric_z(&self, z: S) -> S {
        (self.far * (z - self.near)) / (z * (self.far - self.near))
    }

    /// cast
    pub fn cast<Other: FloatingPointNumber>(self) -> ClippingPlanes<Other> {
        ClippingPlanes {
            near: cast(self.near).unwrap(),
            far: cast(self.far).unwrap(),
        }
    }
}

/// f32 clipping planes
pub type ClippingPlanesF32 = ClippingPlanes<f32>;
/// f64 clipping planes
pub type ClippingPlanesF64 = ClippingPlanes<f64>;

impl ClippingPlanesF64 {
    /// default near clipping plane
    pub const DEFAULT_NEAR: f64 = 1.0;
    /// default far clipping plane
    pub const DEFAULT_FAR: f64 = 1000.0;
}

#[test]
fn clipping_plane_tests() {
    for near in [0.1, 0.5, 1.0, 7.0] {
        for far in [10.0, 5.0, 100.0, 1000.0] {
            for ndc_z in [0.0, 0.1, 0.5, 0.7, 0.99, 1.0] {
                let clipping_planes = ClippingPlanesF64 { near, far };

                let metric_z = clipping_planes.metric_z_from_ndc_z(ndc_z);
                let roundtrip_ndc_z = clipping_planes.ndc_z_from_metric_z(metric_z);

                approx::assert_abs_diff_eq!(roundtrip_ndc_z, ndc_z, epsilon = 0.0001);
            }
        }
    }
}
