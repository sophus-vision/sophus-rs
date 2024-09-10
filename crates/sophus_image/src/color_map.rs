use sophus_core::linalg::SVec;

/// blue to white to red to black

pub struct BlueWhiteRedBlackColorMap;

impl BlueWhiteRedBlackColorMap {
    /// f32 to rgb
    pub fn f32_to_rgb(depth: f32) -> SVec<u8, 3> {
        let depth = depth.clamp(0.0, 1.0);

        let (r, g, b) = if depth < 0.33 {
            // Transition from blue to white
            let t = depth / 0.33;
            (255.0 * t, 255.0 * t, 255.0)
        } else if depth < 0.66 {
            // Transition from white to red
            let t = (depth - 0.33) / 0.33;
            (
                255.0,             // Red stays at 255
                255.0 * (1.0 - t), // Green decreases to 0
                255.0 * (1.0 - t), // Blue stays at 0
            )
        } else {
            // Transition from red to black
            let t = (depth - 0.66) / 0.34;
            (
                255.0 * (1.0 - t), // Red decreases to 0
                0.0,               // Green stays at 0
                0.0,               // Blue stays at 0
            )
        };

        SVec::<u8, 3>::new(r as u8, g as u8, b as u8)
    }
}
