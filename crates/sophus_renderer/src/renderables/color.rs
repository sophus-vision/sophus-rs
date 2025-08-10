/// color hue
#[derive(Debug, Copy, Clone)]
pub enum ColorHue {
    /// Red
    Red,
    /// Orange
    Orange,
    /// Yellow
    Yellow,
    /// Chartreuse
    Chartreuse,
    /// Green
    Green,
    /// Turquoise
    Turquoise,
    /// Cyan
    Cyan,
    /// Azure
    Azure,
    /// Blue
    Blue,
    /// Violet
    Violet,
    /// Magenta
    Magenta,
    /// Pink
    Pink,
}

/// color hue from index
pub fn hue_from_index(index: u8) -> ColorHue {
    match index % 12 {
        0 => ColorHue::Red,
        1 => ColorHue::Orange,
        2 => ColorHue::Yellow,
        3 => ColorHue::Chartreuse,
        4 => ColorHue::Green,
        5 => ColorHue::Turquoise,
        6 => ColorHue::Cyan,
        7 => ColorHue::Azure,
        8 => ColorHue::Blue,
        9 => ColorHue::Violet,
        10 => ColorHue::Magenta,
        _ => ColorHue::Pink,
    }
}

/// color brightness
#[derive(Debug, Copy, Clone)]
pub enum ColorBrightness {
    /// Bright
    Bright,
    /// Medium
    Medium,
    /// Dark
    Dark,
}

/// color saturation
#[derive(Debug, Copy, Clone)]
pub enum ColorSaturation {
    /// Normal
    Normal,
    /// Neon
    Neon,
}

/// color
#[derive(Debug, Copy, Clone)]
pub struct Color {
    /// red
    pub r: f32,
    /// green
    pub g: f32,
    /// blue
    pub b: f32,
    /// alpha
    pub a: f32,
}

impl Color {
    /// new color
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// red
    pub fn red() -> Self {
        Self::from_hue(
            ColorHue::Red,
            ColorBrightness::Bright,
            ColorSaturation::Normal,
            1.0,
        )
    }

    /// orange
    pub fn orange() -> Self {
        Self::from_hue(
            ColorHue::Orange,
            ColorBrightness::Bright,
            ColorSaturation::Normal,
            1.0,
        )
    }

    /// yellow
    pub fn yellow() -> Self {
        Self::from_hue(
            ColorHue::Yellow,
            ColorBrightness::Bright,
            ColorSaturation::Normal,
            1.0,
        )
    }

    /// chartreuse
    pub fn chartreuse() -> Self {
        Self::from_hue(
            ColorHue::Chartreuse,
            ColorBrightness::Bright,
            ColorSaturation::Normal,
            1.0,
        )
    }

    /// green
    pub fn green() -> Self {
        Self::from_hue(
            ColorHue::Green,
            ColorBrightness::Bright,
            ColorSaturation::Normal,
            1.0,
        )
    }

    /// cyan
    pub fn cyan() -> Self {
        Self::from_hue(
            ColorHue::Cyan,
            ColorBrightness::Bright,
            ColorSaturation::Normal,
            1.0,
        )
    }

    /// azure
    pub fn azure() -> Self {
        Self::from_hue(
            ColorHue::Azure,
            ColorBrightness::Bright,
            ColorSaturation::Normal,
            1.0,
        )
    }

    /// blue
    pub fn blue() -> Self {
        Self::from_hue(
            ColorHue::Blue,
            ColorBrightness::Bright,
            ColorSaturation::Normal,
            1.0,
        )
    }

    /// violet
    pub fn violet() -> Self {
        Self::from_hue(
            ColorHue::Violet,
            ColorBrightness::Bright,
            ColorSaturation::Normal,
            1.0,
        )
    }

    /// magenta
    pub fn magenta() -> Self {
        Self::from_hue(
            ColorHue::Magenta,
            ColorBrightness::Bright,
            ColorSaturation::Normal,
            1.0,
        )
    }

    /// red variant
    pub fn red_variant(index: u8) -> Self {
        let hue = match (index / 9) % 3 {
            0 => ColorHue::Red,
            1 => ColorHue::Orange,
            _ => ColorHue::Pink,
        };

        let brightness = match index % 3 {
            0 => ColorBrightness::Bright,
            1 => ColorBrightness::Medium,
            _ => ColorBrightness::Dark,
        };

        let saturation = if (index / 3).is_multiple_of(2) {
            ColorSaturation::Normal
        } else {
            ColorSaturation::Neon
        };

        Self::from_hue(hue, brightness, saturation, 1.0)
    }

    /// green variant
    pub fn green_variant(index: u8) -> Self {
        let hue = match (index / 9) % 3 {
            0 => ColorHue::Green,
            1 => ColorHue::Chartreuse,
            _ => ColorHue::Turquoise,
        };

        let brightness = match index % 3 {
            0 => ColorBrightness::Bright,
            1 => ColorBrightness::Medium,
            _ => ColorBrightness::Dark,
        };

        let saturation = if (index / 3).is_multiple_of(2) {
            ColorSaturation::Normal
        } else {
            ColorSaturation::Neon
        };

        Self::from_hue(hue, brightness, saturation, 1.0)
    }

    /// blue variant
    pub fn blue_variant(index: u8) -> Self {
        let hue = match (index / 9) % 3 {
            0 => ColorHue::Blue,
            1 => ColorHue::Azure,
            _ => ColorHue::Cyan,
        };

        let brightness = match index % 3 {
            0 => ColorBrightness::Bright,
            1 => ColorBrightness::Medium,
            _ => ColorBrightness::Dark,
        };

        let saturation = if (index / 3).is_multiple_of(2) {
            ColorSaturation::Normal
        } else {
            ColorSaturation::Neon
        };

        Self::from_hue(hue, brightness, saturation, 1.0)
    }

    /// color variant
    pub fn color_variant(index: u8) -> Self {
        let hue = hue_from_index(index);
        let brightness = match (index / 12) % 3 {
            0 => ColorBrightness::Bright,
            1 => ColorBrightness::Medium,
            _ => ColorBrightness::Dark,
        };
        let saturation = if (index / 36).is_multiple_of(2) {
            ColorSaturation::Normal
        } else {
            ColorSaturation::Neon
        };

        Self::from_hue(hue, brightness, saturation, 1.0)
    }

    /// from hue
    pub fn from_hue(
        hue: ColorHue,
        brightness: ColorBrightness,
        saturation: ColorSaturation,
        a: f32,
    ) -> Self {
        let mut factor = 1.0;
        let mut zero = 0.0;
        let mut half = 0.5;

        match brightness {
            ColorBrightness::Medium => factor = 0.5,
            ColorBrightness::Dark => factor = 0.3,
            _ => {}
        }

        if let ColorSaturation::Neon = saturation {
            zero = 0.5;
            half = 0.75;
        }

        let base_color = match hue {
            ColorHue::Red => Self::new(1.0, zero, zero, a),
            ColorHue::Orange => Self::new(1.0, half, zero, a),
            ColorHue::Yellow => Self::new(1.0, 1.0, zero, a),
            ColorHue::Chartreuse => Self::new(half, 1.0, zero, a),
            ColorHue::Green => Self::new(zero, 1.0, zero, a),
            ColorHue::Turquoise => Self::new(zero, 1.0, half, a),
            ColorHue::Cyan => Self::new(zero, 1.0, 1.0, a),
            ColorHue::Azure => Self::new(zero, half, 1.0, a),
            ColorHue::Blue => Self::new(zero, zero, 1.0, a),
            ColorHue::Violet => Self::new(half, zero, 1.0, a),
            ColorHue::Magenta => Self::new(1.0, zero, 1.0, a),
            ColorHue::Pink => Self::new(1.0, zero, half, a),
        };
        base_color.scale(factor)
    }

    /// scale color
    pub fn scale(&self, factor: f32) -> Self {
        Self::new(self.r * factor, self.g * factor, self.b * factor, self.a)
    }

    /// black
    pub const fn black(a: f32) -> Self {
        Self::new(0.0, 0.0, 0.0, a)
    }

    /// white
    pub const fn white(a: f32) -> Self {
        Self::new(1.0, 1.0, 1.0, a)
    }

    /// bright gray
    pub const fn bright_gray(a: f32) -> Self {
        Self::new(0.75, 0.75, 0.75, a)
    }

    /// gray
    pub const fn gray(a: f32) -> Self {
        Self::new(0.5, 0.5, 0.5, a)
    }

    /// dark gray
    pub const fn dark_gray(a: f32) -> Self {
        Self::new(0.25, 0.25, 0.25, a)
    }
}
