use crate::prelude::*;
use sophus_core::linalg::SVec;

/// Bilinear interpolated image lookup
pub fn interpolate_impl<'a, I: IsImageView<'a, 2, 0, f32, f32, 1, 1>>(
    img: &'a I,
    uv: SVec<f32, 2>,
) -> f32 {
    let image_size = img.image_size();

    let iu = uv[0].trunc() as usize;
    let iv = uv[1].trunc() as usize;
    let frac_u: f32 = uv[0].fract();
    let frac_v: f32 = uv[1].fract();
    let u = iu;
    let v = iv;

    let u_corner_case = u == image_size.width - 1;
    let v_corner_case = v == image_size.height - 1;

    let val00: f32 = img.pixel(u, v);
    let val01: f32 = if v_corner_case {
        val00
    } else {
        img.pixel(u, v + 1)
    };
    let val10: f32 = if u_corner_case {
        val00
    } else {
        img.pixel(u + 1, v)
    };
    let val11: f32 = if u_corner_case || v_corner_case {
        val00
    } else {
        img.pixel(u + 1, v + 1)
    };

    val00 * (1.0 - frac_u) * (1.0 - frac_v)
        + val01 * ((1.0 - frac_u) * frac_v)
        + val10 * (frac_u * (1.0 - frac_v))
        + val11 * (frac_u * frac_v)
}

/// Bilinear interpolated image lookup
pub fn interpolate<
    'a,
    const ROWS: usize,
    I: IsImageView<'a, 3, 1, f32, SVec<f32, ROWS>, ROWS, 1>,
>(
    img: &'a I,
    uv: nalgebra::Vector2<f32>,
) -> SVec<f32, ROWS> {
    let image_size = img.image_size();

    let iu = uv[0].trunc() as usize;
    let iv = uv[1].trunc() as usize;
    let frac_u: f32 = uv[0].fract();
    let frac_v: f32 = uv[1].fract();
    let u = iu;
    let v = iv;

    let u_corner_case = u == image_size.width - 1;
    let v_corner_case = v == image_size.height - 1;

    let val00: SVec<f32, ROWS> = img.pixel(u, v);
    let val01: SVec<f32, ROWS> = if v_corner_case {
        val00
    } else {
        img.pixel(u, v + 1)
    };
    let val10: SVec<f32, ROWS> = if u_corner_case {
        val00
    } else {
        img.pixel(u + 1, v)
    };
    let val11: SVec<f32, ROWS> = if u_corner_case || v_corner_case {
        val00
    } else {
        img.pixel(u + 1, v + 1)
    };

    val00 * (1.0 - frac_u) * (1.0 - frac_v)
        + val01 * ((1.0 - frac_u) * frac_v)
        + val10 * (frac_u * (1.0 - frac_v))
        + val11 * (frac_u * frac_v)
}
