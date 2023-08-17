use super::{
    pixel::{PixelTrait, ScalarTrait, P},
    view::ImageViewTrait,
};

pub fn interpolate<
    'a,
    const NUM: usize,
    Scalar: ScalarTrait + 'static,
    I: ImageViewTrait<'a, NUM, Scalar>,
>(
    img: &'a I,
    uv: nalgebra::Vector2<f32>,
) -> P<NUM, Scalar> {
    let image_size = img.size();

    let iu = uv[0].trunc() as usize;
    let iv = uv[1].trunc() as usize;
    let frac_u = uv[0].fract();
    let frac_v = uv[1].fract();
    let u = iu;
    let v = iv;

    let u_corner_case = u == image_size.width - 1;
    let v_corner_case = v == image_size.height - 1;

    let val00 = img.pixel(u, v);
    let val01 = if v_corner_case {
        val00
    } else {
        img.pixel(u, v + 1)
    };
    let val10 = if u_corner_case {
        val00
    } else {
        img.pixel(u + 1, v)
    };
    let val11 = if u_corner_case || v_corner_case {
        val00
    } else {
        img.pixel(u + 1, v + 1)
    };

    let val = val00.scale((1.0 - frac_u) * (1.0 - frac_v))
        + val01.scale((1.0 - frac_u) * frac_v)
        + val10.scale(frac_u * (1.0 - frac_v))
        + val11.scale(frac_u * frac_v);
    val
}
