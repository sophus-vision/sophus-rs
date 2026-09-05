@group(0) @binding(0)
var<uniform> camera: CameraProperties;

@group(0) @binding(1)
var<uniform> zoom_2d: Zoom2d;

@group(0) @binding(2)
var<uniform> ortho_camera: PinholeModel;

struct VertexOut {
    @location(0) color: vec4<f32>,
    // signed distance from the center line, and its half width - both in image pixels, so that
    // the fragment shader can work out how much of the pixel the line actually covers
    @location(1) distance: f32,
    @location(2) half_width: f32,
    @location(3) feather: f32,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
     @location(0) p0: vec2<f32>,
     @location(1) p1: vec2<f32>,
     @location(2) color: vec4<f32>,
     @location(3) normal: vec2<f32>,
     @location(4) line_width: f32,
     @builtin(vertex_index) idx: u32)-> VertexOut
{
    var out: VertexOut;
    // one view-port pixel, in image pixels - the width of the antialiased edge
    let feather = ortho_camera.viewport_scale;
    let line_half_width = 0.5 * line_width * ortho_camera.viewport_scale;
    // the quad is grown by the feather, so that the fragments which are only partially covered
    // exist at all
    let outer = line_half_width + feather;

    // Six vertices span the quad of one segment: (p0+n, p0-n, p1+n) and (p0-n, p1+n, p1-n).
    let mod6 = idx % 6u;
    let at_p0 = mod6 == 0u || mod6 == 1u || mod6 == 3u;
    let sign = select(-1.0, 1.0, mod6 % 2u == 0u);

    // 2d renderables are anchored in image pixel coordinates, so the zoom is applied to the
    // anchor only. The line width is a view-port quantity and must not be scaled by the zoom.
    // (The zoom is isotropic, hence the normal stays a normal.)
    let anchor = zoom_apply(select(p1, p0, at_p0), zoom_2d);

    out.position = ortho_pixel_to_clip(anchor + normal * outer * sign,
        vec2<f32>(camera.camera_image_width, camera.camera_image_height));
    out.color = color;
    out.distance = outer * sign;
    out.half_width = line_half_width;
    out.feather = feather;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let coverage = clamp(
        (in.half_width - abs(in.distance)) / in.feather + 0.5,
        0.0,
        1.0
    );
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
