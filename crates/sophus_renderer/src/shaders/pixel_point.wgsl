@group(0) @binding(0)
var<uniform> camera: CameraProperties;

@group(0) @binding(1)
var<uniform> zoom_2d: Zoom2d;

@group(0) @binding(2)
var<uniform> ortho_camera: PinholeModel;

struct VertexOut {
    @location(0) color: vec4<f32>,
    // offset from the center of the point, and its half extent - both in image pixels, so that
    // the fragment shader can work out how much of the pixel the point actually covers
    @location(1) offset: vec2<f32>,
    @location(2) radius: f32,
    @location(3) feather: f32,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
     @location(0) position: vec2<f32>,
     @location(1) point_size: f32,
     @location(2) color: vec4<f32>,
     @builtin(vertex_index) idx: u32)-> VertexOut
{
    var out: VertexOut;
    // one view-port pixel, in image pixels - the width of the antialiased edge
    let feather = ortho_camera.viewport_scale;
    let point_radius = 0.5 * point_size * ortho_camera.viewport_scale;
    // the quad is grown by the feather, so that the fragments which are only partially covered
    // exist at all
    let outer = point_radius + feather;

    // 2d renderables are anchored in image pixel coordinates, so the zoom is applied to the
    // anchor only. The point radius is a view-port quantity and must not be scaled by the zoom.
    let anchor = zoom_apply(position, zoom_2d);

    let mod4 = idx % 6u;
    var offset = vec2<f32>(-outer, -outer);
    if mod4 == 1u || mod4 == 3u {
        offset = vec2<f32>(outer, -outer);
    } else if mod4 == 2u || mod4 == 4u {
        offset = vec2<f32>(-outer, outer);
    } else if mod4 == 5u {
        offset = vec2<f32>(outer, outer);
    }

    out.position = ortho_pixel_to_clip(anchor + offset,
        vec2<f32>(camera.camera_image_width, camera.camera_image_height));
    out.color = color;
    out.offset = offset;
    out.radius = point_radius;
    out.feather = feather;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // how far inside the disc this fragment is
    let inside = in.radius - length(in.offset);
    var coverage = clamp(inside / in.feather + 0.5, 0.0, 1.0);
    if (ortho_camera.wireframe > 0.5) {
        // a ring rather than a disc, as everything else is drawn as its edges
        coverage = coverage
            - clamp((inside - WIREFRAME_WIDTH * in.feather) / in.feather + 0.5, 0.0, 1.0);
    }
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
