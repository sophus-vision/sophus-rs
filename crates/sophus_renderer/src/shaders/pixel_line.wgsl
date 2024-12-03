@group(0) @binding(1)
var<uniform> zoom_2d: Zoom2d;

@group(0) @binding(2)
var<uniform> ortho_camera: PinholeModel;

struct VertexOut {
    @location(0) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
     @location(0) position: vec2<f32>,
     @location(1) color: vec4<f32>,
     @location(2) normal: vec2<f32>,
     @location(3) line_width: f32,
     @builtin(vertex_index) idx: u32)-> VertexOut
{
    var out: VertexOut;

    var line_half_width = 0.5 * line_width * ortho_camera.viewport_scale;
    var p = position;
    var mod6 = idx % 6u;
    if mod6 == 0u {
        // p0
        p += normal * line_half_width;
    } else if mod6 == 1u {
        // p0
        p -= normal * line_half_width;
    } else if mod6 == 2u {
        // p1
        p += normal * line_half_width;
    } else if mod6 == 3u {
        // p1
        p -= normal * line_half_width;
    } else if mod6 == 4u {
        // p1
        p += normal * line_half_width;
    } else if mod6 == 5u {
        // p0
        p -= normal * line_half_width;
    }

    out.position = ortho_pixel_and_z_to_clip(p, zoom_2d, ortho_camera);
    out.color = color;

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
