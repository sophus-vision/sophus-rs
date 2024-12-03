@group(0) @binding(0)
var<uniform> camera: CameraProperties;
@group(0) @binding(1)
var<uniform> zoom: Zoom2d;
@group(0) @binding(2)
var<uniform> pinhole: PinholeModel;
@group(0) @binding(3)
var<uniform> view_uniform: CameraPose;


struct VertexOut {
    @location(0) rgba: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};


@vertex
fn vs_main(
     @location(0) p0: vec3<f32>,
     @location(1) p1: vec3<f32>,
     @location(2) color: vec4<f32>,
     @location(3) line_width: f32,
     @builtin(vertex_index) idx: u32)-> VertexOut
{
    let projection0 = project_point(p0, view_uniform, pinhole, camera, zoom);
    let projection1 = project_point(p1, view_uniform, pinhole, camera, zoom);
    let depth0 = projection0.z;
    let depth1 = projection1.z;
    let uv0 = projection0.uv_undistorted;
    let uv1 = projection1.uv_undistorted;

    var d = normalize(uv0 - uv1);
    var n = vec2<f32>(-d.y, d.x);
    var line_half_width = 0.5 * line_width;
    var uv = vec2<f32>(0.0, 0.0);
    var z = 0.0;
    var mod6 = idx % 6u;
    if mod6 == 0u {
        uv = uv0 + line_half_width * n;
        z = depth0;
    } else if mod6 == 1u {
        uv = uv0 - line_half_width * n;
        z = depth0;
    } else if mod6 == 2u {
        uv = uv1 + line_half_width * n;
        z = depth1;
    } else if mod6 == 3u {
        uv = uv1 - line_half_width * n;
        z = depth1;
    }
    else if mod6 == 4u {
        uv = uv1 + line_half_width * n;
        z = depth1;
    } else if mod6 == 5u {
        uv = uv0 - line_half_width * n;
        z = depth0;
    }

    var out: VertexOut;
    out.position = pixel_and_z_to_clip(uv, z, camera, zoom);
    out.rgba = color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.rgba;
}
