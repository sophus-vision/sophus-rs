@group(0) @binding(0)
var<uniform> camera: CameraProperties;
@group(0) @binding(1)
var<uniform> zoom: Zoom2d;
@group(0) @binding(2)
var<uniform> pinhole: PinholeModel;
@group(0) @binding(3)
var<uniform> view_uniform: CameraPose;


struct VertexOut {
    @location(0) rgbd: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
     @location(0) position: vec3<f32>,
     @location(1) point_size: f32,
     @location(2) color: vec4<f32>,
     @builtin(vertex_index) idx: u32)-> VertexOut
{
    let projection = project_point(position, view_uniform, pinhole, camera, zoom);
    var u = projection.uv_undistorted.x;
    var v = projection.uv_undistorted.y;

    var point_radius = 0.5 * point_size;
    var mod4 = idx % 6u;
    if mod4 == 0u {
        u -= point_radius;
        v -= point_radius;
    } else if mod4 == 1u || mod4 == 3u {
        u += point_radius;
        v -= point_radius;
    } else if mod4 == 2u || mod4 == 4u {
        u -= point_radius;
        v += point_radius;
    } else if mod4 == 5u {
        u += point_radius;
        v += point_radius;
    }
    var out: VertexOut;
    out.position = pixel_and_z_to_clip(vec2<f32>(u, v), projection.z, camera, zoom);
    out.rgbd = color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.rgbd;
}
