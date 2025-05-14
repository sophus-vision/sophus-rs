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
     @location(0) position: vec3<f32>,
     @location(1) normal: vec3<f32>,
     @location(2) color: vec4<f32>)-> VertexOut
{
    let projection = project_point(position, view_uniform, pinhole, camera, zoom);
    var out: VertexOut;
    out.position = pixel_and_z_to_clip(projection.uv_undistorted, projection.z, camera, zoom);
    out.rgba = color;
    return out;
}

@fragment
fn fs_main(frag: VertexOut) -> @location(0) vec4<f32> {
    return frag.rgba;
}
