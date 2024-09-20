@group(0) @binding(0)
var<uniform> camera: CameraProperties;
@group(0) @binding(1)
var<uniform> zoom: Zoom2d;
@group(0) @binding(2)
var<uniform> pinhole: PinholeModel;
@group(0) @binding(3)
var<uniform> view_uniform: CameraPose;


struct VertexOut {
    @location(0) texCoords: vec2<f32>,
    @builtin(position) position: vec4<f32>,
};


@group(1) @binding(0)
var mesh_texture: texture_2d<f32>;

@group(1) @binding(1)
var mesh_texture_sampler: sampler;



@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>
) -> VertexOut {
    let projection = project_point(position, view_uniform, pinhole, camera, zoom);

    var out: VertexOut;
    out.position = pixel_and_z_to_clip(projection.uv_undistorted, projection.z, camera, zoom);
    out.texCoords = tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(mesh_texture, mesh_texture_sampler, in.texCoords);
}
