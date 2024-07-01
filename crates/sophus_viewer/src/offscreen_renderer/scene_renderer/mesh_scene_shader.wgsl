struct VertexOut {
    @location(0) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

@group(1) @binding(0)
var distortion_texture: texture_2d<f32>;


@vertex
fn vs_main(
     @location(0) position: vec3<f32>,
     @location(1) color: vec4<f32>)-> VertexOut
{
    var out: VertexOut;
    var uv_z = scene_point_to_distorted(position, view_uniform, frustum_uniforms, lut_uniform);
    out.position = pixel_and_z_to_clip(uv_z.xy, uv_z.z, frustum_uniforms, zoom);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}

@fragment
fn depth_fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.position.z, 0.0, 0.0, 1.0);
}
