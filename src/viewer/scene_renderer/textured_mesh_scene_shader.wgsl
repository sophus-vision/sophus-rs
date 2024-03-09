struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(1) tex_coords: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> frustum_uniforms: Frustum;

@group(0) @binding(1)
var<uniform> view_uniform: ViewTransform;

@group(0) @binding(2)
var<uniform> lut_uniform: DistortionLut;

@group(1) @binding(0)
var distortion_texture: texture_2d<f32>;


@vertex
fn vs_main(
     @location(0) position: vec3<f32>,
     @location(1) tex_coords: vec2<f32>)-> VertexOut
{
    var out: VertexOut;
    var uv_z = scene_point_to_distorted(position, view_uniform, frustum_uniforms, lut_uniform);
    out.position = pixel_and_z_to_clip(uv_z.xy, uv_z.z, frustum_uniforms);
    out.tex_coords = tex_coords;
    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}

@fragment
fn depth_fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.position.z, 0.0, 0.0, 1.0);
}
