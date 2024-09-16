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
    @location(1) texCoords: vec2<f32>
) -> VertexOut {
    var out: VertexOut;
    var uv_z = scene_point_to_distorted(position, view_uniform, frustum_uniforms);
    out.position = pixel_and_z_to_clip(uv_z.xy, uv_z.z, frustum_uniforms, zoom);
    out.texCoords = texCoords;  // Pass texture coordinates to the fragment shader
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(mesh_texture, mesh_texture_sampler, in.texCoords);
}
