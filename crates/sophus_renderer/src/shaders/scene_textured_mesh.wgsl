@group(0) @binding(0)
var<uniform> camera: CameraProperties;
@group(0) @binding(2)
var<uniform> pinhole: PinholeModel;
@group(0) @binding(3)
var<uniform> view_uniform: CameraPose;


struct VertexOut {
    @location(0) texCoords: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) barycentric: vec3<f32>,
    @builtin(position) position: vec4<f32>,
};


@group(1) @binding(0)
var mesh_texture: texture_2d<f32>;

@group(1) @binding(1)
var mesh_texture_sampler: sampler;



@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @builtin(vertex_index) vertex_index: u32
) -> VertexOut {
    let projection = project_point(position, view_uniform, pinhole, camera);

    var out: VertexOut;
    out.position = pixel_and_z_to_clip(projection.uv_undistorted, projection.z, camera, pinhole);
    out.texCoords = tex_coords;
    out.normal = normal;
    out.barycentric = barycentric_of_corner(vertex_index);
    return out;
}

// Note: the scene is rendered into a texture which the distortion pass composites over the
// background image, so the color has to be *premultiplied* by alpha. Multisample resolve
// averages covered samples with the transparent clear color, which yields premultiplied
// coverage - compositing that as if it were straight alpha darkens every antialiased edge.
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let texel = textureSample(mesh_texture, mesh_texture_sampler, in.texCoords);
    let rgb = shade(in.normal, view_uniform.light_in_entity.xyz, texel.rgb);
    var alpha = texel.a;
    if (draws_as_wireframe(pinhole.wireframe, view_uniform.wireframe)) {
        alpha = alpha * wireframe_coverage(in.barycentric);
    }
    return vec4<f32>(rgb * alpha, alpha);
}
