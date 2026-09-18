@group(0) @binding(0)
var<uniform> camera: CameraProperties;
@group(0) @binding(2)
var<uniform> pinhole: PinholeModel;
@group(0) @binding(3)
var<uniform> view_uniform: CameraPose;

struct VertexOut {
    @location(0) rgba: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) barycentric: vec3<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
     @location(0) position: vec3<f32>,
     @location(1) normal: vec3<f32>,
     @location(2) color: vec4<f32>,
     @builtin(vertex_index) vertex_index: u32)-> VertexOut
{
    let projection = project_point(position, view_uniform, pinhole, camera);
    var out: VertexOut;
    out.position = pixel_and_z_to_clip(projection.uv_undistorted, projection.z, camera, pinhole);
    out.rgba = color;
    out.normal = normal;
    out.barycentric = barycentric_of_corner(vertex_index);
    return out;
}

// Note: the scene is rendered into a texture which the distortion pass composites over the
// background image, so the color has to be *premultiplied* by alpha. Multisample resolve
// averages covered samples with the transparent clear color, which yields premultiplied
// coverage - compositing that as if it were straight alpha darkens every antialiased edge.
@fragment
fn fs_main(frag: VertexOut) -> @location(0) vec4<f32> {
    let rgb = shade(frag.normal, view_uniform.light_in_entity.xyz, frag.rgba.rgb);
    var alpha = frag.rgba.a;
    if (draws_as_wireframe(pinhole.wireframe, view_uniform.wireframe)) {
        // the edges of the triangle rather than its face
        alpha = alpha * wireframe_coverage(frag.barycentric);
    }
    return vec4<f32>(rgb * alpha, alpha);
}
