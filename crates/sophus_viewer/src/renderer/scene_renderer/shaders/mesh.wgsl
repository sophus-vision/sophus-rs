@group(0) @binding(0)
var<uniform> frustum_uniforms: Frustum;
@group(0) @binding(1)
var<uniform> zoom: Zoom2d;
@group(0) @binding(3)
var<uniform> view_uniform: ViewTransform;

struct VertexOut {
    @location(0) rgba: vec4<f32>,
    @builtin(position) position: vec4<f32>,
   // @location(1) uv_distorted: vec2<f32>,
};


@vertex
fn vs_main(
     @location(0) position: vec3<f32>,
     @location(1) color: vec4<f32>)-> VertexOut
{
    let projection = project_point(position, view_uniform, frustum_uniforms, zoom);

    // store distorted image of depth values
   // textureStore(z_distorted_texture,  vec2<u32>(projection.uv), vec4<f32>(projection.z, 0.0, 0.0, 0.0));

    // produce undistorted image of colors
    var out: VertexOut;
    out.position = pixel_and_z_to_clip(projection.uv_undistorted, projection.z, frustum_uniforms, zoom);
    out.rgba = color;
   // out.uv_distorted = projection.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {

   // textureStore(z_distorted_texture,  vec2<u32>(0,0), vec4<f32>(0.7, 0.8, 0.9, 0.99));

//    textureStore(z_distorted_texture,  vec2<u32>(in.uv_distorted), vec4<f32>(0.7, 0.0, 0.0, 0.0));
    return in.rgba;
}
