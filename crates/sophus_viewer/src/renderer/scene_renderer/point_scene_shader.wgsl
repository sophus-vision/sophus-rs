struct VertexOut {
    @location(0) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
     @location(0) position: vec3<f32>,
     @location(1) point_size: f32,
     @location(2) color: vec4<f32>,
     @builtin(vertex_index) idx: u32)-> VertexOut
{
    var out: VertexOut;

    var uv_z = scene_point_to_distorted(position, view_uniform, frustum_uniforms);
    var u = uv_z.x;
    var v = uv_z.y;
    var z = uv_z.z;

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

    // map point from pixel coordinates (Computer Vision convention) to clip space coordinates (WebGPU convention)
    out.position = pixel_and_z_to_clip(vec2<f32>(u, v), z, frustum_uniforms, zoom);
    out.color = color;

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var u = i32(in.position.x);
    var v = i32(in.position.y);
    return in.color;
}
