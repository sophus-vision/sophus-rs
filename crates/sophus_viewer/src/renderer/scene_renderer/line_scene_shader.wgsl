struct VertexOut {
    @location(0) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};


@vertex
fn vs_main(
     @location(0) p0: vec3<f32>,
     @location(1) p1: vec3<f32>,
     @location(2) color: vec4<f32>,
     @location(3) line_width: f32,
     @builtin(vertex_index) idx: u32)-> VertexOut
{
    var out: VertexOut;

    var uv_z0 = scene_point_to_distorted(p0, view_uniform, frustum_uniforms);
    var uv_z1 = scene_point_to_distorted(p1, view_uniform, frustum_uniforms);
    var uv0 = uv_z0.xy;
    var uv1 = uv_z1.xy;
    var d = normalize(uv0 - uv1);
    var n = vec2<f32>(-d.y, d.x);

    // var u0 = uv_z.x;
    // var v = uv_z.y;
    // var z = 0.0;

    var line_half_width = 0.5 * line_width;

    var uv = vec2<f32>(0.0, 0.0);
    var z = 0.0;
    var mod6 = idx % 6u;
    if mod6 == 0u {
        uv = uv0 + line_half_width * n;
        z = uv_z0.z;
    } else if mod6 == 1u {
        uv = uv0 - line_half_width * n;
        z = uv_z0.z;
    } else if mod6 == 2u {
        uv = uv1 + line_half_width * n;
        z = uv_z1.z;
    } else if mod6 == 3u {
        uv = uv1 - line_half_width * n;
        z = uv_z1.z;
    }
    else if mod6 == 4u {
        uv = uv1 + line_half_width * n;
        z = uv_z1.z;
    } else if mod6 == 5u {
        uv = uv0 - line_half_width * n;
        z = uv_z0.z;
    }


    // map point from pixel coordinates (Computer Vision convention) to clip space coordinates (WebGPU convention)
    out.position = pixel_and_z_to_clip(uv, z, frustum_uniforms, zoom);
    out.color = color;

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var u = i32(in.position.x);
    var v = i32(in.position.y);
    return in.color;
}
