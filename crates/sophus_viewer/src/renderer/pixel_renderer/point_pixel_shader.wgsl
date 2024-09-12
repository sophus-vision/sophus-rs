struct VertexOut {
    @location(0) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
     @location(0) position: vec2<f32>,
     @location(1) point_size: f32,
     @location(2) color: vec4<f32>,
     @builtin(vertex_index) idx: u32)-> VertexOut
{
    var out: VertexOut;
    var point_radius = 0.5 * point_size * ortho_camera.viewport_scale;

    var u = position.x;
    var v = position.y;
    var mod4 = idx % 6u;
    if mod4 == 0u {
        u -= point_radius;
        v -= point_radius;
    } else if mod4 == 1u || mod4 == 3u {
        u += point_radius;
        v -= point_radius;
    } else if mod4 == 2u || mod4 == 4u {
        u -= point_radius;
        v +=point_radius;
    } else if mod4 == 5u {
        u += point_radius;
        v += point_radius;
    }

    out.position = pixel_and_z_to_clip(vec2<f32>(u, v), zoom_2d, ortho_camera);
    out.color = color;

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
