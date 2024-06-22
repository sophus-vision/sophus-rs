struct VertexOut {
    @location(0) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

struct Uniforms {
     width_height: vec4<f32>
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;


@vertex
fn vs_main(
     @location(0) position: vec2<f32>,
     @location(1) point_size: f32,
     @location(2) color: vec4<f32>,
     @builtin(vertex_index) idx: u32)-> VertexOut
{
    var out: VertexOut;
        var point_radius = 0.5 * point_size;


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

    out.position = vec4<f32>(2.0 * (u+0.5) / uniforms.width_height.x - 1.0,
                             2.0 - 2.0*(v+0.5) / uniforms.width_height.y - 1.0,
                             0.0,
                             1.0);
    out.color = color;

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
