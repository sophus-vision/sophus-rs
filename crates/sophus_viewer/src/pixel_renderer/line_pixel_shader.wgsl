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
     @location(1) color: vec4<f32>,
     @location(2) normal: vec2<f32>,
     @location(3) line_width: f32,
     @builtin(vertex_index) idx: u32)-> VertexOut
{
    var out: VertexOut;

    var line_half_width = 0.5 * line_width;
    var p = position;
    var mod6 = idx % 6u;
    if mod6 == 0u {
        // p0
        p += normal * line_half_width;
    } else if mod6 == 1u {
        // p0
        p -= normal * line_half_width;
    } else if mod6 == 2u {
        // p1
        p += normal * line_half_width;
    } else if mod6 == 3u {
        // p1
        p -= normal * line_half_width;
    } else if mod6 == 4u {
        // p1
        p += normal * line_half_width;
    } else if mod6 == 5u {
        // p0
        p -= normal * line_half_width;
    }

    out.position = vec4<f32>(2.0 * (p.x+0.5) / uniforms.width_height.x - 1.0,
                             2.0 - 2.0*(p.y+0.5) / uniforms.width_height.y - 1.0,
                             0.0,
                             1.0);
    out.color = color;

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
