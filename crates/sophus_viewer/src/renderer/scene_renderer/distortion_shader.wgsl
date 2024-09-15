struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(vertex_index & 1u) * 2.0 - 1.0;
    let y = f32(vertex_index & 2u) - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.tex_coords = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@group(2) @binding(0) var input_texture: texture_2d<f32>;
@group(2) @binding(1) var input_sampler: sampler;

struct DistortionParams {
    k1: f32,
    k2: f32,
    p1: f32,
    p2: f32,
}

@group(2) @binding(2) var<uniform> distortion_params: DistortionParams;

fn distort(uv: vec2<f32>) -> vec2<f32> {
    let r2 = dot(uv, uv);
    let r4 = r2 * r2;
    let radial_distortion = 1.0 + distortion_params.k1 * r2 + distortion_params.k2 * r4;
    let tangential_x = 2.0 * distortion_params.p1 * uv.x * uv.y + distortion_params.p2 * (r2 + 2.0 * uv.x * uv.x);
    let tangential_y = distortion_params.p1 * (r2 + 2.0 * uv.y * uv.y) + 2.0 * distortion_params.p2 * uv.x * uv.y;
    return uv * radial_distortion + vec2<f32>(tangential_x, tangential_y);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let distorted_uv = distort(in.tex_coords * 2.0 - 1.0) * 0.5 + 0.5;
    return textureSample(input_texture, input_sampler, distorted_uv);
}