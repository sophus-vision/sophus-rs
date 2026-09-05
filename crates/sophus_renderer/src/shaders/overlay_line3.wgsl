@group(0) @binding(0)
var<uniform> camera: CameraProperties;

@group(0) @binding(1)
var<uniform> zoom_2d: Zoom2d;

@group(0) @binding(2)
var<uniform> ortho_camera: PinholeModel;

@group(0) @binding(3)
var<uniform> view_uniform: CameraPose;

@group(1) @binding(0)
var scene_inverse_depth: texture_2d<f32>;

// How many pieces a segment is drawn in.
//
// A straight line in the scene is a *curve* in a distorted image, and the whole point of drawing
// it here rather than through the undistorted intermediate is that it keeps its width in the
// image. So it is drawn as a strip which follows that curve, each joint projected through the
// camera model in its own right. Neighbouring pieces share their joints exactly, so the strip has
// neither gaps nor overlaps to blend twice over.
const PIECES: u32 = 24u;

struct Projected {
    // where it lands in the image, with the zoom applied
    uv: vec2<f32>,
    inverse_distance: f32,
    in_front: bool,
};

fn project_world(point: vec3<f32>) -> Projected {
    var out: Projected;
    let in_camera = (view_uniform.camera_from_entity * vec4<f32>(point, 1.0)).xyz;
    out.in_front = in_camera.z > 1e-6;
    // the 2d zoom is folded into that camera model already, so it is not applied again here
    out.uv = z1_plane_to_distorted(in_camera.xy / max(in_camera.z, 1e-6), camera);
    out.inverse_distance = 1.0 / max(length(in_camera), 1e-9);
    return out;
}

struct VertexOut {
    @location(0) color: vec4<f32>,
    // signed distance from the centre line, and its half width - both in image pixels
    @location(1) distance: f32,
    @location(2) half_width: f32,
    @location(3) feather: f32,
    @location(4) inverse_distance: f32,
    @builtin(position) position: vec4<f32>,
};

// What the scene left behind is written as well as read: these belong to the scene, and the point
// under the pointer needs a distance for anything to be picked off it.
struct FragmentOut {
    @location(0) color: vec4<f32>,
    @location(1) inverse_distance: f32,
};

@vertex
fn vs_main(
     @location(0) p0: vec3<f32>,
     @location(1) p1: vec3<f32>,
     @location(2) color: vec4<f32>,
     @location(3) line_width: f32,
     @builtin(vertex_index) idx: u32) -> VertexOut
{
    var out: VertexOut;
    out.color = color;

    // Six vertices span the quad of one piece: (a+n, a-n, b+n) and (a-n, b+n, b-n).
    let piece = idx / 6u;
    let mod6 = idx % 6u;
    let at_start = mod6 == 0u || mod6 == 1u || mod6 == 3u;
    let sign = select(-1.0, 1.0, mod6 % 2u == 0u);

    let step = 1.0 / f32(PIECES);
    let t = (f32(piece) + select(1.0, 0.0, at_start)) * step;
    let here = project_world(mix(p0, p1, t));
    if (!here.in_front) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    // The way the *projected* curve runs here, taken from the joints on either side rather than
    // from the segment's own direction: it is the curve which has to be followed, and near the
    // edge of a wide image the two part company entirely.
    let before = project_world(mix(p0, p1, max(t - 0.5 * step, 0.0)));
    let after = project_world(mix(p0, p1, min(t + 0.5 * step, 1.0)));
    let along = after.uv - before.uv;
    var normal = vec2<f32>(0.0, 1.0);
    if (length(along) > 1e-9) {
        let tangent = normalize(along);
        normal = vec2<f32>(-tangent.y, tangent.x);
    }

    // one view-port pixel, in image pixels - the width of the antialiased edge
    let feather = ortho_camera.viewport_scale;
    let half_width = 0.5 * line_width * feather;
    let outer = half_width + feather;

    out.position = ortho_pixel_to_clip(here.uv + normal * outer * sign,
        vec2<f32>(camera.camera_image_width, camera.camera_image_height));
    out.distance = outer * sign;
    out.half_width = half_width;
    out.feather = feather;
    out.inverse_distance = here.inverse_distance;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    if (occluded(scene_inverse_depth, in.position.xy, in.inverse_distance)) {
        discard;
    }
    let coverage = clamp(
        (in.half_width - abs(in.distance)) / in.feather + 0.5, 0.0, 1.0);
    // A fragment which covers nothing still writes its depth, so the ones outside the shape are
    // dropped rather than left to claim the pixel.
    if (coverage <= 0.0) {
        discard;
    }

    var out: FragmentOut;
    out.color = vec4<f32>(in.color.rgb, in.color.a * coverage);
    out.inverse_distance = in.inverse_distance;
    return out;
}
